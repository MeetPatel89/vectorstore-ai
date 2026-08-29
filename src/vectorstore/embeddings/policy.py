"""Provider routing, precise budget accounting, and circuit breaking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from threading import RLock
from typing import Protocol, runtime_checkable
from uuid import uuid4

from .base import EmbeddingProvider, EmbeddingSpec
from .pricing import (
    DEFAULT_EMBEDDING_PRICING,
    EmbeddingCharge,
    EmbeddingPrice,
    EmbeddingPricing,
    PricingUnavailableError,
    UsdAmount,
    nanos_to_usd,
    usd_to_nanos,
)
from .pricing import (
    EMBEDDING_COST_PER_MILLION_TOKENS as EMBEDDING_COST_PER_MILLION_TOKENS,
)
from .pricing import estimate_cost_usd as estimate_cost_usd
from .tokenization import estimate_tokens as estimate_tokens

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SelectionReason(StrEnum):
    """Machine-readable reason codes for every routing decision."""

    PRIMARY = "primary"
    MANUAL_OVERRIDE = "manual_override"
    OPENAI_DISABLED = "openai_disabled"
    OPENAI_UNAVAILABLE = "openai_unavailable"
    OPENAI_RATE_LIMITED = "openai_rate_limited"
    BUDGET_DAILY_EXCEEDED = "budget_daily_exceeded"
    BUDGET_MONTHLY_EXCEEDED = "budget_monthly_exceeded"


class BudgetPeriod(StrEnum):
    """The budget window that rejected a reservation."""

    DAILY = "daily"
    MONTHLY = "monthly"


class UsageStatus(StrEnum):
    """Lifecycle state of one budget-ledger entry."""

    RESERVED = "reserved"
    COMMITTED = "committed"
    RELEASED = "released"
    EXPIRED = "expired"


@dataclass(frozen=True)
class BudgetReservation:
    """An atomic hold for a predicted embedding charge."""

    reservation_id: str
    date: str
    charge: EmbeddingCharge
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.reservation_id:
            raise ValueError("reservation_id must be a non-empty string")
        if not self.date:
            raise ValueError("date must be a non-empty string")
        if not self.charge.is_priced:
            raise ValueError("budget reservations require a priced charge")
        if self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")


@dataclass(frozen=True)
class BudgetReservationDecision:
    """The result of an atomic budget-reservation attempt."""

    reservation: BudgetReservation | None = None
    exceeded: BudgetPeriod | None = None

    def __post_init__(self) -> None:
        if (self.reservation is None) == (self.exceeded is None):
            raise ValueError("exactly one of reservation or exceeded must be provided")


@dataclass(frozen=True)
class EmbeddingUsageRecord:
    """One auditable usage or reservation event from a budget ledger."""

    event_id: str
    date: str
    charge: EmbeddingCharge
    status: UsageStatus
    expires_at: datetime | None = None


def _validate_limit_nanos(value: int | None, label: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 0
    ):
        raise ValueError(f"{label} must be a non-negative integer or None")


def _validate_ttl(value: float) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError("reservation_ttl_seconds must be finite and greater than zero")


def _validate_reconciliation(
    reservation: BudgetReservation, actual_charge: EmbeddingCharge
) -> None:
    if not actual_charge.is_priced:
        raise ValueError("a reservation must be committed with a priced charge")
    expected = reservation.charge
    for label in (
        "provider",
        "model",
        "processing_mode",
        "rate_nanos_per_million",
        "price_version",
    ):
        if getattr(expected, label) != getattr(actual_charge, label):
            raise ValueError(f"actual charge {label} does not match the reservation")


def _coerce_record_charge(
    charge_or_provider: EmbeddingCharge | str,
    tokens: int | None,
    usd: UsdAmount | None,
    *,
    model: str,
    processing_mode: str,
    price_version: str,
) -> EmbeddingCharge:
    if isinstance(charge_or_provider, EmbeddingCharge):
        if tokens is not None or usd is not None:
            raise ValueError("tokens and usd must be omitted when recording a charge")
        return charge_or_provider
    if tokens is None or usd is None:
        raise ValueError("legacy record calls require provider, tokens, and usd")
    return EmbeddingCharge.from_total_usd(
        provider=charge_or_provider,
        model=model,
        processing_mode=processing_mode,
        tokens=tokens,
        usd=usd,
        price_version=price_version,
    )


@runtime_checkable
class BudgetLedger(Protocol):
    """Atomically reserve and reconcile embedding usage charges."""

    def reserve(
        self,
        charge: EmbeddingCharge,
        *,
        daily_limit_nanos: int | None,
        monthly_limit_nanos: int | None,
        ttl_seconds: float,
    ) -> BudgetReservationDecision:
        """Atomically reserve a predicted charge under both limits."""

    def commit(
        self,
        reservation: BudgetReservation,
        actual_charge: EmbeddingCharge,
    ) -> None:
        """Replace a reservation with authoritative token usage."""

    def release(self, reservation: BudgetReservation) -> None:
        """Release an unused reservation after a failed or skipped call."""

    def record(
        self,
        charge_or_provider: EmbeddingCharge | str,
        tokens: int | None = None,
        usd: UsdAmount | None = None,
        *,
        model: str = "<unspecified>",
        processing_mode: str = "standard",
        price_version: str = "legacy-explicit-total",
    ) -> None:
        """Record committed usage, including legacy explicit-total calls."""

    def spent_today_nanos(self) -> int:
        """Return committed plus reserved nanodollars for the current UTC day."""

    def spent_month_nanos(self) -> int:
        """Return committed plus reserved nanodollars for the current UTC month."""

    def spent_today(self) -> Decimal:
        """Return exact committed plus reserved USD spend for today."""

    def spent_month(self) -> Decimal:
        """Return exact committed plus reserved USD spend for this month."""

    def tokens_today(self, provider: str, model: str | None = None) -> int:
        """Return committed tokens today, optionally filtered by model."""


class InMemoryBudgetLedger:
    """Thread-safe process-local ledger with atomic budget reservations."""

    def __init__(self, now: Clock = _utc_now) -> None:
        self._now = now
        self._entries: dict[str, EmbeddingUsageRecord] = {}
        self._lock = RLock()

    def reserve(
        self,
        charge: EmbeddingCharge,
        *,
        daily_limit_nanos: int | None,
        monthly_limit_nanos: int | None,
        ttl_seconds: float,
    ) -> BudgetReservationDecision:
        """Atomically hold a predicted charge when both budgets permit it."""
        if not charge.is_priced:
            raise ValueError("budget reservations require a priced charge")
        _validate_limit_nanos(daily_limit_nanos, "daily_limit_nanos")
        _validate_limit_nanos(monthly_limit_nanos, "monthly_limit_nanos")
        _validate_ttl(ttl_seconds)
        assert charge.charge_nanos is not None

        with self._lock:
            moment = self._now()
            self._expire_locked(moment)
            day = moment.strftime("%Y-%m-%d")
            month = moment.strftime("%Y-%m")
            if (
                daily_limit_nanos is not None
                and self._sum_nanos_locked(day=day) + charge.charge_nanos
                > daily_limit_nanos
            ):
                return BudgetReservationDecision(exceeded=BudgetPeriod.DAILY)
            if (
                monthly_limit_nanos is not None
                and self._sum_nanos_locked(month=month) + charge.charge_nanos
                > monthly_limit_nanos
            ):
                return BudgetReservationDecision(exceeded=BudgetPeriod.MONTHLY)

            reservation_id = uuid4().hex
            expires_at = datetime.fromtimestamp(
                moment.timestamp() + ttl_seconds,
                tz=UTC,
            )
            reservation = BudgetReservation(
                reservation_id=reservation_id,
                date=day,
                charge=charge,
                expires_at=expires_at,
            )
            self._entries[reservation_id] = EmbeddingUsageRecord(
                event_id=reservation_id,
                date=day,
                charge=charge,
                status=UsageStatus.RESERVED,
                expires_at=expires_at,
            )
            return BudgetReservationDecision(reservation=reservation)

    def commit(
        self,
        reservation: BudgetReservation,
        actual_charge: EmbeddingCharge,
    ) -> None:
        """Commit actual usage, replacing the reservation estimate."""
        _validate_reconciliation(reservation, actual_charge)
        with self._lock:
            entry = self._entries.get(reservation.reservation_id)
            if entry is None:
                raise ValueError("unknown budget reservation")
            if entry.status in (UsageStatus.COMMITTED, UsageStatus.RELEASED):
                raise ValueError(f"cannot commit a {entry.status} reservation")
            self._entries[reservation.reservation_id] = EmbeddingUsageRecord(
                event_id=reservation.reservation_id,
                date=reservation.date,
                charge=actual_charge,
                status=UsageStatus.COMMITTED,
            )

    def release(self, reservation: BudgetReservation) -> None:
        """Release a live reservation without deleting its audit record."""
        with self._lock:
            entry = self._entries.get(reservation.reservation_id)
            if entry is None:
                raise ValueError("unknown budget reservation")
            if entry.status in (UsageStatus.RELEASED, UsageStatus.EXPIRED):
                return
            if entry.status is UsageStatus.COMMITTED:
                raise ValueError("cannot release a committed reservation")
            self._entries[reservation.reservation_id] = EmbeddingUsageRecord(
                event_id=entry.event_id,
                date=entry.date,
                charge=entry.charge,
                status=UsageStatus.RELEASED,
                expires_at=entry.expires_at,
            )

    def record(
        self,
        charge_or_provider: EmbeddingCharge | str,
        tokens: int | None = None,
        usd: UsdAmount | None = None,
        *,
        model: str = "<unspecified>",
        processing_mode: str = "standard",
        price_version: str = "legacy-explicit-total",
    ) -> None:
        """Record committed usage directly."""
        charge = _coerce_record_charge(
            charge_or_provider,
            tokens,
            usd,
            model=model,
            processing_mode=processing_mode,
            price_version=price_version,
        )
        with self._lock:
            event_id = uuid4().hex
            self._entries[event_id] = EmbeddingUsageRecord(
                event_id=event_id,
                date=self._now().strftime("%Y-%m-%d"),
                charge=charge,
                status=UsageStatus.COMMITTED,
            )

    def spent_today_nanos(self) -> int:
        """Return committed plus reserved nanodollars for today."""
        with self._lock:
            moment = self._now()
            self._expire_locked(moment)
            return self._sum_nanos_locked(day=moment.strftime("%Y-%m-%d"))

    def spent_month_nanos(self) -> int:
        """Return committed plus reserved nanodollars for this month."""
        with self._lock:
            moment = self._now()
            self._expire_locked(moment)
            return self._sum_nanos_locked(month=moment.strftime("%Y-%m"))

    def spent_today(self) -> Decimal:
        """Return exact committed plus reserved USD spend for today."""
        return nanos_to_usd(self.spent_today_nanos())

    def spent_month(self) -> Decimal:
        """Return exact committed plus reserved USD spend for this month."""
        return nanos_to_usd(self.spent_month_nanos())

    def tokens_today(self, provider: str, model: str | None = None) -> int:
        """Return committed tokens today, optionally filtered by model."""
        with self._lock:
            day = self._now().strftime("%Y-%m-%d")
            return sum(
                entry.charge.tokens
                for entry in self._entries.values()
                if entry.date == day
                and entry.status is UsageStatus.COMMITTED
                and entry.charge.provider == provider
                and (model is None or entry.charge.model == model)
            )

    def usage_records(self) -> tuple[EmbeddingUsageRecord, ...]:
        """Return an immutable snapshot of all audit records."""
        with self._lock:
            self._expire_locked(self._now())
            return tuple(self._entries.values())

    def _expire_locked(self, moment: datetime) -> None:
        for event_id, entry in tuple(self._entries.items()):
            if (
                entry.status is UsageStatus.RESERVED
                and entry.expires_at is not None
                and entry.expires_at <= moment
            ):
                self._entries[event_id] = EmbeddingUsageRecord(
                    event_id=entry.event_id,
                    date=entry.date,
                    charge=entry.charge,
                    status=UsageStatus.EXPIRED,
                    expires_at=entry.expires_at,
                )

    def _sum_nanos_locked(
        self,
        *,
        day: str | None = None,
        month: str | None = None,
    ) -> int:
        total = 0
        for entry in self._entries.values():
            if entry.status not in (UsageStatus.RESERVED, UsageStatus.COMMITTED):
                continue
            if day is not None and entry.date != day:
                continue
            if month is not None and not entry.date.startswith(month):
                continue
            if entry.charge.charge_nanos is not None:
                total += entry.charge.charge_nanos
        return total


class CircuitBreaker:
    """Track primary-provider availability from observed call outcomes."""

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        rate_limit_backoff_seconds: float = 60.0,
        now: Clock = _utc_now,
    ) -> None:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be greater than zero")
        if cooldown_seconds <= 0 or rate_limit_backoff_seconds <= 0:
            raise ValueError("cooldown windows must be greater than zero")

        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self._now = now
        self._consecutive_failures = 0
        self._open_until: datetime | None = None
        self._rate_limited_until: datetime | None = None

    def record_success(self) -> None:
        """Close the breaker and reset consecutive failures."""
        self._consecutive_failures = 0
        self._open_until = None
        self._rate_limited_until = None

    def record_failure(self) -> None:
        """Record a failure and open the breaker at its threshold."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            moment = self._now()
            self._open_until = datetime.fromtimestamp(
                moment.timestamp() + self._cooldown_seconds,
                tz=UTC,
            )

    def record_rate_limit(self, retry_after_seconds: float | None = None) -> None:
        """Open the rate-limit window for the requested backoff."""
        backoff = retry_after_seconds or self._rate_limit_backoff_seconds
        moment = self._now()
        self._rate_limited_until = datetime.fromtimestamp(
            moment.timestamp() + backoff,
            tz=UTC,
        )

    @property
    def is_open(self) -> bool:
        """Whether the failure circuit is currently open."""
        return self._window_active(self._open_until)

    @property
    def is_rate_limited(self) -> bool:
        """Whether the rate-limit backoff window is currently active."""
        return self._window_active(self._rate_limited_until)

    def _window_active(self, until: datetime | None) -> bool:
        return until is not None and self._now() < until


@dataclass(frozen=True)
class ProviderSelection:
    """The outcome of one routing decision and any budget reservation."""

    provider: EmbeddingProvider
    spec: EmbeddingSpec
    reason: SelectionReason
    reservation: BudgetReservation | None = None

    @property
    def is_fallback(self) -> bool:
        """Whether the selection represents automatic fallback."""
        return self.reason not in (
            SelectionReason.PRIMARY,
            SelectionReason.MANUAL_OVERRIDE,
        )


class NoProviderAvailableError(RuntimeError):
    """No embedding provider can serve the request."""

    def __init__(self, reason: SelectionReason) -> None:
        super().__init__(
            f"no embedding provider available (primary rejected: {reason})"
        )
        self.reason = reason


class EmbeddingRouter:
    """Select a provider and atomically enforce embedding spend budgets.

    Budgeted calls reserve their predicted charge during selection. Call
    :meth:`record_usage` with the returned reservation after success, or pass
    it to :meth:`record_failure`/:meth:`record_rate_limit` after failure.
    """

    def __init__(
        self,
        primary: EmbeddingProvider,
        fallback: EmbeddingProvider | None = None,
        *,
        ledger: BudgetLedger | None = None,
        breaker: CircuitBreaker | None = None,
        primary_enabled: bool = True,
        override: str | None = None,
        daily_budget_usd: UsdAmount | None = None,
        monthly_budget_usd: UsdAmount | None = None,
        pricing: EmbeddingPricing | None = None,
        processing_mode: str = "standard",
        cost_per_million_tokens: UsdAmount | None = None,
        reservation_ttl_seconds: float = 300.0,
    ) -> None:
        if override not in (None, "force_primary", "force_fallback"):
            raise ValueError(
                "override must be None, 'force_primary', or 'force_fallback'"
            )
        if override == "force_fallback" and fallback is None:
            raise ValueError("override 'force_fallback' requires a fallback provider")
        if fallback is not None and fallback.spec == primary.spec:
            raise ValueError(
                "primary and fallback must belong to different embedding spaces"
            )
        if not isinstance(processing_mode, str) or not processing_mode:
            raise ValueError("processing_mode must be a non-empty string")
        if pricing is not None and cost_per_million_tokens is not None:
            raise ValueError("pass either pricing or cost_per_million_tokens, not both")
        _validate_ttl(reservation_ttl_seconds)

        self._primary = primary
        self._fallback = fallback
        self._ledger = ledger if ledger is not None else InMemoryBudgetLedger()
        self._breaker = breaker if breaker is not None else CircuitBreaker()
        self._primary_enabled = primary_enabled
        self._override = override
        self._processing_mode = processing_mode
        self._daily_limit_nanos = (
            usd_to_nanos(daily_budget_usd, label="daily_budget_usd")
            if daily_budget_usd is not None
            else None
        )
        self._monthly_limit_nanos = (
            usd_to_nanos(monthly_budget_usd, label="monthly_budget_usd")
            if monthly_budget_usd is not None
            else None
        )
        self._reservation_ttl_seconds = float(reservation_ttl_seconds)
        self._price: EmbeddingPrice | None

        if cost_per_million_tokens is not None:
            self._price = EmbeddingPrice.from_usd_per_million(
                primary.spec.provider,
                primary.spec.model,
                cost_per_million_tokens,
                version="explicit-rate",
                processing_mode=processing_mode,
            )
            self._pricing = EmbeddingPricing((self._price,))
        else:
            self._pricing = pricing or DEFAULT_EMBEDDING_PRICING
            self._price = self._pricing.get(
                primary.spec.provider,
                primary.spec.model,
                processing_mode,
            )

        if self._budgets_enabled and self._price is None:
            raise PricingUnavailableError(
                primary.spec.provider,
                primary.spec.model,
                processing_mode,
            )

    @property
    def primary(self) -> EmbeddingProvider:
        """The primary embedding provider."""
        return self._primary

    @property
    def fallback(self) -> EmbeddingProvider | None:
        """The fallback embedding provider, when configured."""
        return self._fallback

    @property
    def ledger(self) -> BudgetLedger:
        """The budget ledger used by routing decisions."""
        return self._ledger

    @property
    def breaker(self) -> CircuitBreaker:
        """The circuit breaker used by routing decisions."""
        return self._breaker

    @property
    def price(self) -> EmbeddingPrice | None:
        """The primary provider's configured price, if available."""
        return self._price

    @property
    def _budgets_enabled(self) -> bool:
        return (
            self._daily_limit_nanos is not None or self._monthly_limit_nanos is not None
        )

    def select(
        self,
        purpose: str = "query",
        *,
        texts: list[str] | None = None,
        estimated_tokens: int | None = None,
    ) -> ProviderSelection:
        """Choose a provider and reserve predicted spend when budgeted."""
        del purpose  # reserved for observability without changing policy
        if estimated_tokens is not None and (
            not isinstance(estimated_tokens, int)
            or isinstance(estimated_tokens, bool)
            or estimated_tokens < 0
        ):
            raise ValueError("estimated_tokens must be a non-negative integer")
        if self._override == "force_primary":
            return self._use_primary(SelectionReason.MANUAL_OVERRIDE)
        if self._override == "force_fallback":
            return self._use_fallback(SelectionReason.MANUAL_OVERRIDE)

        if not self._primary_enabled:
            return self._use_fallback(SelectionReason.OPENAI_DISABLED)
        if self._breaker.is_open:
            return self._use_fallback(SelectionReason.OPENAI_UNAVAILABLE)
        if self._breaker.is_rate_limited:
            return self._use_fallback(SelectionReason.OPENAI_RATE_LIMITED)
        if not self._budgets_enabled:
            return self._use_primary(SelectionReason.PRIMARY)

        if estimated_tokens is None:
            if texts is None:
                raise ValueError(
                    "texts or estimated_tokens is required when budgets are enabled"
                )
            estimated_tokens = self._primary.estimate_tokens(texts)
        assert self._price is not None
        decision = self._ledger.reserve(
            self._price.charge(estimated_tokens),
            daily_limit_nanos=self._daily_limit_nanos,
            monthly_limit_nanos=self._monthly_limit_nanos,
            ttl_seconds=self._reservation_ttl_seconds,
        )
        if decision.exceeded is BudgetPeriod.DAILY:
            return self._use_fallback(SelectionReason.BUDGET_DAILY_EXCEEDED)
        if decision.exceeded is BudgetPeriod.MONTHLY:
            return self._use_fallback(SelectionReason.BUDGET_MONTHLY_EXCEEDED)
        assert decision.reservation is not None
        return self._use_primary(
            SelectionReason.PRIMARY,
            reservation=decision.reservation,
        )

    def record_usage(
        self,
        tokens: int,
        usd: UsdAmount | None = None,
        *,
        reservation: BudgetReservation | None = None,
    ) -> EmbeddingCharge:
        """Record authoritative tokens and reconcile an optional reservation."""
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise ValueError("tokens must be a non-negative integer")
        if reservation is not None:
            if usd is not None:
                raise ValueError("usd cannot override a reserved price")
            reserved = reservation.charge
            assert reserved.rate_nanos_per_million is not None
            assert reserved.price_version is not None
            price = EmbeddingPrice(
                provider=reserved.provider,
                model=reserved.model,
                processing_mode=reserved.processing_mode,
                rate_nanos_per_million=reserved.rate_nanos_per_million,
                version=reserved.price_version,
            )
            charge = price.charge(tokens)
            self._ledger.commit(reservation, charge)
        elif usd is not None:
            charge = EmbeddingCharge.from_total_usd(
                self._primary.spec.provider,
                self._primary.spec.model,
                self._processing_mode,
                tokens,
                usd,
            )
            self._ledger.record(charge)
        elif self._price is not None:
            charge = self._price.charge(tokens)
            self._ledger.record(charge)
        else:
            charge = EmbeddingCharge.unpriced(
                self._primary.spec.provider,
                self._primary.spec.model,
                self._processing_mode,
                tokens,
            )
            self._ledger.record(charge)
        self._breaker.record_success()
        return charge

    def release_reservation(self, reservation: BudgetReservation | None) -> None:
        """Release a reservation when a selected primary call is skipped."""
        if reservation is not None:
            self._ledger.release(reservation)

    def record_failure(
        self,
        reservation: BudgetReservation | None = None,
    ) -> None:
        """Release predicted spend and record a primary-provider failure."""
        try:
            self.release_reservation(reservation)
        finally:
            self._breaker.record_failure()

    def record_rate_limit(
        self,
        retry_after_seconds: float | None = None,
        reservation: BudgetReservation | None = None,
    ) -> None:
        """Release predicted spend and record a primary-provider 429."""
        try:
            self.release_reservation(reservation)
        finally:
            self._breaker.record_rate_limit(retry_after_seconds)

    def _use_primary(
        self,
        reason: SelectionReason,
        *,
        reservation: BudgetReservation | None = None,
    ) -> ProviderSelection:
        return ProviderSelection(
            self._primary,
            self._primary.spec,
            reason,
            reservation,
        )

    def _use_fallback(self, reason: SelectionReason) -> ProviderSelection:
        if self._fallback is None:
            raise NoProviderAvailableError(reason)
        return ProviderSelection(self._fallback, self._fallback.spec, reason)
