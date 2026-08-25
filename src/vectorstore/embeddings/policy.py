"""Provider selection policy: budget ledger, circuit breaker, and router.

The :class:`EmbeddingRouter` is the single place where OpenAI-to-local
fallback logic lives. It makes a deterministic decision with a
machine-readable reason code, based on manual overrides, configuration,
provider availability (circuit breaker), and budget state (ledger). It is
purely threshold-based operational control, not an optimization engine.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .base import EmbeddingProvider, EmbeddingSpec

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


# Published OpenAI embedding prices in USD per one million tokens.
EMBEDDING_COST_PER_MILLION_TOKENS: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


def estimate_tokens(texts: list[str]) -> int:
    """Estimate token usage before an embedding call (~4 characters/token).

    The estimate is corrected with actual ``usage.total_tokens`` from the
    API response when usage is recorded after the call.
    """
    return sum(max(1, math.ceil(len(text) / 4)) for text in texts if text)


def estimate_cost_usd(model: str, tokens: int) -> float:
    """Estimated USD cost of embedding ``tokens`` tokens with ``model``.

    Unknown models cost 0.0 so that budget checks never block a model we
    have no price for; pass explicit rates to the router to override.
    """
    rate = EMBEDDING_COST_PER_MILLION_TOKENS.get(model, 0.0)
    return tokens / 1_000_000 * rate


class SelectionReason(StrEnum):
    """Machine-readable reason codes for every routing decision."""

    PRIMARY = "primary"
    MANUAL_OVERRIDE = "manual_override"
    OPENAI_DISABLED = "openai_disabled"
    OPENAI_UNAVAILABLE = "openai_unavailable"
    OPENAI_RATE_LIMITED = "openai_rate_limited"
    BUDGET_DAILY_EXCEEDED = "budget_daily_exceeded"
    BUDGET_MONTHLY_EXCEEDED = "budget_monthly_exceeded"


@runtime_checkable
class BudgetLedger(Protocol):
    """Track embedding usage aggregates for budget decisions."""

    def record(self, provider: str, tokens: int, usd: float) -> None:
        """Record one usage event."""

    def spent_today(self) -> float:
        """Total estimated USD spent during the current UTC day."""

    def spent_month(self) -> float:
        """Total estimated USD spent during the current UTC month."""


class InMemoryBudgetLedger:
    """Process-local :class:`BudgetLedger` with automatic day/month rollover.

    Suitable for tests and single-process deployments; a catalog-backed
    ledger provides durable aggregates across processes.
    """

    def __init__(self, now: Clock = _utc_now) -> None:
        self._now = now
        self._usd_by_day: dict[str, float] = {}
        self._tokens_by_day: dict[tuple[str, str], int] = {}

    def record(self, provider: str, tokens: int, usd: float) -> None:
        """Record one usage event in the process-local ledger."""
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        if usd < 0:
            raise ValueError("usd must not be negative")
        day = self._now().strftime("%Y-%m-%d")
        self._usd_by_day[day] = self._usd_by_day.get(day, 0.0) + usd
        key = (day, provider)
        self._tokens_by_day[key] = self._tokens_by_day.get(key, 0) + tokens

    def spent_today(self) -> float:
        """Return estimated spend for the current UTC day."""
        day = self._now().strftime("%Y-%m-%d")
        return self._usd_by_day.get(day, 0.0)

    def spent_month(self) -> float:
        """Return estimated spend for the current UTC month."""
        month = self._now().strftime("%Y-%m")
        return sum(
            usd for day, usd in self._usd_by_day.items() if day.startswith(month)
        )

    def tokens_today(self, provider: str) -> int:
        """Return tokens recorded today for *provider*."""
        day = self._now().strftime("%Y-%m-%d")
        return self._tokens_by_day.get((day, provider), 0)


class CircuitBreaker:
    """Track primary-provider availability from observed call outcomes.

    Opens after ``failure_threshold`` consecutive failures and stays open
    for ``cooldown_seconds``. Rate limits open a separate backoff window so
    routing decisions can distinguish outage from throttling.
    """

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
                moment.timestamp() + self._cooldown_seconds, tz=UTC
            )

    def record_rate_limit(self, retry_after_seconds: float | None = None) -> None:
        """Open the rate-limit window for the requested backoff."""
        backoff = retry_after_seconds or self._rate_limit_backoff_seconds
        moment = self._now()
        self._rate_limited_until = datetime.fromtimestamp(
            moment.timestamp() + backoff, tz=UTC
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
    """The outcome of one routing decision."""

    provider: EmbeddingProvider
    spec: EmbeddingSpec
    reason: SelectionReason

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
    """Deterministically select between the primary and fallback provider.

    The primary provider is normally
    :class:`~vectorstore.embeddings.openai.OpenAIEmbedding` and the fallback
    a local Sentence Transformers provider; the reason codes reflect that
    convention.

    Decision order:

    1. Manual override (``force_primary``/``force_fallback``).
    2. Primary disabled in configuration.
    3. Circuit breaker open (consecutive failures) or rate-limit backoff.
    4. Daily then monthly budget: fall back when
       ``spent + estimated_cost > budget``.
    5. Otherwise the primary provider, reason ``primary``.
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
        daily_budget_usd: float | None = None,
        monthly_budget_usd: float | None = None,
        cost_per_million_tokens: float | None = None,
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

        self._primary = primary
        self._fallback = fallback
        self._ledger = ledger or InMemoryBudgetLedger()
        self._breaker = breaker or CircuitBreaker()
        self._primary_enabled = primary_enabled
        self._override = override
        self._daily_budget_usd = daily_budget_usd
        self._monthly_budget_usd = monthly_budget_usd
        if cost_per_million_tokens is not None:
            self._cost_rate = cost_per_million_tokens
        else:
            self._cost_rate = EMBEDDING_COST_PER_MILLION_TOKENS.get(
                primary.spec.model, 0.0
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

    def select(
        self,
        purpose: str = "query",
        *,
        texts: list[str] | None = None,
        estimated_tokens: int | None = None,
    ) -> ProviderSelection:
        """Choose a provider for one embedding call.

        ``purpose`` distinguishes query-time from ingestion-time selection
        for observability; both follow the same policy. Pass ``texts`` or
        ``estimated_tokens`` so budget checks can account for the upcoming
        call, not only past spend.
        """
        if estimated_tokens is None:
            estimated_tokens = estimate_tokens(texts) if texts else 0

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

        estimated_cost = estimated_tokens / 1_000_000 * self._cost_rate
        if (
            self._daily_budget_usd is not None
            and self._ledger.spent_today() + estimated_cost > self._daily_budget_usd
        ):
            return self._use_fallback(SelectionReason.BUDGET_DAILY_EXCEEDED)
        if (
            self._monthly_budget_usd is not None
            and self._ledger.spent_month() + estimated_cost > self._monthly_budget_usd
        ):
            return self._use_fallback(SelectionReason.BUDGET_MONTHLY_EXCEEDED)

        return self._use_primary(SelectionReason.PRIMARY)

    def record_usage(self, tokens: int, usd: float | None = None) -> None:
        """Record actual primary-provider usage after a successful call.

        Prefer the exact ``usage.total_tokens`` from the API response over
        the pre-call estimate.
        """
        if usd is None:
            usd = tokens / 1_000_000 * self._cost_rate
        self._ledger.record(self._primary.spec.provider, tokens, usd)
        self._breaker.record_success()

    def record_failure(self) -> None:
        """Record a failed primary-provider call (network, 5xx, auth)."""
        self._breaker.record_failure()

    def record_rate_limit(self, retry_after_seconds: float | None = None) -> None:
        """Record a 429 from the primary provider."""
        self._breaker.record_rate_limit(retry_after_seconds)

    def _use_primary(self, reason: SelectionReason) -> ProviderSelection:
        return ProviderSelection(self._primary, self._primary.spec, reason)

    def _use_fallback(self, reason: SelectionReason) -> ProviderSelection:
        if self._fallback is None:
            raise NoProviderAvailableError(reason)
        return ProviderSelection(self._fallback, self._fallback.spec, reason)
