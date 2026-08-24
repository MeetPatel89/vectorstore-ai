"""Tests for the budget ledger, circuit breaker, and embedding router."""

from datetime import UTC, datetime

import pytest

from vectorstore.embeddings.policy import (
    CircuitBreaker,
    EmbeddingRouter,
    InMemoryBudgetLedger,
    NoProviderAvailableError,
    SelectionReason,
    estimate_cost_usd,
    estimate_tokens,
)

from conftest import FakeEmbedding


class FakeClock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment

    def advance_days(self, days: int) -> None:
        self.moment = datetime.fromtimestamp(
            self.moment.timestamp() + days * 86400, tz=UTC
        )

    def advance_seconds(self, seconds: float) -> None:
        self.moment = datetime.fromtimestamp(
            self.moment.timestamp() + seconds, tz=UTC
        )


def make_clock(iso: str = "2026-08-23T12:00:00+00:00") -> FakeClock:
    return FakeClock(datetime.fromisoformat(iso))


def primary_embedding() -> FakeEmbedding:
    return FakeEmbedding()


def fallback_embedding() -> FakeEmbedding:
    return FakeEmbedding(provider="st", model="fake-local", dimension=8)


class TestEstimates:
    def test_estimate_tokens_uses_four_chars_per_token(self):
        assert estimate_tokens(["a" * 40]) == 10

    def test_estimate_tokens_rounds_up_and_skips_empty(self):
        assert estimate_tokens(["abc", "", "abcde"]) == 1 + 2

    def test_estimate_cost_known_model(self):
        assert estimate_cost_usd("text-embedding-3-small", 1_000_000) == pytest.approx(
            0.02
        )

    def test_estimate_cost_unknown_model_is_zero(self):
        assert estimate_cost_usd("mystery-model", 1_000_000) == 0.0


class TestInMemoryBudgetLedger:
    def test_accumulates_within_a_day(self):
        ledger = InMemoryBudgetLedger(now=make_clock())
        ledger.record("openai", 1000, 0.01)
        ledger.record("openai", 500, 0.005)
        assert ledger.spent_today() == pytest.approx(0.015)
        assert ledger.tokens_today("openai") == 1500

    def test_day_rollover_resets_daily_but_not_monthly(self):
        clock = make_clock("2026-08-23T23:00:00+00:00")
        ledger = InMemoryBudgetLedger(now=clock)
        ledger.record("openai", 1000, 0.01)
        clock.advance_days(1)
        assert ledger.spent_today() == 0.0
        assert ledger.spent_month() == pytest.approx(0.01)

    def test_month_rollover_resets_monthly(self):
        clock = make_clock("2026-08-31T12:00:00+00:00")
        ledger = InMemoryBudgetLedger(now=clock)
        ledger.record("openai", 1000, 0.01)
        clock.advance_days(1)
        assert ledger.spent_month() == 0.0

    def test_rejects_negative_values(self):
        ledger = InMemoryBudgetLedger()
        with pytest.raises(ValueError):
            ledger.record("openai", -1, 0.0)
        with pytest.raises(ValueError):
            ledger.record("openai", 0, -0.1)


class TestCircuitBreaker:
    def test_closed_by_default(self):
        breaker = CircuitBreaker()
        assert not breaker.is_open
        assert not breaker.is_rate_limited

    def test_opens_after_threshold_consecutive_failures(self):
        breaker = CircuitBreaker(failure_threshold=3, now=make_clock())
        breaker.record_failure()
        breaker.record_failure()
        assert not breaker.is_open
        breaker.record_failure()
        assert breaker.is_open

    def test_success_resets_failure_count(self):
        breaker = CircuitBreaker(failure_threshold=2, now=make_clock())
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert not breaker.is_open

    def test_closes_after_cooldown(self):
        clock = make_clock()
        breaker = CircuitBreaker(
            failure_threshold=1, cooldown_seconds=60.0, now=clock
        )
        breaker.record_failure()
        assert breaker.is_open
        clock.advance_seconds(61)
        assert not breaker.is_open

    def test_rate_limit_window_expires(self):
        clock = make_clock()
        breaker = CircuitBreaker(now=clock)
        breaker.record_rate_limit(retry_after_seconds=30.0)
        assert breaker.is_rate_limited
        clock.advance_seconds(31)
        assert not breaker.is_rate_limited


class TestEmbeddingRouterDecisionTable:
    def make_router(self, clock: FakeClock | None = None, **kwargs) -> EmbeddingRouter:
        clock = clock or make_clock()
        kwargs.setdefault("ledger", InMemoryBudgetLedger(now=clock))
        kwargs.setdefault("breaker", CircuitBreaker(failure_threshold=1, now=clock))
        return EmbeddingRouter(
            primary_embedding(), fallback_embedding(), **kwargs
        )

    def test_default_selects_primary(self):
        selection = self.make_router().select()
        assert selection.reason is SelectionReason.PRIMARY
        assert selection.spec.provider == "fake"
        assert not selection.is_fallback

    def test_force_primary_override(self):
        selection = self.make_router(override="force_primary").select()
        assert selection.reason is SelectionReason.MANUAL_OVERRIDE
        assert selection.spec.provider == "fake"
        assert not selection.is_fallback

    def test_force_fallback_override(self):
        selection = self.make_router(override="force_fallback").select()
        assert selection.reason is SelectionReason.MANUAL_OVERRIDE
        assert selection.spec.provider == "st"

    def test_primary_disabled(self):
        selection = self.make_router(primary_enabled=False).select()
        assert selection.reason is SelectionReason.OPENAI_DISABLED
        assert selection.spec.provider == "st"
        assert selection.is_fallback

    def test_breaker_open_falls_back(self):
        router = self.make_router()
        router.record_failure()
        selection = router.select()
        assert selection.reason is SelectionReason.OPENAI_UNAVAILABLE
        assert selection.spec.provider == "st"

    def test_rate_limited_falls_back(self):
        router = self.make_router()
        router.record_rate_limit(retry_after_seconds=30.0)
        selection = router.select()
        assert selection.reason is SelectionReason.OPENAI_RATE_LIMITED

    def test_recovers_to_primary_after_cooldown(self):
        clock = make_clock()
        router = self.make_router(clock=clock)
        router.record_failure()
        clock.advance_seconds(61)
        assert router.select().reason is SelectionReason.PRIMARY

    def test_success_resets_breaker(self):
        router = self.make_router()
        router.record_failure()
        router.record_usage(tokens=100)
        assert router.select().reason is SelectionReason.PRIMARY

    def test_daily_budget_exceeded(self):
        router = self.make_router(
            daily_budget_usd=0.01, cost_per_million_tokens=0.02
        )
        router.ledger.record("fake", 600_000, 0.012)
        selection = router.select()
        assert selection.reason is SelectionReason.BUDGET_DAILY_EXCEEDED
        assert selection.spec.provider == "st"

    def test_spend_exactly_at_daily_budget_stays_primary(self):
        router = self.make_router(
            daily_budget_usd=0.01, cost_per_million_tokens=0.02
        )
        router.ledger.record("fake", 500_000, 0.01)
        assert router.select().reason is SelectionReason.PRIMARY

    def test_estimated_call_cost_can_tip_daily_budget(self):
        router = self.make_router(
            daily_budget_usd=0.01, cost_per_million_tokens=0.02
        )
        router.ledger.record("fake", 500_000, 0.01)
        selection = router.select(estimated_tokens=100_000)
        assert selection.reason is SelectionReason.BUDGET_DAILY_EXCEEDED

    def test_monthly_budget_exceeded(self):
        clock = make_clock("2026-08-23T12:00:00+00:00")
        ledger = InMemoryBudgetLedger(now=clock)
        ledger.record("fake", 0, 5.0)
        clock.advance_days(1)
        router = self.make_router(
            clock=clock,
            ledger=ledger,
            daily_budget_usd=1.0,
            monthly_budget_usd=5.0,
            cost_per_million_tokens=1.0,
        )
        selection = router.select(estimated_tokens=1_000_000)
        assert selection.reason is SelectionReason.BUDGET_MONTHLY_EXCEEDED

    def test_texts_are_estimated_for_budget_check(self):
        router = self.make_router(
            daily_budget_usd=0.000001, cost_per_million_tokens=1.0
        )
        selection = router.select(texts=["hello world, this is a query"])
        assert selection.reason is SelectionReason.BUDGET_DAILY_EXCEEDED

    def test_record_usage_computes_cost_from_rate(self):
        router = self.make_router(cost_per_million_tokens=0.02)
        router.record_usage(tokens=1_000_000)
        assert router.ledger.spent_today() == pytest.approx(0.02)


class TestEmbeddingRouterValidation:
    def test_no_fallback_raises_when_primary_rejected(self):
        router = EmbeddingRouter(primary_embedding(), primary_enabled=False)
        with pytest.raises(NoProviderAvailableError) as excinfo:
            router.select()
        assert excinfo.value.reason is SelectionReason.OPENAI_DISABLED

    def test_no_fallback_still_serves_primary(self):
        router = EmbeddingRouter(primary_embedding())
        assert router.select().reason is SelectionReason.PRIMARY

    def test_invalid_override_rejected(self):
        with pytest.raises(ValueError):
            EmbeddingRouter(primary_embedding(), override="force_openai")

    def test_force_fallback_requires_fallback(self):
        with pytest.raises(ValueError):
            EmbeddingRouter(primary_embedding(), override="force_fallback")

    def test_same_space_providers_rejected(self):
        with pytest.raises(ValueError):
            EmbeddingRouter(primary_embedding(), primary_embedding())
