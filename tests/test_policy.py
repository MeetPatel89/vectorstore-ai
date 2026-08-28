"""Tests for the budget ledger, circuit breaker, and embedding router."""

from datetime import UTC, datetime
from typing import override

import pytest
from conftest import FakeEmbedding

from vectorstore import TokenCountingUnavailableError
from vectorstore.embeddings.policy import (
    BudgetLedger,
    CircuitBreaker,
    EmbeddingRouter,
    InMemoryBudgetLedger,
    NoProviderAvailableError,
    SelectionReason,
    estimate_cost_usd,
    estimate_tokens,
)


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
        self.moment = datetime.fromtimestamp(self.moment.timestamp() + seconds, tz=UTC)


def make_clock(iso: str = "2026-08-23T12:00:00+00:00") -> FakeClock:
    return FakeClock(datetime.fromisoformat(iso))


def primary_embedding() -> FakeEmbedding:
    return FakeEmbedding()


def fallback_embedding() -> FakeEmbedding:
    return FakeEmbedding(provider="st", model="fake-local", dimension=8)


class TestEstimates:
    def test_estimate_tokens_uses_model_tokenizer(self) -> None:
        assert estimate_tokens(["users can't log in after certificate rotation"]) == 8

    def test_estimate_tokens_handles_identifiers_and_multilingual_text(self) -> None:
        assert estimate_tokens(["INC-1104"]) == 4
        assert estimate_tokens(["业务层 API 客户端收到过多请求"]) == 14

    def test_estimate_tokens_supports_explicit_encoding(self) -> None:
        assert (
            estimate_tokens(
                ["INC-1104"],
                model="custom-embedding-model",
                encoding_name="cl100k_base",
            )
            == 4
        )

    def test_unknown_model_requires_explicit_encoding(self) -> None:
        with pytest.raises(TokenCountingUnavailableError, match="encoding_name"):
            estimate_tokens(["hello"], model="custom-embedding-model")

    def test_invalid_explicit_encoding_raises_typed_error(self) -> None:
        with pytest.raises(TokenCountingUnavailableError, match="missing-encoding"):
            estimate_tokens(
                ["hello"],
                model="custom-embedding-model",
                encoding_name="missing-encoding",
            )

    def test_empty_input_has_no_tokens(self) -> None:
        assert estimate_tokens([""]) == 0

    def test_estimate_cost_known_model(self) -> None:
        assert estimate_cost_usd("text-embedding-3-small", 1_000_000) == pytest.approx(
            0.02
        )

    def test_estimate_cost_unknown_model_is_zero(self) -> None:
        assert estimate_cost_usd("mystery-model", 1_000_000) == 0.0


class TestInMemoryBudgetLedger:
    def test_accumulates_within_a_day(self) -> None:
        ledger = InMemoryBudgetLedger(now=make_clock())
        ledger.record("openai", 1000, 0.01)
        ledger.record("openai", 500, 0.005)
        assert ledger.spent_today() == pytest.approx(0.015)
        assert ledger.tokens_today("openai") == 1500

    def test_day_rollover_resets_daily_but_not_monthly(self) -> None:
        clock = make_clock("2026-08-23T23:00:00+00:00")
        ledger = InMemoryBudgetLedger(now=clock)
        ledger.record("openai", 1000, 0.01)
        clock.advance_days(1)
        assert ledger.spent_today() == 0.0
        assert ledger.spent_month() == pytest.approx(0.01)

    def test_month_rollover_resets_monthly(self) -> None:
        clock = make_clock("2026-08-31T12:00:00+00:00")
        ledger = InMemoryBudgetLedger(now=clock)
        ledger.record("openai", 1000, 0.01)
        clock.advance_days(1)
        assert ledger.spent_month() == 0.0

    def test_rejects_negative_values(self) -> None:
        ledger = InMemoryBudgetLedger()
        with pytest.raises(ValueError):
            ledger.record("openai", -1, 0.0)
        with pytest.raises(ValueError):
            ledger.record("openai", 0, -0.1)


class TestCircuitBreaker:
    def test_closed_by_default(self) -> None:
        breaker = CircuitBreaker()
        assert not breaker.is_open
        assert not breaker.is_rate_limited

    def test_opens_after_threshold_consecutive_failures(self) -> None:
        breaker = CircuitBreaker(failure_threshold=3, now=make_clock())
        breaker.record_failure()
        breaker.record_failure()
        assert not breaker.is_open
        breaker.record_failure()
        assert breaker.is_open

    def test_success_resets_failure_count(self) -> None:
        breaker = CircuitBreaker(failure_threshold=2, now=make_clock())
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert not breaker.is_open

    def test_closes_after_cooldown(self) -> None:
        clock = make_clock()
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0, now=clock)
        breaker.record_failure()
        assert breaker.is_open
        clock.advance_seconds(61)
        assert not breaker.is_open

    def test_rate_limit_window_expires(self) -> None:
        clock = make_clock()
        breaker = CircuitBreaker(now=clock)
        breaker.record_rate_limit(retry_after_seconds=30.0)
        assert breaker.is_rate_limited
        clock.advance_seconds(31)
        assert not breaker.is_rate_limited


class TestEmbeddingRouterDecisionTable:
    def make_router(
        self,
        clock: FakeClock | None = None,
        *,
        ledger: BudgetLedger | None = None,
        breaker: CircuitBreaker | None = None,
        primary_enabled: bool = True,
        override: str | None = None,
        daily_budget_usd: float | None = None,
        monthly_budget_usd: float | None = None,
        cost_per_million_tokens: float | None = None,
    ) -> EmbeddingRouter:
        clock = clock or make_clock()
        return EmbeddingRouter(
            primary_embedding(),
            fallback_embedding(),
            ledger=ledger or InMemoryBudgetLedger(now=clock),
            breaker=breaker or CircuitBreaker(failure_threshold=1, now=clock),
            primary_enabled=primary_enabled,
            override=override,
            daily_budget_usd=daily_budget_usd,
            monthly_budget_usd=monthly_budget_usd,
            cost_per_million_tokens=cost_per_million_tokens,
        )

    def test_default_selects_primary(self) -> None:
        selection = self.make_router().select()
        assert selection.reason is SelectionReason.PRIMARY
        assert selection.spec.provider == "fake"
        assert not selection.is_fallback

    def test_force_primary_override(self) -> None:
        selection = self.make_router(override="force_primary").select()
        assert selection.reason is SelectionReason.MANUAL_OVERRIDE
        assert selection.spec.provider == "fake"
        assert not selection.is_fallback

    def test_force_fallback_override(self) -> None:
        selection = self.make_router(override="force_fallback").select()
        assert selection.reason is SelectionReason.MANUAL_OVERRIDE
        assert selection.spec.provider == "st"

    def test_primary_disabled(self) -> None:
        selection = self.make_router(primary_enabled=False).select()
        assert selection.reason is SelectionReason.OPENAI_DISABLED
        assert selection.spec.provider == "st"
        assert selection.is_fallback

    def test_breaker_open_falls_back(self) -> None:
        router = self.make_router()
        router.record_failure()
        selection = router.select()
        assert selection.reason is SelectionReason.OPENAI_UNAVAILABLE
        assert selection.spec.provider == "st"

    def test_rate_limited_falls_back(self) -> None:
        router = self.make_router()
        router.record_rate_limit(retry_after_seconds=30.0)
        selection = router.select()
        assert selection.reason is SelectionReason.OPENAI_RATE_LIMITED

    def test_recovers_to_primary_after_cooldown(self) -> None:
        clock = make_clock()
        router = self.make_router(clock=clock)
        router.record_failure()
        clock.advance_seconds(61)
        assert router.select().reason is SelectionReason.PRIMARY

    def test_success_resets_breaker(self) -> None:
        router = self.make_router()
        router.record_failure()
        router.record_usage(tokens=100)
        assert router.select().reason is SelectionReason.PRIMARY

    def test_daily_budget_exceeded(self) -> None:
        router = self.make_router(daily_budget_usd=0.01, cost_per_million_tokens=0.02)
        router.ledger.record("fake", 600_000, 0.012)
        selection = router.select()
        assert selection.reason is SelectionReason.BUDGET_DAILY_EXCEEDED
        assert selection.spec.provider == "st"

    def test_spend_exactly_at_daily_budget_stays_primary(self) -> None:
        router = self.make_router(daily_budget_usd=0.01, cost_per_million_tokens=0.02)
        router.ledger.record("fake", 500_000, 0.01)
        assert router.select().reason is SelectionReason.PRIMARY

    def test_estimated_call_cost_can_tip_daily_budget(self) -> None:
        router = self.make_router(daily_budget_usd=0.01, cost_per_million_tokens=0.02)
        router.ledger.record("fake", 500_000, 0.01)
        selection = router.select(estimated_tokens=100_000)
        assert selection.reason is SelectionReason.BUDGET_DAILY_EXCEEDED

    def test_monthly_budget_exceeded(self) -> None:
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

    def test_texts_are_estimated_for_budget_check(self) -> None:
        router = self.make_router(
            daily_budget_usd=0.000001, cost_per_million_tokens=1.0
        )
        selection = router.select(texts=["hello world, this is a query"])
        assert selection.reason is SelectionReason.BUDGET_DAILY_EXCEEDED

    def test_budget_check_uses_primary_provider_estimator(self) -> None:
        class CountingEmbedding(FakeEmbedding):
            def __init__(self) -> None:
                super().__init__()
                self.estimated: list[list[str]] = []

            @override
            def estimate_tokens(self, texts: list[str]) -> int:
                self.estimated.append(texts)
                return 11

        primary = CountingEmbedding()
        router = EmbeddingRouter(
            primary,
            fallback_embedding(),
            daily_budget_usd=0.000010,
            cost_per_million_tokens=1.0,
        )

        selection = router.select(texts=["multilingual query"])

        assert selection.reason is SelectionReason.BUDGET_DAILY_EXCEEDED
        assert primary.estimated == [["multilingual query"]]

    def test_explicit_token_estimate_bypasses_provider_estimator(self) -> None:
        class UnexpectedEstimator(FakeEmbedding):
            @override
            def estimate_tokens(self, texts: list[str]) -> int:
                raise AssertionError("provider estimator should not be called")

        router = EmbeddingRouter(
            UnexpectedEstimator(),
            fallback_embedding(),
            daily_budget_usd=1.0,
            cost_per_million_tokens=1.0,
        )

        assert (
            router.select(texts=["query"], estimated_tokens=10).reason
            is SelectionReason.PRIMARY
        )

    def test_negative_explicit_token_estimate_rejected(self) -> None:
        router = self.make_router(daily_budget_usd=1.0)
        with pytest.raises(ValueError, match="must not be negative"):
            router.select(estimated_tokens=-1)

    def test_record_usage_computes_cost_from_rate(self) -> None:
        router = self.make_router(cost_per_million_tokens=0.02)
        router.record_usage(tokens=1_000_000)
        assert router.ledger.spent_today() == pytest.approx(0.02)


class TestEmbeddingRouterValidation:
    def test_no_fallback_raises_when_primary_rejected(self) -> None:
        router = EmbeddingRouter(primary_embedding(), primary_enabled=False)
        with pytest.raises(NoProviderAvailableError) as excinfo:
            router.select()
        assert excinfo.value.reason is SelectionReason.OPENAI_DISABLED

    def test_no_fallback_still_serves_primary(self) -> None:
        router = EmbeddingRouter(primary_embedding())
        assert router.select().reason is SelectionReason.PRIMARY

    def test_invalid_override_rejected(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingRouter(primary_embedding(), override="force_openai")

    def test_force_fallback_requires_fallback(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingRouter(primary_embedding(), override="force_fallback")

    def test_same_space_providers_rejected(self) -> None:
        with pytest.raises(ValueError):
            EmbeddingRouter(primary_embedding(), primary_embedding())
