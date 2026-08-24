"""Unit tests for weighted Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from vectorstore import rrf


class TestRrf:
    def test_single_list_preserves_order(self) -> None:
        fused = rrf([["a", "b", "c"]])
        assert [candidate for candidate, _ in fused] == ["a", "b", "c"]

    def test_scores_follow_the_formula(self) -> None:
        fused = dict(rrf([["a", "b"], ["b", "a"]], k=60))
        assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)

    def test_agreement_beats_single_signal(self) -> None:
        # "both" appears in both lists at rank 2; "dense_only" leads one list.
        fused = rrf([["dense_only", "both"], ["lex_only", "both"]])
        assert fused[0][0] == "both"

    def test_weights_shift_the_outcome(self) -> None:
        rankings = [["dense_top"], ["lex_top"]]
        balanced = rrf(rankings, weights=[1.0, 1.0])
        assert balanced[0][0] == "dense_top"  # equal score, ID tiebreak
        boosted = rrf(rankings, weights=[1.0, 2.0])
        assert boosted[0][0] == "lex_top"

    def test_missing_candidates_contribute_zero(self) -> None:
        fused = dict(rrf([["a"], []], k=60))
        assert fused == {"a": pytest.approx(1 / 61)}

    def test_ties_broken_by_id_for_determinism(self) -> None:
        fused = rrf([["zeta"], ["alpha"]])
        assert [candidate for candidate, _ in fused] == ["alpha", "zeta"]

    def test_duplicates_within_a_list_keep_best_rank(self) -> None:
        fused = dict(rrf([["a", "a", "b"]], k=60))
        assert fused["a"] == pytest.approx(1 / 61)
        assert fused["b"] == pytest.approx(1 / 63)

    def test_empty_input(self) -> None:
        assert rrf([]) == []
        assert rrf([[], []]) == []

    def test_invalid_k_rejected(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            rrf([["a"]], k=0)

    def test_mismatched_weights_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights must match"):
            rrf([["a"]], weights=[1.0, 2.0])

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            rrf([["a"]], weights=[-1.0])

    def test_non_string_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            rrf([["a", ""]])
