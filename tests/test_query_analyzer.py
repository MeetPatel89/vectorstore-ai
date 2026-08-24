"""Unit tests for the deterministic query analyzer."""

from __future__ import annotations

import pytest

from vectorstore import QueryAnalyzer, QueryKind


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


class TestClassification:
    def test_empty_and_whitespace_queries(self, analyzer: QueryAnalyzer) -> None:
        assert analyzer.analyze("").kind is QueryKind.EMPTY
        assert analyzer.analyze("   \t\n").kind is QueryKind.EMPTY

    def test_natural_language_query(self, analyzer: QueryAnalyzer) -> None:
        profile = analyzer.analyze("why are payment reports missing data")
        assert profile.kind is QueryKind.NATURAL
        assert profile.dense_weight == 1.0
        assert profile.lexical_weight == 1.0
        assert profile.identifiers == ()
        assert profile.phrases == ()

    @pytest.mark.parametrize(
        ("query", "expected_identifier"),
        [
            ("what happened with INC-1104", "INC-1104"),
            ("status of CHG-2407 rollout", "CHG-2407"),
            ("getting ORA-00001 on insert", "ORA-00001"),
            ("duplicate key SQLSTATE 23505 during sync", "SQLSTATE 23505"),
            ("browser shows ERR_CONNECTION_RESET intermittently", "ERR_CONNECTION_RESET"),
        ],
    )
    def test_identifier_queries(
        self, analyzer: QueryAnalyzer, query: str, expected_identifier: str
    ) -> None:
        profile = analyzer.analyze(query)
        assert profile.kind is QueryKind.IDENTIFIER
        assert expected_identifier in profile.identifiers
        assert profile.lexical_weight == 2.0
        assert profile.dense_weight == 1.0

    def test_multiple_identifiers_deduplicated(self, analyzer: QueryAnalyzer) -> None:
        profile = analyzer.analyze("INC-1104 duplicates INC-1104 and CHG-2407")
        assert profile.identifiers == ("INC-1104", "CHG-2407")

    def test_quoted_phrase_query(self, analyzer: QueryAnalyzer) -> None:
        profile = analyzer.analyze('docs about "payment reconciliation" process')
        assert profile.kind is QueryKind.PHRASE
        assert profile.phrases == ("payment reconciliation",)
        assert profile.lexical_weight == 2.0

    def test_identifier_wins_over_phrase(self, analyzer: QueryAnalyzer) -> None:
        profile = analyzer.analyze('INC-1104 "payment reconciliation"')
        assert profile.kind is QueryKind.IDENTIFIER
        assert profile.identifiers == ("INC-1104",)
        assert profile.phrases == ("payment reconciliation",)

    def test_lowercase_identifier_like_text_is_natural(
        self, analyzer: QueryAnalyzer
    ) -> None:
        profile = analyzer.analyze("the inc-1104 style pattern in lowercase")
        assert profile.kind is QueryKind.NATURAL

    def test_empty_quotes_do_not_make_a_phrase(self, analyzer: QueryAnalyzer) -> None:
        profile = analyzer.analyze('search for "" nothing')
        assert profile.kind is QueryKind.NATURAL


class TestConfiguration:
    def test_custom_lexical_boost(self) -> None:
        profile = QueryAnalyzer(lexical_boost=3.5).analyze("INC-1104")
        assert profile.lexical_weight == 3.5

    def test_invalid_boost_rejected(self) -> None:
        with pytest.raises(ValueError, match="lexical_boost"):
            QueryAnalyzer(lexical_boost=0)
