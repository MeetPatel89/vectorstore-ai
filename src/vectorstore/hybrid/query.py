"""Deterministic query analysis for hybrid retrieval weighting.

The analyzer recognizes queries that are strongly identifier- or
keyword-oriented (incident numbers, error codes, quoted phrases) and
up-weights the lexical signal for them. It is intentionally a small set of
regular-expression heuristics — no LLM, no learned routing — so decisions
are reproducible and testable. The seam exists so a learned router can
replace it later without touching the retrieval facade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# Ticket/record identifiers: INC-1104, CHG-2407, ORA-00001, JIRA-42.
_TICKET_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d+\b")
# Error-code constants: ERR_CONNECTION_RESET, E_ACCESS_DENIED.
_CONSTANT_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
# Standardized code prefixes followed by a code: SQLSTATE 23505, HTTP 503.
_CODED_PATTERN = re.compile(r"\b(?:SQLSTATE|HRESULT|ERRNO)\s+[0-9A-Fa-f]+\b")
# Explicitly quoted exact phrases: "payment reconciliation".
_PHRASE_PATTERN = re.compile(r'"([^"]+)"')


class QueryKind(StrEnum):
    """The deterministic classification of one query."""

    EMPTY = "empty"
    IDENTIFIER = "identifier"
    PHRASE = "phrase"
    NATURAL = "natural"


@dataclass(frozen=True)
class QueryProfile:
    """The analyzer's verdict: kind plus fusion weights.

    ``identifiers`` and ``phrases`` list what was detected, for
    observability and tests. Weights feed weighted RRF; they scale rank
    contributions, never raw scores.
    """

    kind: QueryKind
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    identifiers: tuple[str, ...] = ()
    phrases: tuple[str, ...] = ()


class QueryAnalyzer:
    """Classify queries and assign dense/lexical fusion weights.

    Identifier- and phrase-bearing queries get ``lexical_boost`` as their
    lexical weight (dense stays 1.0) because exact tokens are better served
    by the full-text index. Ordinary natural-language queries weight both
    signals equally, keeping dense retrieval the primary semantic signal.
    """

    def __init__(self, lexical_boost: float = 2.0) -> None:
        if lexical_boost <= 0:
            raise ValueError("lexical_boost must be greater than zero")
        self._lexical_boost = lexical_boost

    def analyze(self, query: str) -> QueryProfile:
        if not isinstance(query, str) or not query.strip():
            return QueryProfile(kind=QueryKind.EMPTY)

        identifiers = tuple(
            dict.fromkeys(
                match
                for pattern in (_TICKET_PATTERN, _CODED_PATTERN, _CONSTANT_PATTERN)
                for match in pattern.findall(query)
            )
        )
        phrases = tuple(
            phrase.strip()
            for phrase in _PHRASE_PATTERN.findall(query)
            if phrase.strip()
        )

        if identifiers:
            kind = QueryKind.IDENTIFIER
        elif phrases:
            kind = QueryKind.PHRASE
        else:
            return QueryProfile(kind=QueryKind.NATURAL)

        return QueryProfile(
            kind=kind,
            dense_weight=1.0,
            lexical_weight=self._lexical_boost,
            identifiers=identifiers,
            phrases=phrases,
        )
