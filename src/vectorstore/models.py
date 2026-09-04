"""Core value objects and backend-independent metadata filtering."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeAlias

MetadataValue: TypeAlias = str | int | float | bool
MetadataFilter: TypeAlias = Mapping[str, object]

_COMPARISON_OPERATORS = frozenset({"$gt", "$gte", "$lt", "$lte"})
_SUPPORTED_OPERATORS = _COMPARISON_OPERATORS | {"$in"}


@dataclass(frozen=True)
class Chunk:
    """A pre-made text chunk and the metadata stored alongside it."""

    id: str
    text: str
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("chunk IDs must be non-empty strings")
        if not isinstance(self.text, str):
            raise ValueError("chunk text must be a string")
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, label="chunk metadata"),
        )


@dataclass(frozen=True)
class SearchResult:
    """A chunk returned by similarity search."""

    chunk: Chunk
    score: float


def matches(metadata: Mapping[str, MetadataValue], filter: MetadataFilter) -> bool:
    """Return whether *metadata* satisfies all conditions in *filter*.

    Supported conditions are scalar equality, ``$in``, and the four ordered
    comparison operators. Multiple fields, and multiple comparison operators
    for one field, are combined with AND semantics.
    """
    for key, condition in filter.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata filter keys must be non-empty strings")
        if key not in metadata:
            return False

        actual = metadata[key]
        if not isinstance(condition, dict):
            if actual != condition:
                return False
            continue

        if not condition:
            raise ValueError(f"metadata filter for {key!r} cannot be empty")

        for operator, expected in condition.items():
            if operator not in _SUPPORTED_OPERATORS:
                raise ValueError(f"unsupported metadata filter operator: {operator!r}")

            if operator == "$in":
                if not isinstance(expected, (list, tuple, set, frozenset)):
                    raise ValueError("$in requires a list-like value")
                if actual not in expected:
                    return False
                continue

            try:
                if operator == "$gt" and not actual > expected:
                    return False
                if operator == "$gte" and not actual >= expected:
                    return False
                if operator == "$lt" and not actual < expected:
                    return False
                if operator == "$lte" and not actual <= expected:
                    return False
            except TypeError:
                return False

    return True


def _freeze_metadata(
    metadata: Mapping[str, MetadataValue],
    *,
    label: str,
) -> Mapping[str, MetadataValue]:
    """Validate and snapshot scalar metadata behind a read-only mapping."""
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{label} must be a mapping")
    snapshot: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if isinstance(value, bool) or isinstance(value, (str, int)):
            snapshot[key] = value
            continue
        if isinstance(value, float) and math.isfinite(value):
            snapshot[key] = value
            continue
        raise ValueError(f"{label} values must be finite strings, numbers, or booleans")
    return MappingProxyType(snapshot)
