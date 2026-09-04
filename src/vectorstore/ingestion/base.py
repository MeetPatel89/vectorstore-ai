"""Protocols and shared errors for source ingestion.

Source adapters stop at the :class:`~vectorstore.records.Record` boundary.
They know how to read a file format, but do not know about catalogs, vector
stores, embedding providers, or chunking policy.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from vectorstore.models import MetadataValue
from vectorstore.records import Record

type Source = str | Path


class SourceAdapterError(ValueError):
    """A source could not be decoded into valid records."""


@runtime_checkable
class SourceAdapter(Protocol):
    """Turn one file or directory source into retrieval records."""

    def iter_records(self, source: Source) -> Iterator[Record]:
        """Yield records from *source* in deterministic source order."""


type SemanticFields = Sequence[str] | Mapping[str, str] | None

_SEMANTIC_FIELD_NAMES = (
    "title",
    "question",
    "answer",
    "summary",
    "description",
    "detail_text",
    "policy_text",
    "symptoms",
    "resolution",
    "body",
    "content",
    "text",
)


def _mapped_record(
    values: Mapping[str, object],
    *,
    source: str,
    id_field: str | None,
    semantic_fields: SemanticFields,
    structured_fields: Sequence[str] | None,
    coerce_string_metadata: bool = False,
) -> Record:
    """Map a CSV/JSON object to a Record using a shared, explicit policy."""
    if not values:
        raise SourceAdapterError(f"record in {source!r} is empty")
    keys = tuple(values)
    for key in keys:
        if not isinstance(key, str) or not key:
            raise SourceAdapterError(f"record keys in {source!r} must be strings")

    resolved_id_field = _resolve_id_field(keys, id_field)
    raw_id = values.get(resolved_id_field)
    if raw_id is None or not str(raw_id).strip():
        raise SourceAdapterError(
            f"record in {source!r} has no value for ID field {resolved_id_field!r}"
        )
    record_id = str(raw_id).strip()

    selected_semantic = _resolve_semantic_fields(keys, semantic_fields)
    semantic: dict[str, str] = {}
    semantic_source_fields: set[str] = set()
    for label, field_name in selected_semantic:
        semantic_source_fields.add(field_name)
        value = values.get(field_name)
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple, set)):
            raise SourceAdapterError(
                f"semantic field {field_name!r} in {source!r} must be scalar"
            )
        semantic[label] = str(value)
    if not any(value.strip() for value in semantic.values()):
        raise SourceAdapterError(
            f"record {record_id!r} in {source!r} has no semantic content"
        )

    if structured_fields is None:
        selected_structured = [
            key
            for key in keys
            if key != resolved_id_field and key not in semantic_source_fields
        ]
    else:
        selected_structured = [
            _resolve_field(keys, field_name, "structured")
            for field_name in structured_fields
        ]

    structured: dict[str, MetadataValue] = {}
    for field_name in selected_structured:
        value = values.get(field_name)
        if value is None or value == "":
            continue
        scalar = _metadata_value(value, coerce_strings=coerce_string_metadata)
        if scalar is None:
            if structured_fields is not None:
                raise SourceAdapterError(
                    f"structured field {field_name!r} in {source!r} must be a "
                    "finite scalar"
                )
            continue
        structured[field_name] = scalar

    return Record(
        id=record_id,
        semantic_fields=semantic,
        structured=structured,
        source=source,
    )


def _resolve_id_field(keys: Sequence[str], configured: str | None) -> str:
    if configured is not None:
        return _resolve_field(keys, configured, "ID")
    normalized = {_normalize_field_name(key): key for key in keys}
    for candidate in ("doc_id", "record_id", "document_id", "id"):
        if candidate in normalized:
            return normalized[candidate]
    for key in keys:
        if _normalize_field_name(key).endswith("_id"):
            return key
    raise SourceAdapterError(
        "could not infer an ID field; configure id_field explicitly"
    )


def _resolve_semantic_fields(
    keys: Sequence[str], configured: SemanticFields
) -> list[tuple[str, str]]:
    if isinstance(configured, Mapping):
        return [
            (label, _resolve_field(keys, field_name, "semantic"))
            for label, field_name in configured.items()
        ]
    if configured is not None:
        return [
            (field_name, _resolve_field(keys, field_name, "semantic"))
            for field_name in configured
        ]

    by_normalized = {_normalize_field_name(key): key for key in keys}
    inferred = [
        (by_normalized[name], by_normalized[name])
        for name in _SEMANTIC_FIELD_NAMES
        if name in by_normalized
    ]
    if not inferred:
        raise SourceAdapterError(
            "could not infer semantic fields; configure semantic_fields explicitly"
        )
    return inferred


def _resolve_field(keys: Sequence[str], requested: str, kind: str) -> str:
    if not isinstance(requested, str) or not requested:
        raise ValueError(f"{kind} field names must be non-empty strings")
    if requested in keys:
        return requested
    normalized_requested = _normalize_field_name(requested)
    matches = [
        key for key in keys if _normalize_field_name(key) == normalized_requested
    ]
    if len(matches) == 1:
        return matches[0]
    raise SourceAdapterError(f"{kind} field {requested!r} is missing")


def _normalize_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _metadata_value(
    value: object, *, coerce_strings: bool = False
) -> MetadataValue | None:
    if coerce_strings and isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
        if re.fullmatch(r"[-+]?(?:0|[1-9]\d*)", value.strip()):
            return int(value)
        if re.fullmatch(r"[-+]?(?:0|[1-9]\d*)\.\d+(?:[eE][-+]?\d+)?", value.strip()):
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
    if isinstance(value, bool) or isinstance(value, (str, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None
