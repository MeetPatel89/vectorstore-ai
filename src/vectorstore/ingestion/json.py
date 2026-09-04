"""JSON array/object and JSON Lines source adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from vectorstore.records import Record

from .base import (
    SemanticFields,
    Source,
    SourceAdapterError,
    _mapped_record,
)


class JsonSourceAdapter:
    """Read records from JSON objects, arrays, or newline-delimited JSON.

    A regular JSON file may contain one object, an array of objects, or an
    object containing a configurable ``records_key`` array. Files ending in
    ``.jsonl`` or ``.ndjson`` are decoded one non-empty line at a time.
    """

    def __init__(
        self,
        *,
        id_field: str | None = None,
        semantic_fields: SemanticFields = None,
        structured_fields: Sequence[str] | None = None,
        records_key: str | None = "records",
        encoding: str = "utf-8",
    ) -> None:
        if id_field is not None and (not isinstance(id_field, str) or not id_field):
            raise ValueError("id_field must be a non-empty string or None")
        if records_key is not None and (
            not isinstance(records_key, str) or not records_key
        ):
            raise ValueError("records_key must be a non-empty string or None")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        self._id_field = id_field
        self._semantic_fields = semantic_fields
        self._structured_fields = structured_fields
        self._records_key = records_key
        self._encoding = encoding

    def iter_records(self, source: Source) -> Iterator[Record]:
        """Yield JSON records from a file or supported files below a directory."""
        root = Path(source)
        for path in _json_paths(root):
            source_name = _source_name(path, root)
            items = self._read_items(path, source_name)
            for position, item in enumerate(items, start=1):
                if not isinstance(item, Mapping):
                    raise SourceAdapterError(
                        f"JSON record {position} in {source_name!r} must be an object"
                    )
                try:
                    yield _mapped_record(
                        item,
                        source=source_name,
                        id_field=self._id_field,
                        semantic_fields=self._semantic_fields,
                        structured_fields=self._structured_fields,
                    )
                except SourceAdapterError as exc:
                    raise SourceAdapterError(
                        f"invalid JSON record {position} in {source_name!r}: {exc}"
                    ) from exc

    def _read_items(self, path: Path, source_name: str) -> list[object]:
        try:
            text = path.read_text(encoding=self._encoding)
        except (OSError, UnicodeError) as exc:
            raise SourceAdapterError(
                f"could not read JSON source {source_name!r}: {exc}"
            ) from exc

        try:
            if path.suffix.lower() in {".jsonl", ".ndjson"}:
                items: list[object] = []
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if not line.strip():
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise SourceAdapterError(
                            f"invalid JSON on line {line_number} of "
                            f"{source_name!r}: {exc.msg}"
                        ) from exc
                return items

            decoded: Any = json.loads(text)
        except SourceAdapterError:
            raise
        except json.JSONDecodeError as exc:
            raise SourceAdapterError(
                f"invalid JSON source {source_name!r}: {exc.msg}"
            ) from exc

        if isinstance(decoded, list):
            return list(decoded)
        if isinstance(decoded, dict):
            if self._records_key is not None and self._records_key in decoded:
                records = decoded[self._records_key]
                if not isinstance(records, list):
                    raise SourceAdapterError(
                        f"JSON key {self._records_key!r} in {source_name!r} "
                        "must contain an array"
                    )
                return list(records)
            return [decoded]
        raise SourceAdapterError(
            f"JSON source {source_name!r} must contain an object or array"
        )


def _json_paths(source: Path) -> list[Path]:
    suffixes = {".json", ".jsonl", ".ndjson"}
    if source.is_dir():
        try:
            paths = sorted(
                path for path in source.rglob("*") if path.suffix.lower() in suffixes
            )
        except OSError as exc:
            raise SourceAdapterError(
                f"could not discover JSON sources below {source}"
            ) from exc
        if not paths:
            raise SourceAdapterError(f"no JSON sources found below {source}")
        return paths
    if source.suffix.lower() not in suffixes:
        raise SourceAdapterError(
            f"JSON source must use .json, .jsonl, or .ndjson: {source}"
        )
    if not source.is_file():
        raise SourceAdapterError(f"JSON source does not exist: {source}")
    return [source]


def _source_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if root.is_dir() else path.as_posix()


JSONSourceAdapter = JsonSourceAdapter


__all__ = ["JSONSourceAdapter", "JsonSourceAdapter"]
