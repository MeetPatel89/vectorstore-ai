"""CSV row-per-record source adapter."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from pathlib import Path

from vectorstore.records import Record

from .base import (
    SemanticFields,
    Source,
    SourceAdapterError,
    _mapped_record,
)


class CsvSourceAdapter:
    """Read one :class:`~vectorstore.records.Record` per CSV row.

    ``semantic_fields`` may be a sequence of column names, or a mapping from
    output label to source column. When omitted, common content columns such
    as ``title``, ``description``, ``question``, ``answer``, and ``text`` are
    inferred case-insensitively. ``structured_fields=None`` retains every
    remaining scalar column as filterable metadata.
    """

    def __init__(
        self,
        *,
        id_field: str | None = None,
        semantic_fields: SemanticFields = None,
        structured_fields: Sequence[str] | None = None,
        encoding: str = "utf-8-sig",
        delimiter: str = ",",
    ) -> None:
        if id_field is not None and (not isinstance(id_field, str) or not id_field):
            raise ValueError("id_field must be a non-empty string or None")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ValueError("delimiter must be exactly one character")
        self._id_field = id_field
        self._semantic_fields = semantic_fields
        self._structured_fields = structured_fields
        self._encoding = encoding
        self._delimiter = delimiter

    def iter_records(self, source: Source) -> Iterator[Record]:
        """Yield CSV rows from a file, or every ``*.csv`` below a directory."""
        root = Path(source)
        for path in _csv_paths(root):
            source_name = _source_name(path, root)
            try:
                with path.open("r", encoding=self._encoding, newline="") as source_file:
                    reader = csv.DictReader(source_file, delimiter=self._delimiter)
                    if reader.fieldnames is None:
                        raise SourceAdapterError(f"CSV source {source_name!r} is empty")
                    if any(name is None or not name for name in reader.fieldnames):
                        raise SourceAdapterError(
                            f"CSV source {source_name!r} has an empty header"
                        )
                    if len(set(reader.fieldnames)) != len(reader.fieldnames):
                        raise SourceAdapterError(
                            f"CSV source {source_name!r} has duplicate headers"
                        )
                    for row_number, raw_row in enumerate(reader, start=2):
                        if None in raw_row:
                            raise SourceAdapterError(
                                f"CSV source {source_name!r}, row {row_number}, has "
                                "more values than headers"
                            )
                        row = {str(key): value for key, value in raw_row.items()}
                        try:
                            yield _mapped_record(
                                row,
                                source=source_name,
                                id_field=self._id_field,
                                semantic_fields=self._semantic_fields,
                                structured_fields=self._structured_fields,
                                coerce_string_metadata=True,
                            )
                        except SourceAdapterError as exc:
                            raise SourceAdapterError(
                                f"invalid CSV row {row_number} in "
                                f"{source_name!r}: {exc}"
                            ) from exc
            except SourceAdapterError:
                raise
            except (OSError, UnicodeError, csv.Error) as exc:
                raise SourceAdapterError(
                    f"could not read CSV source {source_name!r}: {exc}"
                ) from exc


def _csv_paths(source: Path) -> list[Path]:
    if source.is_dir():
        try:
            paths = sorted(source.rglob("*.csv"))
        except OSError as exc:
            raise SourceAdapterError(
                f"could not discover CSV sources below {source}"
            ) from exc
        if not paths:
            raise SourceAdapterError(f"no CSV sources found below {source}")
        return paths
    if source.suffix.lower() != ".csv":
        raise SourceAdapterError(f"CSV source must use a .csv extension: {source}")
    if not source.is_file():
        raise SourceAdapterError(f"CSV source does not exist: {source}")
    return [source]


def _source_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if root.is_dir() else path.as_posix()


# Conventional acronym spelling remains available without duplicating behavior.
CSVSourceAdapter = CsvSourceAdapter


__all__ = ["CSVSourceAdapter", "CsvSourceAdapter"]
