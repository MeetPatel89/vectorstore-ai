"""Markdown plus YAML-frontmatter source adapter."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from vectorstore.models import MetadataValue
from vectorstore.records import Record

from .base import Source, SourceAdapterError, _metadata_value

_FRONTMATTER_BOUNDARY = "---"
_H1 = re.compile(r"^#(?!#)\s+(?P<title>.+?)\s*#*\s*$")


class MarkdownSourceAdapter:
    """Read Markdown files with optional YAML-like frontmatter.

    The dependency-free frontmatter parser supports the scalar, nested-map,
    and list subset used by the bundled corpus. Only top-level finite scalar
    values enter ``Record.structured`` because the package metadata contract
    is scalar-valued; nested values remain source-format detail. The semantic
    projection contains ``Title`` plus the Markdown ``Body``.
    """

    def __init__(
        self,
        *,
        id_field: str = "doc_id",
        title_field: str = "title",
        encoding: str = "utf-8",
    ) -> None:
        for label, value in (("id_field", id_field), ("title_field", title_field)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        if not isinstance(encoding, str) or not encoding:
            raise ValueError("encoding must be a non-empty string")
        self._id_field = id_field
        self._title_field = title_field
        self._encoding = encoding

    def iter_records(self, source: Source) -> Iterator[Record]:
        """Yield Markdown records in stable path order."""
        root = Path(source)
        for path in _markdown_paths(root):
            source_name = _source_name(path, root)
            try:
                raw_text = path.read_text(encoding=self._encoding)
            except (OSError, UnicodeError) as exc:
                raise SourceAdapterError(
                    f"could not read Markdown source {source_name!r}: {exc}"
                ) from exc

            try:
                metadata, body = parse_markdown_frontmatter(raw_text)
            except ValueError as exc:
                raise SourceAdapterError(
                    f"invalid frontmatter in {source_name!r}: {exc}"
                ) from exc

            raw_id = metadata.get(self._id_field, path.stem)
            if raw_id is None or not str(raw_id).strip():
                raise SourceAdapterError(
                    f"Markdown source {source_name!r} has an empty document ID"
                )
            doc_id = str(raw_id).strip()

            heading_title = _first_h1(body)
            raw_title = metadata.get(self._title_field, heading_title or path.stem)
            if raw_title is None or not str(raw_title).strip():
                raise SourceAdapterError(
                    f"Markdown source {source_name!r} has an empty title"
                )
            title = str(raw_title).strip()

            structured: dict[str, MetadataValue] = {}
            for key, value in metadata.items():
                scalar = _metadata_value(value)
                if scalar is not None:
                    structured[key] = scalar

            yield Record(
                id=doc_id,
                semantic_fields={
                    "Title": title,
                    "Body": _without_leading_h1(body),
                },
                structured=structured,
                source=source_name,
            )


def parse_markdown_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """Return parsed leading frontmatter and the remaining Markdown body.

    Markdown without a complete leading frontmatter block is treated as plain
    body text. This mirrors common Markdown tooling and avoids dropping text
    merely because a document begins with a horizontal rule.
    """
    if not isinstance(raw_text, str):
        raise ValueError("Markdown source must be text")
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_BOUNDARY:
        return {}, raw_text
    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONTMATTER_BOUNDARY
        ),
        None,
    )
    if closing_index is None:
        return {}, raw_text
    metadata = _parse_simple_yaml(lines[1:closing_index])
    return metadata, "\n".join(lines[closing_index + 1 :]).strip()


def _parse_simple_yaml(lines: Iterable[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, metadata)]
    pending: tuple[int, dict[str, Any], str] | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line[:indent]:
            raise ValueError("tabs are not supported for frontmatter indentation")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        if pending is not None and indent > pending[0]:
            pending_indent, pending_parent, name = pending
            container: dict[str, Any] | list[Any]
            container = [] if stripped.startswith("- ") else {}
            pending_parent[name] = container
            stack.append((pending_indent, container))
            pending = None

        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError("list item appears outside a frontmatter list")
            parent.append(_parse_scalar(stripped[2:].strip()))
            continue

        if ":" not in stripped:
            raise ValueError(f"unsupported frontmatter line: {stripped!r}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError("frontmatter keys must be non-empty")
        if not isinstance(parent, dict):
            raise ValueError("mapping field appears inside a scalar list")
        value = raw_value.strip()
        if not value:
            parent[key] = {}
            pending = (indent, parent, key)
        else:
            parent[key] = _parse_scalar(value)
            pending = None
    return metadata


def _parse_scalar(value: str) -> object:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    normalized = value.lower()
    if normalized in {"null", "~"}:
        return None
    if normalized in {"true", "false"}:
        return normalized == "true"
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    return value


def _markdown_paths(source: Path) -> list[Path]:
    if source.is_dir():
        try:
            paths = sorted(source.rglob("*.md"))
        except OSError as exc:
            raise SourceAdapterError(
                f"could not discover Markdown sources below {source}"
            ) from exc
        if not paths:
            raise SourceAdapterError(f"no Markdown sources found below {source}")
        return paths
    if source.suffix.lower() not in {".md", ".markdown"}:
        raise SourceAdapterError(
            f"Markdown source must use a .md or .markdown extension: {source}"
        )
    if not source.is_file():
        raise SourceAdapterError(f"Markdown source does not exist: {source}")
    return [source]


def _source_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix() if root.is_dir() else path.as_posix()


def _first_h1(body: str) -> str | None:
    for line in body.splitlines():
        match = _H1.fullmatch(line)
        if match is not None:
            return match.group("title").strip()
    return None


def _without_leading_h1(body: str) -> str:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if _H1.fullmatch(line) is not None:
            del lines[index]
        break
    return "\n".join(lines).strip()


__all__ = ["MarkdownSourceAdapter", "parse_markdown_frontmatter"]
