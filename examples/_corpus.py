"""Shared loader for the bundled Nautilus ITSM Markdown corpus."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import TypeGuard

from vectorstore import (
    CatalogChunk,
    CatalogDocument,
    Chunk,
    MetadataValue,
    Record,
    semantic_projection,
)

CORPUS_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / "corpora" / "nautilus" / "raw"
)

type LoadedDocument = tuple[CatalogDocument, list[CatalogChunk], Record]

_TOP_LEVEL_FIELD = re.compile(
    r"^(?P<key>[A-Za-z_][A-Za-z0-9_-]*):(?:\s*(?P<value>.*))?$"
)
_INTEGER = re.compile(r"[-+]?\d+")
_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+)")
_SECOND_LEVEL_HEADING = re.compile(r"^##(?!#)\s+(?P<title>.+?)\s*#*\s*$")

_DOCUMENT_FIELDS = {
    "doc_id",
    "title",
    "source",
    "doc_type",
    "tenant_id",
    "visibility",
    "owner_group",
    "status",
    "created_at",
    "updated_at",
}


def load_documents(root: str | Path = CORPUS_ROOT) -> list[LoadedDocument]:
    """Load documents, section chunks, and source records from *root*.

    The corpus uses a deliberately simple subset of YAML at the top level.
    Scalar frontmatter values become filterable structured attributes; nested
    lists and mappings are left out because the library metadata contract is
    scalar-valued. The title and Markdown body form the semantic projection.
    """
    corpus_root = Path(root)
    print(f"Corpus root: {corpus_root}")
    markdown_files = sorted(corpus_root.rglob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"no Markdown documents found under {corpus_root}")

    loaded: list[LoadedDocument] = []
    for path in markdown_files:
        frontmatter, body = _read_markdown(path)
        doc_id = _required_string(frontmatter, "doc_id", path)
        title = _required_string(frontmatter, "title", path)
        source = path.relative_to(corpus_root).as_posix()

        structured = {
            key: value
            for key, value in frontmatter.items()
            if _is_metadata_value(value)
        }
        record = Record(
            id=doc_id,
            semantic_fields={
                "Title": title,
                "Body": _without_leading_title(body),
            },
            structured=structured,
            source=source,
        )

        attributes = {
            key: value
            for key, value in structured.items()
            if key not in _DOCUMENT_FIELDS
        }
        document = CatalogDocument(
            doc_id=doc_id,
            title=title,
            source=source,
            doc_type=_optional_string(frontmatter, "doc_type", path),
            tenant_id=_optional_string(frontmatter, "tenant_id", path),
            visibility=_optional_string(frontmatter, "visibility", path),
            owner_group=_optional_string(frontmatter, "owner_group", path),
            status=_optional_string(frontmatter, "status", path),
            created_at=_optional_string(frontmatter, "created_at", path),
            updated_at=_optional_string(frontmatter, "updated_at", path),
            attributes=attributes,
        )
        chunks = _section_chunks(record)
        loaded.append((document, chunks, record))

    return loaded


def to_vector_chunks(documents: Iterable[LoadedDocument]) -> list[Chunk]:
    """Convert catalog chunks to dense-store chunks with filter metadata."""
    vector_chunks: list[Chunk] = []
    for document, chunks, record in documents:
        document_metadata = dict(record.structured)
        document_metadata["doc_id"] = document.doc_id
        for chunk in chunks:
            metadata = dict(document_metadata)
            metadata["chunk_index"] = chunk.chunk_index
            if chunk.section_path is not None:
                metadata["section_path"] = chunk.section_path
            vector_chunks.append(
                Chunk(id=chunk.chunk_id, text=chunk.text, metadata=metadata)
            )
    return vector_chunks


def _read_markdown(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} does not start with YAML frontmatter")

    try:
        closing_line = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError(f"{path} has unterminated YAML frontmatter") from exc

    frontmatter: dict[str, object] = {}
    for line in lines[1:closing_line]:
        if not line or line[0].isspace() or line.lstrip().startswith("#"):
            continue
        match = _TOP_LEVEL_FIELD.fullmatch(line)
        if match is None:
            raise ValueError(f"unsupported frontmatter line in {path}: {line!r}")
        raw_value = match.group("value") or ""
        if raw_value:
            frontmatter[match.group("key")] = _parse_scalar(raw_value)

    body = "\n".join(lines[closing_line + 1 :]).strip()
    if not body:
        raise ValueError(f"{path} has no Markdown body")
    return frontmatter, body


def _parse_scalar(raw_value: str) -> object:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if _INTEGER.fullmatch(value):
        return int(value)
    if _FLOAT.fullmatch(value):
        return float(value)
    return value


def _section_chunks(record: Record) -> list[CatalogChunk]:
    body = record.semantic_fields["Body"]
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    section_lines: list[str] = []

    def finish_section() -> None:
        content = "\n".join(section_lines).strip()
        if content or heading is not None:
            sections.append((heading, [content] if content else []))

    for line in body.splitlines():
        match = _SECOND_LEVEL_HEADING.fullmatch(line)
        if match is None:
            section_lines.append(line)
            continue
        finish_section()
        heading = match.group("title")
        section_lines = []
    finish_section()

    if not sections:
        sections.append((None, [body]))

    chunks: list[CatalogChunk] = []
    title = record.semantic_fields["Title"]
    for index, (section_path, content_parts) in enumerate(sections):
        semantic_fields = {"Title": title}
        if section_path is not None:
            semantic_fields["Section"] = section_path
        content = "\n".join(content_parts).strip()
        if content:
            semantic_fields["Content"] = content

        text = semantic_projection(
            Record(
                id=f"{record.id}:{index}",
                semantic_fields=semantic_fields,
                structured=record.structured,
                source=record.source,
            )
        )
        chunks.append(
            CatalogChunk(
                chunk_id=f"{record.id}:{index}",
                doc_id=record.id,
                text=text,
                chunk_index=index,
                section_path=section_path,
            )
        )

    return chunks


def _without_leading_title(body: str) -> str:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line.startswith("# "):
            del lines[index]
        break
    return "\n".join(lines).strip()


def _required_string(frontmatter: dict[str, object], key: str, path: Path) -> str:
    value = _optional_string(frontmatter, key, path)
    if value is None or not value:
        raise ValueError(f"{path} requires a non-empty {key!r} frontmatter value")
    return value


def _optional_string(
    frontmatter: dict[str, object], key: str, path: Path
) -> str | None:
    value = frontmatter.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{path} frontmatter value {key!r} must be a string")
    return value


def _is_metadata_value(value: object) -> TypeGuard[MetadataValue]:
    return isinstance(value, (str, int, float, bool))
