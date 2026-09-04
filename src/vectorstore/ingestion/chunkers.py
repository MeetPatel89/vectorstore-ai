"""Record chunking policies for ingestion."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from vectorstore.catalog.base import CatalogChunk
from vectorstore.records import Record, semantic_projection

DEFAULT_MAX_WORDS = 900
DEFAULT_OVERLAP_WORDS = 75

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@runtime_checkable
class Chunker(Protocol):
    """Split one semantic projection into stable catalog chunks."""

    def chunk(
        self, record: Record, projection: str | None = None
    ) -> list[CatalogChunk]:
        """Chunk *projection*, using *record* for identity and context."""


@dataclass(frozen=True)
class MarkdownSection:
    """One section extracted from a Markdown body."""

    title: str
    section_path: tuple[str, ...]
    content: str


class WholeRecordChunker:
    """Create exactly one chunk from every non-empty record projection."""

    def chunk(
        self, record: Record, projection: str | None = None
    ) -> list[CatalogChunk]:
        """Return one catalog chunk carrying the complete projection."""
        text = semantic_projection(record) if projection is None else projection
        if not isinstance(text, str) or not text.strip():
            raise ValueError("record projection must be non-empty text")
        return [
            CatalogChunk(
                chunk_id=_chunk_id(record.id, 0),
                doc_id=record.id,
                text=text.strip(),
                chunk_index=0,
            )
        ]

    def chunk_record(
        self, record: Record, projection: str | None = None
    ) -> list[CatalogChunk]:
        """Compatibility spelling for callers that prefer an explicit noun."""
        return self.chunk(record, projection)


class WordChunker:
    """Generic paragraph-aware, overlapping word chunker."""

    def __init__(
        self,
        max_words: int = DEFAULT_MAX_WORDS,
        overlap_words: int = DEFAULT_OVERLAP_WORDS,
    ) -> None:
        _validate_window(max_words, overlap_words)
        self._max_words = max_words
        self._overlap_words = overlap_words

    @property
    def max_words(self) -> int:
        """Maximum whitespace-delimited words in one chunk."""
        return self._max_words

    @property
    def overlap_words(self) -> int:
        """Requested trailing-word overlap between adjacent chunks."""
        return self._overlap_words

    def chunk(
        self, record: Record, projection: str | None = None
    ) -> list[CatalogChunk]:
        """Split a record projection while retaining paragraph boundaries."""
        text = semantic_projection(record) if projection is None else projection
        parts = split_large_text(text, self.max_words, self.overlap_words)
        return [
            CatalogChunk(
                chunk_id=_chunk_id(record.id, index),
                doc_id=record.id,
                text=part,
                chunk_index=index,
            )
            for index, part in enumerate(parts)
        ]

    def chunk_record(
        self, record: Record, projection: str | None = None
    ) -> list[CatalogChunk]:
        """Compatibility spelling for :meth:`chunk`."""
        return self.chunk(record, projection)


class MarkdownSectionChunker:
    """Chunk Markdown on H1/H2 sections, then paragraphs and word windows.

    A document title and section label are repeated in every emitted chunk,
    keeping an isolated dense hit understandable. H3-H6 headings stay inside
    their parent section.
    """

    def __init__(
        self,
        max_words: int = DEFAULT_MAX_WORDS,
        overlap_words: int = DEFAULT_OVERLAP_WORDS,
    ) -> None:
        _validate_window(max_words, overlap_words)
        self._max_words = max_words
        self._overlap_words = overlap_words

    @property
    def max_words(self) -> int:
        """Maximum words in one rendered chunk."""
        return self._max_words

    @property
    def overlap_words(self) -> int:
        """Requested word overlap for oversized sections."""
        return self._overlap_words

    def chunk(
        self, record: Record, projection: str | None = None
    ) -> list[CatalogChunk]:
        """Split a Markdown record into section-aware catalog chunks."""
        rendered_projection = (
            semantic_projection(record) if projection is None else projection
        )
        if not isinstance(rendered_projection, str) or not rendered_projection.strip():
            raise ValueError("record projection must be non-empty text")

        title = _semantic_value(record, "title") or record.id
        body = (
            _semantic_value(record, "body")
            or _semantic_value(record, "content")
            or _semantic_value(record, "text")
        )
        if body is None:
            body = rendered_projection

        sections = split_markdown_sections(body, fallback_title=title)
        if not sections:
            sections = [
                MarkdownSection(
                    title=title,
                    section_path=(title,),
                    content=rendered_projection.strip(),
                )
            ]

        chunks: list[CatalogChunk] = []
        for section in sections:
            for part in _split_rendered_section(
                title, section, self.max_words, self.overlap_words
            ):
                index = len(chunks)
                chunks.append(
                    CatalogChunk(
                        chunk_id=_chunk_id(record.id, index),
                        doc_id=record.id,
                        text=part,
                        chunk_index=index,
                        section_path=" > ".join(section.section_path),
                    )
                )
        return chunks

    def chunk_record(
        self, record: Record, projection: str | None = None
    ) -> list[CatalogChunk]:
        """Compatibility spelling for :meth:`chunk`."""
        return self.chunk(record, projection)


def split_markdown_sections(body: str, fallback_title: str) -> list[MarkdownSection]:
    """Split Markdown into document/H2 sections while preserving nested text."""
    if not isinstance(body, str):
        raise ValueError("Markdown body must be text")
    if not isinstance(fallback_title, str) or not fallback_title.strip():
        raise ValueError("fallback_title must be non-empty text")

    document_title = fallback_title.strip()
    h1_seen = False
    sections: list[MarkdownSection] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in body.splitlines():
        heading = _HEADING.fullmatch(line)
        if heading is None:
            current_lines.append(line)
            continue
        level = len(heading.group(1))
        title = heading.group(2).strip()
        if level == 1 and not h1_seen:
            _append_section(sections, document_title, current_title, current_lines)
            document_title = title
            h1_seen = True
            current_title = None
            current_lines = []
            continue
        if level <= 2:
            _append_section(sections, document_title, current_title, current_lines)
            current_title = title
            current_lines = [line]
            continue
        current_lines.append(line)

    _append_section(sections, document_title, current_title, current_lines)
    return sections


def split_large_text(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split text into non-empty paragraph-aware word windows."""
    _validate_window(max_words, overlap_words)
    if not isinstance(text, str):
        raise ValueError("text to chunk must be a string")
    stripped = text.strip()
    if not stripped:
        return []
    if count_words(stripped) <= max_words:
        return [stripped]

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", stripped)
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        paragraph_words = count_words(paragraph)
        current_text = "\n\n".join(current)
        if paragraph_words > max_words:
            if current_text:
                chunks.append(current_text)
                current = []
            chunks.extend(_split_words(paragraph, max_words, overlap_words))
            continue

        if current and count_words(current_text) + paragraph_words > max_words:
            chunks.append(current_text)
            overlap = _tail_words(current_text, overlap_words)
            current = [part for part in (overlap, paragraph) if part]
            # An overlap plus a near-limit paragraph can itself be too large.
            if count_words("\n\n".join(current)) > max_words:
                current = [paragraph]
        else:
            current.append(paragraph)

    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def count_words(text: str) -> int:
    """Count whitespace-delimited words."""
    return len(re.findall(r"\S+", text))


def _split_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    words = re.findall(r"\S+", text)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def _tail_words(text: str, word_count: int) -> str:
    if word_count == 0:
        return ""
    return " ".join(re.findall(r"\S+", text)[-word_count:])


def _append_section(
    sections: list[MarkdownSection],
    document_title: str,
    current_title: str | None,
    current_lines: Sequence[str],
) -> None:
    content = "\n".join(current_lines).strip()
    if not content:
        return
    title = current_title or document_title
    path = (
        (document_title,) if current_title is None else (document_title, current_title)
    )
    sections.append(MarkdownSection(title=title, section_path=path, content=content))


def _render_section(document_title: str, section: MarkdownSection) -> str:
    lines = [f"Title: {document_title}"]
    if section.section_path != (document_title,):
        lines.append(f"Section: {section.title}")
    lines.append(f"Content: {section.content}")
    return "\n".join(lines)


def _split_rendered_section(
    document_title: str,
    section: MarkdownSection,
    max_words: int,
    overlap_words: int,
) -> list[str]:
    prefix = [f"Title: {document_title}"]
    if section.section_path != (document_title,):
        prefix.append(f"Section: {section.title}")
    prefix.append("Content:")
    prefix_text = "\n".join(prefix)
    available = max_words - count_words(prefix_text)
    if available <= 0:
        return split_large_text(
            _render_section(document_title, section), max_words, overlap_words
        )
    content_overlap = min(overlap_words, max(0, available - 1))
    content_parts = split_large_text(section.content, available, content_overlap)
    return [f"{prefix_text} {part}" for part in content_parts]


def _semantic_value(record: Record, wanted: str) -> str | None:
    for label, value in record.semantic_fields.items():
        if label.strip().lower() == wanted and value.strip():
            return value.strip()
    return None


def _chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}::chunk-{index:04d}"


def _validate_window(max_words: int, overlap_words: int) -> None:
    if not isinstance(max_words, int) or isinstance(max_words, bool) or max_words <= 0:
        raise ValueError("max_words must be a positive integer")
    if (
        not isinstance(overlap_words, int)
        or isinstance(overlap_words, bool)
        or overlap_words < 0
    ):
        raise ValueError("overlap_words must be a non-negative integer")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")


# ``TextChunker`` describes the generic policy more directly; ``WordChunker``
# remains the concrete implementation name in documentation and tracebacks.
TextChunker = WordChunker


__all__ = [
    "DEFAULT_MAX_WORDS",
    "DEFAULT_OVERLAP_WORDS",
    "Chunker",
    "MarkdownSection",
    "MarkdownSectionChunker",
    "TextChunker",
    "WholeRecordChunker",
    "WordChunker",
    "count_words",
    "split_large_text",
    "split_markdown_sections",
]
