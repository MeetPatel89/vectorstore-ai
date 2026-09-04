"""Source adapters, chunkers, and lifecycle-aware ingestion."""

from .base import Source, SourceAdapter, SourceAdapterError
from .chunkers import (
    DEFAULT_MAX_WORDS,
    DEFAULT_OVERLAP_WORDS,
    Chunker,
    MarkdownSection,
    MarkdownSectionChunker,
    TextChunker,
    WholeRecordChunker,
    WordChunker,
    count_words,
    split_large_text,
    split_markdown_sections,
)
from .csv import CSVSourceAdapter, CsvSourceAdapter
from .json import JSONSourceAdapter, JsonSourceAdapter
from .markdown import MarkdownSourceAdapter, parse_markdown_frontmatter
from .pipeline import (
    FallbackIndexMode,
    IngestionConfig,
    IngestionError,
    IngestionPipeline,
    IngestionResult,
    PrimaryProviderRequiredError,
    ReembeddingResult,
)

__all__ = [
    "CSVSourceAdapter",
    "DEFAULT_MAX_WORDS",
    "DEFAULT_OVERLAP_WORDS",
    "FallbackIndexMode",
    "JSONSourceAdapter",
    "Chunker",
    "CsvSourceAdapter",
    "IngestionConfig",
    "IngestionError",
    "IngestionPipeline",
    "IngestionResult",
    "JsonSourceAdapter",
    "MarkdownSection",
    "MarkdownSectionChunker",
    "MarkdownSourceAdapter",
    "PrimaryProviderRequiredError",
    "ReembeddingResult",
    "Source",
    "SourceAdapter",
    "SourceAdapterError",
    "TextChunker",
    "WholeRecordChunker",
    "WordChunker",
    "count_words",
    "parse_markdown_frontmatter",
    "split_large_text",
    "split_markdown_sections",
]
