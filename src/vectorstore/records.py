"""Source records, semantic projection, and content hashing for ingestion.

A :class:`Record` separates the two roles a source row or document plays in
retrieval:

- ``semantic_fields`` carry the natural-language content that communicates
  meaning (titles, descriptions, symptoms, resolutions). Only these fields
  are rendered into the text that gets embedded and lexically indexed.
- ``structured`` carries the attributes used for filtering and metadata
  (identifiers, status, category, dates, tenant, visibility). These are
  never embedded merely because they exist on the source record.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from vectorstore.models import MetadataValue


@dataclass(frozen=True)
class Record:
    """A source record split into semantic content and structured attributes.

    ``semantic_fields`` maps human-readable labels to text values; insertion
    order is preserved in the rendered projection. ``structured`` holds the
    filterable attributes that accompany the record through indexing.
    """

    id: str
    semantic_fields: dict[str, str]
    structured: dict[str, MetadataValue] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("record IDs must be non-empty strings")
        if not isinstance(self.semantic_fields, dict):
            raise ValueError("semantic_fields must be a dictionary")
        for label, value in self.semantic_fields.items():
            if not isinstance(label, str) or not label:
                raise ValueError("semantic field labels must be non-empty strings")
            if not isinstance(value, str):
                raise ValueError(
                    f"semantic field {label!r} must be a string, "
                    f"got {type(value).__name__}"
                )
        if not isinstance(self.structured, dict):
            raise ValueError("structured attributes must be a dictionary")


def semantic_projection(record: Record) -> str:
    """Render a record's semantic fields as labeled text for indexing.

    Fields whose values are empty or whitespace-only are omitted. Structured
    attributes are deliberately excluded; they remain available separately
    for filtering and metadata.
    """
    lines = [
        f"{label}: {value.strip()}"
        for label, value in record.semantic_fields.items()
        if value.strip()
    ]
    if not lines:
        raise ValueError(
            f"record {record.id!r} has no non-empty semantic fields to project"
        )
    return "\n".join(lines)


def content_hash(text: str) -> str:
    """Return a stable hex digest of semantic content.

    Paired with an :class:`~vectorstore.embeddings.base.EmbeddingSpec` space
    identifier, this answers whether an existing vector still represents the
    current semantic content, so unchanged content is never re-embedded.
    """
    if not isinstance(text, str):
        raise ValueError("content to hash must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
