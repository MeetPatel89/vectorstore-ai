from __future__ import annotations

import pytest

from vectorstore import (
    Chunker,
    MarkdownSectionChunker,
    Record,
    WholeRecordChunker,
    WordChunker,
    count_words,
    split_markdown_sections,
)


def markdown_record(body: str) -> Record:
    return Record(
        id="RUNBOOK-1",
        semantic_fields={"Title": "Gateway Runbook", "Body": body},
        structured={"doc_type": "runbook"},
    )


def test_whole_record_chunker_uses_stable_id() -> None:
    chunker = WholeRecordChunker()
    record = markdown_record("Use the recovery command.")

    chunks = chunker.chunk(record)

    assert isinstance(chunker, Chunker)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "RUNBOOK-1::chunk-0000"
    assert chunks[0].doc_id == record.id
    assert chunks[0].text.startswith("Title: Gateway Runbook")


def test_word_chunker_splits_long_text_with_overlap() -> None:
    record = Record(
        "DOC-1",
        {"Body": " ".join(f"word-{index}" for index in range(12))},
    )
    chunks = WordChunker(max_words=5, overlap_words=2).chunk(record)

    assert [chunk.chunk_id for chunk in chunks] == [
        "DOC-1::chunk-0000",
        "DOC-1::chunk-0001",
        "DOC-1::chunk-0002",
        "DOC-1::chunk-0003",
    ]
    assert all(count_words(chunk.text) <= 5 for chunk in chunks)
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]


def test_markdown_chunker_splits_h2_and_repeats_context() -> None:
    record = markdown_record(
        "Intro.\n\n## Triage\n\nCollect logs.\n\n## Recovery\n\nRestart safely."
    )

    chunks = MarkdownSectionChunker().chunk(record)

    assert len(chunks) == 3
    assert [chunk.section_path for chunk in chunks] == [
        "Gateway Runbook",
        "Gateway Runbook > Triage",
        "Gateway Runbook > Recovery",
    ]
    assert all(chunk.text.startswith("Title: Gateway Runbook") for chunk in chunks)
    assert "Section: Triage" in chunks[1].text
    assert "Collect logs." in chunks[1].text


def test_split_markdown_sections_honors_first_h1() -> None:
    sections = split_markdown_sections(
        "# Actual title\n\nIntro\n\n## Steps\n\nDo it", "Fallback"
    )
    assert sections[0].section_path == ("Actual title",)
    assert sections[1].section_path == ("Actual title", "Steps")


def test_chunk_window_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        WordChunker(max_words=0)
    with pytest.raises(ValueError, match="smaller"):
        MarkdownSectionChunker(max_words=10, overlap_words=10)
