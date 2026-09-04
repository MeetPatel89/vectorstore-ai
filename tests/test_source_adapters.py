from __future__ import annotations

from pathlib import Path

import pytest

from vectorstore import (
    CSVSourceAdapter,
    JSONSourceAdapter,
    MarkdownSourceAdapter,
    SourceAdapter,
    SourceAdapterError,
    parse_markdown_frontmatter,
)


def test_markdown_adapter_reads_frontmatter_and_nested_values(tmp_path: Path) -> None:
    path = tmp_path / "incident.md"
    path.write_text(
        """---
doc_id: INC-1
title: Login failure
status: open
severity: 2
related_records:
  - CHG-1
servicenow:
  number: INC0001
---
# Login failure

## Symptoms

Users cannot sign in.
""",
        encoding="utf-8",
    )

    adapter = MarkdownSourceAdapter()
    (record,) = list(adapter.iter_records(path))

    assert isinstance(adapter, SourceAdapter)
    assert record.id == "INC-1"
    assert record.semantic_fields["Title"] == "Login failure"
    assert record.semantic_fields["Body"].startswith("## Symptoms")
    assert record.structured == {
        "doc_id": "INC-1",
        "title": "Login failure",
        "status": "open",
        "severity": 2,
    }
    assert "servicenow" not in record.structured


def test_markdown_adapter_loads_directory_in_stable_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# B\n\nBeta", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\nAlpha", encoding="utf-8")

    records = list(MarkdownSourceAdapter().iter_records(tmp_path))

    assert [record.id for record in records] == ["a", "b"]
    assert [record.source for record in records] == ["a.md", "b.md"]


def test_markdown_frontmatter_without_closing_boundary_is_plain_text() -> None:
    raw = "---\ntitle: not frontmatter\n# Body"
    assert parse_markdown_frontmatter(raw) == ({}, raw)


def test_csv_adapter_infers_id_and_semantic_fields(tmp_path: Path) -> None:
    path = tmp_path / "faq.csv"
    path.write_text(
        "ID,Category,Question,Answer,Last Updated\n"
        "101,Shipping,How long?,Three days,2026-09-01\n",
        encoding="utf-8",
    )

    (record,) = list(CSVSourceAdapter().iter_records(path))

    assert record.id == "101"
    assert record.semantic_fields == {
        "Question": "How long?",
        "Answer": "Three days",
    }
    assert record.structured == {
        "Category": "Shipping",
        "Last Updated": "2026-09-01",
    }


def test_csv_adapter_coerces_structured_numbers_and_booleans(tmp_path: Path) -> None:
    path = tmp_path / "terms.csv"
    path.write_text(
        "Term ID,Detail Text,Variable Rate,Fee Amount\n"
        "CC-901,Variable terms,true,495\n",
        encoding="utf-8",
    )

    (record,) = list(CSVSourceAdapter().iter_records(path))

    assert record.structured == {"Variable Rate": True, "Fee Amount": 495}


def test_csv_adapter_supports_explicit_labels_and_fields(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("key,copy,state\na,Hello,live\n", encoding="utf-8")

    adapter = CSVSourceAdapter(
        id_field="key",
        semantic_fields={"Body": "copy"},
        structured_fields=("state",),
    )
    (record,) = list(adapter.iter_records(path))

    assert record.semantic_fields == {"Body": "Hello"}
    assert record.structured == {"state": "live"}


def test_json_adapter_reads_array_wrapper_and_json_lines(tmp_path: Path) -> None:
    wrapped = tmp_path / "records.json"
    wrapped.write_text(
        '{"records":[{"doc_id":"D1","title":"One","status":"live"}]}',
        encoding="utf-8",
    )
    lines = tmp_path / "records.jsonl"
    lines.write_text('{"doc_id":"D2","text":"Two","rank":2}\n\n', encoding="utf-8")

    adapter = JSONSourceAdapter()
    first = list(adapter.iter_records(wrapped))
    second = list(adapter.iter_records(lines))

    assert first[0].id == "D1"
    assert first[0].semantic_fields == {"title": "One"}
    assert first[0].structured == {"status": "live"}
    assert second[0].semantic_fields == {"text": "Two"}
    assert second[0].structured == {"rank": 2}


def test_mapped_adapter_requires_inferable_semantic_content(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('{"id":"1","status":"open"}', encoding="utf-8")

    with pytest.raises(SourceAdapterError, match="semantic fields"):
        list(JSONSourceAdapter().iter_records(path))
