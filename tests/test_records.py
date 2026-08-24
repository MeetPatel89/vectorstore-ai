from __future__ import annotations

import pytest

from vectorstore import Record, content_hash, semantic_projection


def make_record(**overrides) -> Record:
    fields = {
        "id": "INC-1104",
        "semantic_fields": {
            "Title": "Payment reporting data missing",
            "Category": "Reporting",
            "Description": "Dashboard totals lag exports.",
        },
        "structured": {"status": "OPEN", "severity": 3},
        "source": "incidents.csv",
    }
    fields.update(overrides)
    return Record(**fields)


def test_projection_renders_labeled_fields_in_order() -> None:
    assert semantic_projection(make_record()) == (
        "Title: Payment reporting data missing\n"
        "Category: Reporting\n"
        "Description: Dashboard totals lag exports."
    )


def test_projection_skips_empty_values_and_strips_whitespace() -> None:
    record = make_record(
        semantic_fields={
            "Title": "  Payment reporting data missing  ",
            "Symptoms": "   ",
            "Resolution": "",
        },
    )

    assert semantic_projection(record) == "Title: Payment reporting data missing"


def test_projection_excludes_structured_attributes() -> None:
    projection = semantic_projection(make_record())

    assert "OPEN" not in projection
    assert "severity" not in projection


def test_projection_requires_semantic_content() -> None:
    record = make_record(semantic_fields={"Title": "  "})

    with pytest.raises(ValueError, match="no non-empty semantic fields"):
        semantic_projection(record)


def test_record_validation() -> None:
    with pytest.raises(ValueError, match="record IDs"):
        make_record(id="")
    with pytest.raises(ValueError, match="must be a string"):
        make_record(semantic_fields={"Title": 42})
    with pytest.raises(ValueError, match="labels"):
        make_record(semantic_fields={"": "text"})


def test_content_hash_is_stable_and_content_sensitive() -> None:
    text = semantic_projection(make_record())

    assert content_hash(text) == content_hash(text)
    assert len(content_hash(text)) == 64
    assert content_hash(text) != content_hash(text + " updated")
    with pytest.raises(ValueError, match="must be a string"):
        content_hash(None)
