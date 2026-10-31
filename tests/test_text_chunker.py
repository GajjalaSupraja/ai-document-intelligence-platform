import pytest

from app.text_chunker import chunk_text, chunk_sections


def test_chunk_text_returns_empty_for_empty_input():
    assert chunk_text("") == []


def test_chunk_text_splits_long_text():
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert len(chunks) == 3
    assert len(chunks[0].split()) == 500
    assert len(chunks[1].split()) == 500
    assert len(chunks[2].split()) == 400


def test_chunk_text_preserves_overlap():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = chunk_text(text, chunk_size=500, overlap=100)

    assert chunks[0].split()[-100:] == chunks[1].split()[:100]


def test_chunk_text_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0)

    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=10)


def test_chunk_sections_keeps_source_metadata():
    sections = [
        {
            "text": "Shadow BIM affects document traceability.",
            "source": "report.docx",
        }
    ]

    chunks = chunk_sections(sections, chunk_size=10, overlap=2)

    assert len(chunks) == 1
    assert chunks[0]["source"] == "report.docx"
    assert "Shadow BIM" in chunks[0]["text"]