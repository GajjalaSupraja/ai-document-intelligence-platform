
import numpy as np
import pytest

from app.retrieval_service import RetrievalService


class FakeEmbeddingService:
    """A predictable embedding service for testing retrieval."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed([query])

    @staticmethod
    def _embed(texts: list[str]) -> np.ndarray:
        vectors = []

        for text in texts:
            normalized = text.lower()

            if "shadow bim" in normalized or "construction claims" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "project cost" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return np.asarray(vectors, dtype="float32")


def test_index_chunks_returns_count():
    service = RetrievalService(FakeEmbeddingService())

    chunks = [
        {"text": "Shadow BIM affects construction claims.", "source": "report.docx"},
        {"text": "Project cost management information.", "source": "cost.docx"},
    ]

    count = service.index_chunks(chunks)

    assert count == 2


def test_search_returns_relevant_chunk():
    service = RetrievalService(FakeEmbeddingService())

    chunks = [
        {"text": "Shadow BIM affects construction claims.", "source": "report.docx"},
        {"text": "Project cost management information.", "source": "cost.docx"},
    ]

    service.index_chunks(chunks)
    results = service.search("Explain Shadow BIM", top_k=1)

    assert len(results) == 1
    assert results[0]["source"] == "report.docx"
    assert "Shadow BIM" in results[0]["text"]


def test_search_rejects_empty_question():
    service = RetrievalService(FakeEmbeddingService())
    service.index_chunks([{"text": "Example document text.", "source": "test.docx"}])

    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.search("   ")


def test_search_requires_indexing_first():
    service = RetrievalService(FakeEmbeddingService())

    with pytest.raises(ValueError, match="Index a document before searching"):
        service.search("Explain Shadow BIM")


def test_index_chunks_rejects_empty_list():
    service = RetrievalService(FakeEmbeddingService())

    with pytest.raises(ValueError, match="No text chunks provided"):
        service.index_chunks([])
