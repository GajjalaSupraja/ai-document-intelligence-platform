from app.document_store import DocumentStore


def test_document_store_persists_document_and_chunks(tmp_path):
    database_path = tmp_path / "documents.sqlite3"
    original_file = b"original document bytes"
    chunks = [
        {
            "text": "Shadow BIM affects document traceability.",
            "source": "report.docx",
            "chunk_index": 0,
        }
    ]

    DocumentStore(database_path).save(
        document_id="document-1",
        filename="report.docx",
        file_type="docx",
        chunks=chunks,
        original_file=original_file,
    )

    reloaded_store = DocumentStore(database_path)
    saved_document = reloaded_store.get("document-1")

    assert saved_document is not None
    assert saved_document["filename"] == "report.docx"
    assert saved_document["file_type"] == "docx"
    assert saved_document["chunk_count"] == 1
    assert saved_document["chunks"] == chunks
    assert saved_document["created_at"]
    stored_file = reloaded_store.get_original_file("document-1")
    assert stored_file == {
        "filename": "report.docx",
        "file_type": "docx",
        "content": original_file,
    }


def test_document_store_returns_none_for_unknown_original_file(tmp_path):
    store = DocumentStore(tmp_path / "documents.sqlite3")

    assert store.get_original_file("missing-document") is None


def test_document_store_lists_and_deletes_documents(tmp_path):
    store = DocumentStore(tmp_path / "documents.sqlite3")
    store.save(
        document_id="document-1",
        filename="report.pdf",
        file_type="pdf",
        chunks=[{"text": "Document text.", "source": "report.pdf"}],
    )

    listed_documents = store.list_documents()

    assert len(listed_documents) == 1
    assert listed_documents[0]["document_id"] == "document-1"
    assert listed_documents[0]["filename"] == "report.pdf"
    assert "chunks" not in listed_documents[0]
    assert store.delete("document-1") is True
    assert store.get("document-1") is None
    assert store.delete("document-1") is False


def test_document_store_persists_conversation_and_deletes_it_with_document(tmp_path):
    store = DocumentStore(tmp_path / "documents.sqlite3")
    store.save(
        document_id="document-1",
        filename="report.pdf",
        file_type="pdf",
        chunks=[{"text": "Document text.", "source": "report.pdf"}],
    )
    sources = [{"text": "Supporting passage.", "score": 0.9}]

    store.save_conversation(
        document_id="document-1",
        question="What is in this report?",
        answer="It contains a supporting passage.",
        sources=sources,
    )

    history = DocumentStore(tmp_path / "documents.sqlite3").list_conversations(
        "document-1"
    )
    assert len(history) == 1
    assert history[0]["question"] == "What is in this report?"
    assert history[0]["answer"] == "It contains a supporting passage."
    assert history[0]["sources"] == sources
    assert history[0]["created_at"]

    assert store.delete("document-1") is True
    assert store.list_conversations("document-1") == []


def test_document_store_clears_conversation_without_deleting_document(tmp_path):
    store = DocumentStore(tmp_path / "documents.sqlite3")
    store.save(
        document_id="document-1",
        filename="report.pdf",
        file_type="pdf",
        chunks=[{"text": "Document text.", "source": "report.pdf"}],
    )
    store.save_conversation(
        document_id="document-1",
        question="Question",
        answer="Answer",
        sources=[],
    )

    assert store.clear_conversations("document-1") is True
    assert store.list_conversations("document-1") == []
    assert store.get("document-1") is not None
    assert store.clear_conversations("missing-document") is False
