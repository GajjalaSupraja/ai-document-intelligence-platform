from fastapi.testclient import TestClient

from app import main


class FakeAnswerService:
    def summarize_document(self, chunks: list[dict], output_language: str) -> str:
        assert chunks == [{"text": "Full document content."}]
        assert output_language == "Hindi"
        return "यह दस्तावेज़ परियोजना प्रबंधन के बारे में है।"


def test_summarize_api_returns_and_persists_summary(monkeypatch):
    persisted_conversations = []
    document = {
        "filename": "report.docx",
        "chunks": [{"text": "Full document content."}],
    }
    monkeypatch.setattr(main, "get_document", lambda document_id: document)
    monkeypatch.setattr(main, "answer_service", FakeAnswerService())
    monkeypatch.setattr(
        main.document_store,
        "save_conversation",
        lambda **conversation: persisted_conversations.append(conversation),
    )

    response = TestClient(main.app).post(
        "/summarize",
        json={
            "document_id": "document-1",
            "output_language": "Hindi",
        },
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "यह दस्तावेज़ परियोजना प्रबंधन के बारे में है।"
    assert response.json()["output_language"] == "Hindi"
    assert persisted_conversations[0]["question"] == (
        "Summarize this document in Hindi"
    )
