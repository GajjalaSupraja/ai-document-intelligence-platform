import json

from fastapi.testclient import TestClient

from app import main


class FakeAnswerService:
    def stream_answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
        output_language: str,
    ):
        assert question == "What is the main topic?"
        assert retrieved_chunks[0]["source"] == "report.pdf"
        assert output_language == "English"
        yield "The main topic "
        yield "is project management."


class FakeRetrievalService:
    def search(self, question: str, top_k: int) -> list[dict]:
        return [
            {
                "text": "The document is about project management.",
                "source": "report.pdf",
                "score": 0.9,
            }
        ]


def test_streaming_ask_returns_tokens_and_persists_completed_answer(monkeypatch):
    persisted_conversations = []
    document = {
        "filename": "report.pdf",
        "retrieval_service": FakeRetrievalService(),
    }
    monkeypatch.setattr(main, "get_document", lambda document_id: document)
    monkeypatch.setattr(main, "answer_service", FakeAnswerService())
    monkeypatch.setattr(
        main.document_store,
        "save_conversation",
        lambda **conversation: persisted_conversations.append(conversation),
    )

    client = TestClient(main.app)
    response = client.post(
        "/ask/stream",
        json={
            "document_id": "document-1",
            "question": "What is the main topic?",
            "top_k": 3,
            "output_language": "English",
        },
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "sources"
    assert [event["content"] for event in events if event["type"] == "token"] == [
        "The main topic ",
        "is project management.",
    ]
    assert events[-1] == {"type": "done"}
    assert persisted_conversations == [
        {
            "document_id": "document-1",
            "question": "What is the main topic?",
            "answer": "The main topic is project management.",
            "sources": events[0]["sources"],
        }
    ]
