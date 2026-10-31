
from unittest.mock import patch

import pytest

from app.answer_service import AnswerService


def test_rejects_empty_question():
    service = AnswerService()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.generate_answer("   ", [{"text": "Example text."}])


def test_returns_message_when_no_chunks():
    service = AnswerService()

    answer = service.generate_answer(
        "What is Shadow BIM?",
        [],
    )

    assert "could not find relevant information" in answer.lower()


@patch("app.answer_service.ollama.chat")
def test_generates_answer_from_retrieved_chunks(mock_chat):
    mock_chat.return_value = {
        "message": {
            "content": "Shadow BIM refers to unofficial BIM model iterations."
        }
    }

    service = AnswerService()

    answer = service.generate_answer(
        "What is Shadow BIM?",
        [
            {
                "text": "Shadow BIM refers to unofficial model iterations.",
                "source": "report.docx",
            }
        ],
    )

    assert "Shadow BIM" in answer
    mock_chat.assert_called_once()


@patch("app.answer_service.ollama.chat")
def test_rejects_empty_model_response(mock_chat):
    mock_chat.return_value = {
        "message": {"content": "   "}
    }

    service = AnswerService()

    with pytest.raises(RuntimeError, match="empty answer"):
        service.generate_answer(
            "What is Shadow BIM?",
            [{"text": "Example text.", "source": "report.docx"}],
        )


@patch("app.answer_service.ollama.chat")
def test_streams_answer_chunks(mock_chat):
    mock_chat.return_value = iter(
        [
            {"message": {"content": "Shadow BIM "}},
            {"message": {"content": "is an unofficial model iteration."}},
        ]
    )

    service = AnswerService()
    answer_chunks = list(
        service.stream_answer(
            "What is Shadow BIM?",
            [{"text": "Shadow BIM is an unofficial model iteration.", "source": "report.docx"}],
        )
    )

    assert answer_chunks == [
        "Shadow BIM ",
        "is an unofficial model iteration.",
    ]
    mock_chat.assert_called_once()
    assert mock_chat.call_args.kwargs["stream"] is True


def test_stream_returns_fallback_without_retrieved_chunks():
    service = AnswerService()

    answer_chunks = list(service.stream_answer("Question?", []))

    assert len(answer_chunks) == 1
    assert "could not find relevant information" in answer_chunks[0].lower()


@patch("app.answer_service.ollama.chat")
def test_stream_rejects_empty_model_response(mock_chat):
    mock_chat.return_value = iter([{"message": {"content": ""}}])

    service = AnswerService()

    with pytest.raises(RuntimeError, match="empty answer"):
        list(
            service.stream_answer(
                "What is Shadow BIM?",
                [{"text": "Example text.", "source": "report.docx"}],
            )
        )


@patch("app.answer_service.ollama.chat")
def test_summarize_document_summarizes_chunks_in_requested_language(mock_chat):
    mock_chat.side_effect = [
        {"message": {"content": "మొదటి భాగం."}},
        {"message": {"content": "రెండవ భాగం."}},
        {"message": {"content": "మొత్తం సారాంశం."}},
    ]

    answer = AnswerService().summarize_document(
        [
            {"text": "First document section."},
            {"text": "Second document section."},
        ],
        output_language="Telugu",
    )

    assert answer == "మొత్తం సారాంశం."
    assert mock_chat.call_count == 3
    for call in mock_chat.call_args_list:
        assert "Telugu" in call.kwargs["messages"][0]["content"]
