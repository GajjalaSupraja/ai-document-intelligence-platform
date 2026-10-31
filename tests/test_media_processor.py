import pytest

from app import media_processor


def test_extract_media_text_uses_ollama_vision_for_images(monkeypatch):
    monkeypatch.setattr(
        media_processor,
        "extract_image_description",
        lambda filename, content: "A chart with the title Project Costs.",
    )

    result = media_processor.extract_media_text("chart.png", b"image bytes")

    assert result == {
        "file_type": "image",
        "extracted_content": [
            {
                "text": "A chart with the title Project Costs.",
                "source": "chart.png",
            }
        ],
    }


def test_extract_media_text_transcribes_video_audio(monkeypatch):
    monkeypatch.setattr(
        media_processor,
        "transcribe_video",
        lambda filename, content: "The speaker discusses project risks.",
    )

    result = media_processor.extract_media_text("meeting.mp4", b"video bytes")

    assert result["file_type"] == "video"
    assert result["extracted_content"][0]["text"] == (
        "The speaker discusses project risks."
    )


def test_extract_media_text_rejects_unsupported_media():
    with pytest.raises(ValueError, match="Unsupported"):
        media_processor.extract_media_text("audio.mp3", b"audio bytes")


def test_image_model_error_includes_install_command(monkeypatch):
    class FakeOllama:
        @staticmethod
        def chat(**kwargs):
            raise RuntimeError("model not found")

    monkeypatch.setattr(media_processor, "ollama", FakeOllama())

    with pytest.raises(RuntimeError, match="ollama pull"):
        media_processor.extract_image_description("chart.png", b"image bytes")
