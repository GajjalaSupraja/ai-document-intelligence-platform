from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from app.config import settings

try:
    import ollama
except ImportError:
    class _OllamaFallback:
        chat = None

    ollama = _OllamaFallback()


@lru_cache(maxsize=1)
def _get_transcription_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "Video transcription requires faster-whisper. "
            "Install it with: python -m pip install faster-whisper"
        ) from error

    return WhisperModel(
        settings.transcription_model,
        device="cpu",
        compute_type="int8",
    )


def extract_image_description(filename: str, content: bytes) -> str:
    chat_callable = getattr(ollama, "chat", None)
    if chat_callable is None:
        raise RuntimeError(
            "The Ollama client is not available. Install the 'ollama' package."
        )

    try:
        response = chat_callable(
            model=settings.ollama_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Describe this image accurately. Extract all readable text, "
                        "including labels, headings, and visible numbers. Separate "
                        "observations from any interpretation."
                    ),
                    "images": [content],
                }
            ],
        )
    except Exception as error:
        raise RuntimeError(
            "Image analysis could not use Ollama vision model "
            f"'{settings.ollama_vision_model}'. Make sure Ollama is running, "
            f"then run `ollama pull {settings.ollama_vision_model}`. "
            f"Ollama error: {error}"
        ) from error
    description = response.get("message", {}).get("content", "")
    description = str(description).strip() if description is not None else ""

    if not description:
        raise RuntimeError("The vision model returned an empty image description.")

    return description


def transcribe_video(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if not extension:
        raise ValueError("The video filename must include a file extension.")

    with NamedTemporaryFile(suffix=extension, delete=True) as temporary_file:
        temporary_file.write(content)
        temporary_file.flush()
        model = _get_transcription_model()
        segments, _ = model.transcribe(
            temporary_file.name,
            beam_size=5,
            vad_filter=True,
        )
        transcript = "\n".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        )

    if not transcript:
        raise ValueError(
            "No speech was detected in the video. "
            "Videos without a usable audio track are not supported."
        )

    return transcript


def extract_media_text(filename: str, content: bytes) -> dict:
    extension = Path(filename).suffix.lower().lstrip(".")

    if extension in {"png", "jpg", "jpeg", "webp"}:
        text = extract_image_description(filename, content)
        media_type = "image"
    elif extension in {"mp4", "mov", "mkv", "webm"}:
        text = transcribe_video(filename, content)
        media_type = "video"
    else:
        raise ValueError("Unsupported image or video file type.")

    return {
        "file_type": media_type,
        "extracted_content": [{"text": text, "source": filename}],
    }
