
import os

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    app_name: str = "AI Document Intelligence Platform"
    max_file_size_mb: int = 10
    max_video_size_mb: int = 100
    allowed_extensions: tuple[str, ...] = (
        "pdf",
        "docx",
        "png",
        "jpg",
        "jpeg",
        "webp",
        "mp4",
        "mov",
        "mkv",
        "webm",
    )
    ollama_model: str = "llama3.2:3b"
    ollama_vision_model: str = "llama3.2-vision:11b"
    transcription_model: str = "base"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )


settings = Settings()

DATA_DIR.mkdir(parents=True, exist_ok=True)
