from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = REPO_ROOT / "data" / "dream2detect.sqlite3"
DEFAULT_GENERATED_IMAGES_DIR = REPO_ROOT / "generated_images"
DEFAULT_BAND_SPECS_DIR = REPO_ROOT / "docs" / "06-package-severity-bands"


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str | None
    prompt_model: str = "gpt-5.4-mini"
    image_model: str = "gpt-image-2"
    image_quality: str = "low"
    image_size: str = "1024x1024"


@dataclass(frozen=True)
class AppSettings:
    database_path: Path = DEFAULT_DATABASE_PATH
    generated_images_dir: Path = DEFAULT_GENERATED_IMAGES_DIR
    band_specs_dir: Path = DEFAULT_BAND_SPECS_DIR
    openai: OpenAISettings = OpenAISettings(api_key=None)


def load_settings() -> AppSettings:
    load_dotenv(REPO_ROOT / ".env")
    database_path = Path(os.environ.get("DREAM2DETECT_DB_PATH", DEFAULT_DATABASE_PATH))
    generated_images_dir = Path(
        os.environ.get("DREAM2DETECT_GENERATED_IMAGES_DIR", DEFAULT_GENERATED_IMAGES_DIR)
    )
    band_specs_dir = Path(os.environ.get("DREAM2DETECT_BAND_SPECS_DIR", DEFAULT_BAND_SPECS_DIR))
    return AppSettings(
        database_path=database_path,
        generated_images_dir=generated_images_dir,
        band_specs_dir=band_specs_dir,
        openai=OpenAISettings(api_key=os.environ.get("OPENAI_API_KEY")),
    )
