from __future__ import annotations

from openai import OpenAI

from .config import load_settings


def build_openai_client() -> OpenAI:
    settings = load_settings()
    if not settings.openai.api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running prompt drafting or image generation."
        )
    return OpenAI(api_key=settings.openai.api_key)
