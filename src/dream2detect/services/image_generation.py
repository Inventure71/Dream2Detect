from __future__ import annotations

import base64
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

from ..config import load_settings
from ..openai_connector import build_openai_client
from ..storage.repository import Dream2DetectRepository


def _choose_output_dir(output_subdir: str) -> Path:
    settings = load_settings()
    output_dir = settings.generated_images_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _generate_single_image(
    prompt_row,
    *,
    output_dir: Path,
    settings,
) -> tuple[int, Path]:
    client = build_openai_client()
    result = client.images.generate(
        model=settings.openai.image_model,
        prompt=prompt_row["prompt_text"],
        quality=settings.openai.image_quality,
        size=settings.openai.image_size,
    )
    image_data = result.data[0]
    if getattr(image_data, "b64_json", None):
        image_bytes = base64.b64decode(image_data.b64_json)
    elif getattr(image_data, "url", None):
        with urlopen(image_data.url) as response:
            image_bytes = response.read()
    else:
        raise RuntimeError("Image API response contained neither b64_json nor url.")

    image_uid = f"img_{uuid4().hex[:16]}"
    image_path = output_dir / f"{image_uid}.png"
    image_path.write_bytes(image_bytes)
    revised_prompt = getattr(image_data, "revised_prompt", None)
    return int(prompt_row["id"]), image_path, image_uid, revised_prompt


def generate_images_for_pending_prompts(
    repository: Dream2DetectRepository,
    *,
    prompt_status: str = "approved",
    output_subdir: str = "pilot",
    limit: int | None = None,
    max_concurrency: int = 1,
) -> list[Path]:
    settings = load_settings()
    output_dir = _choose_output_dir(output_subdir)
    prompts = repository.list_prompts_for_generation(
        prompt_status=prompt_status,
        image_status="pending",
        limit=limit,
    )
    written_paths: list[Path] = []
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1.")

    for start in range(0, len(prompts), max_concurrency):
        wave = prompts[start : start + max_concurrency]
        for prompt_row in wave:
            repository.mark_prompt_image_in_progress(int(prompt_row["id"]))

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_prompt = {
                executor.submit(
                    _generate_single_image,
                    prompt_row,
                    output_dir=output_dir,
                    settings=settings,
                ): prompt_row
                for prompt_row in wave
            }
            done, not_done = wait(future_to_prompt, return_when=FIRST_EXCEPTION)

            failure: Exception | None = None
            failure_prompt = None

            for future in done:
                prompt_row = future_to_prompt[future]
                prompt_id = int(prompt_row["id"])
                try:
                    generated_prompt_id, image_path, image_uid, revised_prompt = future.result()
                    repository.attach_generated_image(
                        prompt_id=generated_prompt_id,
                        image_uid=image_uid,
                        image_model=settings.openai.image_model,
                        image_quality=settings.openai.image_quality,
                        image_size=settings.openai.image_size,
                        image_ref=str(image_path),
                        revised_prompt=revised_prompt,
                    )
                    written_paths.append(image_path)
                except Exception as exc:
                    repository.mark_prompt_image_failed(prompt_id, str(exc))
                    failure = exc
                    failure_prompt = prompt_row

            if failure is not None:
                for future in not_done:
                    future.cancel()
                raise RuntimeError(
                    f"Image generation failed for prompt id={failure_prompt['id']}, "
                    f"band={failure_prompt['score_band']}, title={failure_prompt['prompt_title']!r}"
                ) from failure

    return written_paths
