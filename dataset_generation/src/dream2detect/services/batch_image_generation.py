from __future__ import annotations

import base64
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from uuid import uuid4

from ..config import load_settings
from ..openai_connector import build_openai_client
from ..storage.repository import Dream2DetectRepository


BATCH_ENDPOINT = "/v1/images/generations"


@dataclass(frozen=True)
class PreparedImageBatch:
    batch_dir: Path
    input_jsonl_path: Path
    manifest_json_path: Path
    metadata_path: Path
    request_count: int


def utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def custom_id_for_prompt_id(prompt_id: int) -> str:
    return f"prompt_{prompt_id}"


def prompt_id_from_custom_id(custom_id: str) -> int:
    prefix = "prompt_"
    if not custom_id.startswith(prefix):
        raise ValueError(f"Unsupported batch custom_id: {custom_id!r}")
    return int(custom_id.removeprefix(prefix))


def build_image_batch_request(
    *,
    prompt_id: int,
    prompt_text: str,
    model: str,
    quality: str,
    size: str,
) -> dict[str, object]:
    return {
        "custom_id": custom_id_for_prompt_id(prompt_id),
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": model,
            "prompt": prompt_text,
            "quality": quality,
            "size": size,
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def prepare_image_generation_batch(
    repository: Dream2DetectRepository,
    *,
    batch_name: str | None = None,
    prompt_status: str = "drafted",
    image_status: str = "pending",
    limit: int | None = None,
    model: str = "gpt-image-2",
    quality: str = "medium",
    size: str = "1024x1024",
    output_subdir: str = "targeted_555_round1",
    batch_root: str | Path = "data/batches/image_generation",
) -> PreparedImageBatch:
    prompts = repository.list_prompts_for_generation(
        prompt_status=prompt_status,
        image_status=image_status,
        limit=limit,
    )
    if not prompts:
        raise ValueError(
            f"No prompts found with prompt_status={prompt_status!r} "
            f"and image_status={image_status!r}."
        )

    resolved_batch_name = batch_name or f"{output_subdir}_{utc_timestamp()}"
    batch_dir = Path(batch_root) / resolved_batch_name
    batch_dir.mkdir(parents=True, exist_ok=False)

    input_jsonl_path = batch_dir / "input.jsonl"
    manifest_json_path = batch_dir / "manifest.json"
    metadata_path = batch_dir / "metadata.json"

    manifest_rows: list[dict[str, object]] = []
    with input_jsonl_path.open("w") as handle:
        for prompt_row in prompts:
            prompt_id = int(prompt_row["id"])
            request = build_image_batch_request(
                prompt_id=prompt_id,
                prompt_text=str(prompt_row["prompt_text"]),
                model=model,
                quality=quality,
                size=size,
            )
            handle.write(json.dumps(request) + "\n")
            manifest_rows.append(
                {
                    "prompt_id": prompt_id,
                    "custom_id": custom_id_for_prompt_id(prompt_id),
                    "score_band": prompt_row["score_band"],
                    "coarse_class": prompt_row["coarse_class"],
                    "prompt_uid": prompt_row["prompt_uid"],
                    "prompt_title": prompt_row["prompt_title"],
                }
            )

    metadata = {
        "batch_name": resolved_batch_name,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "prepared",
        "endpoint": BATCH_ENDPOINT,
        "completion_window": "24h",
        "request_count": len(manifest_rows),
        "input_jsonl_path": str(input_jsonl_path),
        "manifest_json_path": str(manifest_json_path),
        "output_jsonl_path": str(batch_dir / "output.jsonl"),
        "error_jsonl_path": str(batch_dir / "errors.jsonl"),
        "generated_output_subdir": output_subdir,
        "model": model,
        "quality": quality,
        "size": size,
        "prompt_status": prompt_status,
        "image_status": image_status,
        "input_file_id": None,
        "batch_id": None,
        "output_file_id": None,
        "error_file_id": None,
    }
    _write_json(manifest_json_path, {"requests": manifest_rows})
    _write_json(metadata_path, metadata)

    return PreparedImageBatch(
        batch_dir=batch_dir,
        input_jsonl_path=input_jsonl_path,
        manifest_json_path=manifest_json_path,
        metadata_path=metadata_path,
        request_count=len(manifest_rows),
    )


def submit_prepared_image_batch(batch_dir: str | Path) -> dict[str, Any]:
    batch_dir = Path(batch_dir)
    metadata_path = batch_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    if metadata.get("batch_id"):
        raise ValueError(f"Batch already submitted: {metadata['batch_id']}")

    input_jsonl_path = Path(str(metadata["input_jsonl_path"]))
    client = build_openai_client()
    with input_jsonl_path.open("rb") as handle:
        batch_input_file = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=str(metadata["endpoint"]),
        completion_window=str(metadata["completion_window"]),
        metadata={
            "project": "Dream2Detect",
            "batch_name": str(metadata["batch_name"]),
            "description": "Targeted package-defect image generation",
        },
    )

    metadata.update(
        {
            "status": batch.status,
            "input_file_id": batch_input_file.id,
            "batch_id": batch.id,
            "submitted_at": datetime.now(UTC).isoformat(),
        }
    )
    _write_json(metadata_path, metadata)
    return metadata


def refresh_image_batch_status(batch_dir: str | Path) -> dict[str, Any]:
    batch_dir = Path(batch_dir)
    metadata_path = batch_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    batch_id = metadata.get("batch_id")
    if not batch_id:
        raise ValueError(f"Batch has not been submitted yet: {batch_dir}")

    client = build_openai_client()
    batch = client.batches.retrieve(str(batch_id))
    request_counts = None
    if getattr(batch, "request_counts", None) is not None:
        request_counts_obj = batch.request_counts
        if hasattr(request_counts_obj, "model_dump"):
            request_counts = request_counts_obj.model_dump()
        else:
            request_counts = {
                "total": getattr(request_counts_obj, "total", None),
                "completed": getattr(request_counts_obj, "completed", None),
                "failed": getattr(request_counts_obj, "failed", None),
            }
    metadata.update(
        {
            "status": batch.status,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "request_counts": request_counts,
            "last_checked_at": datetime.now(UTC).isoformat(),
        }
    )
    _write_json(metadata_path, metadata)
    return metadata


def _binary_file_content(file_id: str) -> bytes:
    client = build_openai_client()
    content = client.files.content(file_id)
    if hasattr(content, "read"):
        return content.read()
    if hasattr(content, "content"):
        return bytes(content.content)
    if isinstance(content, bytes | bytearray):
        return bytes(content)
    raise TypeError(f"Unsupported file content response type: {type(content)!r}")


def _extract_image_bytes(response_body: dict[str, Any]) -> tuple[bytes, str | None]:
    data = response_body.get("data")
    if not data:
        raise ValueError("Image response body has no data array.")
    image_data = data[0]
    revised_prompt = image_data.get("revised_prompt")
    b64_json = image_data.get("b64_json")
    if b64_json:
        return base64.b64decode(b64_json), revised_prompt

    image_url = image_data.get("url")
    if image_url:
        with urlopen(image_url) as response:
            return response.read(), revised_prompt

    raise ValueError("Image response body contains neither b64_json nor url.")


def ingest_completed_image_batch(
    repository: Dream2DetectRepository,
    *,
    batch_dir: str | Path,
    force: bool = False,
) -> dict[str, int]:
    settings = load_settings()
    batch_dir = Path(batch_dir)
    metadata_path = batch_dir / "metadata.json"
    metadata = refresh_image_batch_status(batch_dir)
    if metadata["status"] != "completed":
        raise ValueError(
            f"Batch is not completed yet: status={metadata['status']!r}, "
            f"batch_id={metadata.get('batch_id')!r}"
        )
    if not metadata.get("output_file_id"):
        raise ValueError("Completed batch has no output_file_id.")

    output_jsonl_path = Path(str(metadata["output_jsonl_path"]))
    output_jsonl_path.write_bytes(_binary_file_content(str(metadata["output_file_id"])))

    if metadata.get("error_file_id"):
        error_jsonl_path = Path(str(metadata["error_jsonl_path"]))
        error_jsonl_path.write_bytes(_binary_file_content(str(metadata["error_file_id"])))

    if metadata.get("pipeline_version") == "v2_real_failure_targeted":
        return _ingest_v2_completed_image_batch(
            metadata=metadata,
            output_jsonl_path=output_jsonl_path,
            output_dir=settings.generated_images_dir / str(metadata["generated_output_subdir"]),
            force=force,
        )

    output_dir = settings.generated_images_dir / str(metadata["generated_output_subdir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    failed = 0
    skipped = 0
    for line in output_jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_id = prompt_id_from_custom_id(str(row["custom_id"]))
        prompt_row = repository.get_prompt(prompt_id)
        if prompt_row is None:
            failed += 1
            continue
        if prompt_row["image_status"] == "generated" and not force:
            skipped += 1
            continue
        if row.get("error"):
            repository.mark_prompt_image_failed(prompt_id, json.dumps(row["error"]))
            failed += 1
            continue

        response = row.get("response") or {}
        status_code = int(response.get("status_code", 0))
        if status_code != 200:
            repository.mark_prompt_image_failed(prompt_id, json.dumps(response))
            failed += 1
            continue

        image_bytes, revised_prompt = _extract_image_bytes(response.get("body") or {})
        image_uid = f"img_{uuid4().hex[:16]}"
        image_path = output_dir / f"{image_uid}.png"
        image_path.write_bytes(image_bytes)
        repository.attach_generated_image(
            prompt_id=prompt_id,
            image_uid=image_uid,
            image_model=str(metadata["model"]),
            image_quality=str(metadata["quality"]),
            image_size=str(metadata["size"]),
            image_ref=str(image_path),
            revised_prompt=revised_prompt,
        )
        generated += 1

    metadata["ingested_at"] = datetime.now(UTC).isoformat()
    metadata["ingest_counts"] = {
        "generated": generated,
        "failed": failed,
        "skipped": skipped,
    }
    _write_json(metadata_path, metadata)
    return {"generated": generated, "failed": failed, "skipped": skipped}


def _ingest_v2_completed_image_batch(
    *,
    metadata: dict[str, Any],
    output_jsonl_path: Path,
    output_dir: Path,
    force: bool,
) -> dict[str, int]:
    prompt_manifest_path = Path(str(metadata["prompt_manifest_path"]))
    prompt_rows = _read_csv_by_key(prompt_manifest_path, key="custom_id")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_manifest_path = output_jsonl_path.parent / "generated_image_manifest.csv"
    generated_rows: list[dict[str, Any]] = []
    generated = 0
    failed = 0
    skipped = 0

    for line in output_jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = str(row["custom_id"])
        prompt_row = prompt_rows.get(custom_id)
        if prompt_row is None:
            failed += 1
            generated_rows.append(
                {
                    "custom_id": custom_id,
                    "image_status": "failed",
                    "failure_reason": "custom_id missing from prompt manifest",
                }
            )
            continue

        base_record: dict[str, Any] = {
            **prompt_row,
            "image_model": metadata["model"],
            "image_quality": metadata["quality"],
            "image_size": metadata["size"],
            "batch_id": metadata["batch_id"],
            "input_file_id": metadata["input_file_id"],
            "output_file_id": metadata["output_file_id"],
        }
        if row.get("error"):
            failed += 1
            generated_rows.append(
                {
                    **base_record,
                    "image_status": "failed",
                    "image_path": "",
                    "revised_prompt": "",
                    "failure_reason": json.dumps(row["error"]),
                }
            )
            continue

        response = row.get("response") or {}
        status_code = int(response.get("status_code", 0))
        if status_code != 200:
            failed += 1
            generated_rows.append(
                {
                    **base_record,
                    "image_status": "failed",
                    "image_path": "",
                    "revised_prompt": "",
                    "failure_reason": json.dumps(response),
                }
            )
            continue

        image_path = output_dir / f"{custom_id}.png"
        if image_path.exists() and not force:
            skipped += 1
            generated_rows.append(
                {
                    **base_record,
                    "image_status": "skipped_existing",
                    "image_path": str(image_path),
                    "revised_prompt": "",
                    "failure_reason": "",
                }
            )
            continue

        image_bytes, revised_prompt = _extract_image_bytes(response.get("body") or {})
        image_path.write_bytes(image_bytes)
        generated += 1
        generated_rows.append(
            {
                **base_record,
                "image_status": "generated",
                "image_path": str(image_path),
                "revised_prompt": revised_prompt or "",
                "failure_reason": "",
            }
        )

    _write_csv(generated_manifest_path, generated_rows)
    metadata["ingested_at"] = datetime.now(UTC).isoformat()
    metadata["generated_image_manifest_path"] = str(generated_manifest_path)
    metadata["ingest_counts"] = {
        "generated": generated,
        "failed": failed,
        "skipped": skipped,
    }
    _write_json(output_jsonl_path.parent / "metadata.json", metadata)
    return {"generated": generated, "failed": failed, "skipped": skipped}


def _read_csv_by_key(path: Path, *, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for field_name in row:
            if field_name not in fieldnames:
                fieldnames.append(field_name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
