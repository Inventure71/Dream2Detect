from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from dream2detect.services.batch_image_generation import (
    BATCH_ENDPOINT,
    _extract_image_bytes,
    build_image_batch_request,
    prepare_image_generation_batch,
    prompt_id_from_custom_id,
)


class FakePromptRepository:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def list_prompts_for_generation(
        self,
        prompt_status: str = "approved",
        image_status: str = "pending",
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        rows = [
            row
            for row in self.rows
            if row["prompt_status"] == prompt_status
            and row["image_status"] == image_status
        ]
        return rows if limit is None else rows[:limit]


class ImageBatchTests(unittest.TestCase):
    def test_build_image_batch_request_uses_images_endpoint_and_medium_quality(self) -> None:
        request = build_image_batch_request(
            prompt_id=241,
            prompt_text="A clean intact cardboard box.",
            model="gpt-image-2",
            quality="medium",
            size="1024x1024",
        )

        self.assertEqual(request["custom_id"], "prompt_241")
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["url"], BATCH_ENDPOINT)
        self.assertEqual(request["body"]["quality"], "medium")  # type: ignore[index]
        self.assertEqual(request["body"]["model"], "gpt-image-2")  # type: ignore[index]

    def test_prepare_image_generation_batch_writes_jsonl_manifest_and_metadata(self) -> None:
        rows = [
            {
                "id": 241,
                "prompt_status": "drafted",
                "image_status": "pending",
                "prompt_text": "A clean intact cardboard box.",
                "score_band": "0-10",
                "coarse_class": "intact",
                "prompt_uid": "prompt_a",
                "prompt_title": "Clean box",
            },
            {
                "id": 242,
                "prompt_status": "drafted",
                "image_status": "pending",
                "prompt_text": "A lightly softened cardboard box corner.",
                "score_band": "11-20",
                "coarse_class": "minor",
                "prompt_uid": "prompt_b",
                "prompt_title": "Minor corner",
            },
        ]
        repository = FakePromptRepository(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = prepare_image_generation_batch(
                repository,  # type: ignore[arg-type]
                batch_name="test_batch",
                batch_root=Path(temp_dir),
            )

            jsonl_rows = [
                json.loads(line)
                for line in prepared.input_jsonl_path.read_text().splitlines()
            ]
            metadata = json.loads(prepared.metadata_path.read_text())
            manifest = json.loads(prepared.manifest_json_path.read_text())

            self.assertEqual(prepared.request_count, 2)
            self.assertEqual(jsonl_rows[0]["custom_id"], "prompt_241")
            self.assertEqual(jsonl_rows[0]["url"], "/v1/images/generations")
            self.assertEqual(jsonl_rows[0]["body"]["quality"], "medium")
            self.assertEqual(metadata["status"], "prepared")
            self.assertEqual(metadata["request_count"], 2)
            self.assertEqual(manifest["requests"][1]["prompt_id"], 242)

    def test_extract_image_bytes_decodes_base64_batch_image_response(self) -> None:
        encoded = base64.b64encode(b"fake-png").decode("ascii")

        image_bytes, revised_prompt = _extract_image_bytes(
            {
                "data": [
                    {
                        "b64_json": encoded,
                        "revised_prompt": "revised",
                    }
                ]
            }
        )

        self.assertEqual(image_bytes, b"fake-png")
        self.assertEqual(revised_prompt, "revised")
        self.assertEqual(prompt_id_from_custom_id("prompt_795"), 795)


if __name__ == "__main__":
    unittest.main()
