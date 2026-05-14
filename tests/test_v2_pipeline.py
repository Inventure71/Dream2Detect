from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dream2detect.pipelines.v2 import (
    ImageGenerationSpec,
    V2PipelineConfig,
    build_v2_pipeline_artifacts,
    generate_v2_prompt_rows,
)


class V2PipelineTests(unittest.TestCase):
    def test_generate_v2_rows_balances_every_score_band(self) -> None:
        rows = generate_v2_prompt_rows(
            V2PipelineConfig(run_name="unit", count_per_band=3, seed=17)
        )

        self.assertEqual(len(rows), 30)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.score_band] = counts.get(row.score_band, 0) + 1

        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(counts["0-10"], 3)
        self.assertTrue(all(row.pipeline_version == "v2_real_failure_targeted" for row in rows))

    def test_v2_rows_include_real_failure_and_hard_negative_targets(self) -> None:
        rows = generate_v2_prompt_rows(
            V2PipelineConfig(run_name="unit", count_per_band=4, seed=23)
        )

        scenario_ids = {row.scenario_id for row in rows}
        self.assertIn("hard_intact_tape_label_clutter", scenario_ids)
        self.assertIn("flat_taped_mailer_subtle_damage", scenario_ids)
        self.assertIn("compression_shadow_occlusion", scenario_ids)
        self.assertIn("minor_low_contrast_real_recall", scenario_ids)
        self.assertIn("moderate_not_intact_real_geometry", scenario_ids)
        self.assertIn("severe_not_intact_opening_visible", scenario_ids)
        self.assertTrue(any(row.hard_negative for row in rows if row.score_band == "0-10"))
        self.assertFalse(any(row.hard_negative for row in rows if row.score_band != "0-10"))

    def test_v2_rows_record_real_gap_priorities(self) -> None:
        rows = generate_v2_prompt_rows(
            V2PipelineConfig(run_name="unit", count_per_band=5, seed=29)
        )

        priorities = {row.real_gap_priority for row in rows}

        self.assertIn("reduce high-confidence real moderate predicted as intact", priorities)
        self.assertIn("repair severe real examples predicted as intact", priorities)
        self.assertTrue(all(row.real_gap_priority for row in rows))

    def test_band_count_overrides_change_prompt_distribution(self) -> None:
        rows = generate_v2_prompt_rows(
            V2PipelineConfig(
                run_name="unit",
                count_per_band=2,
                band_counts={"11-20": 5, "86-100": 1},
                seed=30,
            )
        )

        counts: dict[str, int] = {}
        for row in rows:
            counts[row.score_band] = counts.get(row.score_band, 0) + 1

        self.assertEqual(counts["0-10"], 2)
        self.assertEqual(counts["11-20"], 5)
        self.assertEqual(counts["86-100"], 1)
        self.assertEqual(len(rows), 22)

    def test_severe_high_band_uses_top_band_repair_scenarios(self) -> None:
        rows = generate_v2_prompt_rows(
            V2PipelineConfig(run_name="unit", count_per_band=4, seed=31)
        )

        severe_high_rows = [row for row in rows if row.score_band == "86-100"]
        severe_high_scenarios = {row.scenario_id for row in severe_high_rows}

        self.assertEqual(
            severe_high_scenarios,
            {"severe_high_extreme_collapse_anchor", "severe_high_open_cavity_anchor"},
        )
        self.assertTrue(
            all("top-band" in row.prompt_text or "worst examples" in row.prompt_text for row in severe_high_rows)
        )

    def test_v2_prompt_text_contains_domain_gap_controls(self) -> None:
        rows = generate_v2_prompt_rows(
            V2PipelineConfig(run_name="unit", count_per_band=1, seed=31)
        )

        prompt_text = next(row.prompt_text for row in rows if row.coarse_class != "intact")
        self.assertIn("realistic documentary smartphone photo", prompt_text)
        self.assertIn("Real-domain gap target:", prompt_text)
        self.assertIn("visible at 384px training resolution", prompt_text)
        self.assertIn("Do not create an illustration", prompt_text)
        self.assertIn("unreadable", prompt_text)

    def test_intact_prompts_do_not_ask_for_visible_defects(self) -> None:
        rows = generate_v2_prompt_rows(
            V2PipelineConfig(run_name="unit", count_per_band=3, seed=37)
        )

        intact_prompts = [row.prompt_text for row in rows if row.coarse_class == "intact"]

        self.assertTrue(intact_prompts)
        for prompt_text in intact_prompts:
            self.assertIn("structurally intact", prompt_text)
            self.assertIn("recognizably undamaged", prompt_text)
            self.assertIn("do not add dents, tears, crushed corners", prompt_text)
            self.assertNotIn("assigned condition or defect", prompt_text)
            self.assertNotIn("primary damage", prompt_text)

    def test_v2_prompts_have_global_text_safety_constraint(self) -> None:
        rows = generate_v2_prompt_rows(
            V2PipelineConfig(run_name="unit", count_per_band=3, seed=39)
        )

        for row in rows:
            self.assertIn(
                "All labels, barcodes, addresses, logos, and large text must be absent, generic, or unreadable.",
                row.prompt_text,
            )

    def test_build_v2_artifacts_writes_manifest_jsonl_summary_and_qc(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = build_v2_pipeline_artifacts(
                V2PipelineConfig(
                    run_name="unit_run",
                    count_per_band=2,
                    seed=41,
                    output_dir=Path(temp_dir),
                    image_spec=ImageGenerationSpec(quality="low"),
                )
            )

            self.assertEqual(artifacts.prompt_count, 20)
            self.assertTrue(artifacts.prompt_manifest_path.exists())
            self.assertTrue(artifacts.input_jsonl_path.exists())
            self.assertTrue(artifacts.metadata_path.exists())
            self.assertTrue(artifacts.summary_path.exists())
            self.assertTrue(artifacts.qc_rubric_path.exists())

            with artifacts.prompt_manifest_path.open(newline="", encoding="utf-8") as handle:
                manifest_rows = list(csv.DictReader(handle))
            jsonl_rows = [
                json.loads(line)
                for line in artifacts.input_jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
            metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(len(manifest_rows), 20)
            self.assertEqual(len(jsonl_rows), 20)
            self.assertEqual(jsonl_rows[0]["url"], "/v1/images/generations")
            self.assertEqual(jsonl_rows[0]["body"]["quality"], "low")
            self.assertEqual(jsonl_rows[0]["custom_id"], manifest_rows[0]["custom_id"])
            self.assertEqual(metadata["endpoint"], "/v1/images/generations")
            self.assertEqual(metadata["request_count"], 20)
            self.assertEqual(metadata["status"], "prepared")
            self.assertEqual(summary["hard_negative_count"], 2)
            self.assertIn("Reject the image", artifacts.qc_rubric_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
