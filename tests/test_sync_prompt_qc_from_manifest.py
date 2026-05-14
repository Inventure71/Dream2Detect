from __future__ import annotations

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.sync_prompt_qc_from_manifest import sync_qc_fields


class SyncPromptQcFromManifestTests(unittest.TestCase):
    def test_sync_uses_prompt_uid_and_skips_missing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = temp_path / "test.sqlite3"
            manifest_path = temp_path / "manifest.csv"

            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                CREATE TABLE prompts (
                    id INTEGER PRIMARY KEY,
                    prompt_uid TEXT NOT NULL UNIQUE,
                    qc_status TEXT,
                    reviewed_score_band TEXT,
                    reviewed_coarse_class TEXT,
                    final_score_band TEXT,
                    final_coarse_class TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO prompts (
                    id,
                    prompt_uid,
                    qc_status,
                    reviewed_score_band,
                    reviewed_coarse_class,
                    final_score_band,
                    final_coarse_class,
                    updated_at
                )
                VALUES (99, 'stable_uid', 'unreviewed', '', '', '', '', 'old')
                """
            )
            conn.commit()
            conn.close()

            with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "prompt_id",
                        "prompt_uid",
                        "qc_status",
                        "reviewed_score_band",
                        "reviewed_coarse_class",
                        "final_score_band",
                        "final_coarse_class",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "prompt_id": "1",
                        "prompt_uid": "stable_uid",
                        "qc_status": "accepted_relabel",
                        "reviewed_score_band": "36-45",
                        "reviewed_coarse_class": "moderate",
                        "final_score_band": "36-45",
                        "final_coarse_class": "moderate",
                    }
                )
                writer.writerow(
                    {
                        "prompt_id": "2",
                        "prompt_uid": "missing_uid",
                        "qc_status": "accepted_as_labeled",
                        "reviewed_score_band": "0-10",
                        "reviewed_coarse_class": "intact",
                        "final_score_band": "0-10",
                        "final_coarse_class": "intact",
                    }
                )

            updated, eligible = sync_qc_fields(
                db_path,
                manifest_path,
                only_accepted=True,
                dry_run=False,
                skip_missing=True,
            )

            self.assertEqual(updated, 1)
            self.assertEqual(eligible, 1)

            conn = sqlite3.connect(db_path)
            row = conn.execute(
                """
                SELECT qc_status, final_score_band, final_coarse_class
                FROM prompts
                WHERE prompt_uid = 'stable_uid'
                """
            ).fetchone()
            conn.close()

            self.assertEqual(row, ("accepted_relabel", "36-45", "moderate"))


if __name__ == "__main__":
    unittest.main()
