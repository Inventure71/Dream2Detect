from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from ..config import load_settings
from ..features.catalog import ASSIGNMENT_FIELD_NAMES
from ..features.models import FeatureAssignment
from .schema import schema_statements


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Dream2DetectRepository:
    def __init__(self, database_path: Path | None = None) -> None:
        settings = load_settings()
        self.database_path = database_path or settings.database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        prompt_feature_columns = {
            "damage_profile_primary",
            "damage_profile_secondary",
            "damage_location_primary",
            "box_form_factor",
            "box_pattern",
            "label_presence",
            "tape_profile",
            "background_context",
            "camera_angle",
            "lighting_style",
            "image_error",
            "qc_status",
            "reviewed_score_band",
            "reviewed_coarse_class",
            "final_score_band",
            "final_coarse_class",
            "qc_notes",
        }
        with self.connection() as conn:
            for statement in schema_statements():
                conn.execute(statement)
            existing_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(prompts)").fetchall()
            }
            missing_columns = prompt_feature_columns - existing_columns
            for column_name in sorted(missing_columns):
                conn.execute(f"ALTER TABLE prompts ADD COLUMN {column_name} TEXT")
            conn.execute(
                """
                UPDATE prompts
                SET qc_status = 'unreviewed'
                WHERE qc_status IS NULL
                """
            )

    def insert_prompt(
        self,
        *,
        prompt_uid: str,
        score_band: str,
        coarse_class: str,
        representative_score: int,
        band_spec_file: str,
        damage_profile_primary: str,
        damage_profile_secondary: str,
        damage_location_primary: str,
        box_form_factor: str,
        box_pattern: str,
        label_presence: str,
        tape_profile: str,
        background_context: str,
        camera_angle: str,
        lighting_style: str,
        prompt_model: str,
        prompt_title: str,
        prompt_text: str,
        prompt_rationale: str | None,
        uniqueness_notes: str | None,
        prompt_status: str = "drafted",
    ) -> int:
        now = _utc_now()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO prompts (
                    prompt_uid, score_band, coarse_class, representative_score,
                    band_spec_file,
                    damage_profile_primary, damage_profile_secondary, damage_location_primary,
                    box_form_factor, box_pattern, label_presence, tape_profile,
                    background_context, camera_angle, lighting_style,
                    prompt_model, prompt_title, prompt_text,
                    prompt_rationale, uniqueness_notes, prompt_status, review_notes,
                    image_status, image_ref, image_error,
                    qc_status, reviewed_score_band, reviewed_coarse_class, final_score_band, final_coarse_class, qc_notes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt_uid,
                    score_band,
                    coarse_class,
                    representative_score,
                    band_spec_file,
                    damage_profile_primary,
                    damage_profile_secondary,
                    damage_location_primary,
                    box_form_factor,
                    box_pattern,
                    label_presence,
                    tape_profile,
                    background_context,
                    camera_angle,
                    lighting_style,
                    prompt_model,
                    prompt_title,
                    prompt_text,
                    prompt_rationale,
                    uniqueness_notes,
                    prompt_status,
                    None,
                    "pending",
                    None,
                    None,
                    "unreviewed",
                    None,
                    None,
                    None,
                    None,
                    None,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def list_prompts_for_generation(
        self,
        prompt_status: str = "approved",
        image_status: str = "pending",
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        query = """
            SELECT *
            FROM prompts
            WHERE prompt_status = ? AND image_status = ?
            ORDER BY id ASC
        """
        params: list[object] = [prompt_status, image_status]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return list(rows)

    def update_prompt_review(
        self,
        prompt_id: int,
        *,
        prompt_status: str,
        review_notes: str | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE prompts
                SET prompt_status = ?, review_notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (prompt_status, review_notes, _utc_now(), prompt_id),
            )

    def attach_generated_image(
        self,
        *,
        prompt_id: int,
        image_uid: str,
        image_model: str,
        image_quality: str,
        image_size: str,
        image_ref: str,
        revised_prompt: str | None,
    ) -> None:
        now = _utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO generated_images (
                    image_uid, prompt_id, image_model, image_quality,
                    image_size, image_ref, revised_prompt, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_uid,
                    prompt_id,
                    image_model,
                    image_quality,
                    image_size,
                    image_ref,
                    revised_prompt,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE prompts
                SET image_status = 'generated', image_ref = ?, image_error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (image_ref, now, prompt_id),
            )

    def mark_prompt_image_in_progress(self, prompt_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE prompts
                SET image_status = 'in_progress', image_error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (_utc_now(), prompt_id),
            )

    def mark_prompt_image_failed(self, prompt_id: int, error_message: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE prompts
                SET image_status = 'failed', image_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (error_message[:1000], _utc_now(), prompt_id),
            )

    def update_generated_image_qc(
        self,
        prompt_id: int,
        *,
        qc_status: str,
        reviewed_score_band: str | None = None,
        reviewed_coarse_class: str | None = None,
        final_score_band: str | None = None,
        final_coarse_class: str | None = None,
        qc_notes: str | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE prompts
                SET qc_status = ?, reviewed_score_band = ?, reviewed_coarse_class = ?,
                    final_score_band = ?, final_coarse_class = ?, qc_notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    qc_status,
                    reviewed_score_band,
                    reviewed_coarse_class,
                    final_score_band,
                    final_coarse_class,
                    qc_notes,
                    _utc_now(),
                    prompt_id,
                ),
            )

    def list_prompts(self) -> list[sqlite3.Row]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM prompts ORDER BY id ASC").fetchall()
        return list(rows)

    def list_feature_assignments(self, *, score_band: str) -> list[FeatureAssignment]:
        select_columns = ", ".join(ASSIGNMENT_FIELD_NAMES)
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT {select_columns}
                FROM prompts
                WHERE score_band = ? AND prompt_status != 'rejected'
                """,
                (score_band,),
            ).fetchall()
        assignments: list[FeatureAssignment] = []
        for row in rows:
            values = {column_name: row[column_name] for column_name in ASSIGNMENT_FIELD_NAMES}
            if any(value is None for value in values.values()):
                continue
            assignments.append(FeatureAssignment(**values))
        return assignments

    def get_feature_value_counts(self, *, score_band: str, axis_name: str) -> Counter[str]:
        if axis_name not in ASSIGNMENT_FIELD_NAMES:
            supported = ", ".join(ASSIGNMENT_FIELD_NAMES)
            raise ValueError(f"Unknown feature axis '{axis_name}'. Supported axes: {supported}")
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT {axis_name} AS axis_value
                FROM prompts
                WHERE score_band = ? AND prompt_status != 'rejected' AND {axis_name} IS NOT NULL
                """,
                (score_band,),
            ).fetchall()
        counter: Counter[str] = Counter()
        for row in rows:
            counter[row["axis_value"]] += 1
        return counter

    def get_prompt(self, prompt_id: int) -> sqlite3.Row | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM prompts WHERE id = ?",
                (prompt_id,),
            ).fetchone()
        return row
