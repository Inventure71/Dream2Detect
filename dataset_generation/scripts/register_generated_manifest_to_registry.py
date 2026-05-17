from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream2detect.storage.repository import Dream2DetectRepository


REQUIRED_COLUMNS = {
    "prompt_uid",
    "score_band",
    "coarse_class",
    "representative_score",
    "band_spec_file",
    "prompt_title",
    "prompt_text",
    "image_model",
    "image_quality",
    "image_size",
    "image_status",
    "image_path",
}

FEATURE_COLUMNS = (
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
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register generated-image manifest rows in the SQLite prompt registry."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/dream2detect.sqlite3"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report planned inserts without writing to SQLite.",
    )
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
        rows = list(reader)

    if not rows:
        raise ValueError("Manifest is empty.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        prompt_uid = row["prompt_uid"]
        if prompt_uid in seen:
            duplicates.append(prompt_uid)
        seen.add(prompt_uid)
    if duplicates:
        raise ValueError(f"Manifest contains duplicate prompt_uid values: {duplicates[:20]}")

    missing_images = [
        row["image_path"]
        for row in rows
        if row["image_status"] == "generated" and not Path(row["image_path"]).exists()
    ]
    if missing_images:
        raise FileNotFoundError(
            "Manifest references missing generated images: "
            + ", ".join(missing_images[:20])
        )

    return rows


def _prompt_id_for_uid(conn: sqlite3.Connection, prompt_uid: str) -> int | None:
    row = conn.execute(
        "SELECT id FROM prompts WHERE prompt_uid = ?",
        (prompt_uid,),
    ).fetchone()
    return int(row["id"]) if row is not None else None


def _image_ref_exists(conn: sqlite3.Connection, image_ref: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM generated_images WHERE image_ref = ?",
        (image_ref,),
    ).fetchone()
    return row is not None


def register_manifest(
    *,
    db_path: Path,
    manifest_path: Path,
    dry_run: bool,
) -> tuple[int, int, int]:
    rows = _load_rows(manifest_path)
    repository = Dream2DetectRepository(db_path)
    repository.initialize()

    inserted_prompts = 0
    inserted_images = 0
    skipped_existing_prompts = 0
    now = _utc_now()
    batch_name = manifest_path.parent.name

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in rows:
            prompt_uid = row["prompt_uid"]
            prompt_id = _prompt_id_for_uid(conn, prompt_uid)
            if prompt_id is None:
                if not dry_run:
                    feature_values = [row.get(column, "") for column in FEATURE_COLUMNS]
                    cursor = conn.execute(
                        """
                        INSERT INTO prompts (
                            prompt_uid, score_band, coarse_class, representative_score,
                            band_spec_file,
                            damage_profile_primary, damage_profile_secondary,
                            damage_location_primary, box_form_factor, box_pattern,
                            label_presence, tape_profile, background_context,
                            camera_angle, lighting_style,
                            prompt_model, prompt_title, prompt_text,
                            prompt_rationale, uniqueness_notes, prompt_status, review_notes,
                            image_status, image_ref, image_error,
                            qc_status, reviewed_score_band, reviewed_coarse_class,
                            final_score_band, final_coarse_class, qc_notes,
                            created_at, updated_at
                        ) VALUES (
                            ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?,
                            ?, ?, ?, ?, ?, ?,
                            ?, ?
                        )
                        """,
                        (
                            prompt_uid,
                            row["score_band"],
                            row["coarse_class"],
                            int(row["representative_score"]),
                            row["band_spec_file"],
                            *feature_values,
                            row.get("pipeline_version") or "manifest_import",
                            row["prompt_title"],
                            row["prompt_text"],
                            row.get("target_failure_mode") or "",
                            row.get("real_gap_priority") or "",
                            "approved",
                            "",
                            row["image_status"],
                            row["image_path"] if row["image_status"] == "generated" else "",
                            row.get("failure_reason") or "",
                            row.get("qc_status") or "unreviewed",
                            row.get("reviewed_score_band") or "",
                            row.get("reviewed_coarse_class") or "",
                            row.get("final_score_band") or "",
                            row.get("final_coarse_class") or "",
                            row.get("qc_notes") or "",
                            now,
                            now,
                        ),
                    )
                    prompt_id = int(cursor.lastrowid)
                inserted_prompts += 1
            else:
                existing = conn.execute(
                    "SELECT image_ref FROM prompts WHERE id = ?",
                    (prompt_id,),
                ).fetchone()
                if existing["image_ref"] != row["image_path"]:
                    raise ValueError(
                        f"Existing prompt_uid {prompt_uid!r} points to a different image_ref."
                    )
                skipped_existing_prompts += 1

            if row["image_status"] != "generated" or _image_ref_exists(conn, row["image_path"]):
                continue

            if not dry_run:
                conn.execute(
                    """
                    INSERT INTO generated_images (
                        image_uid, prompt_id, image_model, image_quality,
                        image_size, image_ref, revised_prompt, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"{batch_name}:{prompt_uid}",
                        prompt_id,
                        row["image_model"],
                        row["image_quality"],
                        row["image_size"],
                        row["image_path"],
                        row.get("revised_prompt") or "",
                        now,
                    ),
                )
            inserted_images += 1

        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    finally:
        conn.close()

    return inserted_prompts, inserted_images, skipped_existing_prompts


def main() -> None:
    args = build_argument_parser().parse_args()
    inserted_prompts, inserted_images, skipped_existing_prompts = register_manifest(
        db_path=args.db,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
    )
    action = "Would register" if args.dry_run else "Registered"
    print(f"{action} {inserted_prompts} prompt rows and {inserted_images} generated-image rows.")
    print(f"Skipped existing prompt rows: {skipped_existing_prompts}")
    print(f"Database: {args.db}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
