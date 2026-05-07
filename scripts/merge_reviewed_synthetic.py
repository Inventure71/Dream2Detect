from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


ACCEPTED_QC_STATUSES = ("accepted_as_labeled", "accepted_relabel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reviewed synthetic prompts/images from one SQLite DB into another.")
    parser.add_argument("--source-db", required=True, help="Source calibration SQLite database path")
    parser.add_argument("--dest-db", required=True, help="Destination permanent SQLite database path")
    return parser.parse_args()


def table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]


def prompt_exists(conn: sqlite3.Connection, prompt_uid: str) -> bool:
    row = conn.execute("SELECT 1 FROM prompts WHERE prompt_uid = ?", (prompt_uid,)).fetchone()
    return row is not None


def generated_image_exists(conn: sqlite3.Connection, image_uid: str) -> bool:
    row = conn.execute("SELECT 1 FROM generated_images WHERE image_uid = ?", (image_uid,)).fetchone()
    return row is not None


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_db)
    dest_path = Path(args.dest_db)

    source = sqlite3.connect(source_path)
    source.row_factory = sqlite3.Row
    dest = sqlite3.connect(dest_path)
    dest.row_factory = sqlite3.Row

    try:
        source_prompt_columns = table_columns(source, "prompts")
        dest_prompt_columns = table_columns(dest, "prompts")
        shared_prompt_columns = [
            column
            for column in source_prompt_columns
            if column in dest_prompt_columns and column != "id"
        ]

        source_image_columns = table_columns(source, "generated_images")
        dest_image_columns = table_columns(dest, "generated_images")
        shared_image_columns = [
            column
            for column in source_image_columns
            if column in dest_image_columns and column not in {"id", "prompt_id"}
        ]

        prompt_rows = source.execute(
            """
            SELECT *
            FROM prompts
            WHERE image_status = 'generated'
              AND qc_status IN (?, ?)
            ORDER BY id ASC
            """,
            ACCEPTED_QC_STATUSES,
        ).fetchall()

        imported_prompts = 0
        imported_images = 0

        for prompt_row in prompt_rows:
            prompt_uid = prompt_row["prompt_uid"]
            if not prompt_exists(dest, prompt_uid):
                placeholders = ", ".join("?" for _ in shared_prompt_columns)
                column_sql = ", ".join(shared_prompt_columns)
                values = [prompt_row[column_name] for column_name in shared_prompt_columns]
                dest.execute(
                    f"INSERT INTO prompts ({column_sql}) VALUES ({placeholders})",
                    values,
                )
                imported_prompts += 1

            dest_prompt_id = dest.execute(
                "SELECT id FROM prompts WHERE prompt_uid = ?",
                (prompt_uid,),
            ).fetchone()["id"]

            source_images = source.execute(
                "SELECT * FROM generated_images WHERE prompt_id = ? ORDER BY id ASC",
                (prompt_row["id"],),
            ).fetchall()
            for image_row in source_images:
                image_uid = image_row["image_uid"]
                if generated_image_exists(dest, image_uid):
                    continue
                placeholders = ", ".join("?" for _ in (["prompt_id"] + shared_image_columns))
                column_sql = ", ".join(["prompt_id", *shared_image_columns])
                values = [dest_prompt_id, *[image_row[column_name] for column_name in shared_image_columns]]
                dest.execute(
                    f"INSERT INTO generated_images ({column_sql}) VALUES ({placeholders})",
                    values,
                )
                imported_images += 1

        dest.commit()
        print(f"Imported {imported_prompts} prompt rows and {imported_images} generated-image rows.")
    finally:
        source.close()
        dest.close()


if __name__ == "__main__":
    main()
