from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


REQUIRED_COLUMNS = {
    "prompt_uid",
    "qc_status",
    "reviewed_score_band",
    "reviewed_coarse_class",
    "final_score_band",
    "final_coarse_class",
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sync prompt-level QC fields in the execution SQLite database from "
            "a reviewed synthetic manifest."
        )
    )
    parser.add_argument("--db", type=Path, default=Path("data/dream2detect.sqlite3"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--only-accepted",
        action="store_true",
        help="Skip manifest rows whose qc_status is not accepted_as_labeled or accepted_relabel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the planned updates without writing to SQLite.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip manifest prompt_uids that are not present in SQLite instead of failing.",
    )
    return parser


def _load_manifest_rows(path: Path, only_accepted: bool) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Manifest is missing required columns: " + ", ".join(sorted(missing))
            )
        rows = list(reader)

    if only_accepted:
        rows = [
            row
            for row in rows
            if row["qc_status"] in {"accepted_as_labeled", "accepted_relabel"}
        ]

    if not rows:
        raise ValueError("No manifest rows are eligible for QC sync.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        prompt_uid = row["prompt_uid"]
        if prompt_uid in seen:
            duplicates.append(prompt_uid)
        seen.add(prompt_uid)
    if duplicates:
        raise ValueError(f"Manifest contains duplicate prompt_uid values: {duplicates[:20]}")

    return rows


def _existing_prompt_uids(conn: sqlite3.Connection, prompt_uids: list[str]) -> set[str]:
    placeholders = ",".join("?" for _ in prompt_uids)
    return {
        row[0]
        for row in conn.execute(
            f"SELECT prompt_uid FROM prompts WHERE prompt_uid IN ({placeholders})",
            prompt_uids,
        ).fetchall()
    }


def sync_qc_fields(
    db_path: Path,
    manifest_path: Path,
    *,
    only_accepted: bool,
    dry_run: bool,
    skip_missing: bool,
) -> tuple[int, int]:
    rows = _load_manifest_rows(manifest_path, only_accepted)
    prompt_uids = [row["prompt_uid"] for row in rows]

    conn = sqlite3.connect(db_path)
    try:
        existing_uids = _existing_prompt_uids(conn, prompt_uids)
        missing_uids = sorted(set(prompt_uids) - existing_uids)
        if missing_uids:
            if skip_missing:
                rows = [row for row in rows if row["prompt_uid"] in existing_uids]
            else:
                raise ValueError(
                    "Manifest references prompt_uids that are missing from SQLite: "
                    + ", ".join(missing_uids[:20])
                )
        if missing_uids and skip_missing:
            print(f"Skipped {len(missing_uids)} manifest rows missing from SQLite.")
        elif missing_uids:
            raise ValueError(
                "Manifest references prompt_uids that are missing from SQLite: "
                + ", ".join(missing_uids[:20])
            )

        updated = 0
        for row in rows:
            params = {
                "qc_status": row["qc_status"],
                "reviewed_score_band": row["reviewed_score_band"],
                "reviewed_coarse_class": row["reviewed_coarse_class"],
                "final_score_band": row["final_score_band"],
                "final_coarse_class": row["final_coarse_class"],
                "prompt_uid": row["prompt_uid"],
            }
            if not dry_run:
                conn.execute(
                    """
                    UPDATE prompts
                    SET qc_status = :qc_status,
                        reviewed_score_band = :reviewed_score_band,
                        reviewed_coarse_class = :reviewed_coarse_class,
                        final_score_band = :final_score_band,
                        final_coarse_class = :final_coarse_class,
                        updated_at = datetime('now')
                    WHERE prompt_uid = :prompt_uid
                    """,
                    params,
                )
            updated += 1

        if dry_run:
            conn.rollback()
        else:
            conn.commit()
        return updated, len(rows)
    finally:
        conn.close()


def main() -> None:
    args = build_argument_parser().parse_args()
    updated, eligible = sync_qc_fields(
        args.db,
        args.manifest,
        only_accepted=args.only_accepted,
        dry_run=args.dry_run,
        skip_missing=args.skip_missing,
    )
    action = "Would sync" if args.dry_run else "Synced"
    print(f"{action} QC fields for {updated} prompt rows from {eligible} manifest rows.")
    print(f"Database: {args.db}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
