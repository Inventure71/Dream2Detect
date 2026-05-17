from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = REPO_ROOT / "generated_images"
CANONICAL_DIR = GENERATED_ROOT / "synthetic_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move generated synthetic images into one canonical folder and update database references."
    )
    parser.add_argument(
        "--db",
        action="append",
        default=[],
        help="SQLite database path to update. Can be passed multiple times.",
    )
    return parser.parse_args()


def _update_db_paths(db_path: Path, moved_by_old_path: dict[str, str]) -> int:
    conn = sqlite3.connect(db_path)
    try:
        prompt_updates = 0
        image_updates = 0
        for old_path, new_path in moved_by_old_path.items():
            cur = conn.execute(
                "UPDATE prompts SET image_ref = ?, updated_at = updated_at WHERE image_ref = ?",
                (new_path, old_path),
            )
            prompt_updates += cur.rowcount
            cur = conn.execute(
                "UPDATE generated_images SET image_ref = ? WHERE image_ref = ?",
                (new_path, old_path),
            )
            image_updates += cur.rowcount
        conn.commit()
        return prompt_updates + image_updates
    finally:
        conn.close()


def _move_all_files_to_canonical() -> dict[str, str]:
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
    moved: dict[str, str] = {}
    for path in GENERATED_ROOT.rglob("*.png"):
        if path.parent == CANONICAL_DIR:
            continue
        destination = CANONICAL_DIR / path.name
        if destination.exists():
            if path.resolve() == destination.resolve():
                continue
            raise RuntimeError(f"Destination collision for {destination}")
        old_abs = str(path.resolve())
        shutil.move(str(path), str(destination))
        moved[old_abs] = str(destination.resolve())
    return moved


def _remove_empty_dirs() -> None:
    for path in sorted(GENERATED_ROOT.rglob("*"), reverse=True):
        if path.is_dir() and path != CANONICAL_DIR and path != GENERATED_ROOT:
            try:
                path.rmdir()
            except OSError:
                pass


def main() -> None:
    args = parse_args()
    db_paths = [Path(value).resolve() for value in args.db]
    moved_by_old_path = _move_all_files_to_canonical()
    total_updates = 0
    for db_path in db_paths:
        total_updates += _update_db_paths(db_path, moved_by_old_path)
    _remove_empty_dirs()
    print(f"Moved {len(moved_by_old_path)} image files into {CANONICAL_DIR}")
    print(f"Updated {total_updates} database image_ref fields")


if __name__ == "__main__":
    main()
