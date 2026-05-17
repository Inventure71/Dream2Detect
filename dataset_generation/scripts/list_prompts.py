from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dream2detect.storage.repository import Dream2DetectRepository


def main() -> None:
    repository = Dream2DetectRepository()
    repository.initialize()
    rows = repository.list_prompts()
    if not rows:
        print("No prompts found.")
        return

    for row in rows:
        print(
            f"id={row['id']} uid={row['prompt_uid']} band={row['score_band']} "
            f"status={row['prompt_status']} image_status={row['image_status']} "
            f"qc_status={row['qc_status'] or '-'} "
            f"primary={row['damage_profile_primary'] or '-'} "
            f"bg={row['background_context'] or '-'} "
            f"light={row['lighting_style'] or '-'} "
            f"title={row['prompt_title']}"
        )
        if row["image_error"]:
            print(f"  image_error={row['image_error']}")
        if row["reviewed_score_band"] or row["final_score_band"]:
            print(
                f"  reviewed_band={row['reviewed_score_band'] or '-'} "
                f"final_band={row['final_score_band'] or '-'}"
            )
        if row["qc_notes"]:
            print(f"  qc_notes={row['qc_notes']}")


if __name__ == "__main__":
    main()
