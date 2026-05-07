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
    print(f"Initialized database at {repository.database_path}")


if __name__ == "__main__":
    main()
