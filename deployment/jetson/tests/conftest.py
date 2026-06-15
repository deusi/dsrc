import sys
from pathlib import Path

JETSON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = JETSON_DIR.parents[1]  # the dsrc simulation repo root

for path in (str(JETSON_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
