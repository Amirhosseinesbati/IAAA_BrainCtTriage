"""
make_submission_zip.py — Build the uploadable submission.zip.

Zips the *contents* of ``submission/`` into ``submission.zip`` at the project
root, excluding caches and git placeholders. The zip root contains
``submission.py``, ``model.py``, ``triage.py``, ``models/``, etc. — exactly
what the leaderboard platform expects.

Usage:
    .venv\\Scripts\\python.exe scripts\\make_submission_zip.py
"""

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = PROJECT_ROOT / "submission"
OUTPUT_ZIP = PROJECT_ROOT / "submission.zip"

# Paths/components never shipped in the zip
EXCLUDED_DIR_NAMES = {"__pycache__"}
EXCLUDED_FILE_NAMES = {".gitkeep", "*.pyc", ".DS_Store"}


def _is_excluded(rel_parts: tuple[str, ...], name: str) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in rel_parts):
        return True
    if name in EXCLUDED_FILE_NAMES:
        return True
    if name.endswith(".pyc"):
        return True
    return False


def main() -> None:
    if not SUBMISSION_DIR.is_dir():
        sys.exit(f"ERROR: submission directory not found: {SUBMISSION_DIR}")

    n_files = 0
    n_bytes = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(SUBMISSION_DIR.rglob("*")):
            rel = path.relative_to(SUBMISSION_DIR)
            if _is_excluded(rel.parts, path.name):
                continue
            if path.is_file():
                zf.write(path, arcname=rel)
                n_files += 1
                n_bytes += path.stat().st_size
                if path.stat().st_size > 50 * 1024 * 1024:
                    print(f"  + {rel} ({path.stat().st_size / 1024**2:.1f} MB)")
                else:
                    print(f"  + {rel}")

    size_mb = OUTPUT_ZIP.stat().st_size / 1024**2
    print("-" * 60)
    print(f"✓ Packed {n_files} files ({n_bytes / 1024**2:.1f} MB raw) "
          f"into {OUTPUT_ZIP.name} ({size_mb:.1f} MB)")
    print(f"  → {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
