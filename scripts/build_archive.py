"""Create a clean source archive from this directory."""

from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parent / "residual-aware-position-bias-code.zip"
EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "paper_outputs",
    "results",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return not any(part in EXCLUDED_PARTS for part in relative.parts) and path.suffix not in EXCLUDED_SUFFIXES


def main() -> None:
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and include(path):
                archive.write(path, Path("code") / path.relative_to(ROOT))
        archive.writestr("code/results/.gitkeep", "")
        archive.writestr("code/paper_outputs/.gitkeep", "")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
