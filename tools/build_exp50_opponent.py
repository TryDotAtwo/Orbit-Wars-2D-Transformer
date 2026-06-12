from __future__ import annotations

import json
import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = Path(r"C:\tmp\orbit-wars-exp50\orbit-wars-exp50.ipynb")
DEFAULT_DATASET = Path(r"C:\tmp\producer-orbit-wars-utils")
DEFAULT_OUT = ROOT / "test_results" / "opponents" / "orbit-wars-exp50"


def extract_main_py(notebook_path: Path) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.startswith("%%writefile main.py"):
            return "\n".join(source.splitlines()[1:]).rstrip() + "\n"
    raise RuntimeError(f"main.py cell not found in {notebook_path}")


def build_exp50_opponent(
    notebook_path: Path = DEFAULT_NOTEBOOK,
    dataset_dir: Path = DEFAULT_DATASET,
    out_dir: Path = DEFAULT_OUT,
) -> Path:
    if not notebook_path.exists():
        raise FileNotFoundError(f"missing notebook: {notebook_path}")
    orbit_lite = dataset_dir / "orbit_lite"
    if not orbit_lite.exists():
        raise FileNotFoundError(f"missing orbit_lite dataset dir: {orbit_lite}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "main.py").write_text(extract_main_py(notebook_path), encoding="utf-8")
    shutil.copytree(orbit_lite, out_dir / "orbit_lite")
    return out_dir / "main.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(build_exp50_opponent(args.notebook, args.dataset_dir, args.out_dir))


if __name__ == "__main__":
    main()
