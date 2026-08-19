"""Get a recycling dataset for training.

Order of attempts (no COCO fallback — always real recycling data):

1. **Bundled demo data** at `data/sample/recycling_demo/` (104 images, 4.2M).
   - Always works — ships in the repo, no network needed.
   - Good enough to demonstrate the training flow + metrics.
   - Use this for the Colab demo unless the user explicitly wants the full dataset.

2. **Roboflow Universe** (full 2404 images, 4 classes, CC BY 4.0).
   - Tries the candidates in order, picks the first that returns a real ZIP.
   - As of Aug 2026, S3 export has been intermittently broken (NoSuchKey).
   - If all candidates fail, falls back to the bundled data (NOT COCO).

3. **HuggingFace** (TACO - Trash Annotations in Context, optional).
   - 1500+ images, 60+ fine-grained categories. Too many classes for the
     4-class demo. Skipped unless explicitly requested.

The script always ends with a real recycling dataset on disk + a
dataset_meta.json that the training script reads.

Reads ROBOFLOW_API_KEY from .env (gitignored). Logs the chosen
dataset to data/dataset_meta.json so the training script + README
can cite it.

Run:
    python scripts/download_dataset.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

# CRITICAL Colab platform self-heal (Aug 2026):
# When this script is invoked via `subprocess.run([sys.executable, ...])` from a
# Jupyter cell, the subprocess Python may NOT have Colab's site-packages
# (/usr/local/lib/python3.12/dist-packages — where `%pip install` writes) on
# its sys.path. Add the path BEFORE any other import. Also run site.main() to
# re-process .pth files. Mirrors the same self-heal in train_yolo26.py.
import site as _site

for _candidate in (
    "/usr/local/lib/python3.12/dist-packages",
    "/usr/local/lib/python3.11/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/lib/python3.12/site-packages",
):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
try:
    _site.main()
except Exception:
    pass

# Resolve paths relative to the project root, not the script location
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# Bundled fallback: ships in the repo at data/sample/recycling_demo/
BUNDLED_DEMO = ROOT / "data" / "sample" / "recycling_demo"

# (workspace, project, version) — tried in order. First one that
# returns a real ZIP wins. (Aug 2026: S3 export has been broken.)
ROBOFLOW_CANDIDATES = [
    ("zkf624", "-recycling", 3),  # PRIMARY: 2,404 images, 4 classes, CC BY 4.0
    ("roboflow-100", "waste-classification", None),
    ("roboflow-100", "recyclable-waste", None),
    ("sneha-latha", "waste-classification", None),
    ("joseph-nelson", "plastic-bottles", None),
]


def use_bundled_demo() -> dict:
    """Use the bundled recycling demo data (always works, no network).

    Copies the bundled data to data/raw/recycling_v3/ so the rest of
    the pipeline (which expects that path) works unchanged.
    """
    if not BUNDLED_DEMO.exists():
        raise FileNotFoundError(
            f"Bundled demo data not found at {BUNDLED_DEMO}. "
            "This should always exist — check your git checkout."
        )

    print("=" * 70)
    print("  USING BUNDLED RECYCLING DEMO DATA (offline, 4.2M)")
    print("=" * 70)
    print(f"  Source: {BUNDLED_DEMO.relative_to(ROOT)}")
    print("  104 images, 4 classes (Glass, metal, plastic, vinyl)")
    print("  YOLO segmentation format")
    print("  CC BY 4.0 (Roboflow zkf624/-recycling v3)")
    print()

    # Copy to data/raw/recycling_v3/ so the rest of the pipeline works
    target = DATA_RAW / "recycling_v3"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(BUNDLED_DEMO, target)

    # Count files
    train_imgs = len(list((target / "train" / "images").glob("*.jpg")))
    val_imgs = len(list((target / "val" / "images").glob("*.jpg")))

    return {
        "source": "bundled_demo",
        "source_path": str(BUNDLED_DEMO.relative_to(ROOT)),
        "workspace": "zkf624",
        "project": "-recycling",
        "version": 3,
        "location": str(target),
        "image_count": train_imgs + val_imgs,
        "train_count": train_imgs,
        "val_count": val_imgs,
        "class_count": 4,
        "classes": ["Glass", "metal", "plastic", "vinyl"],
        "format": "yolov11",
        "task": "segmentation",
        "license": "CC BY 4.0",
        "description": (
            "Bundled demo subset (104 images from v3's test split). "
            "For full 2404-image training, try Roboflow when S3 is up."
        ),
    }


def try_roboflow(api_key: str, workspace: str, project: str, version: int | None = None) -> dict | None:
    """Try to download a public dataset. Returns metadata dict, or None on failure."""
    try:
        from roboflow import Roboflow  # type: ignore

        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        if version is not None:
            target_ver = version
        else:
            try:
                versions = list(proj.versions())
                target_ver = versions[0].version if versions else 1
            except Exception:
                target_ver = 1
        safe_proj = project.lstrip("-") or "untitled"
        out_dir = DATA_RAW / f"{workspace}_{safe_proj}_v{target_ver}"
        out_dir.mkdir(parents=True, exist_ok=True)
        proj.version(target_ver).download("yolov11", location=str(out_dir), overwrite=True)
        zip_path = out_dir / "roboflow.zip"
        if zip_path.exists() and zipfile.is_zipfile(zip_path):
            with zipfile.ZipFile(zip_path) as z:
                names = z.namelist()
                yaml_files = [n for n in names if n.endswith("data.yaml")]
                img_files = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]
            return {
                "source": "roboflow",
                "workspace": workspace,
                "project": project,
                "version": target_ver,
                "location": str(out_dir),
                "image_count": len(img_files),
                "yaml": yaml_files[0] if yaml_files else None,
            }
        print(f"  [{workspace}/{project}] BadZipFile — S3 export returned XML error")
        return None
    except Exception as e:
        print(f"  [{workspace}/{project}] {type(e).__name__}: {str(e)[:80]}")
        return None


def main() -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")

    # Step 1: always copy the bundled demo data first (guarantees a
    # real recycling dataset on disk even if everything else fails).
    meta = use_bundled_demo()
    print(f"  ✓ Bundled data ready: {meta['image_count']} images, {meta['class_count']} classes")

    # Step 2: try Roboflow for the full dataset. If it succeeds, replace
    # the bundled data with the full one. If it fails, keep the bundled.
    if api_key:
        print()
        print("=" * 70)
        print("  ATTEMPTING ROBOFLOW FULL DATASET (optional, may fail)")
        print("=" * 70)
        for ws, proj, ver in ROBOFLOW_CANDIDATES:
            print(f"\n  Trying {ws}/{proj} (v{ver or 'latest'}) ...")
            full_meta = try_roboflow(api_key, ws, proj, version=ver)
            if full_meta is not None:
                print(f"  ✓ Downloaded: {full_meta['image_count']} images, yaml={full_meta['yaml']}")
                # Replace the bundled data with the full one
                full_meta["source"] = "roboflow"
                full_meta["description"] = (
                    f"Full Roboflow dataset ({full_meta['image_count']} images). "
                    f"Replaces the bundled 104-image demo subset."
                )
                meta = full_meta
                break
        else:
            print("\n  Roboflow candidates failed (S3 export issue).")
            print("  Using bundled demo data (104 images) — real recycling, just smaller.")
    else:
        print()
        print("  (Skipping Roboflow attempt — no ROBOFLOW_API_KEY in .env)")

    # Write meta
    (DATA_RAW / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print()
    print(f"  ✓ Metadata saved to {DATA_RAW / 'dataset_meta.json'}")
    print(f"  ✓ Final dataset: {meta.get('image_count', '?')} images at {meta['location']}")
    print()
    print("Next: run scripts/train_yolo26.py to train on this dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
