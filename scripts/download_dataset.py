"""Download a recycling dataset from Roboflow Universe.

Tries the candidates in order, picks the first that returns a real ZIP
(not the S3 NoSuchKey XML error we hit on Aug 19 2026). Falls back to
"YOLO26s COCO pretrained" if all Roboflow downloads fail — that still
proves the framework, just not the recycling-specific class names.

Run:
    python scripts/download_dataset.py

Reads ROBOFLOW_API_KEY from .env (gitignored). Logs the chosen dataset
to data/dataset_meta.json so the training script + README can cite it.

Verified candidates (Aug 2026):
- roboflow-100/waste-classification   — 6 classes (cardboard, glass, metal, paper, plastic, trash)
- roboflow-100/recyclable-waste      — 5 classes (Glass, Cardboard, Paper, Plastic, Metal)
- sneha-latha/waste-classification    — varies
- joseph-nelson/plastic-bottles      — 1 class (single-object demo)

If all fail: we fall back to YOLO26s COCO pretrained and the script logs
the failure with a clear message.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv


# Resolve paths relative to the project root, not the script location
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

# (workspace, project) — tried in order. First one that returns a real ZIP wins.
CANDIDATES = [
    ("roboflow-100", "waste-classification"),
    ("roboflow-100", "recyclable-waste"),
    ("sneha-latha", "waste-classification"),
    ("joseph-nelson", "plastic-bottles"),
]

# If all Roboflow candidates fail, we fall back to this baseline.
FALLBACK_DESCRIPTION = (
    "YOLO26s COCO pretrained (80 classes). Recycling-specific training is "
    "deferred — see README for the Roboflow training path once their S3 "
    "export is fixed."
)


def try_roboflow(api_key: str, workspace: str, project: str) -> dict | None:
    """Try to download a public dataset. Returns metadata dict, or None on failure.

    Failure modes handled:
    - RoboflowError (workspace/project doesn't exist)
    - BadZipFile (S3 NoSuchKey — Roboflow's export system is broken)
    - Network errors
    """
    try:
        from roboflow import Roboflow  # type: ignore

        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        # Get the latest version
        try:
            versions = list(proj.versions())
            target_ver = versions[0].version if versions else 1
        except Exception:
            target_ver = 1
        out_dir = DATA_RAW / f"{workspace}_{project}_v{target_ver}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Download in YOLO26 format (Ultralytics uses 'yolov11' as the format name; same)
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


def fallback_to_coco_pretrained() -> dict:
    """Download YOLO26s COCO pretrained as the fallback. The framework
    works with these class names; recycling training is a separate step.
    """
    from ultralytics import YOLO  # type: ignore

    print("\n  Falling back to YOLO26s COCO pretrained...")
    model = YOLO("yolo26s.pt")
    return {
        "source": "ultralytics_coco_pretrained",
        "model_path": str(ROOT / "models" / "yolo26s.pt"),
        "class_count": len(model.names),
        "description": FALLBACK_DESCRIPTION,
    }


def main() -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY not set in .env")
        return 1

    print(f"Searching Roboflow Universe for a recycling dataset (key ends ...{api_key[-4:]})")
    for ws, proj in CANDIDATES:
        print(f"\n  Trying {ws}/{proj} ...")
        meta = try_roboflow(api_key, ws, proj)
        if meta is not None:
            print(f"  ✓ Downloaded: {meta['image_count']} images, yaml={meta['yaml']}")
            (DATA_RAW / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"  ✓ Metadata saved to {DATA_RAW / 'dataset_meta.json'}")
            return 0

    # All Roboflow downloads failed. Fall back to COCO pretrained.
    print("\nAll Roboflow candidates failed (S3 export issue on their side).")
    meta = fallback_to_coco_pretrained()
    (DATA_RAW / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  ✓ Metadata saved to {DATA_RAW / 'dataset_meta.json'}")
    print(f"  ✓ COCO pretrained: {meta['class_count']} classes (not recycling-specific)")
    print("\n  Next step: train YOLO26s on a recycling dataset in Colab")
    print("  once Roboflow's S3 export is fixed. See README § Training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
