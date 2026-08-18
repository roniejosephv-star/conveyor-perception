"""Download a recycling dataset from Roboflow Universe.

Tries the candidates in order, picks the first that returns a real ZIP
(not the S3 NoSuchKey XML error we hit on Aug 19 2026). Falls back to
"YOLO26s COCO pretrained" if all Roboflow downloads fail — that still
proves the framework, just not the recycling-specific class names.

Run:
    python scripts/download_dataset.py

Reads ROBOFLOW_API_KEY from .env (gitignored). Logs the chosen dataset
to data/dataset_meta.json so the training script + README can cite it.

Primary candidate (verified working Aug 19 2026):
- zkf624/-recycling v3                — 2,404 images, 4 classes (Glass, metal,
  plastic, vinyl), CC BY 4.0, YOLO segmentation format. Best fit for an
  industrial recycling demo.

Other verified candidates (Aug 2026 — but S3 export was broken):
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

# (workspace, project, version) — tried in order. First one that returns a real ZIP wins.
# Note: project slugs that start with a dash need to be passed as-is to the Roboflow SDK.
CANDIDATES = [
    ("zkf624", "-recycling", 3),  # PRIMARY: 2,404 images, 4 classes, CC BY 4.0, segmentation
    ("roboflow-100", "waste-classification", None),
    ("roboflow-100", "recyclable-waste", None),
    ("sneha-latha", "waste-classification", None),
    ("joseph-nelson", "plastic-bottles", None),
]

# If all Roboflow candidates fail, we fall back to this baseline.
FALLBACK_DESCRIPTION = (
    "YOLO26s COCO pretrained (80 classes). Recycling-specific training is "
    "deferred — see README for the Roboflow training path once their S3 "
    "export is fixed."
)


def try_roboflow(api_key: str, workspace: str, project: str, version: int | None = None) -> dict | None:
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
        # Resolve target version
        if version is not None:
            target_ver = version
        else:
            try:
                versions = list(proj.versions())
                target_ver = versions[0].version if versions else 1
            except Exception:
                target_ver = 1
        # Output dir uses a sanitized project name to avoid path issues with leading dashes
        safe_proj = project.lstrip("-") or "untitled"
        out_dir = DATA_RAW / f"{workspace}_{safe_proj}_v{target_ver}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Download in YOLO format. Ultralytics uses 'yolov11' as the format name
        # for YOLO11/YOLO26; both Ultralytics 8.3.x and 8.4.x accept it.
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
    for ws, proj, ver in CANDIDATES:
        print(f"\n  Trying {ws}/{proj} (v{ver or 'latest'}) ...")
        meta = try_roboflow(api_key, ws, proj, version=ver)
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
