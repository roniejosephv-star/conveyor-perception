"""Run inference on a single image — the simplest possible demo.

Useful for:
- Sanity-checking a trained model on one image
- Generating a single annotated image for a slide or report
- Quick verification that the framework works after a code change

Usage:
    # Use COCO pretrained (default)
    python scripts/run_inference.py --image data/sample/bus.jpg

    # Use a custom-trained model
    python scripts/run_inference.py \\
        --image data/sample/conveyor.jpg \\
        --model models/yolo26s_recyclable.onnx \\
        --data-yaml data/raw/recyclable-waste/data.yaml \\
        --output output/annotated.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from conveyor_perception.perception.detector import COCO_CLASSES, Detector, _parse_yolo_classes


def main() -> int:
    p = argparse.ArgumentParser(description="Run YOLO26 inference on a single image")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--model", default=None, help="Path to YOLO26 ONNX model")
    p.add_argument("--data-yaml", default=None, help="Path to data.yaml (for custom class names)")
    p.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    p.add_argument("--output", default=None, help="Path to save annotated image")
    args = p.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: image not found: {args.image}")
        return 1

    # Resolve class names
    if args.data_yaml:
        class_names = _parse_yolo_classes(Path(args.data_yaml))
    else:
        class_names = COCO_CLASSES

    # Build detector
    if args.model:
        detector = Detector(
            model_path=args.model,
            class_names=class_names,
            conf_threshold=args.conf,
        )
    else:
        detector = Detector.from_coco_pretrained(conf_threshold=args.conf)

    # Read + infer
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: could not read image: {args.image}")
        return 1

    detections, annotated = detector.detect_and_draw(frame)

    # Print summary
    summary = {
        "image": args.image,
        "model": args.model or "COCO pretrained (YOLO26s)",
        "image_shape": list(frame.shape),
        "class_count": len(class_names),
        "detection_count": len(detections),
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 3),
                "bbox": [round(v, 1) for v in d.bbox],
            }
            for d in detections
        ],
    }
    print(json.dumps(summary, indent=2))

    # Save annotated output
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, annotated)
        print(f"\nSaved annotated image: {args.output}")
    else:
        # Default: save next to the input image
        out_path = Path(args.image).with_name(f"{Path(args.image).stem}_annotated.jpg")
        cv2.imwrite(str(out_path), annotated)
        print(f"\nSaved annotated image: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
