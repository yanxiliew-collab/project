"""Run one isolated RapidOCR request for source diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from rapidocr_onnxruntime import RapidOCR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--bbox", nargs=4, type=int)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.image:
        image = cv2.imread(args.image)
    else:
        capture = cv2.VideoCapture(args.video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, full = capture.read()
        capture.release()
        image = full[args.bbox[1]:args.bbox[3], args.bbox[0]:args.bbox[2]] if ok else None
    if image is None:
        print(json.dumps({"available": False, "reason": "input_unreadable"}))
        return 2
    if args.scale != 1.0:
        image = cv2.resize(
            image,
            (max(1, round(image.shape[1] * args.scale)), max(1, round(image.shape[0] * args.scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    engine = RapidOCR()
    rows, elapsed = engine(image)
    detections = []
    texts = []
    confidences = []
    for row in rows or []:
        box, text, confidence = row
        texts.append(str(text))
        confidences.append(float(confidence))
        detections.append({"text": str(text), "confidence": float(confidence), "box": box})
    print(json.dumps({
        "available": True,
        "detections": detections,
        "text": texts,
        "confidence": confidences,
        "elapsed": elapsed,
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
