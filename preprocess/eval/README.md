# Video track source

`track_source.py` keeps YOLO tracking for the COCO `traffic light` and `stop
sign` classes, and adds a small OpenCV color/shape proposal fallback. Vehicle
detections are intentionally excluded from this sign-focused source. This is
necessary because `yolo11n.pt` is a generic COCO checkpoint whose classes do
not cover Taiwanese road signs reliably.

The fallback is a candidate generator, not a trained classifier. For production
accuracy, annotate sign boxes from `data/road.mp4` and fine-tune a detector with
a `road_sign` class. Until then, inspect `diagnostics`/track outputs and use
the manual ROI tracker when the sign is too small or achromatic for the
candidate masks.

Example:

```powershell
python -m preprocess.eval.run_real --video data\road.mp4 --weights yolo11n.pt
```

For CPU-only runs, `--drop-every 3` reduces detector work while timestamps
remain derived from the source FPS.
