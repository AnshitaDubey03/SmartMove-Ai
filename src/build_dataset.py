"""Sample frames from traffic videos, run detection, and build a feature table
for training the congestion classifier (src/congestion_model.py).

Run: .\\venv\\Scripts\\python.exe src\\build_dataset.py
Output: data/processed/features.csv
"""

import glob
import json
import os

import cv2
import pandas as pd

from src.detection import CONFIDENCE_THRESHOLD, VEHICLE_CLASSES, load_model
from src.density import (
    classify_score,
    compute_composite_score,
    derive_count_thresholds,
    derive_score_thresholds,
    occupancy_ratio,
)

FRAME_SAMPLE_INTERVAL = 10  # take 1 frame every 10 -> ~3 frames/sec at 30fps


def extract_frame_features(model, frame):
    height, width = frame.shape[:2]
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    boxes = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id not in VEHICLE_CLASSES:
            continue
        label = VEHICLE_CLASSES[class_id]
        class_counts[label] += 1
        boxes.append(tuple(map(float, box.xyxy[0])))

    total = sum(class_counts.values())
    ratio = occupancy_ratio(boxes, width, height) if boxes else 0.0

    row = {
        "vehicle_count": total,
        "occupancy_ratio": ratio,
    }
    for name in VEHICLE_CLASSES.values():
        row[f"{name}_ratio"] = class_counts[name] / total if total > 0 else 0.0
    return row


def build_dataset_from_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    rows = []
    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_index % FRAME_SAMPLE_INTERVAL == 0:
            row = extract_frame_features(model, frame)
            row["source"] = os.path.basename(video_path)
            row["frame_index"] = frame_index
            rows.append(row)
        frame_index += 1
    cap.release()
    return rows


if __name__ == "__main__":
    model = load_model()
    all_rows = []

    for video_path in sorted(glob.glob("data/raw/videos/*.ogv")) + sorted(glob.glob("data/raw/videos/*.mp4")):
        print(f"Sampling frames from {video_path} ...")
        all_rows.extend(build_dataset_from_video(model, video_path))

    df = pd.DataFrame(all_rows)

    df["composite_score"] = compute_composite_score(df["vehicle_count"], df["occupancy_ratio"])
    thresholds = derive_score_thresholds(df["composite_score"])
    print("Composite score thresholds (33rd/66th percentile):", thresholds)
    df["congestion_label"] = df["composite_score"].apply(lambda s: classify_score(s, thresholds))

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features.csv", index=False)

    print(f"Saved {len(df)} rows to data/processed/features.csv")
    print(df["congestion_label"].value_counts())

    # Persist simple count-based thresholds too -- used for the transparent
    # rule-based "Density" display in the frontend (Phase 10), kept distinct
    # from the ML-predicted "Congestion" label above.
    density_thresholds = derive_count_thresholds(df["vehicle_count"])
    os.makedirs("models", exist_ok=True)
    with open("models/density_thresholds.json", "w") as f:
        json.dump({"low_cutoff": density_thresholds[0], "high_cutoff": density_thresholds[1]}, f)
    print(f"Saved density thresholds {density_thresholds} to models/density_thresholds.json")
