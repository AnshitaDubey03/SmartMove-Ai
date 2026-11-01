"""Shared frame/video analysis logic used by both app.py (Streamlit UI) and
src/run_tests.py (batch testing) -- kept in one place to avoid duplicating it
across an app script and a test script.
"""

import cv2
import pandas as pd

from src.detection import CONFIDENCE_THRESHOLD, VEHICLE_CLASSES
from src.density import occupancy_ratio

VIDEO_SAMPLE_FRAMES = 10  # evenly spaced frames sampled from an uploaded video


def analyze_frame(model, frame):
    """Run detection on one frame; return annotated frame, counts, and a feature row."""
    height, width = frame.shape[:2]
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    annotated = frame.copy()
    class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    boxes = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id not in VEHICLE_CLASSES:
            continue
        label = VEHICLE_CLASSES[class_id]
        class_counts[label] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        boxes.append((x1, y1, x2, y2))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    total = sum(class_counts.values())
    ratio = occupancy_ratio(boxes, width, height) if boxes else 0.0

    feature_row = {"vehicle_count": total, "occupancy_ratio": ratio}
    for name in VEHICLE_CLASSES.values():
        feature_row[f"{name}_ratio"] = class_counts[name] / total if total > 0 else 0.0

    return annotated, class_counts, feature_row


def analyze_video(model, video_path):
    """Sample a handful of evenly spaced frames; average their features."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = set(
        int(i * frame_count / VIDEO_SAMPLE_FRAMES) for i in range(VIDEO_SAMPLE_FRAMES)
    )

    sampled_rows = []
    display_frame = None
    idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if idx in sample_indices:
            annotated, counts, row = analyze_frame(model, frame)
            sampled_rows.append((counts, row))
            if display_frame is None or idx == frame_count // 2:
                display_frame = annotated
        idx += 1
    cap.release()

    avg_row = pd.DataFrame([r for _, r in sampled_rows]).mean().to_dict()
    avg_row["vehicle_count"] = round(avg_row["vehicle_count"])

    total_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    for counts, _ in sampled_rows:
        for name, c in counts.items():
            total_counts[name] += c
    avg_counts = {name: round(c / len(sampled_rows)) for name, c in total_counts.items()}

    return display_frame, avg_counts, avg_row
