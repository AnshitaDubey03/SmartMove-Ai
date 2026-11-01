
"""Vehicle detection using a pretrained YOLOv8 model."""
""""""

import cv2
from ultralytics import YOLO

# COCO class indices for the vehicle categories we care about
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CONFIDENCE_THRESHOLD = 0.4


def load_model(weights_path: str = "yolov8n.pt") -> YOLO:
    return YOLO(weights_path)


def detect_image(model: YOLO, image_path: str):
    """Run detection on a single image.

    Returns the annotated image (NumPy array) and a dict of vehicle counts per class.
    """
    results = model(image_path, conf=CONFIDENCE_THRESHOLD)[0]

    image = cv2.imread(image_path)
    counts = {name: 0 for name in VEHICLE_CLASSES.values()}

    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id not in VEHICLE_CLASSES:
            continue

        label = VEHICLE_CLASSES[class_id]
        confidence = float(box.conf[0])
        counts[label] += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image, f"{label} {confidence:.2f}", (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )

    return image, counts


def detect_video(model: YOLO, video_path: str, output_path: str):
    """Run detection on every frame of a video and write an annotated copy.

    Returns a list of per-frame vehicle counts (one dict per frame).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    per_frame_counts = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        counts = {name: 0 for name in VEHICLE_CLASSES.values()}

        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASSES:
                continue

            label = VEHICLE_CLASSES[class_id]
            counts[label] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        per_frame_counts.append(counts)
        writer.write(frame)

    cap.release()
    writer.release()
    return per_frame_counts


if __name__ == "__main__":
    model = load_model("yolov8n.pt")
    annotated, counts = detect_image(model, "data/raw/images/sample_bus.jpg")
    cv2.imwrite("reports/figures/sample_detection.jpg", annotated)
    print("Vehicle counts:", counts)
    print("Total vehicles:", sum(counts.values()))
