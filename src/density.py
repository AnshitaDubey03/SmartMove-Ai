"""Traffic density calculation from YOLO detection results."""

import numpy as np


def occupancy_ratio(boxes_xyxy, frame_width: int, frame_height: int) -> float:
    """Fraction of the frame area covered by vehicle bounding boxes."""
    frame_area = frame_width * frame_height
    box_area_sum = 0.0
    for x1, y1, x2, y2 in boxes_xyxy:
        box_area_sum += (x2 - x1) * (y2 - y1)
    return box_area_sum / frame_area


def derive_count_thresholds(vehicle_counts) -> tuple[float, float]:
    """Compute Low/Medium/High cutoffs from the 33rd/66th percentile of observed counts.

    vehicle_counts: a list/array of vehicle counts, one per frame/image,
    collected from OUR OWN dataset (not guessed).
    """
    counts = np.array(vehicle_counts)
    low_cutoff = np.percentile(counts, 33)
    high_cutoff = np.percentile(counts, 66)
    return float(low_cutoff), float(high_cutoff)


def classify_density(vehicle_count: int, thresholds: tuple[float, float]) -> str:
    low_cutoff, high_cutoff = thresholds
    if vehicle_count <= low_cutoff:
        return "Low"
    elif vehicle_count <= high_cutoff:
        return "Medium"
    else:
        return "High"


def _normalize(values) -> np.ndarray:
    values = np.array(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def compute_composite_score(vehicle_counts, occupancy_ratios) -> np.ndarray:
    """Combine count and occupancy (each min-max normalized) into one 0-1 score.

    Using a composite of two independent signals -- instead of vehicle_count alone --
    ensures no single input feature can trivially determine the label, which matters
    once this label is used as a training target for a classifier (see Phase 8).
    """
    norm_counts = _normalize(vehicle_counts)
    norm_occupancy = _normalize(occupancy_ratios)
    return 0.5 * norm_counts + 0.5 * norm_occupancy


def derive_score_thresholds(scores) -> tuple[float, float]:
    scores = np.array(scores)
    return float(np.percentile(scores, 33)), float(np.percentile(scores, 66))


def classify_score(score: float, thresholds: tuple[float, float]) -> str:
    low_cutoff, high_cutoff = thresholds
    if score <= low_cutoff:
        return "Low"
    elif score <= high_cutoff:
        return "Medium"
    else:
        return "High"


if __name__ == "__main__":
    # PLACEHOLDER demo only — not enough real data yet to derive meaningful thresholds.
    sample_counts = [1, 13]  # from sample_bus.jpg and sample_highway.jpg
    thresholds = derive_count_thresholds(sample_counts)
    print("Derived thresholds (placeholder, only 2 samples):", thresholds)

    for count in sample_counts:
        print(f"{count} vehicles -> {classify_density(count, thresholds)}")
