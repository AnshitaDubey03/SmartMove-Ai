"""Phase 12: systematic scenario testing across traffic sources, distances,
and priority profiles. Writes results to reports/test_results.csv.

Run: .\\venv\\Scripts\\python.exe -m src.run_tests
"""

import glob
import json
import os

import cv2
import joblib
import pandas as pd

from src.congestion_model import FEATURE_COLUMNS
from src.density import classify_density
from src.detection import load_model
from src.pipeline import analyze_frame, analyze_video
from src.recommender import recommend_transport

DISTANCES_KM = [5, 15, 30]
WEIGHT_PROFILES = {
    "Balanced": {"time": 0.4, "cost": 0.3, "environment": 0.3},
    "Eco-focused": {"time": 0.2, "cost": 0.2, "environment": 0.6},
    "Time-focused": {"time": 0.7, "cost": 0.15, "environment": 0.15},
}


def analyze_source(detector, path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        frame = cv2.imread(path)
        _, counts, feature_row = analyze_frame(detector, frame)
    else:
        _, counts, feature_row = analyze_video(detector, path)
    return counts, feature_row


if __name__ == "__main__":
    detector = load_model()
    congestion_bundle = joblib.load("models/congestion_clf.pkl")
    with open("models/density_thresholds.json") as f:
        density_thresholds = json.load(f)
    density_cutoffs = (density_thresholds["low_cutoff"], density_thresholds["high_cutoff"])

    sources = sorted(glob.glob("data/raw/images/*.jpg")) + sorted(glob.glob("data/raw/videos/*.ogv"))

    rows = []
    for source_path in sources:
        source_name = os.path.basename(source_path)
        print(f"Analyzing {source_name} ...")
        counts, feature_row = analyze_source(detector, source_path)

        total = feature_row["vehicle_count"]
        density_label = classify_density(total, density_cutoffs)

        X = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
        model = congestion_bundle["model"]
        scaler = congestion_bundle["scaler"]
        X_in = scaler.transform(X) if scaler is not None else X
        congestion_label = model.predict(X_in)[0]

        for distance in DISTANCES_KM:
            for profile_name, weights in WEIGHT_PROFILES.items():
                ranked = recommend_transport(distance, congestion_label, weights)
                best_mode, best_score, breakdown = ranked[0]
                rows.append({
                    "source": source_name,
                    "vehicle_count": total,
                    "density": density_label,
                    "congestion": congestion_label,
                    "distance_km": distance,
                    "priority_profile": profile_name,
                    "recommended_mode": best_mode,
                    "time_min": breakdown["time_min"],
                    "cost": breakdown["cost"],
                    "emission_g": breakdown["emission_g"],
                    "score": best_score,
                })

    df = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/test_results.csv", index=False)

    print(f"\nSaved {len(df)} test rows to reports/test_results.csv")
    print("\nRecommended mode counts overall:")
    print(df["recommended_mode"].value_counts())
    print("\nRecommended mode by priority profile:")
    print(df.groupby("priority_profile")["recommended_mode"].value_counts())
    print("\nCongestion label per source:")
    print(df.drop_duplicates("source")[["source", "vehicle_count", "density", "congestion"]])
