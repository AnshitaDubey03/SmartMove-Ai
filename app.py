"""SmartMove AI -- Streamlit frontend.

Pipeline: upload image/video -> YOLO detection -> density (rule) + congestion (ML)
-> trip inputs -> weighted recommendation.
"""
#http://localhost:8501


import json
import os
import tempfile

import cv2
import joblib
import pandas as pd
import streamlit as st

from src.congestion_model import FEATURE_COLUMNS
from src.density import classify_density
from src.detection import load_model
from src.pipeline import analyze_frame, analyze_video
from src.recommender import DEFAULT_WEIGHTS, recommend_transport

MODE_ICONS = {"Car": "🚗", "Bus": "🚌", "Metro": "🚇", "Carpool": "🚙"}


@st.cache_resource
def load_resources():
    detector = load_model()
    congestion_bundle = joblib.load("models/congestion_clf.pkl")
    with open("models/density_thresholds.json") as f:
        density_thresholds = json.load(f)
    return detector, congestion_bundle, density_thresholds


st.set_page_config(page_title="SmartMove AI", page_icon="🚦", layout="wide")

st.markdown("## 🚦 SmartMove AI")
st.caption("AI-Powered Traffic Intelligence and Sustainable Mobility Recommendation System")
st.divider()

if not os.path.exists("models/congestion_clf.pkl"):
    st.error("Congestion model not found. Run `python -m src.congestion_model` first (Phase 8).")
    st.stop()

detector, congestion_bundle, density_thresholds = load_resources()
density_cutoffs = (density_thresholds["low_cutoff"], density_thresholds["high_cutoff"])

with st.sidebar:
    st.header("Trip Details")
    distance = st.slider("Distance (km)", 1, 50, 10)

    st.subheader("Priorities")
    time_w = st.slider("⏱️ Time", 0.0, 1.0, DEFAULT_WEIGHTS["time"])
    cost_w = st.slider("💰 Cost", 0.0, 1.0, DEFAULT_WEIGHTS["cost"])
    env_w = st.slider("🌱 Environmental impact", 0.0, 1.0, DEFAULT_WEIGHTS["environment"])

    total_w = time_w + cost_w + env_w
    weights = (
        {"time": time_w / total_w, "cost": cost_w / total_w, "environment": env_w / total_w}
        if total_w > 0
        else DEFAULT_WEIGHTS
    )

    st.divider()
    st.caption(
        "Speed, cost, and emission reference values are sourced from real Indian "
        "urban transport data (Delhi Metro fares, government bus fare notifications, "
        "national traffic speed surveys) -- see `src/recommender.py` for citations."
    )

st.markdown("### 1. Traffic Analysis")
uploaded = st.file_uploader(
    "Upload a traffic image or video", type=["jpg", "jpeg", "png", "mp4", "ogv", "avi"]
)

analysis = None

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Running vehicle detection..."):
        if suffix.lower() in [".jpg", ".jpeg", ".png"]:
            frame = cv2.imread(tmp_path)
            annotated, counts, feature_row = analyze_frame(detector, frame)
        else:
            annotated, counts, feature_row = analyze_video(detector, tmp_path)

    total = feature_row["vehicle_count"]
    density_label = classify_density(total, density_cutoffs)

    X = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
    model = congestion_bundle["model"]
    scaler = congestion_bundle["scaler"]
    X_in = scaler.transform(X) if scaler is not None else X
    congestion_label = model.predict(X_in)[0]

    img_col, metrics_col = st.columns([2, 1])
    with img_col:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected Vehicles", use_container_width=True)
    with metrics_col:
        st.metric("🚘 Vehicle Count", total)
        st.dataframe(
            pd.DataFrame(counts.items(), columns=["Class", "Count"]),
            hide_index=True, use_container_width=True,
        )
        st.metric("📊 Traffic Density (rule-based)", density_label)
        st.metric("🧠 Predicted Congestion (ML)", congestion_label)

    analysis = {"congestion": congestion_label}
else:
    st.info("Upload a traffic image or video to begin analysis.")

st.divider()
st.markdown("### 2. Transportation Recommendation")

if analysis is None:
    st.info("Complete traffic analysis above to get a recommendation.")
else:
    ranked = recommend_transport(distance, analysis["congestion"], weights)
    best_mode, best_score, best_breakdown = ranked[0]

    with st.container(border=True):
        st.markdown(f"#### {MODE_ICONS[best_mode]} Recommended: {best_mode}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Time", f"{best_breakdown['time_min']} min")
        c2.metric("Cost", f"₹{best_breakdown['cost']}")
        c3.metric("Emissions", f"{best_breakdown['emission_g']} g CO₂")

    st.markdown("##### All options compared")
    table_rows = [
        {
            "Mode": f"{MODE_ICONS[mode]} {mode}",
            "Time (min)": breakdown["time_min"],
            "Cost (₹)": breakdown["cost"],
            "Emissions (g CO₂)": breakdown["emission_g"],
            "Score (lower = better)": score,
        }
        for mode, score, breakdown in ranked
    ]
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
    st.caption(
        "Score combines normalized time, cost, and environmental impact using your "
        "selected priorities from the sidebar."
    )
