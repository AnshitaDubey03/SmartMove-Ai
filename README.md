# SmartMove AI

**AI-Powered Traffic Intelligence and Sustainable Mobility Recommendation System**

*BTech CSE (AI/ML), 3rd Year — Deep Learning College Project*

SmartMove AI is an end-to-end pipeline that takes a traffic image or video, detects
and counts vehicles using deep learning, estimates traffic density and predicts
congestion level, and recommends the best transport mode (Car, Bus, Metro, or
Carpool) for a trip based on congestion, distance, cost, and environmental impact —
all through a simple Streamlit web interface.

The project deliberately combines exactly one deep learning component (detection),
one classical machine learning component (congestion prediction), and one
transparent rule-based component (recommendation) — each chosen because it is the
right tool for that sub-problem, not to maximize apparent model complexity.

---

## Pipeline

```
Traffic Image/Video
      │
      ▼
Vehicle Detection (YOLOv8n)              — Deep Learning (CNN)
      │
      ▼
Vehicle Counting + Density (rule-based)  — Percentile thresholds on count
      │
      ▼
Congestion Prediction (Random Forest)    — Classical ML on tabular features
      │
      ▼
Transportation Recommendation            — Weighted multi-criteria scoring
(weighted by time / cost / environment)
      │
      ▼
Streamlit Web Interface
```

---

## Models Used

| Stage | Model | Category | Why this one |
|---|---|---|---|
| Vehicle detection | **YOLOv8n** (pretrained, `yolov8n.pt`) | Deep Learning (single-stage CNN detector) | Transfer learning on COCO weights — COCO already includes car/bus/truck/motorcycle classes, so no custom training was needed |
| Traffic density | Percentile-threshold rule (`f(vehicle_count)`) | Rule/statistics-based | Kept transparent by design; not learned |
| Congestion prediction | **Random Forest Classifier** (scikit-learn) | Classical ML | Empirically beat Logistic Regression (90.9% vs 81.8% test accuracy) on 6 low-dimensional tabular features; a neural net is unjustified with only ~100 training rows |
| Transport recommendation | **Weighted Sum Model** (Simple Additive Weighting / MCDA) | Rule-based | No training data exists for a learned recommender; this is the standard technique for this problem class |

### Detection details
- Filters COCO's 80 classes down to 4 vehicle classes: `car`, `motorcycle`, `bus`, `truck`
- Confidence threshold: **0.25** (tuned from an initial 0.4 after testing showed it more than doubled recall with no observed false positives)

### Congestion classifier details
- **Input features (6):** `vehicle_count`, `occupancy_ratio`, `car_ratio`, `motorcycle_ratio`, `bus_ratio`, `truck_ratio`
- **Output:** Low / Medium / High
- **Training data:** 106 frames sampled (1 every ~0.33s) from a 35-second, 1920x1080 traffic video ("Traffic.ogv", Kumasi, Ghana; CC BY-SA 4.0, Wikimedia Commons)
- **Label design:** composite score = 0.5 × normalized vehicle count + 0.5 × normalized occupancy ratio, thresholded at the 33rd/66th percentile — deliberately *not* a direct threshold on vehicle count alone (an earlier version that did this leaked the label into the input and produced a misleading 100% accuracy)
- **Train/test split:** 80/20, stratified

### Recommendation engine details
For each mode (Car / Bus / Metro / Carpool), computes:
- **Time** — congestion-tiered speed + fixed access/waiting overhead (walking to stop/station, parking)
- **Cost** — real Delhi Metro (DMRC) distance-based fare slabs, government bus fare rates, fuel cost for car/carpool
- **Emissions** — published g CO₂/passenger-km figures (India fleet average, CNG bus, Mumbai Metro lifecycle assessment)

Each metric is min-max normalized across the 4 modes, then combined via a
user-adjustable weighted sum (default: time 0.4, cost 0.3, environment 0.3). The
lowest-scoring mode is recommended.

---

## Tech Stack & All Packages

**Language:** Python (Python-only stack — no MERN/database/auth; see [Scope](#scope--exclusions))

### Core ML / Deep Learning
| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.14.0 | Backend tensor/DL framework for YOLOv8 |
| `torchvision` | 0.29.0 | Vision utilities for torch |
| `ultralytics` | 8.4.165 | YOLOv8 implementation (detector) |
| `scikit-learn` | 1.9.1 | Random Forest, Logistic Regression, train/test split, scaling, metrics |
| `scipy`, `sympy`, `mpmath` | latest | Numerical support (sklearn/torch dependencies) |

### Computer Vision & Data Processing
| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | 5.0.0.93 | Image/video I/O, drawing bounding boxes |
| `numpy` | 2.5.3 | Array/numeric operations |
| `pandas` | 3.0.6 | Feature tables, CSV read/write |
| `pillow` | 12.3.0 | Image handling |
| `joblib` | 1.6.0 | Saving/loading the trained congestion model (`.pkl`) |

### Frontend
| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.64.0 | Web UI |
| `altair`, `pydeck`, `pyarrow`, `narwhals`, `polars` | latest | Streamlit's internal charting/data dependencies |
| `watchdog` | 6.0.0 | Streamlit hot-reload |

### Supporting / Transitive Dependencies
`matplotlib`, `networkx`, `PyYAML`, `requests`, `httpx`, `httpcore`, `uvicorn`,
`starlette`, `jsonschema`, `jsonschema-specifications`, `protobuf`, `psutil`,
`nvidia-ml-py`, `filelock`, `fsspec`, `click`, `Jinja2`, `MarkupSafe`, `attrs`,
`certifi`, `charset-normalizer`, `idna`, `urllib3`, `six`, `packaging`,
`typing_extensions`, `python-dateutil`, `tzdata`, `threadpoolctl`, `cloudpickle`,
`contourpy`, `cycler`, `kiwisolver`, `fonttools`, `pyparsing`, `toml`, `websockets`,
`h11`, `anyio`, `rpds-py`, `referencing`, `python-multipart`, `httptools`.

> Note: only PyTorch (not PyTorch + TensorFlow) was used, since Ultralytics YOLOv8
> is PyTorch-native and a second DL framework would add install complexity with no
> benefit.

To regenerate the exact environment:
```
.\venv\Scripts\python.exe -m pip freeze > requirements.txt
```

---

## Project Structure

```
SmartMove Ai/
├── app.py                     # Streamlit frontend — main entry point
├── src/
│   ├── detection.py           # YOLOv8 wrapper: load model, run inference, draw boxes
│   ├── density.py             # vehicle count -> density label (rule-based)
│   ├── build_dataset.py       # video frame sampling -> feature table for training
│   ├── congestion_model.py    # scikit-learn classifier: train/compare/predict
│   ├── recommender.py         # weighted scoring -> transport recommendation
│   ├── pipeline.py            # shared analysis logic (used by app.py and run_tests.py)
│   └── run_tests.py           # systematic scenario testing (45 scenarios)
├── models/
│   ├── yolo_weights/yolov8n.pt      # pretrained YOLOv8 nano weights
│   ├── congestion_clf.pkl           # trained Random Forest bundle (model + scaler)
│   └── density_thresholds.json      # persisted Low/Medium/High count cutoffs
├── data/
│   ├── raw/images/                  # 4 test images (bus, highway, mainstreet, trafficlight)
│   ├── raw/videos/sample_traffic.ogv
│   └── processed/features.csv       # 106-row training feature table
├── reports/
│   ├── project_report.md            # full academic project report
│   ├── evaluation_summary.md        # detection + classifier evaluation results
│   ├── limitations_log.md           # 4 documented, observed limitations
│   ├── test_results.csv             # 45-scenario recommendation test output
│   └── figures/                     # detection sample images
└── venv/                            # Python virtual environment
```

---

## Setup & Installation

```powershell
# 1. Create and activate the virtual environment (already present as venv/)
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (if setting up fresh)
pip install torch torchvision ultralytics scikit-learn opencv-python pandas numpy streamlit joblib
```

## Usage

**Run the web app:**
```powershell
.\venv\Scripts\streamlit.exe run app.py
```
Opens at `http://localhost:8501` — upload an image/video, enter trip distance and
priority weights, and get a ranked transport recommendation.

**Rebuild the training dataset** (samples frames from the source video, runs
detection, computes features + labels):
```powershell
.\venv\Scripts\python.exe src\build_dataset.py
```

**Retrain the congestion classifier** (compares Random Forest vs Logistic
Regression, saves the better one):
```powershell
.\venv\Scripts\python.exe -m src.congestion_model
```

**Run the systematic scenario test suite** (writes `reports/test_results.csv`):
```powershell
.\venv\Scripts\python.exe -m src.run_tests
```

---

## Dataset

- **Detection:** no custom dataset — pretrained YOLOv8n (COCO-trained) used directly via transfer learning.
- **Congestion classifier training:** 106 frames sampled from one 35-second traffic video (Kumasi, Ghana; CC BY-SA 4.0, Wikimedia Commons, Wiki Loves Africa 2020).
- **Test images:** 4 Wikimedia Commons images (CC BY-SA/CC BY) — a bus/street scene, a highway traffic jam, a moderate-traffic intersection (Nairobi), and a mostly-parked street scene (Ocean Grove, NJ).
- **Scope:** demo-scale (~100 frames + a handful of images), not production-scale.

---

## Results

- **Detection (YOLOv8n):** official benchmark mAP@0.5:0.95 = 37.3 (COCO val2017). On our own images: 100% precision (zero false positives), recall 100% on close-range scenes dropping to well under 30% on an aerial highway jam with small, distant, overlapping vehicles.
- **Congestion classifier (Random Forest):** **90.9% in-domain accuracy** vs. **25% out-of-domain accuracy** (4 unseen images, evaluated against the rule-based density label as proxy ground truth). This large, formally quantified gap is the project's central evaluation finding — direct evidence of overfitting to a single training video's camera framing.
- **Recommendation engine:** 45 scenarios tested (5 traffic sources × 3 distances × 3 priority profiles). Metro recommended in 80% of cases (expected, given its cost/emissions/congestion-resilience advantage); Car correctly recommended in time-heavily-weighted, short-distance scenarios — confirming the recommender responds to genuine trade-offs.

Full detail: `reports/evaluation_summary.md`, `reports/project_report.md`.

---

## Limitations

1. **Undercounting small/distant/occluded vehicles** — e.g. only 13 of ~17-19 vehicles detected in an aerial highway jam cross-section. Worse with the "nano" (smallest) YOLO variant.
2. **Parked vs. moving vehicles not distinguished** — a street scene with 11 mostly-parked vehicles was classified "High" density.
3. **Extreme close-up/unusual cargo missed entirely** — a large truck with an atypical scrap-metal load, very close and at an oblique angle, was not detected.
4. **Distribution shift in the congestion classifier** — trained only on one dashcam video where count and occupancy were positively correlated (r=0.46); fails on wide/aerial images with high count but low occupancy, a combination never seen in training (formally measured: 25% out-of-domain accuracy).
5. Recommender reference parameters (speed/cost/emissions) are real, cited **national/representative averages**, not city-specific measurements.
6. All training/test data comes from a small, narrow-domain sample — appropriate for a college project, not production deployment.

Full detail: `reports/limitations_log.md`.

---

## Scope & Exclusions

**In scope:** single-image/short-video analysis, 4 vehicle classes (car/bus/truck/motorcycle), recommendation from user-entered trip parameters, demo-scale dataset.

**Explicitly excluded:** MERN stack (React/Node/Express), any database, user authentication, payment systems, live camera/API integration, multi-city deployment, mobile app. Reason: unnecessary complexity for a solo DL college project — adds no academic value.

## Future Scope

- Train the congestion classifier on footage from multiple camera angles/locations to close the distribution-shift gap.
- Fine-tune YOLO on a labeled traffic-specific dataset (e.g. via Roboflow Universe).
- Integrate live traffic camera feeds and real-time GPS/Maps distance calculation.
- Replace assumed reference parameters with city-specific, regularly updated data sources.
- Explore a learned recommendation model once real usage/feedback data exists.

---

## References

1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
2. Ultralytics YOLOv8 Documentation. https://docs.ultralytics.com/models/yolov8
3. Mittal, U., & Chawla, P. (2023). *Vehicle detection and traffic density estimation using ensemble of deep learning models.* Multimedia Tools and Applications. [Springer](https://link.springer.com/article/10.1007/s11042-022-13659-5)
4. *Deep Learning Techniques for Vehicle Detection and Classification from Images/Videos: A Survey* (2023). [PMC10224508](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10224508/)
5. *Traffic Congestion Prediction Using Machine Learning Techniques* (2022). [arXiv:2206.10983](https://arxiv.org/html/2206.10983v5)
6. Weighted Sum Model. [Wikipedia](https://en.wikipedia.org/wiki/Weighted_sum_model); [DTU Transport Compendium, Part 2: MCDA](https://backend.orbit.dtu.dk/ws/files/104276012/DTU_Transport_Compendium_Part_2_MCDA_.pdf)
7. "Traffic.ogv", Kumasi, Ghana, Wiki Loves Africa 2020. Wikimedia Commons. CC BY-SA 4.0.
8. Delhi Metro Rail Corporation (DMRC) fare chart (2026).
9. Global Traffic Index 2025 (Indian city traffic speed data).
10. India corporate average fleet emissions figure, 2021-22.
