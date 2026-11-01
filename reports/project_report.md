# SmartMove AI: AI-Powered Traffic Intelligence and Sustainable Mobility Recommendation System

*Project report — BTech CSE (AI/ML), 3rd Year*

---

## Abstract

SmartMove AI is an end-to-end system that analyzes traffic images and video to detect
and count vehicles using deep learning, estimates traffic density and predicts
congestion level, and recommends a transportation mode (car, bus, metro, or carpool)
based on congestion, trip distance, cost, and environmental impact. The system uses
YOLOv8 (a convolutional neural network-based object detector) for vehicle detection,
a Random Forest classifier for congestion prediction, and a weighted multi-criteria
scoring model for transport recommendation, presented through a Streamlit web
interface. The project deliberately combines exactly one deep learning component
(detection), one classical machine learning component (congestion prediction), and
one transparent rule-based component (recommendation) — each chosen because it is the
right tool for that sub-problem, not to maximize apparent model complexity. Testing
across varied traffic scenes revealed a genuine distribution-shift limitation (90.9%
in-domain vs. 25% out-of-domain congestion classification accuracy), which is reported
honestly as a finding rather than concealed.

## 1. Introduction

Urban commuters routinely face a combination of unpredictable traffic congestion and
a lack of tools that connect *visual* traffic conditions to *actionable* transport
decisions. Existing traffic apps show congestion levels but rarely reason about which
transport mode is actually best for a specific trip given the current conditions,
cost, and environmental impact. SmartMove AI addresses this gap by building a single
pipeline from raw traffic imagery to a concrete, explained recommendation.

## 2. Problem Statement

Commuters lack a simple, integrated tool that (a) assesses real-time traffic
conditions from visual data and (b) translates that assessment, combined with trip
parameters, into a personalized, explainable transport mode recommendation.

## 3. Objectives

1. Detect and count vehicles from traffic images/video using a deep learning object
   detector.
2. Quantify traffic density and classify congestion level.
3. Recommend a transport mode (car / bus / metro / carpool) using congestion and trip
   parameters (distance, cost, time, environmental impact).
4. Demonstrate the complete pipeline through a working, simple web interface.

## 4. Scope

- Single-image or short-video traffic analysis (not live citywide camera feeds).
- Four vehicle classes: car, bus, truck, motorcycle (COCO-defined classes covered by
  the pretrained detector).
- Recommendation driven by user-entered trip parameters, not live GPS/Maps
  integration.
- Demo-scale dataset (~100 frames from one video plus a handful of stock images), not
  production-scale.

**Explicitly out of scope:** live camera/API integration, user accounts, payments,
multi-city deployment, database-backed historical analytics, mobile app. These are
listed under Future Scope (Section 14).

## 5. Literature Survey

1. **Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).** "You Only Look
   Once: Unified, Real-Time Object Detection." *CVPR 2016.* [arXiv:1506.02640](https://arxiv.org/abs/1506.02640).
   Introduced the single-stage detection paradigm this project's detector (YOLOv8,
   the architecture's fifth-generation descendant) is built on — framing detection as
   one regression problem over the whole image instead of a two-stage
   propose-then-classify pipeline.

2. **Mittal, U., & Chawla, P. (2023).** "Vehicle detection and traffic density
   estimation using ensemble of deep learning models." *Multimedia Tools and
   Applications* (Springer). [Link](https://link.springer.com/article/10.1007/s11042-022-13659-5).
   Combines Faster R-CNN and SSD for vehicle detection and density estimation,
   reporting up to 94% mAP on a thermal imaging dataset — relevant prior work on the
   detection + density estimation pairing this project also implements, though with a
   single detector (YOLOv8) rather than an ensemble, for scope reasons.

3. **"Deep Learning Techniques for Vehicle Detection and Classification from
   Images/Videos: A Survey"** (2023). [PMC10224508](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10224508/).
   Surveys deep learning architectures for vehicle detection/classification,
   confirming CNN-based detectors (including YOLO variants) as the dominant approach
   for this task class.

4. **"Traffic Congestion Prediction Using Machine Learning Techniques"** (2022).
   [arXiv:2206.10983](https://arxiv.org/html/2206.10983v5).
   Compares classical ML models for congestion prediction; motivates this project's
   Phase 8 decision to evaluate Random Forest against Logistic Regression rather than
   assume a deep learning model is necessary for a low-dimensional tabular prediction
   task.

5. **Weighted Sum Model (Simple Additive Weighting)** — see [Wikipedia summary](https://en.wikipedia.org/wiki/Weighted_sum_model)
   and the DTU Transport Compendium on [Multi-Criteria Decision Analysis for transport](https://backend.orbit.dtu.dk/ws/files/104276012/DTU_Transport_Compendium_Part_2_MCDA_.pdf).
   The weighted sum model is a well-established multi-criteria decision analysis
   (MCDA) technique; this project's recommendation engine (Section 10.4) is a direct,
   domain-adapted application of it.

## 6. Methodology / Workflow

```
Traffic Image/Video -> Vehicle Detection (YOLOv8) -> Vehicle Counting + Density (rule-based)
  -> Congestion Prediction (Random Forest) -> Transportation Recommendation (weighted scoring)
  -> Streamlit Web Interface
```

Component breakdown by technique category:

| Component | Category | Justification |
|---|---|---|
| Vehicle detection | Deep Learning (CNN) | YOLOv8; genuine DL contribution |
| Image/video handling | Classical Computer Vision (OpenCV) | Pixel manipulation, not learned |
| Traffic density | Rule/statistics-based | Percentile thresholds on vehicle count; transparent by design |
| Congestion prediction | Classical Machine Learning | Random Forest on low-dimensional tabular features |
| Transport recommendation | Rule-based weighted scoring | No training data exists for a learned recommender; MCDA is the standard technique for this problem class |

## 7. System Architecture

```
SmartMove Ai/
├── app.py                     # Streamlit frontend
├── src/
│   ├── detection.py           # YOLO wrapper: load model, run inference, draw boxes
│   ├── density.py             # vehicle count -> density label (rule-based)
│   ├── build_dataset.py       # video frame sampling -> feature table for training
│   ├── congestion_model.py    # scikit-learn classifier: train/predict congestion
│   ├── recommender.py         # weighted scoring -> transport recommendation
│   ├── pipeline.py            # shared analysis logic (used by app.py and run_tests.py)
│   └── run_tests.py           # systematic scenario testing
├── models/                    # trained congestion_clf.pkl, YOLO weights, density thresholds
├── data/                      # raw images/videos, processed feature CSVs
└── reports/                   # figures, evaluation summary, limitations log, this report
```

## 8. Technology Stack

Python, OpenCV (image/video I/O), Ultralytics YOLOv8 (PyTorch-backed detector),
NumPy/Pandas (numerical/tabular processing), scikit-learn (congestion classifier),
Streamlit (frontend). PyTorch alone (not PyTorch + TensorFlow) was used since
Ultralytics YOLOv8 is PyTorch-native and adding a second DL framework would add
install complexity with no benefit. MERN/MongoDB/authentication were explicitly
excluded as unnecessary complexity for a single-user DL demonstration project.

## 9. Dataset Description

**Detection:** no custom dataset was collected for vehicle detection — the
pretrained YOLOv8n weights (trained on COCO, which includes `car`, `bus`, `truck`,
`motorcycle` classes) were used directly via transfer learning, avoiding unnecessary
data collection for a task the pretrained model already handles well.

**Congestion classifier training data:** built by sampling 1 frame every ~0.33
seconds (10 frames) from a 35-second, 1920x1080 traffic video ("Traffic.ogv", filmed
in Kumasi, Ghana, showing slowed traffic on a poor-quality road; CC BY-SA 4.0,
Wikimedia Commons, Wiki Loves Africa 2020), yielding 106 labeled frames. Each frame's
label (Low/Medium/High) was derived from a composite score combining normalized
vehicle count and occupancy ratio, thresholded at the 33rd/66th percentile of the
observed distribution — not arbitrary cutoffs.

**Test images** (for evaluation and testing, not training): 4 additional images
sourced from Wikimedia Commons under CC BY-SA/CC BY licenses — a bus/street scene,
a highway traffic jam ("401 Traffic Jam.jpg"), a moderate-traffic intersection
("Cars at a traffic light.jpg", Nairobi), and a mostly-parked street scene
("Main Street from intersection...jpg", Ocean Grove, NJ).

**Train/test split:** 80/20, stratified by class, for the congestion classifier.

## 10. Algorithms

### 10.1 Vehicle Detection — YOLOv8

A single-stage CNN-based object detector. Given an image, it outputs bounding boxes,
class labels, and confidence scores in one forward pass. This project filters the
80 COCO classes down to 4 vehicle classes (car, motorcycle, bus, truck) and applies a
confidence threshold of 0.25 (tuned from an initial 0.4 after empirical testing
showed it more than doubled recall on a multi-vehicle scene with no observed false
positives).

### 10.2 Traffic Density — Rule-Based Classification

`density = f(vehicle_count)`, where thresholds are the 33rd/66th percentile of
vehicle counts observed across the training frame set (2.0 and 4.0 vehicles),
persisted to `models/density_thresholds.json`. Deliberately not a learned model —
transparency was prioritized over marginal accuracy gains for this sub-problem.

### 10.3 Congestion Prediction — Random Forest Classifier

**Why not a neural network:** the input is 6 low-dimensional tabular features
(vehicle count, occupancy ratio, 4 class ratios) with ~100 training rows — exactly
the regime where classical ML is standard practice and a neural network would be
both unjustified and prone to worse performance from insufficient data.

**Why Random Forest over Logistic Regression:** empirically compared (not assumed) —
Random Forest achieved 90.9% test accuracy vs. Logistic Regression's 81.8% on the
same held-out split, plausibly because it captures non-linear interactions between
count/occupancy/vehicle-mix that a linear model cannot.

**A methodological note on label design:** an initial version defined the congestion
label as a direct threshold on vehicle count — the same feature also used as model
input — which produced a misleading 100% test accuracy (the model had merely
re-learned the same threshold it was labeled with). The label was redesigned to use
a composite of count *and* occupancy ratio, after which genuine (90.9%, non-trivial)
accuracy was observed. This is documented because catching this kind of label leakage
is itself a core ML competency, not just a detail.

### 10.4 Transportation Recommendation — Weighted Sum Model

For each of 4 modes, time/cost/emissions are computed from reference parameters
(congestion-tiered speeds, real Delhi Metro fare slabs, government bus fare rates,
published emission factors — see `src/recommender.py` docstring for full citations),
each min-max normalized across the 4 modes for the specific query, then combined via
a user-weighted sum (time/cost/environment priorities). The mode with the lowest
score is recommended. Access/waiting time overhead (walking to a stop/station,
parking) is included per mode — without it, mass transit unrealistically dominated
every scenario in initial testing, since it lacked a "door to door" penalty.

## 11. Results

A systematic test (`src/run_tests.py`) across 5 traffic sources x 3 distances (5/15/
30 km) x 3 priority profiles (Balanced/Eco-focused/Time-focused) produced 45
scenarios. Metro was recommended in 36/45 (80%) — expected, given the project's
sustainability framing and Metro's typically lower cost/emissions/congestion
sensitivity in the modeled scenarios — but Car was correctly recommended in
time-heavily-weighted, short-distance scenarios (5/45), confirming the recommender
responds to genuine trade-offs rather than always favoring one mode. Full results:
`reports/test_results.csv`.

## 12. Evaluation

See `reports/evaluation_summary.md` for full detail. Key results:

- **Detection:** official YOLOv8n benchmark, mAP@0.5:0.95 = 37.3 (COCO val2017,
  Ultralytics). Manual ground-truth spot-checks on our own images: 100% recall on
  close-range scenes, dropping sharply (well under 30%) on an aerial highway jam with
  small, distant, overlapping vehicles. Zero false positives observed across all
  test images.
- **Congestion classifier:** 90.9% in-domain accuracy vs. **25% out-of-domain
  accuracy** (on 4 images never seen in training, compared against the rule-based
  density label). This large, formally quantified gap is the project's central
  evaluation finding — direct evidence of overfitting to a single training video's
  camera framing.
- **MAE/RMSE:** not applicable — congestion labels are categorical, not continuous.

## 13. Limitations

1. Small/distant/heavily occluded vehicles are undercounted, especially in
   aerial/wide-angle views (a known characteristic of one-stage detectors).
2. Count-based density cannot distinguish parked vehicles from moving traffic.
3. Extreme close-up crops with unusual vehicle framing/cargo can be missed entirely.
4. The congestion classifier, trained on a single video source, does not generalize
   to structurally different camera framings (formally measured: 25% out-of-domain
   accuracy).
5. Reference parameters for the recommender (speeds, costs, emissions) are real,
   cited national/representative averages, not measurements for a specific city.
6. All training/test data comes from a small, narrow-domain sample (one video, a
   handful of images) — appropriate for a college project demonstration, not for
   production deployment.

Full detail: `reports/limitations_log.md`.

## 14. Future Scope

- Train the congestion classifier on footage from multiple camera angles, distances,
  and locations to close the distribution-shift gap identified in Section 12.
- Fine-tune YOLO on a labeled traffic-specific dataset (e.g., via Roboflow Universe)
  for improved domain-specific detection accuracy.
- Integrate live traffic camera feeds and real-time GPS/Maps distance calculation.
- Replace assumed reference parameters (speed/cost/emissions) with city-specific,
  regularly updated data sources.
- Explore a learned recommendation model once real usage/feedback data exists to
  train one (not currently available or fabricable).

## 15. Conclusion

SmartMove AI demonstrates a complete, working pipeline from raw traffic imagery to
an explainable transportation recommendation, built with a deliberate, defensible
split between deep learning (vehicle detection), classical machine learning
(congestion prediction), and transparent rule-based logic (density and
recommendation) — using each technique where it is genuinely appropriate rather than
where it looks most impressive. Rigorous testing surfaced a real, formally quantified
limitation (out-of-domain generalization failure in the congestion classifier), which
is documented honestly as a finding rather than hidden — itself a demonstration of
sound ML practice.

## 16. References

1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once:
   Unified, Real-Time Object Detection. CVPR 2016. https://arxiv.org/abs/1506.02640
2. Ultralytics YOLOv8 Documentation. https://docs.ultralytics.com/models/yolov8
3. Mittal, U., & Chawla, P. (2023). Vehicle detection and traffic density estimation
   using ensemble of deep learning models. Multimedia Tools and Applications.
   https://link.springer.com/article/10.1007/s11042-022-13659-5
4. Deep Learning Techniques for Vehicle Detection and Classification from
   Images/Videos: A Survey (2023). PMC10224508.
5. Traffic Congestion Prediction Using Machine Learning Techniques (2022).
   arXiv:2206.10983.
6. Weighted Sum Model. Wikipedia. https://en.wikipedia.org/wiki/Weighted_sum_model
7. DTU Transport Compendium, Part 2: Multi-Criteria Decision Analysis.
   https://backend.orbit.dtu.dk/ws/files/104276012/DTU_Transport_Compendium_Part_2_MCDA_.pdf
8. "Traffic.ogv" (training video), Kumasi, Ghana, Wiki Loves Africa 2020. Wikimedia
   Commons. CC BY-SA 4.0. https://commons.wikimedia.org/wiki/File:Traffic.ogv
9. "401 Traffic Jam.jpg". Wikimedia Commons. CC BY-SA 2.0.
10. "Cars at a traffic light.jpg", Nairobi. Wikimedia Commons.
11. "Main Street from intersection - view of intersection.jpg", Ocean Grove, NJ.
    Wikimedia Commons.
12. Delhi Metro Rail Corporation (DMRC) fare chart (2026).
13. Global Traffic Index 2025 (Indian city traffic speed data).
14. India corporate average fleet emissions figure, 2021-22; Down To Earth reporting
    on urban mobility and emissions.
