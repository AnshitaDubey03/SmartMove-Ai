# Phase 13: Model Evaluation Summary

## A. Detection (YOLOv8n)

No ground-truth bounding box annotations exist for our own images (pretrained COCO
weights were used directly, per Phase 5's decision to avoid unnecessary data
collection), so a formal mAP on our data is not something we can honestly compute.
Instead:

- **Official published benchmark** (not our result, cited for context): YOLOv8n
  achieves mAP@0.5:0.95 = 37.3 on COCO val2017 (Ultralytics official documentation).
- **Manual ground-truth spot-check** (our own, computed):

| Image | Scenario | Manual count | Detected | Recall (proxy) |
|---|---|---|---|---|
| sample_bus.jpg | close-range street | 1 | 1 | 100% |
| sample_trafficlight.jpg | close-range dashcam | ~10-11 | 8 | ~73-80% |
| sample_mainstreet.jpg | medium-range parked | 11 | 11 | 100% |
| sample_highway.jpg | aerial/wide highway jam | impractical to count fully; sample cross-section alone had 17-19 vs. 13 detected total | 13 | severely low |

Precision (proxy): 100% observed -- zero false positives across all test images.
Recall is the weak point, degrading sharply with distance/scale (see
reports/limitations_log.md item 1).

## B. Congestion Classifier (Random Forest)

- **In-domain** (train/test split from the same training video, Phase 8): accuracy
  90.9%, precision/recall/F1 >= 0.86 per class, errors confined to adjacent
  High/Medium confusion.
- **Out-of-domain** (4 still images never seen in training, compared against the
  rule-based density label as proxy ground truth): accuracy **1/4 = 25%**.

This 90.9% vs. 25% gap is the key evaluation finding: it's formal, quantified proof
of the distribution-shift problem identified in Phase 12, not just a qualitative
observation.

**MAE/RMSE: not applicable.** Congestion labels are categorical (Low/Medium/High),
not continuous -- regression error metrics would not be meaningful here.
