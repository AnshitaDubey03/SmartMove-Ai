# Known Limitations Log

Running notes captured during development, for use in Phase 14 (Documentation) and
Phase 15 (Viva prep). Each entry is a real, observed issue -- not a hypothetical.

## 1. Small/distant/occluded vehicle undercounting (Phase 6)
On `sample_highway.jpg` (aerial highway jam), YOLOv8n at conf=0.4 detected only 6 of
dozens of visible vehicles; lowering to conf=0.25 improved this to 13, still an
undercount. Root cause: small, heavily overlapping distant vehicles are the hardest
case for a one-stage detector, worsened by using the "nano" (smallest/fastest) model
variant. Less severe on close-range/street-level footage.

## 2. Parked vs. moving vehicles not distinguished (Phase 7)
`sample_mainstreet.jpg` was classified "High" density with 11 vehicles, but most were
parked at the curb, not congested moving traffic. Our count-based density has no way
to tell parked from moving vehicles.

## 3. Extreme close-up / unusual cargo miss (Phase 11)
In one frame of `sample_traffic.ogv`, a large truck filling most of the frame (very
close, oblique angle, carrying an atypical scrap-metal load) was not detected at all,
despite being large and unambiguous to a human viewer. Likely due to the unusual
crop/angle/cargo not resembling typical COCO truck training examples.

## 4. Distribution shift: congestion classifier fails on out-of-domain framing (Phase 12)
The congestion classifier was trained exclusively on frames from one 35-second
close-up dashcam video, where vehicle count and occupancy ratio were positively
correlated (r=0.46; count range 0-9, occupancy range 0-0.80). On wide/aerial test
images with high count but low occupancy (a combination never seen in training --
e.g. 13 vehicles at 3.7% occupancy), the model predicts "Low" congestion while the
simple, transparent density rule correctly says "High". This is a genuine
distribution-shift failure: the model extrapolates poorly outside its narrow
training distribution, and a same-video train/test split (Phase 8's 90.9% accuracy)
could not have caught it, since it never tested on a different-domain sample.
**Recommended fix (future work):** train on footage from multiple camera angles/
sources, not a single video.

## Common thread
All four issues trace back to training/testing on a small, narrow-domain dataset
(one video, a handful of stock images). This is an honest, defensible limitation
for a college project -- explicitly note in the report that results are demonstrative,
and production use would require a larger, more diverse dataset.
