"""Temporal-signature FP classifier for solar PV detections.

Trains a small tabular classifier (LightGBM, 3-class: pv / becoming_pv /
not_pv) on year-long aggregate_spatial traces of known PV polygons plus
sampled negatives, then applies it as a post-U-Net filter on detection
polygons. See the project plan and README.md for the full design.

Modules:
    polygons   — Phase 1: sample positives + 3 negative classes
    timeseries — Phase 2: extract aggregate_spatial traces via CDSE OpenEO
    features   — Phase 3: compute features over a time-series trace
    train      — Phase 3: train the LightGBM classifier
    inference  — Phase 4: runtime filter (trace -> features -> classify)
    eval_eu_unseen — Phase 5: evaluate on the cached EU-unseen sites
"""
