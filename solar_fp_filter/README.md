# solar_fp_filter

Temporal-signature false-positive filter for the solar-PV U-Net detections.

## What it does

The R3 R101 U-Net is a per-image segmentation model — it sees one 3-month mosaic of a scene and decides which pixels are PV. That works well for most cases but still produces visible FPs on things like greenhouses, reflective industrial roofs, or unusual water surfaces, because those classes share *spatial* signatures with PV.

This module adds a second stage that exploits *temporal* signatures via CDSE OpenEO's `aggregate_spatial` reducer. For each detection polygon coming out of the U-Net, it:

1. Pulls the most recent 12 months of mean reflectance per band (a small table of scalars — ~50 timestamps × 11 bands — costing ~1% of the credit of a raster pull).
2. Computes ~40 features over the trace, focused on the last 6 months (so newly-built sites still pass).
3. Classifies into `pv` (stable PV signature), `becoming_pv` (construction → PV transition), or `not_pv` (anything else — vegetation phenology, water variability, persistent industrial reflectance, persistent bare ground).
4. Keeps detections where `P(pv) + P(becoming_pv) ≥ 0.5`.

## Why this catches FPs the U-Net misses

- **Vineyards / cropland**: have a strong NDVI seasonal swing that PV never has. Falls into `not_pv`.
- **Greenhouses**: similar low-NDVI signature to PV but at much higher absolute reflectance in the visible bands. The features include per-band mean reflectance levels, so the classifier learns to require *dark* low-NDVI signatures.
- **Reservoirs / water bodies**: highly variable reflectance with illumination/wind. Falls into `not_pv` via per-band CV features.
- **Reflective industrial roofs**: persistent high VIS reflectance + flat NDVI. The classifier learns "PV ends DARK and low-NDVI", not just "ends low-NDVI".

## Why it doesn't reject newly-built PV

A naïve classifier trained only on stable, fully-built PV would reject sites where the last few months are PV but earlier months were construction or bare ground — defeating the deployment purpose. We address this by giving the classifier **three windows per positive polygon at training time** (pre-build / transition / fully-built) so it learns the trajectory shape of a real PV site as well as the stable signature. The `becoming_pv` class explicitly accepts construction → PV traces. See the plan section "Why 'phase' is implicit, not an explicit input" for the full reasoning.

## How to retrain (for Apex)

```bash
# Phase 1: build the polygon training set (offline, CPU, ~5 min)
python scripts/run_fp_phase1_build_polygons.py

# Phase 2: extract aggregate_spatial traces via CDSE (~6-10K credits)
sbatch scripts/slurm/fp_phase2_extract.sh

# Phase 3: train LightGBM
sbatch scripts/slurm/fp_phase3_train.sh

# Phase 5: evaluate on the EU-unseen cache
sbatch scripts/slurm/fp_phase5_eval.sh
```

Artifacts land in `outputs/fp_classifier/`. The runtime path is `solar_fp_filter/inference.py:apply_filter` — called from the production UDF in `openeo_udp/udf/fp_classifier_udf.py` (after Phase 6 packaging).
