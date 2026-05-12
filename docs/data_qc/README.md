# Data QC quickviews

Spot-check images for the in-flight **final** training dataset
(`outputs/final/chips/*_image.tif` + `*_mask.tif`).

- `quickview_montage_<UTC-stamp>.png` — grid of randomly sampled chips. For
  each chip: **left** = RGB quicklook (B4/B3/B2, 2–98 % stretch), **right** =
  the same RGB with the annotated solar-panel polygons overlaid (translucent
  red fill + yellow outline). Chips are drawn from batches spread across the
  whole generation run, so different time periods are represented.
- `samples/<chip_id>_qc.png` — the same pair, one file per sampled chip, for
  zooming in.

Regenerate / resample:

```bash
python scripts/17_qc_quickview_montage.py --n 20 --seed 42 --cols 4
# different random draw:
python scripts/17_qc_quickview_montage.py --n 24 --seed 7
```

The montage tile titles show `chip_id`, source `batch`, continent, polygon
count and masked-pixel fraction.
