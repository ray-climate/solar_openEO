# Solar PV Detection — Inference Specification

This document defines the input/output contract for the U-Net solar PV
detection model.  All parameters are controlled by `model_registry.yaml`.

## Model Summary

| Property           | Value                                      |
|--------------------|--------------------------------------------|
| Architecture       | U-Net with pretrained encoder              |
| Backbone           | ResNet101 (ImageNet weights)               |
| Decoder filters    | [256, 128, 96, 64, 32]                    |
| Decoder dropout    | 0.1                                        |
| Attention gates    | No                                         |
| SE blocks          | No                                         |
| Input adapter      | 1x1 Conv projecting 13 bands → 3 channels |
| Output activation  | Sigmoid                                    |
| Training loss      | Dice loss                                  |
| Training chips     | 4,922 (Sentinel-2 L1C, Europe)             |
| Best test Dice     | 0.8853 (at threshold 0.80)                |
| Best test IoU      | 0.7942                                     |
| Weights file       | ~999 MB (H5 format)                        |

## Input Specification

| Property        | Value                          |
|-----------------|--------------------------------|
| Shape           | (256, 256, 13)                 |
| Dtype           | float32                        |
| CRS             | EPSG:3857                      |
| Resolution      | 10 m/pixel                     |
| Chip size       | 2560 m × 2560 m               |
| Temporal window | Single cloud-free composite    |
| Data level      | Sentinel-2 L1C (TOA reflectance) |

### Band Order (must be exact)

| Index | Band | Wavelength   | Native res |
|-------|------|-------------|-----------|
| 0     | B1   | 443 nm      | 60 m      |
| 1     | B2   | 490 nm      | 10 m      |
| 2     | B3   | 560 nm      | 10 m      |
| 3     | B4   | 665 nm      | 10 m      |
| 4     | B5   | 705 nm      | 20 m      |
| 5     | B6   | 740 nm      | 20 m      |
| 6     | B7   | 783 nm      | 10 m      |
| 7     | B8   | 842 nm      | 10 m      |
| 8     | B8A  | 865 nm      | 20 m      |
| 9     | B9   | 945 nm      | 60 m      |
| 10    | B10  | 1375 nm     | 60 m      |
| 11    | B11  | 1610 nm     | 20 m      |
| 12    | B12  | 2190 nm     | 20 m      |

All bands are resampled to 10 m using bilinear interpolation and aligned
to the EPSG:3857 grid before normalisation.

### Normalisation

Z-score normalisation, per band:

```
normalised[band] = (raw[band] - mean[band]) / std[band]
```

The `mean` and `std` arrays are stored in `band_stats.npz` (computed from
the training set).  The UDF applies this automatically.

**Important:** The model was trained on L1C TOA reflectance in raw DN
(digital number) units — NOT surface reflectance, NOT scaled to [0,1].
Typical S2 L1C values are in range [0, 10000+].  If your OpenEO
collection returns values in different units, the `band_stats.npz` will
need to be recomputed or a scaling factor applied.

## Output Specification

| Property  | Value                         |
|-----------|-------------------------------|
| Shape     | (256, 256)                    |
| Dtype     | uint8 (binary) or float32 (probability) |
| Values    | 0 = background, 1 = solar PV |
| Threshold | 0.80 (configurable)          |

The model outputs sigmoid probabilities in [0, 1].  The threshold of
0.80 was determined by sweeping thresholds 0.10–0.90 on the held-out
test set and selecting the threshold that maximises pixel-level Dice.

### Threshold Sensitivity

| Threshold | Dice   | IoU    | Precision | Recall |
|-----------|--------|--------|-----------|--------|
| 0.50      | 0.8831 | 0.7907 | 0.876     | 0.890  |
| 0.65      | 0.8849 | 0.7935 | 0.891     | 0.879  |
| 0.80      | 0.8853 | 0.7942 | 0.903     | 0.868  |
| 0.90      | 0.8831 | 0.7907 | 0.914     | 0.854  |

The curve is flat between 0.50–0.85, so threshold choice trades off
precision vs recall without significantly affecting Dice.

## Tiling for Large AOIs

The model operates on 256×256 chips.  For areas larger than a single chip
(2560 m), tile the AOI into a regular grid aligned to:

- Origin: (0, 0) in EPSG:3857
- Cell size: 2560 m (= 256 px × 10 m/px)

The grid formula:

```
chip_xmin = floor(aoi_xmin / 2560) * 2560
chip_ymin = floor(aoi_ymin / 2560) * 2560
```

In OpenEO, use `apply_neighborhood()` with size 256×256 px and overlap 0.

## Future Model Updates

When a new model is trained:

1. The architecture parameters (backbone, decoder_filters, etc.) are read
   from `model_registry.yaml` — not hardcoded in the UDF.
2. Update `model_registry.yaml` with new weights URL and threshold.
3. The UDF code does NOT need to change.

The one constraint: the new model must accept 13-band Sentinel-2 L1C
input at 256×256 pixels.  If input shape changes, the tiling logic in
the process graph must also be updated.
