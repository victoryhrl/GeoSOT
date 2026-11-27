# GeoSOT: Geometry-Aware Cross-View Geo-Localization via Phase Priors and Spatial Optimal Transport

This repository contains the official PyTorch implementation for the paper: **"GeoSOT: Geometry-Aware Cross-View Geo-Localization via Phase Priors and Spatial Optimal Transport"**.

## Abstract

Cross-view geo-localization requires representations that match ground images to satellite references despite large changes in viewpoint and appearance. An image-level retrieval objective does not specify which regions support a match. GeoSOT learns cross-view representations through phase-guided local correspondence. Fourier phase provides structural cues alongside RGB appearance in the regional matching costs. Differentiable spatial optimal transport uses these costs to assign regions jointly, with a learnable dustbin for unmatched content. The aligned features provide a training signal for the retrieval encoders without requiring patch-level correspondence labels. On VIGOR, the contextual and aligned features are used to predict a Gaussian offset distribution. Its mean estimates the query position within a candidate reference, and its covariance helps re-rank candidates after global retrieval. We evaluate GeoSOT on CVUSA, CVACT, and VIGOR. It reaches 98.92% R@1 on CVUSA. On VIGOR, covariance-based refinement raises R@1 from 80.96% to 82.39% on Same-Area and from 66.34% to 68.30% on Cross-Area with the same ten candidates per query.

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.10+ installed. The required packages can be installed via:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `torch >= 1.13.1`
- `timm >= 0.8.8`
- `scipy >= 1.10.0`
- `albumentations >= 1.3.0`
- `transformers >= 4.27.0`

---

## 📂 Dataset Preparation

1. **Download Datasets:**
Download CVUSA, CVACT, and VIGOR datasets and organize them into the `data` directory according to the following structure:

```text
├─ GeoSOT
  ├── pretrained/     # Place pretrained weights here
  ├── data/
    ├── VIGOR/ 
    ├── CVUSA/ 
    └── CVACT/
  └── geosot/     # Core library components
```

1. **Distance Pre-computation:**
Before training, pre-compute the GPS-based distances using the provided scripts:

```bash
python calc_distance_cvusa.py
python calc_distance_cvact.py
python calc_distance_vigor.py
```

---

## 🚀 Training

To train the GeoSOT models from scratch, execute the corresponding training scripts. The dual-stream encoders are initialized with ImageNet-pretrained ConvNeXt-B weights.

### CVUSA

```bash
python train_cvusa.py
```

### CVACT

```bash
python train_cvact.py
```

### VIGOR (Same-Area and Cross-Area)

```bash
python train_vigor.py
```

---

## 📊 Evaluation

We provide dedicated evaluation scripts to evaluate models on respective datasets and splits. Ensure your model weights are in the appropriate save directory before running.

### CVUSA Evaluation

```bash
python eval_cvusa.py
```

### CVACT Evaluation

```bash
python eval_cvact.py
```

### VIGOR Evaluation

Evaluate on the Same-Area and Cross-Area splits separately:

```bash
python eval_vigor_same.py
python eval_vigor_cross.py
```

*Note: Evaluation includes coarse global retrieval and the uncertainty-guided geometric re-ranking phase.

---

## Acknowledgments

This repository builds upon the foundational code of Sample4Geo and incorporates advanced spatial optimal transport matching techniques for cross-view tasks.
