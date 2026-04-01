# GeoSOT: Geometry-Aware Cross-View Geo-Localization via Phase Priors and Spatial Optimal Transport

This repository contains the official PyTorch implementation for the paper: **"GeoSOT: Geometry-Aware Cross-View Geo-Localization via Phase Priors and Spatial Optimal Transport"**.

## Abstract

Cross-view geo-localization aims to match ground-level imagery
against a geo-tagged satellite database, a task fundamentally
challenged by the extreme viewpoint discrepancy between ground
and overhead perspectives. While recent methods have achieved
impressive retrieval performance, they predominantly rely on
global pooling to produce compact descriptors, inherently
discarding the spatial geometric arrangements that are critical
under large viewpoint shifts. To address this, we propose GeoSOT,
a geometry-aware cross-view matching framework that integrates
frequency-domain structural priors with optimal transport-based
spatial correspondence modeling. GeoSOT introduces three tightly
integrated components: a Phase Extraction Module that provides
view-robust geometric priors via the Fourier phase spectrum;
a Geometry-Aware Spatial Optimal Transport module that explicitly
establishes region-level spatial correspondences while suppressing
unmatchable regions via a learnable dustbin mechanism; and a
Probabilistic Regression Head that models localization uncertainty
for precise coordinate prediction and uncertainty-guided geometric
re-ranking at inference time. Extensive experiments on CVUSA,
CVACT, and VIGOR demonstrate the effectiveness of GeoSOT, with
particularly pronounced gains on the challenging VIGOR Cross-Area
split, demonstrating that explicit spatial reasoning under large
viewpoint discrepancies is key to robust cross-view geo-localization.

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
