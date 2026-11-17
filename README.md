# 🩺 Breast Ultrasound Classifier (BUSI)

Educational deep learning project. **Not a diagnostic tool.**

**Live demo:** <PASTE YOUR STREAMLIT LINK HERE>

## Overview
Classifies breast ultrasound images as **benign / malignant / normal** using transfer learning
(EfficientNetB0, TensorFlow/Keras), with Grad-CAM explanations in a Streamlit dashboard.

## Dataset
- BUSI (Breast Ultrasound Images), 780 images, 3 classes
- Removed 1 exact-duplicate pair carrying conflicting labels (SHA-256)
- 778 images, split 70/15/15 (stratified) into train/validation/test
- Test set was locked and evaluated once

## Method
- EfficientNetB0 pretrained on ImageNet, 224x224 input
- Stage 1: train the classification head with a frozen backbone
- Stage 2: fine-tune the top ~80 layers at learning rate 1e-4 (BatchNorm layers frozen)
- Augmentation: horizontal flip, small rotation, zoom
- Class weights to handle imbalance
- Malignant decision threshold (0.4) chosen on the validation set only

## Results (test set, evaluated once)
| Metric | Value |
|---|---|
| Accuracy | __ |
| Malignant recall | __ |
| Malignant precision | __ |
| Macro ROC-AUC | __ |

Confusion matrix: `<add image or table>`

## Explainability
Grad-CAM examples: `<add images>`

Observation: `<one sentence on whether heatmaps focus on the lesion or on artifacts>`

## Limitations
- Single public dataset from one source; generalization to other scanners or hospitals is untested
- BUSI has no patient IDs, so a true patient-level split was not possible
- Small test set (~31 malignant cases), so metrics have wide uncertainty
- Educational project; not validated for clinical use

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – Streamlit dashboard (prediction, Grad-CAM, performance tab)
- `best_stage2_v2.keras` – trained model
- `metrics.json` – test metrics shown in the dashboard
