
# 🩺 Breast Ultrasound Classifier (BUSI)

Educational deep learning project. **Not a diagnostic tool and not for clinical use.**

**Live demo:** <PASTE YOUR STREAMLIT LINK HERE>

## Overview
This project classifies breast ultrasound images as **benign**, **malignant** or **normal**
using transfer learning (EfficientNetB0, TensorFlow/Keras). A Streamlit dashboard lets you
upload an image, see the prediction with class probabilities, and view a Grad-CAM heatmap
of the regions that influenced the decision.

## Dataset
- BUSI (Breast Ultrasound Images): 780 images in 3 classes (437 benign, 210 malignant, 133 normal)
- Exact duplicates were detected with SHA-256 hashing. One pair appeared under two different
  labels (benign and malignant) and was removed, leaving **778 images**
- Segmentation masks were excluded; only the original images are used
- Stratified split: **544 train / 117 validation / 117 test**
- The test set was locked until all model decisions were finished and evaluated once

## Method
- **Backbone:** EfficientNetB0 pretrained on ImageNet, 224x224 input
- **Stage 1:** train the classification head with the backbone frozen
- **Stage 2:** fine-tune the top ~80 backbone layers at learning rate 1e-4 (BatchNorm layers kept frozen)
- **Augmentation:** horizontal flip, small rotation, zoom, contrast
- **Imbalance handling:** class
