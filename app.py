import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras

st.set_page_config(page_title="Breast Ultrasound Classifier", page_icon="🩺", layout="wide")

CLASSES = ["benign", "malignant", "normal"]
MODEL_PATH = "best_stage2_v2.keras"
METRICS_PATH = "metrics.json"
DEFAULT_THRESH = 0.4


def show_table(df):
    """Render a DataFrame as plain HTML (avoids the pyarrow dependency)."""
    st.markdown(df.to_html(classes="table"), unsafe_allow_html=True)



@st.cache_resource(show_spinner="Loading model...")
def load():
    model = keras.models.load_model(MODEL_PATH, compile=False)

    # Find the nested EfficientNet by name/type instead of a fixed index
    base = next(
        l for l in model.layers
        if isinstance(l, keras.Model) and "efficientnet" in l.name.lower()
    )
    gap = next(l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D))
    dense = model.layers[-1]

    grad_model = keras.Model(
        base.inputs, [base.get_layer("top_activation").output, base.output]
    )
    return model, grad_model, gap, dense


def gradcam(grad_model, gap, dense, x, class_idx):
    with tf.GradientTape() as tape:
        conv, feat = grad_model(x)
        preds = dense(gap(feat))
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0]
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy()


if not os.path.exists(MODEL_PATH):
    st.error(f"Model file `{MODEL_PATH}` not found. Put it in the same folder as app.py.")
    st.stop()

model, grad_model, gap_layer, dense_layer = load()

st.title("🩺 Breast Ultrasound Classifier")
st.warning("Educational project only. Not a diagnostic tool and not for clinical use.")

tab1, tab2, tab3 = st.tabs(["Predict", "Model performance", "About"])

# ---------------- Predict ----------------
with tab1:
    thresh = st.sidebar.slider(
        "Malignant threshold", 0.1, 0.9, DEFAULT_THRESH, 0.05,
        help="Lower = catches more cancers but gives more false alarms.",
    )
    # Built-in samples so visitors can try the app without having an image
    sample_files = sorted(
        glob.glob("samples/*.png") + glob.glob("samples/*.jpg") + glob.glob("samples/*.jpeg")
    )
    file, truth = None, None
    if sample_files:
        mode = st.radio(
            "Image source", ["Try a sample image", "Upload your own"], horizontal=True
        )
    else:
        mode = "Upload your own"

    if mode == "Try a sample image":
        choice = st.selectbox(
            "Choose a sample ultrasound from the public BUSI dataset",
            sample_files,
            format_func=lambda f: f"Sample {sample_files.index(f) + 1}",
        )
        file = choice
        truth = os.path.basename(choice).split("_")[0]  # e.g. benign_1.png -> benign
    else:
        file = st.file_uploader("Upload a breast ultrasound image", type=["png", "jpg", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB").resize((224, 224))
        x = np.array(img, dtype="float32")[None]

        probs = model.predict(x, verbose=0)[0]
        pred = 1 if probs[1] >= thresh else int(np.argmax(probs))

        cam = gradcam(grad_model, gap_layer, dense_layer, tf.constant(x), pred)
        cam = np.array(Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)) / 255.0
        heat = matplotlib.colormaps["jet"](cam)[..., :3] * 255
        overlay = np.uint8(0.6 * np.array(img) + 0.4 * heat)

        c1, c2, c3 = st.columns(3)
        c1.image(img, caption="Input", use_container_width=True)
        c2.image(overlay, caption=f"Grad-CAM ({CLASSES[pred]})", use_container_width=True)
        with c3:
            st.metric("Prediction", CLASSES[pred].upper())
            st.metric("Malignant probability", f"{probs[1]:.1%}")
            for cname, p in zip(CLASSES, probs):
                st.write(f"{cname}: {p:.1%}")
                st.progress(float(p))
        if truth:
            st.info(f"Ground-truth label from the dataset: **{truth}**")
        st.caption("The heatmap shows the image regions that most influenced the prediction.")
    else:
        st.info("Choose a sample or upload an image to see the prediction and Grad-CAM heatmap.")

# ---------------- Performance ----------------
with tab2:
    if os.path.exists(METRICS_PATH):
        m = json.load(open(METRICS_PATH))
        rep = m["report"]
        a, b, c = st.columns(3)
        a.metric("Test accuracy", f"{rep['accuracy']:.1%}")
        b.metric("Malignant recall", f"{rep['malignant']['recall']:.1%}")
        c.metric("Malignant precision", f"{rep['malignant']['precision']:.1%}")

        st.subheader("Confusion matrix (rows = true, columns = predicted)")
        show_table(
            pd.DataFrame(
                m["confusion_matrix"],
                index=[f"true {c}" for c in CLASSES],
                columns=[f"pred {c}" for c in CLASSES],
            )
        )
        st.subheader("Per-class report")
        show_table(pd.DataFrame(rep).T.loc[CLASSES].round(3))
        st.caption(f"Evaluated once on a held-out test set. Malignant threshold: {m.get('threshold', DEFAULT_THRESH)}.")
    else:
        st.info("metrics.json not found. Export it from the notebook (see README).")

# ---------------- About ----------------
with tab3:
    st.markdown(
        """
**Model:** EfficientNetB0 pretrained on ImageNet, fine-tuned on the BUSI dataset (TensorFlow/Keras).

**Pipeline:** duplicate removal, stratified train/val/test split, two-stage transfer learning,
class weighting, threshold chosen on the validation set, Grad-CAM explanations.

**Limitations**
- Single public dataset from one source; other scanners or hospitals are untested
- BUSI has no patient IDs, so near-duplicates were grouped by image similarity
- Small test set, so metrics have wide uncertainty
- Educational project; not validated for clinical use
"""
    )
