import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fruit Mould Classifier",
    page_icon="🍎",
    layout="centered"
)

# ── Model paths (relative to repo root) ───────────────────────────────────────
MULTICLASS_MODEL_PATH = "densenet121_run1_best.keras"
BINARY_MODEL_PATH     = "densenet121_binary_run1_best.keras"

# ── Class labels ──────────────────────────────────────────────────────────────
FRUIT_CLASSES = [
    'Blackberry', 'Blueberry', 'Carrots', 'Cheese',
    'Cream Cheese', 'Mixed Bread', 'Onion', 'Orange',
    'Raspberry', 'Toast', 'Tomatoes'
]

MOLD_CLASSES = ['Mold', 'No mold']

# ── Model verification ────────────────────────────────────────────────────────
def verify_models():
    issues = []
    for path, label in [(MULTICLASS_MODEL_PATH, "Multiclass"), (BINARY_MODEL_PATH, "Binary")]:
        if not os.path.exists(path):
            issues.append(f"❌ {label} model file not found: `{path}`")
        else:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            issues.append(f"✅ {label} model found: `{path}` ({size_mb:.1f} MB)")
    return issues

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    multiclass_model = tf.keras.models.load_model(MULTICLASS_MODEL_PATH)
    binary_model     = tf.keras.models.load_model(BINARY_MODEL_PATH)

    # Verify output shapes match expectations
    # Multiclass: should output 11 classes
    # Binary: should output 1 (sigmoid)
    mc_output_shape  = multiclass_model.output_shape
    bin_output_shape = binary_model.output_shape

    if mc_output_shape[-1] != 11:
        raise ValueError(
            f"Multiclass model output shape is {mc_output_shape} — "
            f"expected 11 classes. Wrong model file may be loaded."
        )
    if bin_output_shape[-1] != 1:
        raise ValueError(
            f"Binary model output shape is {bin_output_shape} — "
            f"expected 1 output (sigmoid). Wrong model file may be loaded."
        )

    return multiclass_model, binary_model

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.densenet import preprocess_input
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr   = np.array(image, dtype=np.float32)
    arr   = np.expand_dims(arr, axis=0)
    arr   = preprocess_input(arr)
    return arr

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🍎 Fruit & Mould Classifier")
st.markdown(
    "Upload an image of a fruit or food item. "
    "The app will first identify the food type, "
    "then determine whether mould is present."
)

st.divider()

# ── Model status sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Model Status")
    for msg in verify_models():
        st.markdown(msg)

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:

        with st.spinner("Loading models..."):
            try:
                multiclass_model, binary_model = load_models()
                st.sidebar.success("✅ Both models loaded successfully")
            except ValueError as e:
                st.sidebar.error(f"⚠️ Model mismatch: {e}")
                st.error(f"Model loading failed: {e}")
                st.stop()
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load models: {e}")
                st.error(f"Model loading failed: {e}")
                st.stop()

        arr = preprocess(image)

        # ── Stage 1 — Fruit type ──────────────────────────────────────────────
        st.subheader("Stage 1 — Food Type")

        with st.spinner("Identifying food type..."):
            mc_preds  = multiclass_model.predict(arr, verbose=0)
            mc_idx    = int(np.argmax(mc_preds[0]))
            mc_label  = FRUIT_CLASSES[mc_idx]
            mc_conf   = float(mc_preds[0][mc_idx]) * 100

        st.success(f"**{mc_label}**")
        st.caption(f"Confidence: {mc_conf:.1f}%")

        top3_idx  = np.argsort(mc_preds[0])[::-1][:3]
        st.markdown("**Top 3 predictions:**")
        for idx in top3_idx:
            label = FRUIT_CLASSES[idx]
            conf  = float(mc_preds[0][idx]) * 100
            st.progress(int(conf), text=f"{label}: {conf:.1f}%")

        st.divider()

        # ── Stage 2 — Mould detection ─────────────────────────────────────────
        st.subheader("Stage 2 — Mould Detection")

        with st.spinner("Checking for mould..."):
            bin_pred  = binary_model.predict(arr, verbose=0)
            bin_prob  = float(bin_pred[0][0])

            if bin_prob < 0.5:
                bin_label = "Mold"
                mold_conf = (1 - bin_prob) * 100
            else:
                bin_label = "No mold"
                mold_conf = bin_prob * 100

        if bin_label == "Mold":
            st.error(f"⚠️ **{bin_label} detected**")
        else:
            st.success(f"✅ **{bin_label}**")

        st.caption(f"Confidence: {mold_conf:.1f}%")

        st.markdown("**Mould probability:**")
        mold_probability = (1 - bin_prob) * 100
        st.progress(
            int(mold_probability),
            text=f"Mold: {mold_probability:.1f}%  |  No mold: {bin_prob * 100:.1f}%"
        )

    st.divider()

    st.subheader("Summary")

    if bin_label == "Mold":
        st.error(
            f"🔴 This image appears to be **{mc_label}** "
            f"and **mould has been detected**. "
            f"It is not safe for consumption."
        )
    else:
        st.success(
            f"🟢 This image appears to be **{mc_label}** "
            f"and **no mould was detected**. "
            f"It appears safe for consumption."
        )

else:
    st.info("Please upload an image to begin.")

    st.divider()
    st.markdown("### How it works")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "**Stage 1 — Food Type**\n\n"
            "DenseNet121 trained on 11 food classes:\n\n"
            + "\n".join([f"- {c}" for c in FRUIT_CLASSES])
        )

    with col2:
        st.markdown(
            "**Stage 2 — Mould Detection**\n\n"
            "DenseNet121 binary classifier:\n\n"
            "- 🔴 Mold detected\n"
            "- 🟢 No mold detected\n\n"
            "Both models trained on a dataset of ~4,300 images "
            "split 70/15/15 for train, validation and test."
        )
