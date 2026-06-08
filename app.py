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

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    multiclass_model = tf.keras.models.load_model(MULTICLASS_MODEL_PATH)
    binary_model     = tf.keras.models.load_model(BINARY_MODEL_PATH)
    return multiclass_model, binary_model

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """
    Resize to 224x224, convert to RGB, apply DenseNet121 preprocess_input.
    Same preprocessing used during training for both models.
    """
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
            multiclass_model, binary_model = load_models()

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

        # Top 3 predictions
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
            bin_prob  = float(bin_pred[0][0])   # sigmoid output

            # class_indices: Mold=0, No mold=1
            # sigmoid output close to 0 → Mold, close to 1 → No mold
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

        # Probability bar
        st.markdown("**Mould probability:**")
        mold_probability = (1 - bin_prob) * 100
        st.progress(
            int(mold_probability),
            text=f"Mold: {mold_probability:.1f}%  |  "
                 f"No mold: {bin_prob * 100:.1f}%"
        )

    st.divider()

    # ── Combined result ───────────────────────────────────────────────────────
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
