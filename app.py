import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fruit Mould Classifier",
    page_icon="🍎",
    layout="centered"
)

# ── Hugging Face model URLs (USING .h5 FILES) ─────────────────────────────────
MULTICLASS_MODEL_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/densenet121_multiclass_final.h5"
BINARY_MODEL_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/densenet121_binary_final.h5"

# ── Class labels ──────────────────────────────────────────────────────────────
FRUIT_CLASSES = [
    'Blackberry', 'Blueberry', 'Carrots', 'Cheese',
    'Cream Cheese', 'Mixed Bread', 'Onion', 'Orange',
    'Raspberry', 'Toast', 'Tomatoes'
]

# ── Download model from Hugging Face ──────────────────────────────────────────
@st.cache_resource
def download_model(url, filename):
    """Download model from Hugging Face if not already downloaded"""
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename} (33 MB)..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(filename, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for data in response.iter_content(chunk_size=8192):
                        downloaded += len(data)
                        f.write(data)
                        progress = (downloaded / total_size) * 100
                        st.sidebar.text(f"Downloading: {progress:.1f}%")
    return filename

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load models from Hugging Face (downloads if not cached)"""
    
    try:
        # Download models from Hugging Face
        multiclass_path = download_model(MULTICLASS_MODEL_URL, "densenet121_multiclass_final.h5")
        binary_path = download_model(BINARY_MODEL_URL, "densenet121_binary_final.h5")
        
        # Load models
        multiclass_model = load_model(multiclass_path, compile=False)
        binary_model = load_model(binary_path, compile=False)
        
        # Verify output shapes
        mc_output_shape = multiclass_model.output_shape
        bin_output_shape = binary_model.output_shape
        
        if mc_output_shape[-1] != 11:
            raise ValueError(f"Expected 11 classes, got {mc_output_shape[-1]}")
        if bin_output_shape[-1] != 1:
            raise ValueError(f"Expected 1 output, got {bin_output_shape[-1]}")
        
        st.sidebar.success("✅ Models loaded successfully!")
        return multiclass_model, binary_model
        
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
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
    st.markdown("📦 Using .h5 model files for better compatibility")
    st.markdown("🔗 [Model Repository](https://huggingface.co/NdahTah/MoldTwoPhaseClassification)")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        
        with st.spinner("Loading models (first time may take 1-2 minutes)..."):
            try:
                multiclass_model, binary_model = load_models()
                st.sidebar.success("✅ Both models loaded successfully")
            except Exception as e:
                st.error(f"Model loading failed: {e}")
                st.stop()
        
        arr = preprocess(image)
        
        # ── Stage 1 — Fruit type ──────────────────────────────────────────────
        st.subheader("Stage 1 — Food Type")
        
        with st.spinner("Identifying food type..."):
            mc_preds = multiclass_model.predict(arr, verbose=0)
            mc_idx = int(np.argmax(mc_preds[0]))
            mc_label = FRUIT_CLASSES[mc_idx]
            mc_conf = float(mc_preds[0][mc_idx]) * 100
        
        st.success(f"**{mc_label}**")
        st.caption(f"Confidence: {mc_conf:.1f}%")
        
        top3_idx = np.argsort(mc_preds[0])[::-1][:3]
        st.markdown("**Top 3 predictions:**")
        for idx in top3_idx:
            label = FRUIT_CLASSES[idx]
            conf = float(mc_preds[0][idx]) * 100
            st.progress(int(conf), text=f"{label}: {conf:.1f}%")
        
        st.divider()
        
        # ── Stage 2 — Mould detection ─────────────────────────────────────────
        st.subheader("Stage 2 — Mould Detection")
        
        with st.spinner("Checking for mould..."):
            bin_pred = binary_model.predict(arr, verbose=0)
            bin_prob = float(bin_pred[0][0])
            
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
            "Models hosted on Hugging Face as .h5 files."
        )
