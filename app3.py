import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fruit Mould Classifier (3-Model Ensemble)",
    page_icon="🍎",
    layout="centered"
)

# Food Classification Models (Multiclass)
DENSENET_FOOD_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/densenet121_multiclass_final.h5"
EFFICIENTNET_FOOD_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/efficientnetb0_run5_best_final.h5"
MOBILENET_FOOD_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/mobilenetv2_run5_best_final.h5"

# Mould Detection Models (Binary)
DENSENET_MOLD_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/densenet121_binary_final.h5"
EFFICIENTNET_MOLD_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/efficientnetb0_binary_run5_best_final.h5"
MOBILENET_MOLD_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/mobilenetv2_binary_run5_best_final.h5"

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
        with st.spinner(f"Downloading {filename}..."):
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

# ── Load all 6 ensemble models ────────────────────────────────────────────────
@st.cache_resource
def load_ensemble_models():
    """Load all 3 food models and 3 mold models"""
    
    with st.spinner("Downloading ensemble models (6 models, ~200MB total)..."):
        # Food models
        densenet_food_path = download_model(DENSENET_FOOD_URL, "densenet_food.h5")
        efficientnet_food_path = download_model(EFFICIENTNET_FOOD_URL, "efficientnet_food.h5")
        mobilenet_food_path = download_model(MOBILENET_FOOD_URL, "mobilenet_food.h5")
        
        # Mold models
        densenet_mold_path = download_model(DENSENET_MOLD_URL, "densenet_mold.h5")
        efficientnet_mold_path = download_model(EFFICIENTNET_MOLD_URL, "efficientnet_mold.h5")
        mobilenet_mold_path = download_model(MOBILENET_MOLD_URL, "mobilenet_mold.h5")
    
    with st.spinner("Loading models into memory..."):
        # Load food models
        food_models = {
            'densenet': load_model(densenet_food_path, compile=False),
            'efficientnet': load_model(efficientnet_food_path, compile=False),
            'mobilenet': load_model(mobilenet_food_path, compile=False)
        }
        
        # Load mold models
        mold_models = {
            'densenet': load_model(densenet_mold_path, compile=False),
            'efficientnet': load_model(efficientnet_mold_path, compile=False),
            'mobilenet': load_model(mobilenet_mold_path, compile=False)
        }
    
    st.sidebar.success("✅ All 6 ensemble models loaded!")
    return food_models, mold_models

# ── Preprocessing for different model architectures ───────────────────────────
def preprocess_for_model(image, model_type):
    """Preprocess image according to model's requirements"""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    
    if model_type == 'densenet':
        from tensorflow.keras.applications.densenet import preprocess_input
    elif model_type == 'efficientnet':
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif model_type == 'mobilenet':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    return preprocess_input(arr)

# ── Ensemble prediction ───────────────────────────────────────────────────────
def ensemble_predict(food_models, mold_models, image):
    """Get predictions from all models and combine with weights"""
    
    # Store predictions
    food_predictions = []
    food_confidences = []
    mold_predictions = []
    
    # Weight for each model (DenseNet: 0.4, EfficientNet: 0.35, MobileNet: 0.25)
    # Based on typical performance
    weights = {'densenet': 0.4, 'efficientnet': 0.35, 'mobilenet': 0.25}
    
    # Get food predictions from each model
    for model_name, model in food_models.items():
        proc_image = preprocess_for_model(image, model_name)
        pred = model.predict(proc_image, verbose=0)
        food_predictions.append(pred[0])
        
        # Get confidence for the predicted class
        pred_class = np.argmax(pred[0])
        confidence = pred[0][pred_class]
        food_confidences.append(confidence * weights[model_name])
    
    # Weighted average for food
    weighted_food_pred = np.average(food_predictions, axis=0, weights=[weights[n] for n in food_models.keys()])
    final_food_class = np.argmax(weighted_food_pred)
    final_food_confidence = weighted_food_pred[final_food_class] * 100
    
    # Get mold predictions
    mold_results = []
    for model_name, model in mold_models.items():
        proc_image = preprocess_for_model(image, model_name)
        pred = model.predict(proc_image, verbose=0)[0][0]
        
        # For mold: pred < 0.5 means mold, > 0.5 means no mold
        is_mold = pred < 0.5
        confidence = (1 - pred) * 100 if is_mold else pred * 100
        mold_results.append({
            'model': model_name,
            'raw_pred': pred,
            'is_mold': is_mold,
            'confidence': confidence,
            'vote': 1 if is_mold else 0
        })
    
    # Weighted voting for mold
    mold_vote = 0
    for result in mold_results:
        mold_vote += result['vote'] * weights[result['model']]
    
    has_mold = mold_vote >= 0.5
    mold_confidence = sum([r['confidence'] * weights[r['model']] for r in mold_results if r['is_mold'] == has_mold])
    
    # Calculate agreement percentage
    mold_agreement = mold_vote * 100 if has_mold else (1 - mold_vote) * 100
    
    return {
        'food_class': final_food_class,
        'food_confidence': final_food_confidence,
        'food_model_predictions': food_predictions,
        'has_mold': has_mold,
        'mold_confidence': mold_confidence,
        'mold_agreement': mold_agreement,
        'mold_details': mold_results
    }

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🍎 Fruit & Mould Classifier")
st.markdown("**3-Model Ensemble** | DenseNet121 + EfficientNetB0 + MobileNetV2")

st.divider()

with st.sidebar:
    st.markdown("### 🔍 Ensemble Status")
    st.markdown("**Models:**")
    st.markdown("- DenseNet121 (40% weight)")
    st.markdown("- EfficientNetB0 (35% weight)")
    st.markdown("- MobileNetV2 (25% weight)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("Running ensemble inference..."):
            try:
                food_models, mold_models = load_ensemble_models()
                results = ensemble_predict(food_models, mold_models, image)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        
        # ── Stage 1 — Fruit type ──────────────────────────────────────────────
        st.subheader("Stage 1 — Food Type (Ensemble)")
        food_label = FRUIT_CLASSES[results['food_class']]
        st.success(f"**{food_label}**")
        st.caption(f"Confidence: {results['food_confidence']:.1f}%")
        
        with st.expander("View individual model predictions"):
            for i, (model_name, pred) in enumerate(zip(food_models.keys(), results['food_model_predictions'])):
                pred_class = FRUIT_CLASSES[np.argmax(pred)]
                pred_conf = pred[np.argmax(pred)] * 100
                st.text(f"{model_name.title()}: {pred_class} ({pred_conf:.1f}%)")
        
        st.divider()
        
        # ── Stage 2 — Mould detection ─────────────────────────────────────────
        st.subheader("Stage 2 — Mould Detection (Ensemble)")
        
        if results['has_mold']:
            st.error(f"⚠️ **MOLD DETECTED**")
        else:
            st.success(f"✅ **NO MOLD DETECTED**")
        
        st.caption(f"Confidence: {results['mold_confidence']:.1f}%")
        
        # Agreement indicator
        if results['mold_agreement'] >= 80:
            st.info(f"🎯 Strong ensemble agreement: {results['mold_agreement']:.0f}%")
        elif results['mold_agreement'] >= 60:
            st.warning(f"📊 Moderate ensemble agreement: {results['mold_agreement']:.0f}%")
        else:
            st.error(f"⚡ Low agreement ({results['mold_agreement']:.0f}%) - Manual inspection recommended")
        
        with st.expander("View individual model votes"):
            for r in results['mold_details']:
                status = "⚠️ MOLD" if r['is_mold'] else "✅ NO MOLD"
                st.text(f"{r['model'].title()}: {status} ({r['confidence']:.1f}%)")
    
    st.divider()
    st.subheader("Summary")
    
    if results['has_mold']:
        st.error(f"🔴 **Ensemble Result**: {food_label} with **mould detected**")
    else:
        st.success(f"🟢 **Ensemble Result**: {food_label} with **no mould detected**")
else:
    st.info("Please upload an image to begin")
