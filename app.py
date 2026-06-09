import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input

# ── Page config with custom theme ───────────────────────────────────────────
st.set_page_config(
    page_title="Fruit Mould Classifier",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for better aesthetics ────────────────────────────────────────
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Gradient background for headers */
    .gradient-text {
        background: linear-gradient(120deg, #ff6b6b, #4ecdc4, #45b7d1);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        font-weight: bold;
        padding: 0.5rem 0;
    }
    
    /* Card-like containers */
    .custom-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: white;
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 5px solid #4ecdc4;
    }
    
    /* Confidence bar styling */
    .confidence-bar {
        background: linear-gradient(90deg, #4ecdc4, #45b7d1);
        border-radius: 10px;
        padding: 0.3rem;
        color: white;
        text-align: center;
        font-weight: bold;
        transition: width 0.5s ease;
    }
    
    /* Animated button */
    .stButton > button {
        background: linear-gradient(120deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        transition: transform 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #4ecdc4;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(78, 205, 196, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #ff6b6b;
        background: rgba(255, 107, 107, 0.05);
    }
    
    /* Metrics styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4ecdc4, #45b7d1);
        border-radius: 10px;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.8rem;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #4ecdc4 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header with animation ─────────────────────────────────────────────────────
st.markdown('<p class="gradient-text" style="font-size: 3rem; text-align: center;">🍎 Fruit & Mould Classifier</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced AI-Powered Food Safety Analysis</p>', unsafe_allow_html=True)

# Create two columns for metrics
col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

with col_metrics1:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Model</h3>
        <p style="font-size: 1.5rem; font-weight: bold;">DenseNet121</p>
        <p>State-of-the-art architecture</p>
    </div>
    """, unsafe_allow_html=True)

with col_metrics2:
    st.markdown("""
    <div class="metric-card">
        <h3>🍎 Classes</h3>
        <p style="font-size: 1.5rem; font-weight: bold;">11 Food Types</p>
        <p>+ Binary Mould Detection</p>
    </div>
    """, unsafe_allow_html=True)

with col_metrics3:
    st.markdown("""
    <div class="metric-card">
        <h3>⚡ Speed</h3>
        <p style="font-size: 1.5rem; font-weight: bold;">Quick Analysis</p>
        <p>~2-3 seconds per image</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Model URLs ───────────────────────────────────────────────────────────────
MULTICLASS_MODEL_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/densenet121_multiclass_final.h5"
BINARY_MODEL_URL = "https://huggingface.co/NdahTah/MoldTwoPhaseClassification/resolve/main/densenet121_binary_final.h5"

# ── Class labels with emojis ─────────────────────────────────────────────────
FRUIT_CLASSES_WITH_EMOJIS = {
    'Blackberry': '🖤 Blackberry',
    'Blueberry': '🫐 Blueberry',
    'Carrots': '🥕 Carrots',
    'Cheese': '🧀 Cheese',
    'Cream Cheese': '🍦 Cream Cheese',
    'Mixed Bread': '🍞 Mixed Bread',
    'Onion': '🧅 Onion',
    'Orange': '🍊 Orange',
    'Raspberry': '❤️ Raspberry',
    'Toast': '🍞 Toast',
    'Tomatoes': '🍅 Tomatoes'
}

FRUIT_CLASSES = list(FRUIT_CLASSES_WITH_EMOJIS.keys())

# ── Download model ───────────────────────────────────────────────────────────
@st.cache_resource
def download_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"📥 Downloading {filename}..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    return filename

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        multiclass_path = download_model(MULTICLASS_MODEL_URL, "densenet121_multiclass_final.h5")
        binary_path = download_model(BINARY_MODEL_URL, "densenet121_binary_final.h5")
        
        multiclass_model = load_model(multiclass_path, compile=False)
        binary_model = load_model(binary_path, compile=False)
        
        st.sidebar.success("✅ Models ready!")
        return multiclass_model, binary_model
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

# ── Sidebar with helpful info ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎨 About This App")
    st.markdown("""
    This AI-powered tool helps you:
    - 🍎 **Identify** fruits and food items
    - 🦠 **Detect** mould presence
    - ⚡ **Get quick** safety assessments
    
    ---
    
    ### 📊 Model Performance
    - **Accuracy:** 95%+
    - **Training Images:** ~4,300
    - **Framework:** TensorFlow 2.15
    - **Model:** DenseNet121
    
    ---
    
    ### 💡 Tips
    - Use clear, well-lit images
    - Center the food item
    - Avoid blurry photos
    
    ---
    
    ### 🔗 Links
    [📚 Documentation](https://github.com/Frankline27/Mold-Classification)
    [🤗 Model Repository](https://huggingface.co/NdahTah/MoldTwoPhaseClassification)
    """)
    
    st.markdown(f"<p style='text-align: center; font-size: 0.8rem;'>🕐 Last analyzed: {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

# ── Main upload area ─────────────────────────────────────────────────────────
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "**📸 Click or drag an image here**",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload clear images for best results"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create two columns with better proportion
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 📷 Your Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Image quality indicators
        img_size = uploaded_file.size / 1024
        st.caption(f"📏 Size: {img_size:.1f} KB | 📐 Dimensions: {image.size[0]}x{image.size[1]}")
    
    with col2:
        with st.spinner("🧠 Analyzing image..."):
            multiclass_model, binary_model = load_models()
            arr = preprocess(image)
            
            # Stage 1 Predictions
            mc_preds = multiclass_model.predict(arr, verbose=0)
            mc_idx = int(np.argmax(mc_preds[0]))
            mc_label = FRUIT_CLASSES[mc_idx]
            mc_conf = float(mc_preds[0][mc_idx]) * 100
            
            # Stage 2 Predictions
            bin_pred = binary_model.predict(arr, verbose=0)
            bin_prob = float(bin_pred[0][0])
            has_mold = bin_prob < 0.5
            mold_conf = (1 - bin_prob) * 100 if has_mold else bin_prob * 100
        
        # Display results with animations
        st.markdown("## 🔍 Analysis Results")
        
        # Food Type Card
        st.markdown("### 🍽️ Food Identification")
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="margin: 0; color: #2c3e50;">{FRUIT_CLASSES_WITH_EMOJIS[mc_label]}</h2>
            <p style="margin: 0.5rem 0; color: #666;">Confidence</p>
            <div class="confidence-bar" style="width: {mc_conf}%;">
                {mc_conf:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 3 predictions
        with st.expander("📊 View All Predictions"):
            top3_idx = np.argsort(mc_preds[0])[::-1][:3]
            for idx in top3_idx:
                label = FRUIT_CLASSES_WITH_EMOJIS[FRUIT_CLASSES[idx]]
                conf = float(mc_preds[0][idx]) * 100
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <p style="margin: 0;"><strong>{label}</strong></p>
                    <div class="confidence-bar" style="width: {conf}%; background: linear-gradient(90deg, #95a5a6, #7f8c8d);">
                        {conf:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Mould Detection Card
        st.markdown("### 🦠 Mould Analysis")
        
        if has_mold:
            st.markdown(f"""
            <div class="prediction-box" style="border-left-color: #e74c3c;">
                <h2 style="margin: 0; color: #e74c3c;">⚠️ MOLD DETECTED</h2>
                <p style="margin: 0.5rem 0; color: #666;">Risk Level: High</p>
                <div class="confidence-bar" style="width: {mold_conf}%; background: linear-gradient(90deg, #e74c3c, #c0392b);">
                    {mold_conf:.1f}% Confidence
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box" style="border-left-color: #27ae60;">
                <h2 style="margin: 0; color: #27ae60;">✅ NO MOLD DETECTED</h2>
                <p style="margin: 0.5rem 0; color: #666;">Risk Level: Low</p>
                <div class="confidence-bar" style="width: {mold_conf}%; background: linear-gradient(90deg, #27ae60, #2ecc71);">
                    {mold_conf:.1f}% Confidence
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability gauge
        mold_probability = (1 - bin_prob) * 100
        st.markdown("**Probability Distribution:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🦠 Mold Probability", f"{mold_probability:.1f}%")
        with col_b:
            st.metric("✅ No Mold Probability", f"{(1 - mold_probability/100)*100:.1f}%")
    
    # Summary and Recommendations
    st.markdown("---")
    st.markdown("### 📋 Safety Summary")
    
    if has_mold:
        st.error(f"""
        ⚠️ **SAFETY ADVISORY**
        
        This item has been identified as **{FRUIT_CLASSES_WITH_EMOJIS[mc_label]}** with mould detected.
        
        **Recommendation:** Do not consume. Dispose of the item properly.
        """)
    else:
        st.success(f"""
        ✅ **SAFETY CONFIRMED**
        
        This item has been identified as **{FRUIT_CLASSES_WITH_EMOJIS[mc_label]}** with no mould detected.
        
        **Recommendation:** Safe for consumption under normal conditions.
        """)
    
    # Additional tips
    if mold_conf < 70:
        st.info("💡 **Note:** Lower confidence predictions may occur with unusual angles or lighting. For critical decisions, manually inspect the item.")

else:
    # Welcome screen with instructions
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h1 style="font-size: 4rem;">📸</h1>
        <h2>Ready to analyze food items?</h2>
        <p style="color: #666;">Upload a clear image of your fruit or food item to get started</p>
        <p style="font-size: 0.9rem; color: #999;">Supported formats: JPG, PNG, JPEG, BMP, WEBP</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("""
        #### 🍎 **11 Food Classes**
        - Fruits (Blackberry, Blueberry, Orange, Raspberry, Tomatoes)
        - Vegetables (Carrots, Onion)
        - Dairy (Cheese, Cream Cheese)
        - Bakery (Mixed Bread, Toast)
        """)
    
    with col_f2:
        st.markdown("""
        #### 🦠 **Mould Detection**
        - Binary classification
        - Mould vs. No mould
        - Probability score
        - Safety recommendations
        """)
    
    with col_f3:
        st.markdown("""
        #### ⚡ **Key Benefits**
        - Instant analysis
        - High accuracy (95%+)
        - User-friendly interface
        - Mobile responsive
        """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Powered by TensorFlow & Streamlit | Model trained on 4,300+ images</p>
    <p style="font-size: 0.7rem;">⚠️ This is an AI-powered tool. For medical or safety-critical decisions, always consult experts.</p>
</div>
""", unsafe_allow_html=True)
