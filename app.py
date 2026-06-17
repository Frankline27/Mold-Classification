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
    page_title="Dementia Classifier",
    page_icon="🧠",
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
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
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
        border-left: 5px solid #764ba2;
    }
    
    /* Confidence bar styling */
    .confidence-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        padding: 0.3rem;
        color: white;
        text-align: center;
        font-weight: bold;
        transition: width 0.5s ease;
    }
    
    /* Animated button */
    .stButton > button {
        background: linear-gradient(120deg, #667eea, #764ba2);
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
        border: 2px dashed #764ba2;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(118, 75, 162, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
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
        background: linear-gradient(90deg, #667eea, #764ba2);
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
        border-top-color: #764ba2 !important;
    }
    
    /* Risk indicator */
    .risk-high {
        background: #fee;
        border-left: 5px solid #e74c3c;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .risk-low {
        background: #e8f8f5;
        border-left: 5px solid #27ae60;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .risk-moderate {
        background: #fef9e7;
        border-left: 5px solid #f39c12;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header with animation ─────────────────────────────────────────────────────
st.markdown('<p class="gradient-text" style="font-size: 3rem; text-align: center;">🧠 Dementia Classifier</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Brain Health Assessment</p>', unsafe_allow_html=True)

# Create two columns for metrics
col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

with col_metrics1:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Model</h3>
        <p style="font-size: 1.5rem; font-weight: bold;">DenseNet121</p>
        <p>Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)

with col_metrics2:
    st.markdown("""
    <div class="metric-card">
        <h3>🧠 Classes</h3>
        <p style="font-size: 1.5rem; font-weight: bold;">2 Categories</p>
        <p>Demented / Non-Demented</p>
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
MODEL_URL = "https://huggingface.co/NdahTah/DementiaClassifier/resolve/main/densenet121_dementia_final.h5"

# ── Class labels with descriptions ────────────────────────────────────────────
CLASS_INFO = {
    'Demented': {
        'emoji': '⚠️',
        'description': 'Shows signs consistent with dementia',
        'color': '#e74c3c',
        'risk_level': 'High',
        'recommendation': 'Please consult a healthcare professional for proper diagnosis and care.',
        'symptoms': [
            'Memory loss affecting daily activities',
            'Difficulty planning or solving problems',
            'Confusion with time or place',
            'Trouble understanding visual images',
            'Problems with speaking or writing'
        ]
    },
    'Non-Demented': {
        'emoji': '✅',
        'description': 'Shows no signs consistent with dementia',
        'color': '#27ae60',
        'risk_level': 'Low',
        'recommendation': 'Continue maintaining a healthy lifestyle and regular check-ups.',
        'symptoms': [
            'Normal cognitive function',
            'No significant memory issues',
            'Ability to plan and solve problems',
            'Good spatial awareness',
            'Effective communication skills'
        ]
    }
}

CLASS_NAMES = list(CLASS_INFO.keys())

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

# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    try:
        model_path = download_model(MODEL_URL, "densenet121_dementia_final.h5")
        model = load_model(model_path, compile=False)
        st.sidebar.success("✅ Model ready!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
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
    - 🧠 **Analyze** brain health from MRI scans
    - 📊 **Detect** potential dementia indicators
    - 💡 **Get** initial health insights
    - 🏥 **Understand** next steps
    
    ---
    
    ### 📊 Model Performance
    - **Accuracy:** 85%+
    - **Training Images:** ~6,400 MRI scans
    - **Framework:** TensorFlow 2.15
    - **Model:** DenseNet121
    
    ---
    
    ### 💡 Important Notes
    - Use clear MRI images
    - Ensure proper orientation
    - This is a screening tool only
    
    ---
    
    ### 🏥 Medical Disclaimer
    This is an AI-powered screening tool, not a diagnostic device. Always consult healthcare professionals for proper diagnosis.
    """)
    
    st.markdown(f"<p style='text-align: center; font-size: 0.8rem;'>🕐 Last analyzed: {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

# ── Main upload area ─────────────────────────────────────────────────────────
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "**📸 Upload MRI Scan Image**",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload clear brain MRI scans for best results"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create two columns with better proportion
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### 🧠 MRI Scan")
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)
        
        # Image quality indicators
        img_size = uploaded_file.size / 1024
        st.caption(f"📏 Size: {img_size:.1f} KB | 📐 Dimensions: {image.size[0]}x{image.size[1]}")
    
    with col2:
        with st.spinner("🧠 Analyzing brain scan..."):
            model = load_model_cached()
            arr = preprocess(image)
            
            # Predictions
            preds = model.predict(arr, verbose=0)
            pred_idx = int(np.argmax(preds[0]))
            predicted_class = CLASS_NAMES[pred_idx]
            confidence = float(preds[0][pred_idx]) * 100
            
            # Get class information
            class_info = CLASS_INFO[predicted_class]
        
        # Display results with animations
        st.markdown("## 🔍 Analysis Results")
        
        # Classification Card
        st.markdown("### 🧠 Classification")
        st.markdown(f"""
        <div class="prediction-box" style="border-left-color: {class_info['color']};">
            <h2 style="margin: 0; color: {class_info['color']};">{class_info['emoji']} {predicted_class}</h2>
            <p style="margin: 0.5rem 0; color: #666;">{class_info['description']}</p>
            <p style="margin: 0.5rem 0; color: #666;">Confidence</p>
            <div class="confidence-bar" style="width: {confidence}%; background: linear-gradient(90deg, {class_info['color']}, #764ba2);">
                {confidence:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability distribution
        st.markdown("### 📊 Probability Distribution")
        col_prob1, col_prob2 = st.columns(2)
        
        with col_prob1:
            demented_prob = float(preds[0][0]) * 100
            st.metric(
                label="⚠️ Demented",
                value=f"{demented_prob:.1f}%",
                delta=None
            )
        
        with col_prob2:
            non_demented_prob = float(preds[0][1]) * 100
            st.metric(
                label="✅ Non-Demented",
                value=f"{non_demented_prob:.1f}%",
                delta=None
            )
        
        # Risk Assessment
        st.markdown("---")
        st.markdown("### 🏥 Risk Assessment")
        
        if predicted_class == 'Demented':
            st.markdown(f"""
            <div class="risk-high">
                <h3 style="color: #e74c3c;">⚠️ HIGH RISK</h3>
                <p><strong>Assessment:</strong> The analysis suggests patterns consistent with dementia.</p>
                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                <p><strong>Recommendation:</strong> {class_info['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h3 style="color: #27ae60;">✅ LOW RISK</h3>
                <p><strong>Assessment:</strong> The analysis shows no significant signs of dementia.</p>
                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                <p><strong>Recommendation:</strong> {class_info['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Symptoms/Indicators
        with st.expander("📋 Detailed Indicators"):
            st.markdown(f"#### {class_info['emoji']} {predicted_class} - Key Indicators")
            for symptom in class_info['symptoms']:
                st.markdown(f"- {symptom}")
        
        # Confidence gauge
        st.markdown("### 📈 Confidence Gauge")
        st.progress(confidence / 100)
        
        if confidence < 70:
            st.warning("💡 **Note:** Lower confidence predictions may occur with unclear images. Consider rescanning or consulting a specialist.")
        
        # Additional Insights
        st.markdown("---")
        st.markdown("### 🏥 Next Steps")
        
        if predicted_class == 'Demented':
            st.markdown("""
            **Based on this screening, we recommend:**
            1. 👨‍⚕️ **Schedule an appointment** with a neurologist or healthcare provider
            2. 📋 **Bring this report** to your consultation
            3. 🧠 **Consider additional testing** for comprehensive evaluation
            4. ❤️ **Seek support** from family and caregivers
            5. 📚 **Learn more** about dementia management and care options
            """)
        else:
            st.markdown("""
            **Based on this screening, we recommend:**
            1. 🏋️ **Maintain a healthy lifestyle** with regular exercise
            2. 🧩 **Keep mentally active** through learning and puzzles
            3. 🥗 **Eat a brain-healthy diet** rich in omega-3s
            4. 💤 **Get adequate sleep** (7-9 hours per night)
            5. 🏥 **Continue regular check-ups** for preventive care
            """)
    
    # Summary Card
    st.markdown("---")
    st.markdown("### 📋 Summary Report")
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    
    with col_sum1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧠 Classification</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: {class_info['color']};">{predicted_class}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sum2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Confidence</h4>
            <p style="font-size: 1.5rem; font-weight: bold;">{confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sum3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚠️ Risk Level</h4>
            <p style="font-size: 1.5rem; font-weight: bold; color: {class_info['color']};">{class_info['risk_level']}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome screen with instructions
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h1 style="font-size: 4rem;">🧠</h1>
        <h2>Ready to analyze brain health?</h2>
        <p style="color: #666;">Upload a brain MRI scan for AI-powered dementia screening</p>
        <p style="font-size: 0.9rem; color: #999;">Supported formats: JPG, PNG, JPEG, BMP, WEBP</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("""
        #### 🧠 **Dementia Screening**
        - Binary classification
        - Demented vs. Non-Demented
        - Confidence scoring
        - Risk assessment
        """)
    
    with col_f2:
        st.markdown("""
        #### 📊 **Detailed Analysis**
        - Probability distribution
        - Confidence metrics
        - Key indicators
        - Next steps guidance
        """)
    
    with col_f3:
        st.markdown("""
        #### 🏥 **Health Insights**
        - Risk level assessment
        - Recommendations
        - Follow-up guidance
        - Educational information
        """)
    
    st.markdown("---")
    st.markdown("### 📚 About Dementia")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.info("""
        **What is Dementia?**
        
        Dementia is not a single disease but a general term for a decline in mental ability severe enough to interfere with daily life. It affects:
        - Memory
        - Thinking skills
        - Social abilities
        - Daily functioning
        """)
    
    with col_info2:
        st.info("""
        **Early Detection Matters**
        
        Early detection of dementia can:
        - Allow for better treatment options
        - Enable advanced planning
        - Improve quality of life
        - Help manage symptoms effectively
        """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Powered by TensorFlow & Streamlit | Model trained on 6,400+ MRI scans</p>
    <p style="font-size: 0.7rem;">⚠️ This is an AI-powered screening tool, not a diagnostic device. Always consult healthcare professionals for proper diagnosis.</p>
</div>
""", unsafe_allow_html=True)
