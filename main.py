"""
Plant Disease Detection - Streamlit Web Application
Upload a plant leaf image to detect if it's healthy or diseased.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# Use tf.keras for compatibility
keras = tf.keras

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        color: #2e7d32 !important;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(46, 125, 50, 0.4);
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #4a5568 !important;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-style: italic;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Upload area styling */
    .upload-area {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Result box styling */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        margin-top: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border: 2px solid #c8e6c9;
    }
    
    .result-box-healthy {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-color: #81c784;
    }
    
    .result-box-disease {
        background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
        border-color: #ef9a9a;
    }
    
    /* Prediction text */
    .prediction-text {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1b5e20;
        margin: 1rem 0;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-text-disease {
        color: #c62828;
    }
    
    /* Confidence text */
    .confidence-text {
        font-size: 1.3rem;
        color: #388e3c;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    
    .status-healthy {
        background: linear-gradient(135deg, #4caf50 0%, #66bb6a 100%);
        color: white;
    }
    
    .status-disease {
        background: linear-gradient(135deg, #f44336 0%, #ef5350 100%);
        color: white;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Info card */
    .info-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Image container */
    .image-container {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title with better styling
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 2rem 0; background: linear-gradient(135deg, #f0f7f0 0%, #e8f5e9 100%); border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
        <h1 class="main-header" style="color: #2e7d32 !important; -webkit-text-fill-color: #2e7d32 !important;">🌿 Plant Disease Detection</h1>
        <p class="sub-header" style="color: #4a5568 !important;">AI-Powered Plant Health Analysis System</p>
    </div>
""", unsafe_allow_html=True)

# Load class names
@st.cache_data
def load_class_names():
    """Load class names from file."""
    if os.path.exists('class_names.txt'):
        with open('class_names.txt', 'r') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines
            return names if names else ['Healthy', 'Disease A (Spots)', 'Disease B (Yellow Patches)', 'Disease C (Brown Edges)']
    else:
        # Default class names if file doesn't exist
        return ['Healthy', 'Disease A (Spots)', 'Disease B (Yellow Patches)', 'Disease C (Brown Edges)']

# Load model
@st.cache_resource
def load_model():
    """Load the trained CNN model."""
    if os.path.exists('model.h5'):
        try:
            model = keras.models.load_model('model.h5')
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    else:
        st.warning("⚠️ Model file (model.h5) not found. Please run train_model.py first.")
        return None

# Preprocess image
def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for model prediction."""
    # Resize image
    image = image.resize(target_size)
    # Convert to array
    img_array = np.array(image)
    # Normalize to [0, 1]
    if img_array.max() > 1:
        img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load model and class names
model = load_model()
class_names = load_class_names()

# Upload section with better styling
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload a plant leaf image for analysis",
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, JPEG, PNG. Upload a clear image of a plant leaf for best results."
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Create two columns for image and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if model is not None:
            # Preprocess and predict
            with st.spinner("🔍 Analyzing image with AI..."):
                try:
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predictions = model.predict(processed_image, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx]
                    
                    # Determine box class based on prediction
                    box_class = "result-box-healthy" if predicted_class_idx == 0 else "result-box-disease"
                    text_class = "" if predicted_class_idx == 0 else "prediction-text-disease"
                    
                    # Display results
                    st.markdown(f'<div class="result-box {box_class}">', unsafe_allow_html=True)
                    st.markdown("### 📊 Analysis Results")
                    
                    # Status badge
                    if predicted_class_idx == 0:
                        status_badge = '<span class="status-badge status-healthy">✅ HEALTHY</span>'
                        status_emoji = "✅"
                    else:
                        status_badge = '<span class="status-badge status-disease">⚠️ DISEASED</span>'
                        status_emoji = "⚠️"
                    
                    st.markdown(status_badge, unsafe_allow_html=True)
                    
                    # Display prediction
                    st.markdown(
                        f'<div class="prediction-text {text_class}">{status_emoji} <strong>{class_names[predicted_class_idx]}</strong></div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display confidence with progress bar
                    confidence_percent = confidence * 100
                    st.markdown(f'<p class="confidence-text">🎯 Confidence: <strong>{confidence_percent:.2f}%</strong></p>', unsafe_allow_html=True)
                    # Convert numpy float32 to Python float for Streamlit progress bar
                    st.progress(float(confidence))
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
        else:
            st.error("⚠️ Model not loaded. Please ensure model.h5 exists.")
    
    # Show detailed probabilities in a separate section
    if model is not None:
        try:
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            
            st.markdown("---")
            st.markdown("### 📈 Detailed Probability Analysis")
            
            # Ensure arrays have the same length
            num_classes = len(predictions[0])
            prob_values = [p * 100 for p in predictions[0]]
            display_names = class_names[:num_classes] if len(class_names) >= num_classes else class_names
            
            # Create a better visualization
            prob_data = {
                'Class': display_names,
                'Probability (%)': prob_values
            }
            prob_df = pd.DataFrame(prob_data)
            
            # Display as columns for better visualization
            cols = st.columns(len(display_names))
            for idx, (col, name, prob) in enumerate(zip(cols, display_names, prob_values)):
                with col:
                    # Color based on probability
                    if prob > 50:
                        color = "#4caf50" if idx == predicted_class_idx else "#81c784"
                    else:
                        color = "#bdbdbd"
                    
                    st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{name}</div>
                            <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{prob:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Bar chart
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.bar_chart(prob_df.set_index('Class'), height=300)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            pass

# Instructions with better styling
st.markdown("---")
with st.expander("ℹ️ How to Use This App", expanded=False):
    st.markdown("""
    <div class="info-card">
        <h4>📋 Step-by-Step Guide:</h4>
        <ol style="line-height: 2;">
            <li><strong>Upload an Image</strong>: Click the upload button above and select a clear image of a plant leaf</li>
            <li><strong>Wait for Analysis</strong>: Our AI model will automatically analyze the image (takes 2-3 seconds)</li>
            <li><strong>View Results</strong>: See the prediction, confidence score, and detailed probability breakdown</li>
        </ol>
        
        <h4>💡 Tips for Best Results:</h4>
        <ul style="line-height: 2;">
            <li>Use clear, well-lit images</li>
            <li>Ensure the leaf is in focus</li>
            <li>Upload images in JPG, JPEG, or PNG format</li>
            <li>For best accuracy, use images similar to training data</li>
        </ul>
        
        <p style="margin-top: 1rem; padding: 1rem; background: white; border-radius: 5px;">
            <strong>⚠️ Note:</strong> This model was trained on synthetic data for demonstration purposes. 
            For production use, train with a real plant disease dataset like PlantVillage.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer with better styling
st.markdown("""
    <div class="footer">
        <p>🌿 <strong>Plant Disease Detection System</strong></p>
        <p>Powered by TensorFlow/Keras & Streamlit | AI-Powered Plant Health Analysis</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">© 2024 Plant Disease Detection App</p>
    </div>
""", unsafe_allow_html=True)

