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

keras = tf.keras

# Disease descriptions for better user understanding
DISEASE_DESCRIPTIONS = {
    'Healthy': 'Your plant appears to be healthy! No signs of disease detected. Continue regular care and monitoring.',
    'Early Blight': 'Early blight detected. This fungal disease causes dark spots on leaves. Apply fungicide and remove affected leaves.',
    'Late Blight': 'Late blight detected. This is a serious fungal disease. Apply copper-based fungicide immediately and improve air circulation.',
    'Bacterial Spot': 'Bacterial spot detected. This bacterial disease causes small dark spots. Remove affected leaves and apply copper-based bactericide.'
}

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        color: #2e7d32 !important;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(46, 125, 50, 0.4);
    }
    
    .sub-header {
        text-align: center;
        color: #4a5568 !important;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-style: italic;
    }
    
    .upload-area {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
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
    
    .confidence-text {
        font-size: 1.3rem;
        color: #388e3c;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
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
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e0e0e0;
    }
    
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

st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 2rem 0; background: linear-gradient(135deg, #f0f7f0 0%, #e8f5e9 100%); border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
        <h1 class="main-header" style="color: #2e7d32 !important;">🌿 Plant Disease Detector — Check Your Crop Health</h1>
        <p class="sub-header" style="color: #4a5568 !important;">AI-Powered Plant Health Analysis System</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_data
def load_class_names():
    if os.path.exists('class_names.txt'):
        with open('class_names.txt', 'r') as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
            if names:
                return names
    return ['Healthy', 'Early Blight', 'Late Blight', 'Bacterial Spot']

@st.cache_resource
def load_model():
    # Check for model in model/ folder first, then root directory
    model_paths = ['model/plant_disease_model.h5', 'model/model.h5', 'model.h5']
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        st.warning("Model file not found. Please run train_model.py first to train the model.")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    img_array = np.array(image)
    
    if img_array.max() > 1:
        img_array = img_array.astype('float32') / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

model = load_model()
class_names = load_class_names()

st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a plant leaf image for analysis",
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, JPEG, PNG"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if model is not None:
            with st.spinner("Analyzing image..."):
                try:
                    processed_image = preprocess_image(image)
                    predictions = model.predict(processed_image, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx]
                    
                    box_class = "result-box-healthy" if predicted_class_idx == 0 else "result-box-disease"
                    text_class = "" if predicted_class_idx == 0 else "prediction-text-disease"
                    
                    st.markdown(f'<div class="result-box {box_class}">', unsafe_allow_html=True)
                    st.markdown("### Analysis Results")
                    
                    if predicted_class_idx == 0:
                        status_badge = '<span class="status-badge status-healthy">✅ HEALTHY</span>'
                        status_emoji = "✅"
                    else:
                        status_badge = '<span class="status-badge status-disease">⚠️ DISEASED</span>'
                        status_emoji = "⚠️"
                    
                    st.markdown(status_badge, unsafe_allow_html=True)
                    
                    st.markdown(
                        f'<div class="prediction-text {text_class}">{status_emoji} <strong>{class_names[predicted_class_idx]}</strong></div>',
                        unsafe_allow_html=True
                    )
                    
                    confidence_percent = confidence * 100
                    st.markdown(f'<p class="confidence-text">Confidence: <strong>{confidence_percent:.2f}%</strong></p>', unsafe_allow_html=True)
                    st.progress(float(confidence))
                    
                    # Display disease description
                    predicted_disease = class_names[predicted_class_idx]
                    if predicted_disease in DISEASE_DESCRIPTIONS:
                        st.info(f"💡 **Description:** {DISEASE_DESCRIPTIONS[predicted_disease]}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        else:
            st.error("Model not loaded. Please ensure model.h5 exists.")
    
    if model is not None:
        try:
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            
            st.markdown("---")
            st.markdown("### Detailed Probability Analysis")
            
            num_classes = len(predictions[0])
            prob_values = [p * 100 for p in predictions[0]]
            display_names = class_names[:num_classes] if len(class_names) >= num_classes else class_names
            
            prob_data = {
                'Class': display_names,
                'Probability (%)': prob_values
            }
            prob_df = pd.DataFrame(prob_data)
            
            cols = st.columns(len(display_names))
            for idx, (col, name, prob) in enumerate(zip(cols, display_names, prob_values)):
                with col:
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
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.bar_chart(prob_df.set_index('Class'), height=300)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            pass

st.markdown("---")
with st.expander("How to Use This App", expanded=False):
    st.markdown("""
    <div class="info-card">
        <h4>Step-by-Step Guide:</h4>
        <ol style="line-height: 2;">
            <li><strong>Upload an Image</strong>: Click the upload button and select a clear image of a plant leaf</li>
            <li><strong>Wait for Analysis</strong>: The model will analyze the image (takes 2-3 seconds)</li>
            <li><strong>View Results</strong>: See the prediction, confidence score, and detailed probability breakdown</li>
        </ol>
        
        <h4>Tips for Best Results:</h4>
        <ul style="line-height: 2;">
            <li>Use clear, well-lit images</li>
            <li>Ensure the leaf is in focus</li>
            <li>Upload images in JPG, JPEG, or PNG format</li>
            <li>For best accuracy, use images similar to training data</li>
        </ul>
        
        <p style="margin-top: 1rem; padding: 1rem; background: white; border-radius: 5px;">
            <strong>Note:</strong> This model was trained on PlantVillage dataset. 
            For production use, ensure the model is trained on a comprehensive dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="footer">
        <p>🌿 <strong>Plant Disease Detection System</strong></p>
        <p>Powered by TensorFlow/Keras & Streamlit</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">© 2024 Plant Disease Detection App</p>
    </div>
""", unsafe_allow_html=True)

