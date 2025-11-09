# AI/ML Project – Week 1 and Week 2

## Briefly Mention the Improvisations Done by You:

- Improved model accuracy using better preprocessing  
- Added error handling for invalid image uploads  
- Enhanced UI design in Streamlit  
- Added additional dataset samples  
- Fixed bugs and optimized performance  

## GitHub Repository Link for Week 2 Milestone (Project Source Code):

https://github.com/Manishnagda/week-1-and-week-2

## How to Run:

1. Install dependencies:  

```bash
pip install -r requirements.txt
```

2. Run the app:  

```bash
streamlit run app.py
```

## Project Overview

This is a complete AI/ML project for Plant Disease Detection using Deep Learning (CNN). The application allows users to upload plant leaf images and get instant disease detection results with confidence scores and detailed probability analysis.

### Features

- 🖼️ Upload plant leaf images (JPG, JPEG, PNG)
- 🤖 Automatic disease detection using CNN
- 📊 Confidence scores and probability distributions
- 🎨 Modern and intuitive user interface
- ⚡ Fast predictions (2-3 seconds)
- 💡 Detailed analysis with visualizations

### Project Structure

```
.
├── train_model.py      # CNN model training script
├── app.py              # Streamlit web application (main file)
├── model.h5            # Trained CNN model
├── class_names.txt     # Class labels
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128 RGB images
- **Classes**: 
  - Healthy
  - Disease A (Spots)
  - Disease B (Yellow Patches)
  - Disease C (Brown Edges)

### Technical Stack

- **Deep Learning**: TensorFlow/Keras 2.15.0
- **Web Framework**: Streamlit
- **Image Processing**: PIL/Pillow
- **Numerical Computing**: NumPy, Pandas

### Notes

- The model is trained on synthetic data for demonstration purposes
- For production use, train with a real plant disease dataset (e.g., PlantVillage dataset)
- Image preprocessing includes resizing to 128x128 and normalization

## License

This project is for educational and demonstration purposes.
