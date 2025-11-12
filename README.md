# AI/ML Project – Week 1 and Week 2

## Project Overview

This is an AI-powered plant disease detection system that uses Convolutional Neural Networks (CNN) to classify plant leaf images into healthy or diseased categories. The system can detect multiple types of plant diseases including Early Blight, Late Blight, and Bacterial Spot.

## Key Features

- **CNN-based Classification**: Deep learning model trained on PlantVillage dataset
- **User-friendly Web Interface**: Beautiful Streamlit web application
- **Real-time Prediction**: Upload images and get instant disease detection results
- **Detailed Analysis**: View confidence scores and probability breakdown for all classes
- **Data Augmentation**: Enhanced training with image augmentation techniques
- **Comprehensive Evaluation**: Includes classification reports and confusion matrices

## Improvements Made

- Merged Week 1 and Week 2 work into a single improved project  
- Integrated PlantVillage dataset for realistic training and testing  
- Enhanced Streamlit UI with modern design and better user experience  
- Added data augmentation to improve model generalization  
- Implemented comprehensive evaluation metrics (precision, recall, F1-score)  
- Improved error handling and image validation  
- Optimized model architecture and training pipeline  

## Dataset Used

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Classes**: 4 categories
  - Healthy
  - Early Blight
  - Late Blight
  - Bacterial Spot
- **Note**: The dataset folder is excluded from git due to size. Download it separately if needed.


## GitHub Repository Link for Week 2 Milestone (Project Source Code):

https://github.com/Manishnagda/week-1-and-week-2

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Kaggle API credentials for downloading dataset

### Step 1: Clone the Repository

```bash
git clone https://github.com/Manishnagda/week-1-and-week-2.git
cd week-1-and-week-2
```

### Step 2: Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Dataset Setup

The project uses the PlantVillage dataset. You have two options:

**Option A: Use Existing Dataset**
- If you already have the PlantVillage dataset, place it in the `PlantVillage/PlantVillage/` folder
- The dataset should be organized by class folders (e.g., `Tomato_healthy/`, `Potato___Early_blight/`, etc.)

**Option B: Download Dataset**
- Download from [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Extract and place in `PlantVillage/PlantVillage/` folder

**Option C: Generate Synthetic Data (for testing)**
- If dataset is not available, the training script will automatically generate synthetic data
- Or run: `python create_dataset.py`

### Step 4: Train the Model

Train the CNN model on the PlantVillage dataset:

```bash
python train_model.py
```

This will:
- Load images from the PlantVillage dataset
- Train a CNN model with data augmentation
- Save the model to `model/plant_disease_model.h5`
- Generate evaluation metrics (accuracy, precision, recall, F1-score)
- Save class names to `class_names.txt`

**Note:** Training may take 10-30 minutes depending on your hardware and dataset size.

### Step 5: Run the Streamlit Web Application

Start the web app:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

**Alternative:** You can also run `streamlit run main.py` (both apps are available)

---

## Project Structure

```
week-1-and-week-2/
├── app.py                     # Main Streamlit web application
├── main.py                    # Alternative Streamlit web application
├── train_model.py             # CNN model training script with data augmentation
├── create_dataset.py          # Script to generate synthetic dataset subset
├── class_names.txt            # Class labels (generated after training)
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
├── README.md                  # Project documentation
├── model/                     # Model directory (created after training)
│   └── plant_disease_model.h5 # Trained CNN model
└── PlantVillage/              # Dataset folder (not in git)
    └── PlantVillage/          # Actual dataset images organized by class
```

**Note**: 
- `model/` folder and `model.h5` are excluded from git due to size
- `PlantVillage/` dataset folder is excluded from git
- You need to train the model or download it separately

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
  - 4 Convolutional layers (32, 64, 128, 128 filters)
  - MaxPooling layers for dimensionality reduction
  - Dropout (0.5) for regularization
  - Dense layers with softmax activation
- **Input Size**: 128x128 RGB images
- **Classes**: 4 (Healthy, Early Blight, Late Blight, Bacterial Spot)
- **Training Features**:
  - Data augmentation (rotation, shift, flip, zoom)
  - Train/validation split (80/20)
  - Early stopping and model checkpointing
  - Comprehensive evaluation metrics

## Technical Stack

- **Deep Learning**: TensorFlow/Keras 2.15.0
- **Web Framework**: Streamlit
- **Data Processing**: NumPy, Pandas, PIL/Pillow
- **Evaluation**: scikit-learn, matplotlib
- **Python**: 3.8+

## Model Performance

After training, the model provides comprehensive evaluation metrics:

- **Validation Accuracy**: Overall model accuracy on validation set
- **Per-class Metrics**: Precision, recall, and F1-score for each disease class
- **Confusion Matrix**: Detailed analysis of classification performance
- **Classification Report**: Complete statistics including support for each class

### Expected Performance

With the PlantVillage dataset and proper training:
- Validation accuracy typically ranges from 85-95%
- Model performs well on clear, well-lit images
- Best results with images similar to training data distribution

---

## Using the Web Application

### How to Use

1. **Upload an Image**: Click the upload button and select a clear image of a plant leaf
2. **Wait for Analysis**: The CNN model will analyze the image (takes 2-3 seconds)
3. **View Results**: See the prediction, confidence score, disease description, and detailed probability breakdown

### Tips for Best Results

- Use clear, well-lit images of plant leaves
- Ensure the leaf is in focus and centered
- Supported formats: JPG, JPEG, PNG
- The model works best with images similar to the PlantVillage training data
- For production use, ensure the model is trained on a comprehensive dataset

### Example Output

The app displays:
- **Predicted Disease**: Name of the detected disease or "Healthy"
- **Confidence Score**: Percentage confidence in the prediction
- **Disease Description**: Helpful information about the detected disease and treatment recommendations
- **Probability Breakdown**: Detailed probabilities for all disease classes
- **Visual Charts**: Bar chart showing probability distribution

## Future Enhancements

- Transfer learning with pre-trained models (ResNet, EfficientNet)
- Support for more disease classes
- Real-time camera integration
- Mobile app development
- Cloud deployment

## License

This project is part of the Edunet Foundation - Shell-Edunet Skills4Future Internship program.

---

🧠 *This project combines Week 1 and Week 2 work and uses the PlantVillage dataset for realistic model training and testing.*

