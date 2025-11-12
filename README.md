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

## How to Run:

1. Install dependencies:  

```bash
pip install -r requirements.txt
```

2. (Optional) Generate dataset if not present:

```bash
python create_dataset.py
```

3. Train the model (if model.h5 doesn't exist):

```bash
python train_model.py
```

4. Run the app:  

```bash
streamlit run main.py
```

---

## Project Structure

```
week-1-and-week-2/
├── main.py                    # Streamlit web application
├── train_model.py             # CNN model training script with data augmentation
├── create_dataset.py          # Script to generate synthetic dataset subset
├── class_names.txt            # Class labels (generated after training)
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore file
├── README.md                  # Project documentation

```

**Note**: `model.h5` and dataset folders are excluded from git due to size. You need to train the model or download it separately.

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

After training, the model provides:
- Validation accuracy metrics
- Per-class precision, recall, and F1-scores
- Confusion matrix for detailed analysis
- Classification report with comprehensive statistics

---

## Usage Tips

- Upload clear, well-lit images of plant leaves for best results
- Supported formats: JPG, JPEG, PNG
- The model works best with images similar to the training data
- For production use, ensure the model is trained on a comprehensive dataset

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

