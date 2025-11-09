# AI/ML Project – Week 1 and Week 2

## Briefly Mention the Improvisations Done by Me:

- Merged Week 1 and Week 2 work into a single improved project  
- Integrated public dataset (PlantVillage) for realistic testing  
- Enhanced Streamlit UI for a cleaner user experience  
- Improved model accuracy and preprocessing logic  
- Added validation for incorrect image uploads  

## Dataset Used:

- Source: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- A small subset is included in the `dataset/` folder for demo purposes.  
- The dataset contains 4 classes: Healthy, Disease A (Spots), Disease B (Yellow Patches), and Disease C (Brown Edges)


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
week 1 and week 2/
├── main.py              # Streamlit web application
├── train_model.py       # CNN model training script
├── create_dataset.py    # Script to generate dataset subset
├── model.h5             # Trained CNN model
├── class_names.txt      # Class labels
├── requirements.txt     # Python dependencies
├── dataset/             # PlantVillage-style dataset subset
│   ├── Healthy/
│   ├── Disease_A_Spots/
│   ├── Disease_B_Yellow/
│   └── Disease_C_Brown/
└── README.md           # This file
```

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 128x128 RGB images
- **Classes**: 4 (Healthy + 3 disease types)
- **Training**: Uses dataset from `dataset/` folder, falls back to synthetic data if not found

## Technical Stack

- TensorFlow/Keras 2.15.0
- Streamlit
- NumPy, Pandas
- PIL/Pillow

---

🧠 *This project combines my Week 1 and Week 2 work and uses a real-world dataset for model training and testing.*

