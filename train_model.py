"""
Plant Disease Detection - CNN Model Training Script
Trains a CNN model using PlantVillage dataset for plant disease classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

keras = tf.keras

def load_dataset_from_folder(dataset_dir='PlantVillage/PlantVillage', img_size=128, max_samples_per_class=1000):
    """
    Load images from PlantVillage dataset folder.
    Maps PlantVillage classes to 4 categories: Healthy, Early Blight, Late Blight, Bacterial Spot
    """
    X_train = []
    y_train = []
    
    class_mapping = {
        'Pepper__bell___healthy': 0,
        'Potato___healthy': 0,
        'Tomato_healthy': 0,
        'Potato___Early_blight': 1,
        'Tomato_Early_blight': 1,
        'Potato___Late_blight': 2,
        'Tomato_Late_blight': 2,
        'Pepper__bell___Bacterial_spot': 3,
        'Tomato_Bacterial_spot': 3
    }
    
    class_names = ['Healthy', 'Early Blight', 'Late Blight', 'Bacterial Spot']
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset folder '{dataset_dir}' not found. Generating synthetic data...")
        return generate_synthetic_data(num_samples=800, img_size=img_size), class_names
    
    print(f"Loading dataset from '{dataset_dir}' folder...")
    
    # Get all subdirectories (classes) in the dataset folder
    if not os.path.isdir(dataset_dir):
        print(f"Error: '{dataset_dir}' is not a directory. Generating synthetic data...")
        return generate_synthetic_data(num_samples=800, img_size=img_size), class_names
    
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for subdir in subdirs:
        if subdir not in class_mapping:
            continue
        
        class_idx = class_mapping[subdir]
        class_dir = os.path.join(dataset_dir, subdir)
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
        
        if len(image_files) > max_samples_per_class:
            import random
            random.shuffle(image_files)
            image_files = image_files[:max_samples_per_class]
        
        print(f"Loading {len(image_files)} images from {subdir} -> {class_names[class_idx]}...")
        
        loaded_count = 0
        for img_file in image_files:
            try:
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path)
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize((img_size, img_size))
                img_array = np.array(img).astype('float32') / 255.0
                
                X_train.append(img_array)
                y_train.append(class_idx)
                loaded_count += 1
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
        
        print(f"  Successfully loaded {loaded_count} images")
    
    if len(X_train) == 0:
        print("No images loaded. Generating synthetic data instead...")
        return generate_synthetic_data(num_samples=800, img_size=img_size), class_names
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print("\nClass distribution:")
    for i, name in enumerate(class_names):
        count = np.sum(y_train == i)
        print(f"  {name}: {count} images")
    
    num_classes = len(class_names)
    y_train = utils.to_categorical(y_train, num_classes)
    
    split_idx = int(0.8 * len(X_train))
    indices = np.random.permutation(len(X_train))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    return (X_train, y_train), (X_val, y_val), class_names

def generate_synthetic_data(num_samples=1000, img_size=128):
    """Generate synthetic plant leaf images for training if dataset not found."""
    print("Generating synthetic training data...")
    
    X_train = []
    y_train = []
    num_classes = 4
    class_names = ['Healthy', 'Early Blight', 'Late Blight', 'Bacterial Spot']
    
    for i in range(num_samples):
        img = np.random.rand(img_size, img_size, 3)
        img[:, :, 0] = np.random.uniform(0.2, 0.4)
        img[:, :, 1] = np.random.uniform(0.4, 0.6)
        img[:, :, 2] = np.random.uniform(0.1, 0.3)
        
        noise = np.random.randn(img_size, img_size, 3) * 0.1
        img = np.clip(img + noise, 0, 1)
        
        class_idx = i % num_classes
        
        if class_idx == 1:
            spots = np.random.randint(0, img_size, (20, 2))
            for spot in spots:
                y, x = spot
                if 0 <= y < img_size and 0 <= x < img_size:
                    img[max(0, y-3):min(img_size, y+3), max(0, x-3):min(img_size, x+3), :] = [0.8, 0.2, 0.2]
        elif class_idx == 2:
            patches = np.random.randint(0, img_size, (15, 2))
            for patch in patches:
                y, x = patch
                if 0 <= y < img_size and 0 <= x < img_size:
                    img[max(0, y-5):min(img_size, y+5), max(0, x-5):min(img_size, x+5), :] = [0.9, 0.9, 0.3]
        elif class_idx == 3:
            edge_width = 10
            img[:edge_width, :, :] = [0.5, 0.3, 0.2]
            img[-edge_width:, :, :] = [0.5, 0.3, 0.2]
            img[:, :edge_width, :] = [0.5, 0.3, 0.2]
            img[:, -edge_width:, :] = [0.5, 0.3, 0.2]
        
        X_train.append(img)
        y_train.append(class_idx)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    y_train = utils.to_categorical(y_train, num_classes)
    
    split_idx = int(0.8 * len(X_train))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    return (X_train, y_train), (X_val, y_val), class_names

def create_cnn_model(input_shape=(128, 128, 3), num_classes=4):
    """Create CNN model for plant disease classification."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("=" * 50)
    print("Plant Disease Detection - Model Training")
    print("Using PlantVillage-style dataset")
    print("=" * 50)
    
    img_size = 128
    num_classes = 4
    batch_size = 32
    epochs = 10
    
    result = load_dataset_from_folder(
        dataset_dir='PlantVillage/PlantVillage',
        img_size=img_size,
        max_samples_per_class=1000
    )
    
    (X_train, y_train), (X_val, y_val), class_names = result
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Image shape: {X_train[0].shape}")
    
    print("\nCreating CNN model...")
    model = create_cnn_model(input_shape=(img_size, img_size, 3), num_classes=num_classes)
    model.summary()
    
    print("\nSetting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    print("\nTraining model with data augmentation...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    print("\nEvaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    print("\nGenerating detailed evaluation metrics...")
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(cm)
    
    print("\nPer-class Accuracy:")
    for i, name in enumerate(class_names):
        class_mask = y_true_classes == i
        if np.sum(class_mask) > 0:
            class_acc = np.sum((y_pred_classes[class_mask] == i)) / np.sum(class_mask)
            print(f"  {name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    model_path = 'model/plant_disease_model.h5'
    model.save(model_path)
    print(f"\n[SUCCESS] Model saved to {model_path}")
    
    # Also save to root for backward compatibility
    model.save('model.h5')
    print(f"[SUCCESS] Model also saved to model.h5 (backward compatibility)")
    
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    
    print("[SUCCESS] Class names saved to class_names.txt")
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
