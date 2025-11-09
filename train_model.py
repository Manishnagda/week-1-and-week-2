"""
Plant Disease Detection - CNN Model Training Script
This script trains a CNN model using the PlantVillage-style dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Use tf.keras for compatibility
keras = tf.keras

def load_dataset_from_folder(dataset_dir='dataset', img_size=128):
    """
    Load images from dataset folder organized by class.
    Falls back to synthetic data if dataset folder doesn't exist.
    """
    X_train = []
    y_train = []
    
    class_names = ['Healthy', 'Disease_A_Spots', 'Disease_B_Yellow', 'Disease_C_Brown']
    class_mapping = {
        'Healthy': 0,
        'Disease_A_Spots': 1,
        'Disease_B_Yellow': 2,
        'Disease_C_Brown': 3
    }
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset folder '{dataset_dir}' not found. Generating synthetic data...")
        return generate_synthetic_data(num_samples=800, img_size=img_size)
    
    print(f"Loading dataset from '{dataset_dir}' folder...")
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found. Skipping...")
            continue
        
        class_idx = class_mapping[class_name]
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Loading {len(image_files)} images from {class_name}...")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(class_dir, img_file)
                img = Image.open(img_path)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize
                img = img.resize((img_size, img_size))
                
                # Convert to array and normalize
                img_array = np.array(img).astype('float32') / 255.0
                
                X_train.append(img_array)
                y_train.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
    
    if len(X_train) == 0:
        print("No images loaded. Generating synthetic data instead...")
        return generate_synthetic_data(num_samples=800, img_size=img_size)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Convert to categorical
    num_classes = len(class_names)
    y_train = utils.to_categorical(y_train, num_classes)
    
    # Split into train and validation
    split_idx = int(0.8 * len(X_train))
    indices = np.random.permutation(len(X_train))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    return (X_train, y_train), (X_val, y_val)

def generate_synthetic_data(num_samples=1000, img_size=128):
    """Generate synthetic plant leaf images for training (fallback)."""
    print("Generating synthetic training data...")
    
    X_train = []
    y_train = []
    num_classes = 4
    
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
    
    return (X_train, y_train), (X_val, y_val)

def create_cnn_model(input_shape=(128, 128, 3), num_classes=4):
    """Create a CNN model for plant disease classification."""
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
    
    # Load dataset from folder
    (X_train, y_train), (X_val, y_val) = load_dataset_from_folder(
        dataset_dir='dataset',
        img_size=img_size
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Image shape: {X_train[0].shape}")
    
    # Create model
    print("\nCreating CNN model...")
    model = create_cnn_model(input_shape=(img_size, img_size, 3), num_classes=num_classes)
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Save model
    model_path = 'model.h5'
    model.save(model_path)
    print(f"\n[SUCCESS] Model saved to {model_path}")
    
    # Save class names
    class_names = ['Healthy', 'Disease A (Spots)', 'Disease B (Yellow Patches)', 'Disease C (Brown Edges)']
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')
    
    print("[SUCCESS] Class names saved to class_names.txt")
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
