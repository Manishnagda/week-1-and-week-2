"""
Plant Disease Detection - CNN Model Training Script
This script creates and trains a CNN model for plant disease detection.
Uses synthetic data generation for quick training.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, utils
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Use tf.keras for compatibility
keras = tf.keras

def generate_synthetic_data(num_samples=1000, img_size=128):
    """
    Generate synthetic plant leaf images for training.
    In production, replace this with real dataset loading.
    """
    print("Generating synthetic training data...")
    
    # Create synthetic images with different patterns
    X_train = []
    y_train = []
    
    # Class labels: 0 = Healthy, 1 = Disease A, 2 = Disease B, 3 = Disease C
    num_classes = 4
    
    for i in range(num_samples):
        # Create a base green leaf-like image
        img = np.random.rand(img_size, img_size, 3)
        img[:, :, 0] = np.random.uniform(0.2, 0.4)  # Green channel
        img[:, :, 1] = np.random.uniform(0.4, 0.6)   # Green channel
        img[:, :, 2] = np.random.uniform(0.1, 0.3)   # Blue channel
        
        # Add some texture
        noise = np.random.randn(img_size, img_size, 3) * 0.1
        img = np.clip(img + noise, 0, 1)
        
        # Determine class
        class_idx = i % num_classes
        
        # Add disease patterns for diseased classes
        if class_idx == 1:  # Disease A - spots
            spots = np.random.randint(0, img_size, (20, 2))
            for spot in spots:
                y, x = spot
                if 0 <= y < img_size and 0 <= x < img_size:
                    img[max(0, y-3):min(img_size, y+3), max(0, x-3):min(img_size, x+3), :] = [0.8, 0.2, 0.2]
        
        elif class_idx == 2:  # Disease B - yellow patches
            patches = np.random.randint(0, img_size, (15, 2))
            for patch in patches:
                y, x = patch
                if 0 <= y < img_size and 0 <= x < img_size:
                    img[max(0, y-5):min(img_size, y+5), max(0, x-5):min(img_size, x+5), :] = [0.9, 0.9, 0.3]
        
        elif class_idx == 3:  # Disease C - brown edges
            edge_width = 10
            img[:edge_width, :, :] = [0.5, 0.3, 0.2]
            img[-edge_width:, :, :] = [0.5, 0.3, 0.2]
            img[:, :edge_width, :] = [0.5, 0.3, 0.2]
            img[:, -edge_width:, :] = [0.5, 0.3, 0.2]
        
        X_train.append(img)
        y_train.append(class_idx)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Convert to categorical
    y_train = utils.to_categorical(y_train, num_classes)
    
    # Split into train and validation
    split_idx = int(0.8 * len(X_train))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    return (X_train, y_train), (X_val, y_val)

def create_cnn_model(input_shape=(128, 128, 3), num_classes=4):
    """Create a CNN model for plant disease classification."""
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Fourth convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Flatten and dense layers
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
    print("=" * 50)
    
    # Parameters
    img_size = 128
    num_classes = 4
    batch_size = 32
    epochs = 10  # Reduced for quick training
    
    # Generate synthetic data
    (X_train, y_train), (X_val, y_val) = generate_synthetic_data(
        num_samples=800,  # Reduced for faster training
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

