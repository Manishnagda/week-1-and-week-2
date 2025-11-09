"""
Create a small subset of PlantVillage-style dataset for demo purposes
This script generates synthetic plant disease images similar to PlantVillage dataset
"""

import numpy as np
from PIL import Image
import os

def create_plant_disease_dataset(output_dir='dataset', samples_per_class=8):
    """
    Create a small subset of plant disease images for training.
    This simulates the PlantVillage dataset structure.
    """
    print("Creating PlantVillage-style dataset subset...")
    
    # Class names matching PlantVillage structure
    classes = {
        'Healthy': (0.3, 0.6, 0.2),  # Green
        'Disease_A_Spots': (0.8, 0.2, 0.2),  # Red spots
        'Disease_B_Yellow': (0.9, 0.9, 0.3),  # Yellow patches
        'Disease_C_Brown': (0.5, 0.3, 0.2),  # Brown edges
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name, base_color in classes.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Creating {samples_per_class} samples for {class_name}...")
        
        for i in range(samples_per_class):
            # Create base image
            img = np.random.rand(128, 128, 3)
            img[:, :, 0] = np.random.uniform(base_color[0] - 0.1, base_color[0] + 0.1)
            img[:, :, 1] = np.random.uniform(base_color[1] - 0.1, base_color[1] + 0.1)
            img[:, :, 2] = np.random.uniform(base_color[2] - 0.1, base_color[2] + 0.1)
            
            # Add texture
            noise = np.random.randn(128, 128, 3) * 0.08
            img = np.clip(img + noise, 0, 1)
            
            # Add disease patterns
            if 'Spots' in class_name:
                # Add red spots
                spots = np.random.randint(0, 128, (15, 2))
                for spot in spots:
                    y, x = spot
                    if 0 <= y < 128 and 0 <= x < 128:
                        img[max(0, y-2):min(128, y+2), max(0, x-2):min(128, x+2), :] = [0.9, 0.1, 0.1]
            
            elif 'Yellow' in class_name:
                # Add yellow patches
                patches = np.random.randint(0, 128, (10, 2))
                for patch in patches:
                    y, x = patch
                    if 0 <= y < 128 and 0 <= x < 128:
                        img[max(0, y-4):min(128, y+4), max(0, x-4):min(128, x+4), :] = [0.95, 0.95, 0.4]
            
            elif 'Brown' in class_name:
                # Add brown edges
                edge_width = 8
                img[:edge_width, :, :] = [0.4, 0.25, 0.15]
                img[-edge_width:, :, :] = [0.4, 0.25, 0.15]
                img[:, :edge_width, :] = [0.4, 0.25, 0.15]
                img[:, -edge_width:, :] = [0.4, 0.25, 0.15]
            
            # Convert to PIL Image and save
            img_uint8 = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            
            filename = f"{class_name}_{i+1:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            pil_img.save(filepath, 'JPEG', quality=95)
        
        print(f"[SUCCESS] Created {samples_per_class} images for {class_name}")
    
    print(f"\n[SUCCESS] Dataset created successfully in '{output_dir}' folder!")
    print(f"Total images: {len(classes) * samples_per_class}")

if __name__ == '__main__':
    create_plant_disease_dataset(samples_per_class=8)

