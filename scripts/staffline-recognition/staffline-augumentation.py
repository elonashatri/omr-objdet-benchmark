# staffline_augmentation.py
import cv2
import numpy as np
import os
from pathlib import Path
import random

def augment_stafflines(dataset_path, output_path, staffline_class_id):
    """Create augmented versions of images with stafflines"""
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
    
    # Get all training images and labels
    train_labels = list(Path(os.path.join(dataset_path, "labels", "train")).glob("*.txt"))
    
    for label_path in train_labels:
        # Read labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Check if file contains stafflines
        staffline_present = any(line.split()[0] == str(staffline_class_id) for line in lines)
        
        if staffline_present:
            # Get corresponding image
            img_path = os.path.join(dataset_path, "images", "train", 
                                    label_path.stem + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(dataset_path, "images", "train", 
                                       label_path.stem + ".png")
            
            if os.path.exists(img_path):
                # Create staffline-focused augmentations
                img = cv2.imread(img_path)
                
                # Augmentation 1: Contrast enhancement
                contrast_img = cv2.convertScaleAbs(img, alpha=1.2, beta=0)
                
                # Augmentation 2: Noise reduction
                denoised_img = cv2.GaussianBlur(img, (3, 3), 0)
                
                # Augmentation 3: Small shear
                h, w = img.shape[:2]
                shear_factor = random.uniform(-0.05, 0.05)
                M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
                sheared_img = cv2.warpAffine(img, M, (w, h))
                
                # Save augmented images and labels
                augmentations = [
                    (contrast_img, "_contrast"),
                    (denoised_img, "_denoised"),
                    (sheared_img, "_sheared")
                ]
                
                for aug_img, suffix in augmentations:
                    # Save image
                    aug_img_path = os.path.join(output_path, "images", "train", 
                                              label_path.stem + suffix + ".jpg")
                    cv2.imwrite(aug_img_path, aug_img)
                    
                    # Save labels (same as original for these augmentations)
                    aug_label_path = os.path.join(output_path, "labels", "train",
                                                label_path.stem + suffix + ".txt")
                    with open(aug_label_path, 'w') as f:
                        f.writelines(lines)