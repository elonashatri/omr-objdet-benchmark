from ultralytics import YOLO
import argparse
import os
import yaml
import cv2
import numpy as np
import torch
import random
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

class MusicNotationAlbumentations:
    """Custom augmentations for music notation datasets that preserve staffline integrity"""
    
    @staticmethod
    def apply_music_augmentations(img, labels):
        """Apply music-specific augmentations that preserve staffline characteristics
        
        Args:
            img: PIL Image
            labels: numpy array of [class, x, y, w, h] normalized coordinates
            
        Returns:
            augmented image and labels
        """
        # Convert PIL to numpy if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        # Skip augmentation randomly (30% of the time)
        if random.random() < 0.3:
            return img, labels
            
        h, w = img.shape[:2]
        augmented_img = img.copy()
        augmented_labels = labels.copy()
        
        # Identify staffline classes - assume class index for "kStaffLine"
        # You'll need to update this based on your actual class mapping
        staffline_indices = np.where(labels[:, 0] == 2)[0]  # Assuming class 2 is kStaffLine
        
        # 1. Apply slight line thickness variation (only if stafflines exist)
        if len(staffline_indices) > 0 and random.random() < 0.5:
            # This would need to be implemented with morphological operations
            # For demonstration purposes, we simply apply a slight blur and threshold
            if len(augmented_img.shape) == 3:  # Color image
                gray = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2GRAY)
            else:  # Already grayscale
                gray = augmented_img.copy()
                
            # Apply slight erosion or dilation randomly
            kernel_size = random.choice([1, 2])
            if random.random() < 0.5:
                kernel = np.ones((1, kernel_size), np.uint8)  # Horizontal kernel
                gray = cv2.dilate(gray, kernel, iterations=1)
            else:
                kernel = np.ones((1, kernel_size), np.uint8)  # Horizontal kernel
                gray = cv2.erode(gray, kernel, iterations=1)
                
            # Convert back to original format
            if len(augmented_img.shape) == 3:
                augmented_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                augmented_img = gray
        
        # 2. Add subtle noise to simulate real-world scores (20% chance)
        if random.random() < 0.2:
            noise_type = random.choice(['gaussian', 'speckle', 'salt_pepper'])
            
            if noise_type == 'gaussian':
                # Add slight Gaussian noise
                mean = 0
                sigma = random.uniform(1, 5)
                noise = np.random.normal(mean, sigma, augmented_img.shape).astype(np.uint8)
                augmented_img = cv2.add(augmented_img, noise)
                
            elif noise_type == 'speckle':
                # Add multiplicative speckle noise
                noise = np.random.normal(0, random.uniform(0.01, 0.05), augmented_img.shape)
                augmented_img = augmented_img + augmented_img * noise
                augmented_img = np.clip(augmented_img, 0, 255).astype(np.uint8)
                
            elif noise_type == 'salt_pepper':
                # Add salt and pepper noise
                prob = random.uniform(0.001, 0.01)
                black = np.random.random(augmented_img.shape[:2])
                white = np.random.random(augmented_img.shape[:2])
                augmented_img[black < prob/2] = 0
                augmented_img[white > 1 - prob/2] = 255
        
        # 3. Slight horizontal stretching (5% chance)
        if random.random() < 0.05:
            stretch_factor = random.uniform(0.95, 1.05)
            augmented_img = cv2.resize(augmented_img, 
                                     (int(w * stretch_factor), h),
                                     interpolation=cv2.INTER_LINEAR)
            
            # Adjust labels for horizontal stretching
            new_w = int(w * stretch_factor)
            # Scale x coordinates and width
            if stretch_factor != 1.0:
                augmented_labels[:, 1] = augmented_labels[:, 1] * (w / new_w)  # x center
                augmented_labels[:, 3] = augmented_labels[:, 3] * (w / new_w)  # width
            
            # Resize back to original dimensions
            augmented_img = cv2.resize(augmented_img, (w, h), interpolation=cv2.INTER_LINEAR)
                
        # 4. Add slight paper texture or aging effect (10% chance)
        if random.random() < 0.1:
            # Convert to PIL for texture blending
            pil_img = Image.fromarray(augmented_img)
            
            # Apply slight sepia tone or aging
            aging_factor = random.uniform(0.05, 0.2)
            enhancer = ImageDraw.Draw(pil_img)
            width, height = pil_img.size
            
            # Create a semi-transparent overlay
            overlay = Image.new('RGBA', pil_img.size, (200, 180, 150, int(255 * aging_factor)))
            
            # Convert images to RGBA if needed
            if pil_img.mode != 'RGBA':
                pil_img = pil_img.convert('RGBA')
            
            # Blend images
            augmented_img = Image.alpha_composite(pil_img, overlay)
            augmented_img = augmented_img.convert('RGB')
            augmented_img = np.array(augmented_img)
        
        # 5. Simulate slight page curvature (5% chance) 
        # This is complex and would involve perspective transformation
        # We'll skip detailed implementation here
            
        return augmented_img, augmented_labels

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on music notation dataset with optimizations for stafflines")
    parser.add_argument("--yaml", type=str, required=True, help="Path to dataset.yaml file")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="YOLO model to use")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--img-size", type=int, default=1600, help="Image size for training")
    parser.add_argument("--name", type=str, default="", help="Experiment name")
    parser.add_argument("--custom-aug", action="store_true", help="Use custom music notation augmentations")
    
    args = parser.parse_args()
    
    # Create experiment name if not provided
    if not args.name:
        args.name = f"music_notation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Training {args.model} on {args.yaml} for {args.epochs} epochs")
    print(f"Using GPU device {args.device}, image size {args.img_size}")
    print(f"Experiment name: {args.name}")
    print(f"Custom music augmentations: {'Enabled' if args.custom_aug else 'Disabled'}")
    
    # Get class names from dataset.yaml to identify staffline class
    with open(args.yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
        class_names = dataset_config.get('names', {})
        
    # Find the class index for "kStaffLine" or similar
    staffline_class = None
    for idx, name in class_names.items():
        if 'staff' in name.lower() and 'line' in name.lower():
            staffline_class = int(idx)
            print(f"Detected staffline class: {name} (index {staffline_class})")
            break
    
    if staffline_class is None:
        print("Warning: Could not identify staffline class. Custom augmentations may not work correctly.")
    
    # Load model
    model = YOLO(args.model)
    
    # Set up training arguments
    train_args = {
        "data": args.yaml,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.img_size,
        "device": args.device,
        "name": args.name,
        "rect": True,              # Use rectangular training (important for extreme aspect ratios)
        "mosaic": 0.0,             # Disable mosaic augmentation (can break long horizontal lines)
        "scale": 0.3,              # Minimal scale augmentation
        "fliplr": 0.0,             # Disable horizontal flips (can affect staffline detection)
        "flipud": 0.0,             # Disable vertical flips (music notation is orientation-sensitive)
        "hsv_h": 0.0,              # Disable hue augmentation (preserve black/white contrast)
        "hsv_s": 0.0,              # Disable saturation augmentation
        "hsv_v": 0.1,              # Minimal brightness augmentation
        "degrees": 0,              # Disable rotation (music notation is orientation-sensitive)
        "translate": 0.1,          # Minimal translation augmentation
        "perspective": 0.0,        # Disable perspective transform (can distort stafflines)
        "patience": 50,            # More patience for early stopping
        "cos_lr": True,            # Use cosine learning rate scheduler
        "overlap_mask": True,      # Better for overlapping objects
        "val": True,               # Validate during training
        "save": True,              # Save checkpoints
        "plots": True              # Generate performance plots
    }
    
    # If custom augmentations enabled, use a custom dataloader (implementation depends on YOLOv8 version)
    if args.custom_aug:
        print("Note: Custom music notation augmentations are implemented but require YOLOv8 source modification")
        print("      This script provides the augmentation code but can't apply it without modifying YOLOv8")
        
        # In practice, you would need to create a custom dataset class for YOLOv8
        # or modify the YOLOv8 source to integrate these custom augmentations
    
    # Train model with optimizations for stafflines
    results = model.train(**train_args)
    
    # Generate deployment model
    best_model_path = os.path.join(model.trainer.save_dir, 'weights/best.pt')
    if os.path.exists(best_model_path):
        print(f"Generating optimized model from {best_model_path}")
        best_model = YOLO(best_model_path)
        
        # Export to ONNX format for deployment
        best_model.export(format="onnx")
        
        # Also save a TorchScript version
        best_model.export(format="torchscript")
    
    print("Training complete!")
    print(f"Results saved to: {model.trainer.save_dir}")
    
    # Print summary of results
    metrics = results.results_dict
    print("\nTraining Results Summary:")
    print(f"Best mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"Best mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"Best Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"Best Recall: {metrics.get('metrics/recall(B)', 0):.4f}")

if __name__ == "__main__":
    main()