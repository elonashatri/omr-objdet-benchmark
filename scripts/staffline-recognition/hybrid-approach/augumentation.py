import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import yaml
import shutil
import random
from tqdm import tqdm

def enhance_staffline_dataset(data_yaml, output_dir, staffline_class_id):
    """
    Generate an enhanced dataset with staffline-specific augmentations
    """
    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get dataset path
    dataset_path = data_config.get('path', '.')
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Copy validation set directly (no augmentation)
    print("Copying validation set...")
    val_img_dir = os.path.join(dataset_path, 'images', 'val')
    val_label_dir = os.path.join(dataset_path, 'labels', 'val')
    
    for img_file in Path(val_img_dir).glob('*.*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            shutil.copy(img_file, os.path.join(output_dir, 'images', 'val', img_file.name))
    
    for label_file in Path(val_label_dir).glob('*.txt'):
        shutil.copy(label_file, os.path.join(output_dir, 'labels', 'val', label_file.name))
    
    # Process training set with augmentations
    print("Processing training set with staffline augmentations...")
    train_img_dir = os.path.join(dataset_path, 'images', 'train')
    train_label_dir = os.path.join(dataset_path, 'labels', 'train')
    
    # Get all label files
    label_files = list(Path(train_label_dir).glob('*.txt'))
    
    # Track staffline statistics
    total_images = 0
    staffline_images = 0
    total_stafflines = 0
    augmented_images = 0
    
    for label_file in tqdm(label_files, desc="Augmenting training images"):
        total_images += 1
        
        # Check if file contains stafflines
        has_stafflines = False
        staffline_count = 0
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5 and int(parts[0]) == staffline_class_id:
                    has_stafflines = True
                    staffline_count += 1
        
        # Copy original files first
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png']:
            possible_img = os.path.join(train_img_dir, label_file.stem + ext)
            if os.path.exists(possible_img):
                img_file = possible_img
                break
        
        if img_file is None:
            continue
        
        # Copy original image and label
        shutil.copy(img_file, os.path.join(output_dir, 'images', 'train', os.path.basename(img_file)))
        shutil.copy(label_file, os.path.join(output_dir, 'labels', 'train', label_file.name))
        
        # Update statistics
        if has_stafflines:
            staffline_images += 1
            total_stafflines += staffline_count
            
            # Create augmentations focused on stafflines
            # Read the image
            img = cv2.imread(img_file)
            if img is None:
                continue
            
            # Create staffline-focused augmentations
            augmentations = []
            
            # 1. Contrast adjustment (helps with thin lines)
            contrast_img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            augmentations.append(("contrast", contrast_img))
            
            # 2. Sharpening (helps emphasize lines)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharp_img = cv2.filter2D(img, -1, kernel)
            augmentations.append(("sharp", sharp_img))
            
            # 3. Horizontal stretching (makes stafflines more prominent)
            h, w = img.shape[:2]
            stretch_factor = random.uniform(1.05, 1.15)  # 5-15% stretch
            stretch_matrix = np.float32([[stretch_factor, 0, 0], [0, 1, 0]])
            stretch_img = cv2.warpAffine(img, stretch_matrix, (int(w * stretch_factor), h))
            augmentations.append(("stretch", stretch_img))
            
            # 4. Minor affine transform (slight perspective change)
            rows, cols = img.shape[:2]
            src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
            # Slight shift of top-right and bottom-left corners
            dst_points = np.float32([[0, 0], 
                                    [cols-1, random.randint(-5, 5)], 
                                    [random.randint(-5, 5), rows-1]])
            affine_matrix = cv2.getAffineTransform(src_points, dst_points)
            affine_img = cv2.warpAffine(img, affine_matrix, (cols, rows))
            augmentations.append(("affine", affine_img))
            
            # Save augmented images and adjust labels
            for aug_name, aug_img in augmentations:
                # Create filenames
                aug_img_name = f"{label_file.stem}_{aug_name}{os.path.splitext(img_file)[1]}"
                aug_label_name = f"{label_file.stem}_{aug_name}.txt"
                
                # Save image
                cv2.imwrite(os.path.join(output_dir, 'images', 'train', aug_img_name), aug_img)
                
                # Copy label file (for basic augmentations)
                # For more complex augmentations like stretch, we would need to adjust coordinates
                if aug_name == "stretch":
                    # Adjust x-coordinates for horizontal stretching
                    with open(os.path.join(output_dir, 'labels', 'train', aug_label_name), 'w') as out_f:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls = int(parts[0])
                                x_center = float(parts[1]) / stretch_factor
                                y_center = float(parts[2])
                                width = float(parts[3]) / stretch_factor
                                height = float(parts[4])
                                
                                # Write adjusted coordinates
                                out_f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
                elif aug_name == "affine":
                    # Simple approximation for affine transform
                    # For full accuracy, would need to transform each box
                    shutil.copy(label_file, os.path.join(output_dir, 'labels', 'train', aug_label_name))
                else:
                    # For contrast and sharp, labels stay the same
                    shutil.copy(label_file, os.path.join(output_dir, 'labels', 'train', aug_label_name))
                
                augmented_images += 1
    
    # Create new dataset.yaml
    new_config = data_config.copy()
    new_config['path'] = os.path.abspath(output_dir)
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    # Print statistics
    print(f"\nStaffline Dataset Enhancement Complete:")
    print(f"Total original images: {total_images}")
    print(f"Images with stafflines: {staffline_images} ({staffline_images/total_images*100:.1f}%)")
    print(f"Total stafflines: {total_stafflines}")
    print(f"Augmented images added: {augmented_images}")
    print(f"New dataset size: {total_images + augmented_images} images")
    print(f"New dataset saved to: {output_dir}")
    print(f"New dataset.yaml: {os.path.join(output_dir, 'dataset.yaml')}")
    
    return os.path.join(output_dir, 'dataset.yaml')

def main():
    parser = argparse.ArgumentParser(description="Enhance dataset with staffline-focused augmentations")
    parser.add_argument("--yaml", type=str, required=True, help="Path to original dataset.yaml file")
    parser.add_argument("--output", type=str, default="staffline_enhanced_dataset", 
                        help="Output directory for enhanced dataset")
    parser.add_argument("--staffline_class_id", type=int, required=True,
                        help="Class ID for stafflines in your dataset (0-indexed)")
    
    args = parser.parse_args()
    
    print(f"Enhancing dataset for staffline detection")
    print(f"Original dataset: {args.yaml}")
    print(f"Staffline class ID: {args.staffline_class_id}")
    
    # Create enhanced dataset
    new_yaml = enhance_staffline_dataset(args.yaml, args.output, args.staffline_class_id)
    
    print(f"\nNext steps:")
    print(f"1. Train YOLOv8 on the enhanced dataset:")
    print(f"   python train.py --yaml {new_yaml} --model yolov8x.pt --epochs 300 --batch 8 --device 0")

if __name__ == "__main__":
    main()