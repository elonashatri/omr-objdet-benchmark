import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import yaml
import shutil

def preprocess_image_for_stafflines(image_path, output_path, enhance_stafflines=True):
    """
    Preprocess an image to enhance staffline detection
    
    Args:
        image_path: Path to input image
        output_path: Path to save processed image
        enhance_stafflines: Whether to apply staffline enhancement
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error reading image: {image_path}")
        return False
    
    # Create a copy for staffline enhancement
    processed_img = img.copy()
    
    if enhance_stafflines:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blurring to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5
        )
        
        # Create horizontal kernel for staffline detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        
        # Detect horizontal lines (stafflines)
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Make stafflines slightly thicker for better detection
        dilated_lines = cv2.dilate(detected_lines, np.ones((1, 3), np.uint8), iterations=1)
        
        # Convert back to BGR for overlay
        stafflines_overlay = cv2.cvtColor(dilated_lines, cv2.COLOR_GRAY2BGR)
        
        # FIX: Create a proper mask using boolean indexing
        mask = dilated_lines > 0
        
        # Apply the mask - this is the corrected section
        alpha = 0.3
        for c in range(3):  # Loop through each color channel
            processed_img[:,:,c] = np.where(
                mask, 
                processed_img[:,:,c] * (1-alpha) + stafflines_overlay[:,:,c] * alpha,
                processed_img[:,:,c]
            )
        
    # Save processed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), processed_img)
    return True

def preprocess_dataset(data_yaml, output_dir, enhance_stafflines=True):
    """
    Preprocess an entire YOLO dataset to enhance staffline detection
    
    Args:
        data_yaml: Path to dataset.yaml file
        output_dir: Path to save processed dataset
        enhance_stafflines: Whether to apply staffline enhancement
    """
    # Load dataset config
    with open(data_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Get dataset base path
    dataset_path = Path(dataset_config.get('path', os.path.dirname(data_yaml)))
    
    # Create output directory structure
    output_dataset_path = Path(output_dir)
    os.makedirs(output_dataset_path, exist_ok=True)
    
    # Create images and labels directories
    for split in ['train', 'val', 'test']:
        # Only process splits that exist in the config
        if split not in dataset_config:
            continue
            
        # Create directories
        images_dir = output_dataset_path / 'images' / split
        labels_dir = output_dataset_path / 'labels' / split
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Get source directories
        src_images_dir = dataset_path / dataset_config[split]
        src_labels_dir = dataset_path / 'labels' / split
        
        # Process images
        image_files = list(src_images_dir.glob('*.jpg')) + list(src_images_dir.glob('*.png'))
        print(f"Processing {len(image_files)} images in {split} set")
        
        for img_path in tqdm(image_files, desc=f"Processing {split} images"):
            # Get corresponding label file
            label_path = src_labels_dir / f"{img_path.stem}.txt"
            
            # Skip if label doesn't exist
            if not label_path.exists():
                print(f"Warning: No label file for {img_path}")
                continue
            
            # Process image
            output_img_path = images_dir / img_path.name
            success = preprocess_image_for_stafflines(img_path, output_img_path, enhance_stafflines)
            
            if success:
                # Copy label file (no modification needed)
                shutil.copy(label_path, labels_dir / f"{img_path.stem}.txt")
    
    # Create new dataset.yaml
    new_config = dataset_config.copy()
    new_config['path'] = str(output_dataset_path.absolute())
    
    # Update paths to be relative to the new dataset location
    for split in ['train', 'val', 'test']:
        if split in new_config:
            new_config[split] = f'images/{split}'
    
    # Save new dataset.yaml
    output_yaml = output_dataset_path / 'dataset.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    print(f"Preprocessing complete. New dataset saved to: {output_dataset_path}")
    print(f"New dataset configuration: {output_yaml}")
    return str(output_yaml)

def main():
    parser = argparse.ArgumentParser(description="Preprocess music notation dataset for better staffline detection")
    parser.add_argument("--yaml", type=str, required=True, help="Path to dataset.yaml file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--no-enhance", action="store_true", help="Disable staffline enhancement")
    
    args = parser.parse_args()
    
    # Process dataset
    preprocess_dataset(args.yaml, args.output, enhance_stafflines=not args.no_enhance)

if __name__ == "__main__":
    main()