"""
Preprocessing Module for Staff Line Detection

This module handles the conversion of XML annotations to binary masks
for training the staff line detection model.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import concurrent.futures
from pathlib import Path
import random


def parse_xml_to_mask(xml_path, output_mask_dir, image_dir=None):
    """
    Parse an XML annotation file and create a binary mask for staff lines.
    
    Args:
        xml_path (str): Path to the XML annotation file
        output_mask_dir (str): Directory to save the masks
        image_dir (str, optional): Directory containing corresponding images
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get base filename
        base_name = os.path.splitext(os.path.basename(xml_path))[0]
        
        # If image directory is provided, load the image to get dimensions
        image_shape = None
        if image_dir:
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                img_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(img_path):
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image_shape = image.shape
                        break
        
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Determine mask dimensions
        if image_shape is None:
            # Determine dimensions from XML
            height = 0
            width = 0
            for page in root.findall('.//Page'):
                for node in page.findall('.//Nodes/Node'):
                    class_name_elem = node.find('ClassName')
                    if class_name_elem is not None and class_name_elem.text == 'kStaffLine':
                        top = int(node.find('Top').text)
                        left = int(node.find('Left').text)
                        node_width = int(node.find('Width').text)
                        node_height = int(node.find('Height').text)
                        
                        height = max(height, top + node_height)
                        width = max(width, left + node_width)
            
            # Add padding
            height += 20
            width += 20
        else:
            height, width = image_shape
        
        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process staff line nodes
        for page in root.findall('.//Page'):
            for node in page.findall('.//Nodes/Node'):
                class_name_elem = node.find('ClassName')
                if class_name_elem is not None and class_name_elem.text == 'kStaffLine':
                    top = int(node.find('Top').text)
                    left = int(node.find('Left').text)
                    node_width = int(node.find('Width').text)
                    node_height = int(node.find('Height').text)
                    
                    # Draw the staff line on the mask
                    if top < height and left < width:
                        end_top = min(top + node_height, height)
                        end_left = min(left + node_width, width)
                        mask[top:end_top, left:end_left] = 255
        
        # Save the mask
        output_path = os.path.join(output_mask_dir, base_name + '.png')
        cv2.imwrite(output_path, mask)
        
        return True
    
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False


def process_directory(xml_dir, output_mask_dir, image_dir=None, num_workers=4, subset_fraction=1.0, subset_seed=42):
    """
    Process all XML files in a directory to create binary masks.
    
    Args:
        xml_dir (str): Directory containing XML annotation files
        output_mask_dir (str): Directory to save the masks
        image_dir (str, optional): Directory containing corresponding images
        num_workers (int): Number of parallel workers
        subset_fraction (float): Fraction of dataset to use (0.0-1.0)
        subset_seed (int): Random seed for subset selection
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Get all XML files
    xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) 
                if f.lower().endswith('.xml')]
    
    # Take a subset if requested
    if subset_fraction < 1.0:
        random.seed(subset_seed)
        subset_size = max(1, int(len(xml_files) * subset_fraction))
        xml_files = random.sample(xml_files, subset_size)
        print(f"Processing {subset_size} XML files ({subset_fraction:.1%} of total)")
    
    print(f"Found {len(xml_files)} XML files to process")
    
    # Process files in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(parse_xml_to_mask, xml_file, output_mask_dir, image_dir) 
                  for xml_file in xml_files]
        
        # Show progress
        success_count = 0
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print(f"Error in worker: {e}")
        
        print(f"Successfully processed {success_count} out of {len(xml_files)} XML files")


def verify_dataset(image_dir, mask_dir):
    """
    Verify that the dataset is correctly prepared by checking that each image has a corresponding mask.
    
    Args:
        image_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
    """
    # Get image and mask files
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))}
    mask_files = {os.path.splitext(f)[0] for f in os.listdir(mask_dir) 
                if f.lower().endswith('.png')}
    
    # Check for missing masks
    missing_masks = image_files - mask_files
    if missing_masks:
        print(f"Warning: {len(missing_masks)} images do not have corresponding masks.")
        if len(missing_masks) < 10:
            print("Missing masks for:", missing_masks)
        else:
            print("First 10 missing masks:", list(missing_masks)[:10])
    else:
        print("All images have corresponding masks.")
    
    # Check for extra masks
    extra_masks = mask_files - image_files
    if extra_masks:
        print(f"Warning: {len(extra_masks)} masks do not have corresponding images.")
        if len(extra_masks) < 10:
            print("Extra masks:", extra_masks)
        else:
            print("First 10 extra masks:", list(extra_masks)[:10])
    else:
        print("All masks have corresponding images.")


def visualize_samples(image_dir, mask_dir, output_dir, num_samples=5):
    """
    Visualize random samples to verify the dataset.
    
    Args:
        image_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    import random
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get common files (both image and mask exist)
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))}
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(mask_dir) 
                if f.lower().endswith('.png')}
    
    common_keys = set(image_files.keys()) & set(mask_files.keys())
    
    # Select random samples
    if len(common_keys) <= num_samples:
        sample_keys = list(common_keys)
    else:
        sample_keys = random.sample(list(common_keys), num_samples)
    
    for key in sample_keys:
        # Load image and mask
        img_path = os.path.join(image_dir, image_files[key])
        mask_path = os.path.join(mask_dir, mask_files[key])
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Staff Line Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Create RGB image for overlay
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Create mask overlay (red channel)
        overlay = np.zeros_like(rgb_image)
        overlay[mask > 0] = [255, 0, 0]  # Red for staff lines
        # Blend
        alpha = 0.5
        blended = cv2.addWeighted(rgb_image, 1, overlay, alpha, 0)
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{key}_visualization.png"), dpi=150)
        plt.close()
    
    print(f"Saved {len(sample_keys)} visualizations to {output_dir}")


def preprocess_xmls_to_masks(xml_dir, output_mask_dir, image_dir=None, num_workers=4, 
                             verify=True, visualize=False, num_vis_samples=5, 
                             subset_fraction=1.0):
    """
    Main preprocessing function to convert XML annotations to binary masks.
    
    Args:
        xml_dir (str): Directory containing XML annotations
        output_mask_dir (str): Directory to save masks
        image_dir (str, optional): Directory containing corresponding images
        num_workers (int): Number of parallel workers
        verify (bool): Whether to verify the dataset after preprocessing
        visualize (bool): Whether to visualize sample images and masks
        num_vis_samples (int): Number of samples to visualize
        subset_fraction (float): Fraction of dataset to use (0.0-1.0)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create output directory
        os.makedirs(output_mask_dir, exist_ok=True)
        
        # Process XML annotations to create masks
        print("Processing XML annotations...")
        process_directory(xml_dir, output_mask_dir, image_dir, num_workers, subset_fraction)
        
        # Verify the dataset
        if verify and image_dir:
            print("\nVerifying dataset...")
            verify_dataset(image_dir, output_mask_dir)
        
        # Visualize samples
        if visualize and image_dir:
            vis_dir = os.path.join(os.path.dirname(output_mask_dir), "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            print(f"\nGenerating {num_vis_samples} visualizations...")
            visualize_samples(image_dir, output_mask_dir, vis_dir, num_vis_samples)
        
        print("\nPreprocessing complete!")
        return True
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess XML annotations for staff line detection")
    parser.add_argument("--xml_dir", type=str, required=True, help="Directory containing XML annotations")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the preprocessed data")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--verify", action="store_true", help="Verify the dataset after preprocessing")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample images and masks")
    parser.add_argument("--num_vis_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of dataset to use (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Create mask directory
    mask_dir = os.path.join(args.output_dir, "masks")
    
    # Run preprocessing
    preprocess_xmls_to_masks(
        args.xml_dir, 
        mask_dir, 
        args.image_dir, 
        args.num_workers,
        args.verify,
        args.visualize,
        args.num_vis_samples,
        args.subset_fraction
    )