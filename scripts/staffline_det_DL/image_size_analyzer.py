#!/usr/bin/env python3
"""
Script to analyze image sizes in a directory and print statistics.
"""

import os
import sys
import cv2
import numpy as np
from collections import Counter
from tqdm import tqdm

def analyze_image_sizes(image_dir):
    """
    Analyze all images in a directory and print size statistics.
    
    Args:
        image_dir (str): Directory containing images
    """
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    print(f"Found {len(image_files)} image files in {image_dir}")
    
    # Collect image sizes
    sizes = []
    failed = 0
    
    for img_file in tqdm(image_files, desc="Analyzing image sizes"):
        img_path = os.path.join(image_dir, img_file)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # OpenCV returns height, width for grayscale images
                height, width = img.shape
                sizes.append((height, width))
            else:
                print(f"Warning: Could not read {img_file}")
                failed += 1
        except Exception as e:
            print(f"Error reading {img_file}: {e}")
            failed += 1
    
    # Count frequency of each size
    size_counter = Counter(sizes)
    
    # Report results
    print(f"\nImage Size Analysis Results:")
    print(f"Successfully read {len(sizes)} images")
    print(f"Failed to read {failed} images")
    
    print(f"\nUnique image sizes (height × width):")
    for size, count in size_counter.most_common():
        height, width = size
        percentage = (count / len(sizes)) * 100
        print(f"  {height} × {width}: {count} images ({percentage:.2f}%)")
    
    if len(size_counter) > 0:
        print("\nRecommended --filter_dimensions settings:")
        most_common_size = size_counter.most_common(1)[0][0]
        height, width = most_common_size
        print(f"  --filter_dimensions {height} {width}  # Most common size")
        
        # If there's a second most common size, also report it
        if len(size_counter) > 1:
            second_common = size_counter.most_common(2)[1][0]
            height, width = second_common
            print(f"  --filter_dimensions {height} {width}  # Second most common size")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image_directory>")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    if not os.path.isdir(image_dir):
        print(f"Error: {image_dir} is not a valid directory")
        sys.exit(1)
    
    analyze_image_sizes(image_dir)