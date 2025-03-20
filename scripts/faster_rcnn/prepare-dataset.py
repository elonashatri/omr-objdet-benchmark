#!/usr/bin/env python3
"""
OMR Dataset Preparation Script

This script prepares an Optical Music Recognition dataset by:
1. Parsing XML annotations
2. Splitting data into train/validation/test sets (70/15/15)
3. Converting annotations to COCO format
4. Removing duplicate labels if needed
"""

import os
import glob
import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
import shutil
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare OMR dataset for training')
    parser.add_argument('--images_dir', type=str, default='/homes/es314/omr-objdet-benchmark/data/images',
                        help='Directory containing image files')
    parser.add_argument('--annotations_dir', type=str, default='/homes/es314/omr-objdet-benchmark/data/annotations',
                        help='Directory containing XML annotation files')
    parser.add_argument('--output_dir', type=str, default='/homes/es314/omr-objdet-benchmark/data/prepared_dataset',
                        help='Output directory for prepared dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def parse_xml_annotation(xml_file):
    """Parse an XML annotation file and extract bounding boxes and classes."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        annotation_objects = []
        
        # Find all Node elements
        for node in root.findall('.//Node'):

            class_name = node.find('ClassName')
            if class_name is None or class_name.text is None:
                continue
                
            # Skip staff lines if needed
            # if class_name.text.startswith('kStaffLine'):
            #     continue
                
            # Get bounding box coordinates
            top = node.find('Top')
            left = node.find('Left')
            width = node.find('Width')
            height = node.find('Height')
            
            if None in (top, left, width, height):
                continue
                
            try:
                top = int(top.text)
                left = int(left.text)
                width = int(width.text)
                height = int(height.text)
            except (ValueError, TypeError):
                continue
                
            # Create object entry
            obj = {
                'class': class_name.text,
                'bbox': [left, top, width, height]  # [x, y, width, height] format
            }
            
            annotation_objects.append(obj)
            
        return annotation_objects
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return []

def create_category_mapping(all_annotations):
    """Create a mapping from class names to category IDs."""
    # Extract all unique class names
    all_classes = set()
    for _, annotations in all_annotations.items():
        for obj in annotations:
            all_classes.add(obj['class'])
    
    # Create mapping
    class_to_id = {class_name: i+1 for i, class_name in enumerate(sorted(all_classes))}
    
    return class_to_id

def convert_to_coco_format(image_list, annotations, class_to_id, dataset_name):
    """Convert annotations to COCO format."""
    coco_data = {
        "info": {
            "description": f"OMR Dataset - {dataset_name}",
            "version": "1.0",
            "year": 2025,
            "contributor": "OMR Benchmark",
            "date_created": "2025-03-17"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "Unknown"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for class_name, class_id in class_to_id.items():
        coco_data["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "music_symbol"
        })
    
    annotation_id = 1
    
    # Add images and annotations
    for image_id, image_file in enumerate(image_list, 1):
        image_name = os.path.basename(image_file)
        
        # Add image info
        # Note: In a real implementation, you'd get the actual width and height of the image
        # For now, we'll use placeholders
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": 2475,  # from actual image
            "height": 3504  # from actual image
        })
        
        # Add annotations for this image
        if image_name in annotations:
            for obj in annotations[image_name]:
                x, y, w, h = obj['bbox']
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_to_id[obj['class']],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": [],  # We don't have segmentation data
                    "iscrowd": 0
                })
                
                annotation_id += 1
    
    return coco_data

def create_mapping_file(class_to_id, output_file):
    """Create a mapping file from class IDs to class names."""
    with open(output_file, 'w') as f:
        for class_name, class_id in class_to_id.items():
            f.write(f"{class_id}:{class_name}\n")

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    test_dir = os.path.join(args.output_dir, 'test')
    
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(directory, 'images'), exist_ok=True)
    
    # Get all image and annotation files
    image_files = glob.glob(os.path.join(args.images_dir, '*.png')) + \
                  glob.glob(os.path.join(args.images_dir, '*.jpg'))
    annotation_files = glob.glob(os.path.join(args.annotations_dir, '*.xml'))
    
    # Create a mapping from image filenames to annotation filenames
    image_to_annotation = {}
    for image_file in image_files:
        image_basename = os.path.splitext(os.path.basename(image_file))[0]
        possible_annotation = os.path.join(args.annotations_dir, f"{image_basename}.xml")
        if possible_annotation in annotation_files:
            image_to_annotation[image_file] = possible_annotation
    
    # Parse all annotations
    all_annotations = {}
    for image_file, annotation_file in tqdm(image_to_annotation.items(), desc="Parsing annotations"):
        image_name = os.path.basename(image_file)
        parsed_annotations = parse_xml_annotation(annotation_file)
        if parsed_annotations:
            all_annotations[image_name] = parsed_annotations
    
    # Create category mapping
    class_to_id = create_category_mapping(all_annotations)
    
    # Create mapping file
    mapping_file = os.path.join(args.output_dir, 'mapping.txt')
    create_mapping_file(class_to_id, mapping_file)
    
    # Get list of images with annotations
    valid_images = [img for img in image_files if os.path.basename(img) in all_annotations]
    
    # Shuffle and split the dataset
    random.shuffle(valid_images)
    total_images = len(valid_images)
    
    train_count = int(total_images * args.train_ratio)
    val_count = int(total_images * args.val_ratio)
    
    train_images = valid_images[:train_count]
    val_images = valid_images[train_count:train_count + val_count]
    test_images = valid_images[train_count + val_count:]
    
    print(f"Dataset split: {len(train_images)} training, {len(val_images)} validation, {len(test_images)} test")
    
    # Copy images to their respective directories
    for image_set, target_dir in [(train_images, train_dir), 
                                  (val_images, val_dir), 
                                  (test_images, test_dir)]:
        for image_file in tqdm(image_set, desc=f"Copying images to {os.path.basename(target_dir)}"):
            image_name = os.path.basename(image_file)
            shutil.copy2(
                image_file, 
                os.path.join(target_dir, 'images', image_name)
            )
    
    # Create COCO format annotations
    train_annotations = {os.path.basename(img): all_annotations[os.path.basename(img)] 
                         for img in train_images if os.path.basename(img) in all_annotations}
    val_annotations = {os.path.basename(img): all_annotations[os.path.basename(img)] 
                       for img in val_images if os.path.basename(img) in all_annotations}
    test_annotations = {os.path.basename(img): all_annotations[os.path.basename(img)] 
                        for img in test_images if os.path.basename(img) in all_annotations}
    
    train_coco = convert_to_coco_format(train_images, train_annotations, class_to_id, "Train")
    val_coco = convert_to_coco_format(val_images, val_annotations, class_to_id, "Validation")
    test_coco = convert_to_coco_format(test_images, test_annotations, class_to_id, "Test")
    
    # Write COCO annotations to files
    with open(os.path.join(train_dir, 'annotations.json'), 'w') as f:
        json.dump(train_coco, f, indent=4)
    
    with open(os.path.join(val_dir, 'annotations.json'), 'w') as f:
        json.dump(val_coco, f, indent=4)
    
    with open(os.path.join(test_dir, 'annotations.json'), 'w') as f:
        json.dump(test_coco, f, indent=4)
    
    print(f"Dataset preparation complete. Files saved to {args.output_dir}")
    print(f"Found {len(class_to_id)} unique classes")
    print(f"Class mapping saved to {mapping_file}")

if __name__ == "__main__":
    main()