import os
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
import shutil
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
import re

def parse_xml_filename(xml_filename):
    """
    Parse XML filename to extract image base name information.
    Example: 'Parsed_accidental tucking-layout-0-muscima_Page_2.xml' 
    would extract 'accidental tucking' and page number 2
    """
    # Extract the main part between 'Parsed_' and '-layout'
    match = re.match(r'Parsed_(.*?)-layout-\d+-muscima_Page_(\d+)\.xml', xml_filename)
    if match:
        base_name = match.group(1)  # e.g., 'accidental tucking'
        page_num = int(match.group(2))  # e.g., 2
        return base_name, page_num
    return None, None

def find_matching_image(base_name, page_num, data_dir):
    """
    Find the matching image file for a given XML annotation.
    Maps from 'accidental tucking' and page 2 to something like 'accidental tucking-002.png'
    """
    # Format the page number with leading zeros
    formatted_page = f"{page_num:03d}"
    
    # Create potential image filename patterns
    potential_patterns = [
        f"{base_name}-{formatted_page}",  # e.g., "accidental tucking-002"
    ]
    
    # Check for each pattern with different extensions
    for pattern in potential_patterns:
        for ext in [".png", ".jpg", ".jpeg"]:
            # Look for direct matches
            img_path = Path(data_dir) / f"{pattern}{ext}"
            if img_path.exists():
                return img_path
            
            # Try alternative formats if needed
            # Check for files that might have different formatting but similar content
            possible_matches = list(Path(data_dir).glob(f"{base_name}*{formatted_page}*{ext}"))
            if possible_matches:
                return possible_matches[0]
    
    return None

def create_mapping_dict(annotation_dir, data_dir):
    """
    Create a dictionary mapping annotation files to their corresponding image files
    """
    mapping = {}
    unmapped_annotations = []
    
    annotation_files = list(Path(annotation_dir).glob("*.xml"))
    print(f"Creating mapping for {len(annotation_files)} annotation files")
    
    for ann_file in tqdm(annotation_files, desc="Building file mapping"):
        base_name, page_num = parse_xml_filename(ann_file.name)
        
        if base_name is None:
            unmapped_annotations.append(ann_file)
            continue
        
        img_path = find_matching_image(base_name, page_num, data_dir)
        
        if img_path:
            mapping[ann_file] = img_path
        else:
            unmapped_annotations.append(ann_file)
    
    print(f"Successfully mapped {len(mapping)} of {len(annotation_files)} annotation files")
    print(f"Failed to map {len(unmapped_annotations)} annotation files")
    
    return mapping, unmapped_annotations

def create_simplified_dataset(data_dir, annotation_dir, output_dir, class_mapping_file, split_ratio=0.2):
    """
    Create a simplified YOLO dataset with minimal box filtering
    """
    # Load class mapping
    with open(class_mapping_file, 'r') as f:
        class_mapping = json.load(f)
    
    # Handle different formats of class mapping file
    if isinstance(class_mapping, list):
        # Check if it's a list of dictionaries with 'id' and 'name' fields
        if class_mapping and isinstance(class_mapping[0], dict) and 'id' in class_mapping[0] and 'name' in class_mapping[0]:
            # Convert list of dicts to a dict mapping class names to IDs
            class_to_idx = {item['name']: item['id'] for item in class_mapping}
        else:
            # If it's a list of class names, create a dict with 1-based indices
            class_to_idx = {name: i+1 for i, name in enumerate(class_mapping)}
    elif isinstance(class_mapping, dict):
        # If it's already a dict, use it directly
        class_to_idx = class_mapping
    else:
        raise ValueError(f"Unsupported class mapping format: {type(class_mapping)}")
        
    print(f"Loaded {len(class_to_idx)} classes from mapping file")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset structure
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    for split in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Create mapping between annotation files and image files
    file_mapping, unmapped = create_mapping_dict(annotation_dir, data_dir)
    
    # Get paired files
    paired_files = list(file_mapping.items())  # List of (ann_file, img_path) tuples
    
    # Shuffle and split
    random.shuffle(paired_files)
    split_idx = int(len(paired_files) * (1 - split_ratio))
    train_pairs = paired_files[:split_idx]
    val_pairs = paired_files[split_idx:]
    
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Process each split
    for split, file_pairs in [("train", train_pairs), ("val", val_pairs)]:
        for ann_file, img_path in tqdm(file_pairs, desc=f"Processing {split} set"):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            height, width, _ = img.shape
            
            # Copy image to dataset
            out_img_path = os.path.join(images_dir, split, img_path.name)
            shutil.copy(img_path, out_img_path)
            
            # Create label file path (use image stem for consistency)
            out_label_path = os.path.join(labels_dir, split, img_path.stem + ".txt")
            
            # Parse XML
            try:
                tree = ET.parse(ann_file)
                root = tree.getroot()
                
                # Find all Node elements
                nodes = root.findall(".//Node")
                
                # If no nodes found, try other tags
                if not nodes:
                    nodes = root.findall(".//Object") or root.findall(".//object")
                
                # Skip if no nodes found
                if not nodes:
                    continue
                
                # Write labels to file
                with open(out_label_path, "w") as f:
                    valid_boxes = False
                    
                    for node in nodes:
                        # Get class name
                        class_name = None
                        for tag in ["ClassName", "name", "Name", "class"]:
                            elem = node.find(f".//{tag}")
                            if elem is not None:
                                class_name = elem.text
                                break
                        
                        if class_name is None or class_name not in class_to_idx:
                            continue
                        
                        # Get box coordinates
                        box_info = {}
                        
                        # Try Pattern 1: Top, Left, Width, Height
                        for coord, tags in [
                            ("top", ["Top", "top", "ymin"]),
                            ("left", ["Left", "left", "xmin"]),
                            ("width", ["Width", "width"]),
                            ("height", ["Height", "height"])
                        ]:
                            for tag in tags:
                                elem = node.find(f".//{tag}")
                                if elem is not None:
                                    box_info[coord] = float(elem.text)
                                    break
                        
                        # If not all coordinates found, try Pattern 2: bndbox
                        if len(box_info) < 4:
                            bndbox = node.find(".//bndbox")
                            if bndbox is not None:
                                xmin = float(bndbox.find(".//xmin").text if bndbox.find(".//xmin") is not None else 0)
                                ymin = float(bndbox.find(".//ymin").text if bndbox.find(".//ymin") is not None else 0)
                                xmax = float(bndbox.find(".//xmax").text if bndbox.find(".//xmax") is not None else 0)
                                ymax = float(bndbox.find(".//ymax").text if bndbox.find(".//ymax") is not None else 0)
                                
                                box_info = {
                                    "top": ymin,
                                    "left": xmin,
                                    "width": xmax - xmin,
                                    "height": ymax - ymin
                                }
                        
                        # Skip if not all coordinates found
                        if len(box_info) < 4:
                            continue
                        
                        # Skip boxes with zero or negative dimensions
                        if box_info["width"] <= 0 or box_info["height"] <= 0:
                            continue
                        
                        # Convert to YOLO format with extra safeguards
                        try:
                            # Get class index (0-based for YOLO)
                            if class_name in class_to_idx:
                                idx_value = class_to_idx[class_name]
                                # Convert to integer if it's a string
                                if isinstance(idx_value, str):
                                    idx_value = int(idx_value)
                                # Convert to 0-based index for YOLO
                                class_idx = idx_value - 1
                            else:
                                # Skip if class name not in mapping
                                continue
                            
                            # Calculate center coordinates and normalized dimensions
                            center_x = min(max((box_info["left"] + box_info["width"] / 2) / width, 0), 1)
                            center_y = min(max((box_info["top"] + box_info["height"] / 2) / height, 0), 1)
                            norm_width = min(max(box_info["width"] / width, 0), 1)
                            norm_height = min(max(box_info["height"] / height, 0), 1)
                            
                            # Write to file
                            f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                            valid_boxes = True
                        except Exception as e:
                            print(f"Error processing box: {e}")
                            continue
            except Exception as e:
                print(f"Error processing {ann_file}: {e}")
                continue
    
    # Create dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    
    # Create class names dictionary for YOLO format
    names_dict = {}
    for name, idx in class_to_idx.items():
        # Convert to 0-based index for YOLO
        idx_zero_based = idx - 1 if isinstance(idx, int) else int(idx) - 1
        names_dict[idx_zero_based] = name
    
    dataset_config = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": names_dict
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    # Check dataset
    train_labels = list(Path(os.path.join(labels_dir, "train")).glob("*.txt"))
    val_labels = list(Path(os.path.join(labels_dir, "val")).glob("*.txt"))
    
    # Count non-empty label files
    non_empty_train = 0
    for label_file in train_labels:
        with open(label_file, 'r') as f:
            if f.read().strip():
                non_empty_train += 1
    
    non_empty_val = 0
    for label_file in val_labels:
        with open(label_file, 'r') as f:
            if f.read().strip():
                non_empty_val += 1
    
    print(f"\nDataset Creation Results:")
    print(f"Train labels: {len(train_labels)} files, {non_empty_train} non-empty")
    print(f"Val labels: {len(val_labels)} files, {non_empty_val} non-empty")
    print(f"Dataset saved to: {output_dir}")
    print(f"Config file: {yaml_path}")
    
    # Save mapping information for debugging
    mapping_info = {
        "total_annotations": len(list(Path(annotation_dir).glob("*.xml"))),
        "successfully_mapped": len(file_mapping),
        "unmapped": len(unmapped),
        "unmapped_files": [str(f.name) for f in unmapped[:10]]  # First 10 for brevity
    }
    
    with open(os.path.join(output_dir, "mapping_info.json"), "w") as f:
        json.dump(mapping_info, f, indent=2)
    
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Simplified YOLO dataset creator")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory with XML annotations")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Create dataset
    create_simplified_dataset(
        args.data_dir,
        args.annotation_dir,
        args.output_dir,
        args.class_mapping,
        args.val_ratio
    )

if __name__ == "__main__":
    main()