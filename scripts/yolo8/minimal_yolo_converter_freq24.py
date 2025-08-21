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
from collections import Counter

def create_simplified_dataset(data_dir, annotation_dir, output_dir, class_mapping_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, min_class_count=24):
    """
    Create a simplified YOLO dataset with train/val/test splits
    
    Args:
        data_dir: Directory with images
        annotation_dir: Directory with XML annotations
        output_dir: Output directory
        class_mapping_file: Path to class mapping JSON file
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        min_class_count: Minimum number of instances required for a class to be included (default: 24)
    """
    # Validate split ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Load class mapping
    with open(class_mapping_file, 'r') as f:
        class_to_idx = json.load(f)
    
    # Get XML annotation files
    annotation_files = list(Path(annotation_dir).glob("*.xml"))
    print(f"Found {len(annotation_files)} annotation files")
    
    # First pass: Count class occurrences
    class_counter = Counter()
    
    print("Counting class occurrences...")
    for ann_file in tqdm(annotation_files, desc="Counting classes"):
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
            
            for node in nodes:
                # Get class name
                class_name = None
                for tag in ["ClassName", "name", "Name", "class"]:
                    elem = node.find(f".//{tag}")
                    if elem is not None:
                        class_name = elem.text
                        break
                
                if class_name is not None and class_name in class_to_idx:
                    class_counter[class_name] += 1
                    
        except Exception as e:
            print(f"Error parsing {ann_file}: {e}")
    
    # Filter classes with at least min_class_count instances
    valid_classes = {cls: count for cls, count in class_counter.items() if count >= min_class_count}
    filtered_class_to_idx = {cls: class_to_idx[cls] for cls in valid_classes}
    
    # Create a new contiguous mapping for the filtered classes
    new_class_to_idx = {}
    for i, cls in enumerate(sorted(filtered_class_to_idx.keys(), key=lambda x: filtered_class_to_idx[x])):
        new_class_to_idx[cls] = i + 1  # 1-indexed
    
    print(f"Original classes: {len(class_to_idx)}")
    print(f"Filtered classes (min count {min_class_count}): {len(new_class_to_idx)}")
    
    # Print classes that were removed
    removed_classes = set(class_to_idx.keys()) - set(new_class_to_idx.keys())
    if removed_classes:
        print(f"Removed {len(removed_classes)} classes with fewer than {min_class_count} instances:")
        for cls in sorted(removed_classes):
            print(f"  - {cls}: {class_counter[cls]} instances")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset structure
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    # Add splits to directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Shuffle and split
    random.shuffle(annotation_files)
    train_idx = int(len(annotation_files) * train_ratio)
    val_idx = train_idx + int(len(annotation_files) * val_ratio)
    
    train_files = annotation_files[:train_idx]
    val_files = annotation_files[train_idx:val_idx]
    test_files = annotation_files[val_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    # Save the filtered class mapping to the output directory
    filtered_mapping_path = os.path.join(output_dir, "filtered_class_mapping.json")
    with open(filtered_mapping_path, 'w') as f:
        json.dump(new_class_to_idx, f, indent=2)
    
    # Process each split
    for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        for ann_file in tqdm(files, desc=f"Processing {split} set"):
            # Find image file
            found_image = False
            for ext in [".png", ".jpg", ".jpeg"]:
                img_path = Path(data_dir) / (ann_file.stem + ext)
                if img_path.exists():
                    found_image = True
                    break
            
            if not found_image:
                continue
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            height, width, _ = img.shape
            
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
                
                # Check if there are any valid boxes in filtered classes
                valid_boxes = []
                for node in nodes:
                    # Get class name
                    class_name = None
                    for tag in ["ClassName", "name", "Name", "class"]:
                        elem = node.find(f".//{tag}")
                        if elem is not None:
                            class_name = elem.text
                            break
                    
                    if class_name is None or class_name not in new_class_to_idx:
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
                    
                    # Calculate YOLO format values
                    try:
                        # Get class index (0-based for YOLO)
                        class_idx = new_class_to_idx[class_name] - 1
                        
                        # Calculate center coordinates and normalized dimensions
                        center_x = min(max((box_info["left"] + box_info["width"] / 2) / width, 0), 1)
                        center_y = min(max((box_info["top"] + box_info["height"] / 2) / height, 0), 1)
                        norm_width = min(max(box_info["width"] / width, 0), 1)
                        norm_height = min(max(box_info["height"] / height, 0), 1)
                        
                        valid_boxes.append((class_idx, center_x, center_y, norm_width, norm_height))
                    except Exception as e:
                        print(f"Error processing box: {e}")
                        continue
                
                # Only save images and labels if there are valid boxes
                if valid_boxes:
                    # Copy image to dataset
                    out_img_path = os.path.join(images_dir, split, img_path.name)
                    shutil.copy(img_path, out_img_path)
                    
                    # Create label file path
                    out_label_path = os.path.join(labels_dir, split, ann_file.stem + ".txt")
                    
                    # Write to file
                    with open(out_label_path, "w") as f:
                        for box in valid_boxes:
                            class_idx, center_x, center_y, norm_width, norm_height = box
                            f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                
            except Exception as e:
                print(f"Error processing {ann_file}: {e}")
                continue
    
    # Create dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    
    dataset_config = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {new_class_to_idx[name] - 1: name for name in new_class_to_idx}
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    # Check dataset
    train_labels = list(Path(os.path.join(labels_dir, "train")).glob("*.txt"))
    val_labels = list(Path(os.path.join(labels_dir, "val")).glob("*.txt"))
    test_labels = list(Path(os.path.join(labels_dir, "test")).glob("*.txt"))
    
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
    
    non_empty_test = 0
    for label_file in test_labels:
        with open(label_file, 'r') as f:
            if f.read().strip():
                non_empty_test += 1
    
    print(f"\nDataset Creation Results:")
    print(f"Train labels: {len(train_labels)} files, {non_empty_train} non-empty")
    print(f"Val labels: {len(val_labels)} files, {non_empty_val} non-empty")
    print(f"Test labels: {len(test_labels)} files, {non_empty_test} non-empty")
    print(f"Dataset saved to: {output_dir}")
    print(f"Config file: {yaml_path}")
    print(f"Filtered class mapping: {filtered_mapping_path}")
    
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Simplified YOLO dataset creator with class filtering")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory with XML annotations")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--min_class_count", type=int, default=24, help="Minimum number of instances required for a class")
    
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
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.min_class_count
    )

if __name__ == "__main__":
    main()