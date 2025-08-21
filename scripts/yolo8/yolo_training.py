import os
import json
import yaml
import argparse
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm
import random

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def analyze_xml_annotations(annotation_dir, min_box_size=5, sample_size=100):
    """
    Analyze XML annotations to understand the dataset
    
    Args:
        annotation_dir (str): Directory containing XML annotation files
        min_box_size (int): Minimum bounding box size to consider
        sample_size (int): Number of files to sample for analysis
    """
    annotation_files = list(Path(annotation_dir).glob("*.xml"))
    if len(annotation_files) == 0:
        print(f"No XML files found in {annotation_dir}")
        return
    
    # Sample files for analysis
    if len(annotation_files) > sample_size:
        sample_files = random.sample(annotation_files, sample_size)
    else:
        sample_files = annotation_files
    
    print(f"Analyzing {len(sample_files)} sample annotation files...")
    
    # Statistics
    total_files = 0
    total_objects = 0
    files_with_objects = 0
    class_counts = {}
    small_boxes = 0
    classes_with_small_boxes = set()
    
    for ann_file in tqdm(sample_files):
        total_files += 1
        file_has_objects = False
        
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            
            # Try different node patterns
            for node_tag in ["Node", "Object", "object"]:
                nodes = root.findall(f".//{node_tag}")
                if nodes:
                    for node in nodes:
                        # Try different class name patterns
                        class_name = None
                        for class_tag in ["ClassName", "name", "Name", "class"]:
                            class_elem = node.find(f".//{class_tag}")
                            if class_elem is not None:
                                class_name = class_elem.text
                                break
                        
                        if class_name is None:
                            continue
                        
                        # Count classes
                        if class_name not in class_counts:
                            class_counts[class_name] = 0
                        class_counts[class_name] += 1
                        
                        total_objects += 1
                        file_has_objects = True
                        
                        # Check bounding box size
                        box_found = False
                        box_width = box_height = 0
                        
                        # Try different bbox patterns
                        # Pattern 1: Direct Top, Left, Width, Height
                        if not box_found:
                            top_elem = node.find(".//Top") or node.find(".//top")
                            left_elem = node.find(".//Left") or node.find(".//left")
                            width_elem = node.find(".//Width") or node.find(".//width")
                            height_elem = node.find(".//Height") or node.find(".//height")
                            
                            if all([top_elem, left_elem, width_elem, height_elem]):
                                box_width = float(width_elem.text)
                                box_height = float(height_elem.text)
                                box_found = True
                        
                        # Pattern 2: bndbox with xmin, ymin, xmax, ymax
                        if not box_found:
                            bndbox = node.find(".//bndbox")
                            if bndbox is not None:
                                xmin_elem = bndbox.find(".//xmin")
                                ymin_elem = bndbox.find(".//ymin")
                                xmax_elem = bndbox.find(".//xmax")
                                ymax_elem = bndbox.find(".//ymax")
                                
                                if all([xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                                    box_width = float(xmax_elem.text) - float(xmin_elem.text)
                                    box_height = float(ymax_elem.text) - float(ymin_elem.text)
                                    box_found = True
                        
                        if box_found and (box_width < min_box_size or box_height < min_box_size):
                            small_boxes += 1
                            classes_with_small_boxes.add(class_name)
        
        except Exception as e:
            print(f"Error parsing {ann_file}: {e}")
            continue
        
        if file_has_objects:
            files_with_objects += 1
    
    # Print statistics
    print(f"\nAnnotation Analysis Results:")
    print(f"Total files analyzed: {total_files}")
    print(f"Files with objects: {files_with_objects} ({files_with_objects/total_files*100:.1f}%)")
    print(f"Total objects: {total_objects}")
    print(f"Average objects per file: {total_objects/total_files:.1f}")
    print(f"Small boxes (< {min_box_size}px): {small_boxes} ({small_boxes/total_objects*100:.1f}%)")
    print(f"Classes with small boxes: {len(classes_with_small_boxes)}")
    
    # Print top 10 classes by frequency
    print("\nTop 10 classes by frequency:")
    top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for class_name, count in top_classes:
        print(f"  {class_name}: {count} ({count/total_objects*100:.1f}%)")
    
    # Print classes with small boxes
    if classes_with_small_boxes:
        print("\nClasses with small boxes:")
        for class_name in sorted(classes_with_small_boxes):
            print(f"  {class_name}")
    
    return {
        "total_files": total_files,
        "files_with_objects": files_with_objects,
        "total_objects": total_objects,
        "small_boxes": small_boxes,
        "class_counts": class_counts
    }

def check_yaml_dataset(yaml_path):
    """
    Check YAML dataset file for issues
    
    Args:
        yaml_path (str): Path to YAML dataset file
    """
    if not os.path.exists(yaml_path):
        print(f"YAML file not found: {yaml_path}")
        return False
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['path', 'train', 'val', 'names']
        for key in required_keys:
            if key not in data:
                print(f"Missing required key in YAML: {key}")
                return False
        
        # Check paths exist
        train_path = os.path.join(data['path'], data['train'])
        val_path = os.path.join(data['path'], data['val'])
        
        if not os.path.exists(train_path):
            print(f"Train path not found: {train_path}")
            return False
        
        if not os.path.exists(val_path):
            print(f"Val path not found: {val_path}")
            return False
        
        # Check class names
        if not data['names'] or not isinstance(data['names'], dict):
            print(f"Invalid class names format in YAML")
            return False
        
        # Check if class IDs are consecutive integers starting from 0
        class_ids = sorted(int(k) for k in data['names'].keys())
        expected_ids = list(range(len(class_ids)))
        if class_ids != expected_ids:
            print(f"Warning: Class IDs are not consecutive integers starting from 0")
            print(f"Expected: {expected_ids[:10]}...")
            print(f"Found: {class_ids[:10]}...")
        
        print(f"YAML check passed: {yaml_path}")
        print(f"  Classes: {len(data['names'])}")
        print(f"  Train path: {train_path}")
        print(f"  Val path: {val_path}")
        
        return True
        
    except Exception as e:
        print(f"Error checking YAML file: {e}")
        return False

def check_label_files(dataset_dir):
    """
    Check label files for issues
    
    Args:
        dataset_dir (str): Directory containing the dataset
    """
    labels_dir = os.path.join(dataset_dir, "labels")
    train_dir = os.path.join(labels_dir, "train")
    val_dir = os.path.join(labels_dir, "val")
    
    # Check train labels
    train_labels = list(Path(train_dir).glob("*.txt"))
    empty_train_labels = 0
    
    for label_file in train_labels:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if not content:
                empty_train_labels += 1
    
    # Check val labels
    val_labels = list(Path(val_dir).glob("*.txt"))
    empty_val_labels = 0
    
    for label_file in val_labels:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            if not content:
                empty_val_labels += 1
    
    print(f"\nLabel Files Analysis:")
    print(f"Train labels: {len(train_labels)}")
    print(f"Empty train labels: {empty_train_labels} ({empty_train_labels/len(train_labels)*100:.1f}% if {len(train_labels)} > 0)")
    print(f"Val labels: {len(val_labels)}")
    print(f"Empty val labels: {empty_val_labels} ({empty_val_labels/len(val_labels)*100:.1f}% if {len(val_labels)} > 0)")
    
    # Return True if all seems OK
    return (empty_val_labels < len(val_labels))

def create_yolo_dataset(data_dir, annotation_dir, output_dir, class_mapping_file, 
                       split_ratio=0.2, min_box_size=5, min_size_multiplier=1.5,
                       debug=False):
    """
    Create YOLO format dataset from Doremi dataset
    
    Args:
        data_dir (str): Directory with all the images
        annotation_dir (str): Directory with XML annotation files
        output_dir (str): Directory to save YOLO format dataset
        class_mapping_file (str): Path to JSON file with class mapping
        split_ratio (float): Validation set ratio
        min_box_size (int): Minimum size for bounding boxes
        min_size_multiplier (float): Multiplier for minimum size when checking for filtering
        debug (bool): Enable debug mode
    """
    # Run analysis if in debug mode
    if debug:
        analyze_xml_annotations(annotation_dir, min_box_size)
    
    # Load class mapping
    with open(class_mapping_file, 'r') as f:
        class_to_idx = json.load(f)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset structure
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    for split in ["train", "val"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    # Get XML annotation files
    annotation_files = list(Path(annotation_dir).glob("*.xml"))
    
    if len(annotation_files) == 0:
        raise ValueError(f"No XML files found in {annotation_dir}. Please check the directory path.")
    
    # Shuffle files
    np.random.shuffle(annotation_files)
    
    # Split files
    split_idx = int(len(annotation_files) * (1 - split_ratio))
    train_files = annotation_files[:split_idx]
    val_files = annotation_files[split_idx:]
    
    print(f"Found {len(annotation_files)} total annotation files")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Track statistics
    total_processed = 0
    total_boxes = 0
    total_filtered = 0
    valid_image_count = 0
    empty_label_files = 0
    
    # Process each annotation file
    for split, files in [("train", train_files), ("val", val_files)]:
        for ann_file in tqdm(files, desc=f"Processing {split} set"):
            processed_boxes = 0
            filtered_boxes = 0
            
            # Try different image extensions
            img_file = None
            for ext in [".png", ".jpg", ".jpeg"]:
                potential_img = Path(data_dir) / (ann_file.stem + ext)
                if potential_img.exists():
                    img_file = potential_img
                    break
            
            if img_file is None:
                if debug:
                    print(f"Warning: No image found for {ann_file.stem}")
                continue
            
            # Read image to get dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                if debug:
                    print(f"Warning: Could not read image {img_file}")
                continue
                
            height, width, _ = img.shape
            
            # Create output image and label paths
            out_img_path = os.path.join(images_dir, split, img_file.name)
            out_label_path = os.path.join(labels_dir, split, ann_file.stem + ".txt")
            
            # Copy image
            shutil.copy(img_file, out_img_path)
            
            # Parse annotation
            try:
                tree = ET.parse(ann_file)
                root = tree.getroot()
                
                # Check for different XML structures
                nodes = []
                
                # Try different node patterns
                for node_tag in ["Node", "Object", "object"]:
                    nodes = root.findall(f".//{node_tag}")
                    if nodes:
                        break
                
                if not nodes:
                    if debug:
                        print(f"Warning: No nodes found in {ann_file}")
                    empty_label_files += 1
                    # Still create an empty label file
                    with open(out_label_path, "w") as f:
                        pass
                    continue
                
                # Open label file for writing
                valid_boxes = 0
                with open(out_label_path, "w") as f:
                    for node in nodes:
                        processed_boxes += 1
                        
                        # Try different class name patterns
                        class_name = None
                        for class_tag in ["ClassName", "name", "Name", "class"]:
                            class_elem = node.find(f".//{class_tag}")
                            if class_elem is not None:
                                class_name = class_elem.text
                                break
                        
                        if class_name is None:
                            filtered_boxes += 1
                            continue
                        
                        # Skip if class not in mapping
                        if class_name not in class_to_idx:
                            filtered_boxes += 1
                            continue
                        
                        # Get class index (0-based for YOLO)
                        class_idx = class_to_idx[class_name] - 1
                        
                        # Try different bounding box patterns
                        box_found = False
                        
                        # Pattern 1: Direct Top, Left, Width, Height
                        if not box_found:
                            top_elem = node.find(".//Top") or node.find(".//top")
                            left_elem = node.find(".//Left") or node.find(".//left")
                            width_elem = node.find(".//Width") or node.find(".//width")
                            height_elem = node.find(".//Height") or node.find(".//height")
                            
                            if all([top_elem, left_elem, width_elem, height_elem]):
                                top = float(top_elem.text)
                                left = float(left_elem.text)
                                box_width = float(width_elem.text)
                                box_height = float(height_elem.text)
                                box_found = True
                        
                        # Pattern 2: bndbox with xmin, ymin, xmax, ymax
                        if not box_found:
                            bndbox = node.find(".//bndbox")
                            if bndbox is not None:
                                xmin_elem = bndbox.find(".//xmin")
                                ymin_elem = bndbox.find(".//ymin")
                                xmax_elem = bndbox.find(".//xmax")
                                ymax_elem = bndbox.find(".//ymax")
                                
                                if all([xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                                    left = float(xmin_elem.text)
                                    top = float(ymin_elem.text)
                                    box_width = float(xmax_elem.text) - left
                                    box_height = float(ymax_elem.text) - top
                                    box_found = True
                        
                        # Pattern 3: bbox attribute
                        if not box_found:
                            bbox = node.get("bbox")
                            if bbox:
                                try:
                                    left, top, box_width, box_height = map(float, bbox.split(','))
                                    box_found = True
                                except:
                                    pass
                        
                        if not box_found:
                            filtered_boxes += 1
                            continue
                        
                        # Higher threshold for validation set to ensure good quality examples
                        threshold = min_box_size
                        if split == "val":
                            threshold = min_box_size * min_size_multiplier
                        
                        # Skip small boxes
                        if box_width < threshold or box_height < threshold:
                            filtered_boxes += 1
                            continue
                        
                        # Convert to YOLO format (normalized center x, center y, width, height)
                        center_x = (left + box_width / 2) / width
                        center_y = (top + box_height / 2) / height
                        norm_width = box_width / width
                        norm_height = box_height / height
                        
                        # Check for invalid values
                        if (center_x < 0 or center_x > 1 or 
                            center_y < 0 or center_y > 1 or
                            norm_width <= 0 or norm_width > 1 or
                            norm_height <= 0 or norm_height > 1):
                            filtered_boxes += 1
                            continue
                        
                        # Write to label file
                        f.write(f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                        valid_boxes += 1
                
                # Update statistics
                total_boxes += processed_boxes
                total_filtered += filtered_boxes
                if valid_boxes > 0:
                    valid_image_count += 1
                else:
                    empty_label_files += 1
                
            except Exception as e:
                if debug:
                    print(f"Error processing {ann_file}: {e}")
                continue
            
            total_processed += 1
    
    # Count the number of labels created
    train_labels = list(Path(os.path.join(labels_dir, "train")).glob("*.txt"))
    val_labels = list(Path(os.path.join(labels_dir, "val")).glob("*.txt"))
    
    print(f"\nDataset Creation Statistics:")
    print(f"Created {len(train_labels)} train labels and {len(val_labels)} val labels")
    print(f"Images with valid annotations: {valid_image_count} ({valid_image_count/total_processed*100:.1f}%)")
    print(f"Total boxes processed: {total_boxes}")
    print(f"Boxes filtered out: {total_filtered} ({total_filtered/total_boxes*100:.1f}%)")
    print(f"Empty label files (no valid boxes): {empty_label_files}")
    
    # Check if enough valid labels were created
    if len(train_labels) == 0 or len(val_labels) == 0:
        raise ValueError("No valid labels were created. Please check the annotation files and minimum box size.")
    
    # Create dataset.yaml
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    
    # Map class indices to names (0-based)
    names_dict = {class_to_idx[name] - 1: name for name in class_to_idx}
    
    dataset_config = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": names_dict
    }
    
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset prepared at {output_dir}")
    print(f"Dataset config saved to {yaml_path}")
    
    # Verify YAML file
    check_yaml_dataset(yaml_path)
    
    # Verify label files
    check_label_files(output_dir)
    
    return yaml_path

def train_yolo(dataset_yaml, output_dir, model="yolov8s.pt", epochs=30, batch_size=16, 
               imgsz=640, device=0, conf_threshold=0.001, 
               workers=8, verbose=False):
    """
    Train YOLOv8 on custom dataset with error handling
    
    Args:
        dataset_yaml (str): Path to dataset.yaml file
        output_dir (str): Directory to save training results
        model (str): YOLOv8 model to use
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        imgsz (int): Image size
        device (int): GPU device ID
        conf_threshold (float): Confidence threshold
        workers (int): Number of worker threads
        verbose (bool): Enable verbose output
    """
    try:
        from ultralytics import YOLO
        
        # Verify YAML before starting training
        if not check_yaml_dataset(dataset_yaml):
            print("YAML validation failed, aborting training")
            return None
        
        print(f"Starting YOLOv8 training with {model} for {epochs} epochs")
        
        # Load YOLO model
        yolo = YOLO(model)
        
        # Setup training arguments
        train_args = {
            "data": dataset_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": imgsz,
            "device": device,
            "project": output_dir,
            "name": "yolo_train",
            "exist_ok": True,
            "conf": conf_threshold,  # Low confidence threshold to avoid empty detection lists
            "verbose": verbose,
            "workers": workers,
            "patience": 50,  # Early stopping patience
            "seed": 42,  # Fixed seed for reproducibility
        }
        
        # Train model
        try:
            results = yolo.train(**train_args)
            print(f"Training completed. Results saved to {os.path.join(output_dir, 'yolo_train')}")
            return os.path.join(output_dir, "yolo_train", "weights", "best.pt")
        except Exception as e:
            print(f"Error training model: {e}")
            return None
    
    except ImportError:
        print("Error: ultralytics package not installed. Please install it with: pip install ultralytics")
        return None

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Doremi dataset with improved error handling")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory containing XML annotations")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    
    # Dataset parameters
    parser.add_argument("--min_box_size", type=int, default=5, help="Minimum size for bounding boxes")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    
    # Training parameters
    parser.add_argument("--model", type=str, default="yolov8s.pt", 
                       help="YOLOv8 model to use (options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--img_size", type=int, default=640, help="Image size")
    parser.add_argument("--conf_threshold", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    
    # Device parameters
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    
    # Execution parameters
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip dataset preprocessing")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose YOLOv8 output")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze dataset, don't train")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command-line arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Set random seed
    set_seed(42)
    
    # If only analyzing, run analysis and exit
    if args.analyze_only:
        analyze_xml_annotations(args.annotation_dir, args.min_box_size)
        return
    
    # Create dataset
    dataset_dir = os.path.join(args.output_dir, "dataset")
    dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")
    
    if not args.skip_preprocessing or not os.path.exists(dataset_yaml):
        try:
            dataset_yaml = create_yolo_dataset(
                args.data_dir,
                args.annotation_dir,
                dataset_dir,
                args.class_mapping,
                args.val_ratio,
                args.min_box_size,
                debug=args.debug
            )
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return
    else:
        print(f"Using existing dataset at {dataset_yaml}")
        
        # Still verify the dataset
        if args.debug:
            check_yaml_dataset(dataset_yaml)
            check_label_files(dataset_dir)
    
    # Train model
    model_path = train_yolo(
        dataset_yaml,
        args.output_dir,
        args.model,
        args.epochs,
        args.batch_size,
        args.img_size,
        args.gpu_id,
        args.conf_threshold,
        args.workers,
        args.verbose
    )
    
    if model_path:
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()