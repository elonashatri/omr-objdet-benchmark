import os
import sys
import json
import yaml
import torch
import argparse
import numpy as np
import cv2
import random
import time
import csv
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh
from ultralytics.utils.torch_utils import select_device


def bbox_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes [x1, y1, x2, y2]
    """
    # Get coordinates of intersection
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    # Calculate area of intersection
    w = max(0, x2_min - x1_max)
    h = max(0, y2_min - y1_max)
    intersection = w * h
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    if union > 0:
        return intersection / union
    else:
        return 0.0


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate multiple YOLOv8 models on test data')
    
    # Model directories
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='List of directories containing model weights to evaluate')
    
    # Default dataset parameters
    parser.add_argument('--default_data_yaml', type=str, 
                        default='/homes/es314/omr-objdet-benchmark/data/yolo-9654-data-splits/dataset.yaml',
                        help='Default path to dataset.yaml if not specified in args.yaml')
    
    parser.add_argument('--special_data_yaml', type=str, 
                        default='/homes/es314/omr-objdet-benchmark/data/202-24classes-yolo-9654-data-splits/dataset.yaml',
                        help='Special dataset for certain models')
                        
    # Add MUSCIMA++ parameters
    parser.add_argument('--muscima_data_dir', type=str, 
                        default=None,
                        help='Path to MUSCIMA++ dataset directory')
                        
    parser.add_argument('--muscima_mapping', type=str, 
                        default=None,
                        help='Path to MUSCIMA++ class mapping JSON file (optional)')
    
    parser.add_argument('--muscima_annotation_dir', type=str, 
                        default=None,
                        help='Path to MUSCIMA++ annotations directory (XML files)')
    
    # Retain DOREMI parameters for backward compatibility
    parser.add_argument('--doremi_data_dir', type=str, 
                        default=None,
                        help='Path to DOREMI dataset directory for DOREMI model')
                        
    parser.add_argument('--doremi_mapping', type=str, 
                        default=None,
                        help='Path to DOREMI class mapping JSON file')
    
    parser.add_argument('--special_models', type=str, nargs='+', 
                        default=['train3-yolo-9654-data-splits', 'train-202-24classes-yolo-9654-data-splits'],
                        help='Model names that should use the special dataset')
    
    # Add parameter to identify MUSCIMA++ models
    parser.add_argument('--muscima_models', type=str, nargs='+', 
                        default=['muscima', 'muscima-pp'],
                        help='Model names that should use MUSCIMA++ dataset')
    
    # Evaluation parameters
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of test images to evaluate per model (0 for all)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for considering a detection as correct')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for detections')
    parser.add_argument('--map_iou_thresholds', type=str, default='0.5,0.75',
                        help='IoU thresholds for mAP calculation, comma-separated')
    
    # Output parameters
    parser.add_argument('--save_visualization', action='store_true',
                        help='Save detection visualizations')
    parser.add_argument('--vis_dir', type=str, default='yolo_evaluation_visualizations',
                        help='Directory to save visualizations')
    
    # Class-specific confidence thresholds
    parser.add_argument('--use_class_conf_thresholds', action='store_true', default=True,
                        help='Use class-specific confidence thresholds for difficult classes')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use (e.g., "0" for GPU 0, "cpu" for CPU)')
    
    # Random seed for reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for image selection')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_model_info(model_dir, default_data_yaml, special_data_yaml, special_models, 
                   muscima_models=None, muscima_data_dir=None, muscima_mapping=None,
                   doremi_data_dir=None, doremi_mapping=None):
    """
    Load model information including args.yaml and model weights
    Also determine the dataset.yaml file for this specific model
    """
    model_info = {}
    
    # Extract model name from directory
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_dir)))
    model_subfolder = os.path.basename(os.path.dirname(model_dir))
    if model_subfolder != "weights":
        model_name = f"{model_name}/{model_subfolder}"
    model_info['name'] = model_name
    
    # Find model file
    model_path = os.path.join(model_dir, 'best.pt')
    if not os.path.exists(model_path) and '/' in model_dir:
        # Try with explicitly provided file
        model_basename = os.path.basename(model_dir)
        model_dirname = os.path.dirname(model_dir)
        model_path = os.path.join(model_dirname, model_basename)
    
    if not os.path.exists(model_path):
        print(f"Warning: Could not find model file at {model_path}")
        model_info['model_path'] = None
        return model_info
    
    model_info['model_path'] = model_path
    
    # Load args.yaml if it exists
    args_path = os.path.join(os.path.dirname(model_dir), 'args.yaml')
    if os.path.exists(args_path):
        try:
            with open(args_path, 'r') as f:
                model_info['args'] = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not parse args.yaml in {os.path.dirname(model_dir)}: {e}")
            model_info['args'] = {}
    else:
        print(f"Warning: Could not find args.yaml in {os.path.dirname(model_dir)}")
        model_info['args'] = {}
    
    # Determine dataset info based on model name
    
    # Check if this is a MUSCIMA++ model
    if muscima_models and muscima_data_dir and any(muscima_name in model_name for muscima_name in muscima_models):
        print(f"Using MUSCIMA++ dataset for {model_name}")
        model_info['data_yaml'] = None  
        model_info['is_muscima'] = True
        model_info['is_doremi'] = False
        model_info['muscima_data_dir'] = muscima_data_dir
        model_info['muscima_mapping'] = muscima_mapping
    # Check if this is the DOREMI model (keeping for backward compatibility)
    elif doremi_data_dir and "doremiv1-94classes" in model_name:
        print(f"Using DOREMI dataset for {model_name}")
        model_info['data_yaml'] = None
        model_info['is_doremi'] = True
        model_info['is_muscima'] = False
        model_info['doremi_data_dir'] = doremi_data_dir
        model_info['doremi_mapping'] = doremi_mapping
    # Check if this model should use the special dataset
    elif any(special_name in model_name for special_name in special_models):
        model_info['data_yaml'] = special_data_yaml
        model_info['is_doremi'] = False
        model_info['is_muscima'] = False
        print(f"Using special dataset for {model_name}: {special_data_yaml}")
    else:
        model_info['data_yaml'] = default_data_yaml
        model_info['is_doremi'] = False
        model_info['is_muscima'] = False
        print(f"Using default dataset for {model_name}: {default_data_yaml}")
    
    return model_info


def load_muscima_class_mapping(mapping_file=None):
    """
    Load class mapping for MUSCIMA++ dataset
    If no mapping file is provided, create a default mapping from class names
    Supports multiple formats:
    1. JSON format (like DOREMI) - can be either {id: name} or {name: id}
    2. TensorFlow Object Detection API pbtxt-like format
    """
    if mapping_file and os.path.exists(mapping_file):
        try:
            # Determine file format based on extension
            if mapping_file.endswith('.json'):
                # Try JSON format first
                with open(mapping_file, 'r') as f:
                    mapping = json.load(f)
                
                # Check format: {id: name} or {name: id}
                if mapping and isinstance(list(mapping.keys())[0], str):
                    # Format is {name: id}, convert to {id: name}
                    if all(isinstance(v, int) for v in mapping.values()):
                        print(f"Detected {name: id} format, converting to {id: name} format")
                        mapping = {v: k for k, v in mapping.items()}
                
                print(f"Loaded class mapping from {mapping_file} with {len(mapping)} classes")
                return mapping
            else:
                # Try TensorFlow Object Detection API pbtxt-like format
                mapping = {}
                with open(mapping_file, 'r') as f:
                    content = f.read()
                
                # Parse item blocks
                import re
                item_blocks = re.findall(r'item\s*\{[^}]*\}', content)
                for item in item_blocks:
                    # Extract id and name
                    id_match = re.search(r'id:\s*(\d+)', item)
                    name_match = re.search(r"name:\s*'([^']*)'", item)
                    
                    if id_match and name_match:
                        class_id = int(id_match.group(1))
                        class_name = name_match.group(1)
                        mapping[class_id] = class_name
                
                if mapping:
                    print(f"Loaded {len(mapping)} classes from pbtxt format file: {mapping_file}")
                    return mapping
                else:
                    raise ValueError("Could not parse class mapping from file")
        except Exception as e:
            print(f"Error loading mapping from {mapping_file}: {e}")
            # Try to read the file line by line for debugging
            try:
                with open(mapping_file, 'r') as f:
                    lines = f.readlines()
                print(f"First 10 lines of the file:")
                for i, line in enumerate(lines[:10]):
                    print(f"{i+1}: {line.strip()}")
            except:
                pass
    
    # Default: Create mapping based on discovered class names
    print("No mapping file provided or failed to load, will create dynamically from annotations")
    return {}


def load_dataset_info(model_info, muscima_annotation_dir=None):
    """Load dataset information based on model info"""
    
    # Handle MUSCIMA++ dataset specially
    if model_info.get('is_muscima', False):
        muscima_data_dir = model_info.get('muscima_data_dir')
        muscima_mapping_file = model_info.get('muscima_mapping')
        
        if not muscima_data_dir:
            print("Error: MUSCIMA++ model requires muscima_data_dir path")
            return {
                'dataset_dir': '',
                'test_dir': '',
                'class_names': {},
                'annotation_dir': muscima_annotation_dir
            }
        
        # Load MUSCIMA++ class mapping
        class_mapping = load_muscima_class_mapping(muscima_mapping_file)
        
        # If we have a mapping file, use it directly
        if class_mapping and isinstance(class_mapping, dict):
            if all(k.isdigit() for k in class_mapping.keys()):
                # If keys are strings of digits, convert to int keys
                class_names = {int(k): v for k, v in class_mapping.items()}
            else:
                # If keys are already class names, invert the mapping
                class_names = {i: name for i, name in enumerate(sorted(set(class_mapping.values())))}
        else:
            # Initialize empty - we'll populate from annotations later
            class_names = {}
                
        # Set paths for MUSCIMA++ dataset
        dataset_dir = muscima_data_dir
        test_dir = os.path.join(muscima_data_dir, 'images/test')  # Adjust based on your structure
        if not os.path.exists(test_dir):
            test_dir = os.path.join(muscima_data_dir, 'images/val')  # Fallback to val set
            if not os.path.exists(test_dir):
                test_dir = os.path.join(muscima_data_dir, 'images')  # Final fallback
        
        annotation_dir = muscima_annotation_dir or os.path.join(muscima_data_dir, 'annotations')
        
        print(f"MUSCIMA++ dataset dir: {dataset_dir}")
        print(f"MUSCIMA++ test dir: {test_dir}")
        print(f"MUSCIMA++ annotation dir: {annotation_dir}")
        if class_names:
            print(f"Loaded {len(class_names)} classes for MUSCIMA++")
        
        return {
            'dataset_dir': dataset_dir,
            'test_dir': test_dir,
            'class_names': class_names,
            'annotation_dir': annotation_dir
        }
    
    # Handle DOREMI dataset specially (keeping for backward compatibility)
    if model_info.get('is_doremi', False):
        doremi_data_dir = model_info.get('doremi_data_dir')
        doremi_mapping = model_info.get('doremi_mapping')
        
        if not doremi_data_dir or not doremi_mapping:
            print("Error: DOREMI model requires doremi_data_dir and doremi_mapping paths")
            return {
                'dataset_dir': '',
                'test_dir': '',
                'class_names': {}
            }
        
        # Load DOREMI class mapping from JSON file
        try:
            with open(doremi_mapping, 'r') as f:
                class_mapping = json.load(f)
                class_names = {int(k): v for k, v in class_mapping.items() if k.isdigit()}
                
            # Set paths for DOREMI dataset
            dataset_dir = doremi_data_dir
            test_dir = os.path.join(doremi_data_dir, 'images/val')  # Using val set for testing
            
            print(f"Loaded {len(class_names)} classes from DOREMI mapping file: {doremi_mapping}")
            
            return {
                'dataset_dir': dataset_dir,
                'test_dir': test_dir,
                'class_names': class_names
            }
        except Exception as e:
            print(f"Error loading DOREMI mapping from {doremi_mapping}: {e}")
            return {
                'dataset_dir': doremi_data_dir,
                'test_dir': os.path.join(doremi_data_dir, 'images/val'),
                'class_names': {}
            }
    
    # Regular YAML-based datasets
    data_yaml_path = model_info.get('data_yaml')
    if not data_yaml_path:
        print("Error: No data YAML path specified")
        return {
            'dataset_dir': '',
            'test_dir': '',
            'class_names': {}
        }
    
    try:
        with open(data_yaml_path, 'r') as f:
            data_info = yaml.safe_load(f)
        
        # Get dataset root directory
        dataset_dir = data_info.get('path', os.path.dirname(data_yaml_path))
        
        # Get test directory
        test_dir = data_info.get('test', 'images/test')
        if not os.path.isabs(test_dir):
            test_dir = os.path.join(dataset_dir, test_dir)
        
        # Get class names - handle both dictionary and list formats
        class_names = data_info.get('names', {})
        
        # Convert list format to dictionary if needed
        if isinstance(class_names, list):
            class_names = {i: name for i, name in enumerate(class_names)}
        
        print(f"Loaded {len(class_names)} classes from {data_yaml_path}")
        
        return {
            'dataset_dir': dataset_dir,
            'test_dir': test_dir,
            'class_names': class_names
        }
    except Exception as e:
        print(f"Error loading dataset info from {data_yaml_path}: {e}")
        return {
            'dataset_dir': os.path.dirname(data_yaml_path),
            'test_dir': os.path.join(os.path.dirname(data_yaml_path), 'images/test'),
            'class_names': {}
        }


def load_test_dataset(test_dir, num_images=0, seed=42):
    """Load test dataset from directory"""
    # Get image paths
    image_paths = []
    if os.path.exists(test_dir):
        image_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_paths:
        print(f"Warning: No images found in {test_dir}")
        return []
    
    # Select subset of images if requested
    if num_images > 0 and num_images < len(image_paths):
        random.seed(seed)
        image_paths = random.sample(image_paths, num_images)
    
    print(f"Loaded {len(image_paths)} test images")
    return image_paths


def get_muscima_annotation_path(image_path, annotation_dir):
    """Get the corresponding annotation file path for an image in MUSCIMA++"""
    image_name = os.path.basename(image_path)
    image_base = os.path.splitext(image_name)[0]
    
    # Check for direct XML file match
    annotation_path = os.path.join(annotation_dir, f"{image_base}.xml")
    if os.path.exists(annotation_path):
        return annotation_path
    
    # Try to find by matching image name in XML files
    for xml_file in os.listdir(annotation_dir):
        if xml_file.endswith('.xml'):
            full_path = os.path.join(annotation_dir, xml_file)
            try:
                tree = ET.parse(full_path)
                root = tree.getroot()
                # Check if the XML contains the filename we're looking for
                filename_elem = root.find('.//filename')
                if filename_elem is not None and image_name in filename_elem.text:
                    return full_path
            except Exception as e:
                print(f"Error checking XML file {xml_file}: {e}")
    
    # No matching annotation found
    return None


def parse_muscima_annotation(annotation_path, class_mapping=None):
    """Parse MUSCIMA++ XML annotation file and return bounding boxes"""
    if not os.path.exists(annotation_path):
        return np.zeros((0, 5)), {}
    
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Get image size
        size_elem = root.find('./size')
        if size_elem is None:
            print(f"Warning: No size information in {annotation_path}")
            img_width, img_height = 1, 1  # Default to prevent division by zero
        else:
            img_width = int(size_elem.find('width').text)
            img_height = int(size_elem.find('height').text)
        
        # Extract all class names if we don't have a mapping
        discovered_classes = set()
        
        # Process each object
        annotations = []
        for obj in root.findall('./object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
                
            class_name = name_elem.text
            discovered_classes.add(class_name)
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
                
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Convert to YOLO format (x_center, y_center, width, height)
            x_center = (xmin + xmax) / (2 * img_width)
            y_center = (ymin + ymax) / (2 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # Assign class ID
            if class_mapping is not None and isinstance(class_mapping, dict):
                # If we have a mapping dict, try to use it
                if class_name in class_mapping:
                    # If mapping is name -> id
                    class_id = class_mapping[class_name]
                elif class_name in class_mapping.values():
                    # If mapping is id -> name, find the id
                    for k, v in class_mapping.items():
                        if v == class_name:
                            class_id = int(k)
                            break
                    else:
                        # Not found, assign a new ID
                        class_id = len(discovered_classes) - 1
                else:
                    # Not in mapping, assign a new ID
                    class_id = len(discovered_classes) - 1
            else:
                # If no mapping, assign sequential IDs as we discover classes
                class_id = len(discovered_classes) - 1
            
            # Add to annotations [class_id, x_center, y_center, width, height, confidence]
            annotations.append([class_id, x_center, y_center, width, height, 1.0])
        
        # Create or update the class mapping
        updated_mapping = {}
        if not class_mapping or not isinstance(class_mapping, dict):
            # Create new mapping
            for i, name in enumerate(sorted(discovered_classes)):
                updated_mapping[i] = name
        else:
            # Use existing mapping and extend if needed
            updated_mapping = class_mapping.copy()
            # Add any new classes
            if isinstance(next(iter(class_mapping.keys())), int):
                # If mapping is id -> name
                next_id = max(class_mapping.keys()) + 1 if class_mapping else 0
                for name in discovered_classes:
                    if name not in class_mapping.values():
                        updated_mapping[next_id] = name
                        next_id += 1
            else:
                # If mapping is name -> id, convert to id -> name
                id_to_name = {v: k for k, v in class_mapping.items()}
                next_id = max(id_to_name.keys()) + 1 if id_to_name else 0
                for name in discovered_classes:
                    if name not in class_mapping:
                        updated_mapping[name] = next_id
                        next_id += 1
        
        return np.array(annotations), updated_mapping
    
    except Exception as e:
        print(f"Error parsing MUSCIMA++ annotation {annotation_path}: {e}")
        return np.zeros((0, 5)), {}


def load_yolo_annotations(image_path):
    """Load YOLO format annotations for an image"""
    # Convert image path to label path
    label_path = image_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
    
    # Check if label file exists
    if not os.path.exists(label_path):
        return np.zeros((0, 5))  # Return empty array if no annotations
    
    # Load annotations
    try:
        with open(label_path, 'r') as f:
            annotations = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    confidence = 1.0  # Ground truth has perfect confidence
                    
                    annotations.append([class_id, x_center, y_center, width, height, confidence])
            
            return np.array(annotations)
    except Exception as e:
        print(f"Error loading annotations from {label_path}: {e}")
        return np.zeros((0, 5))


def load_annotations(image_path, dataset_info):
    """
    Load annotations for an image based on dataset type
    Returns annotations in YOLO format [class_id, x_center, y_center, width, height, confidence]
    """
    # Check if this is MUSCIMA++ format
    if 'annotation_dir' in dataset_info and dataset_info['annotation_dir']:
        # Get XML annotation path
        annotation_path = get_muscima_annotation_path(image_path, dataset_info['annotation_dir'])
        if annotation_path:
            # Parse MUSCIMA++ annotation
            annotations, updated_mapping = parse_muscima_annotation(
                annotation_path, 
                dataset_info.get('class_names', {})
            )
            
            # Update class_names if we discovered new classes
            if updated_mapping:
                dataset_info['class_names'].update(updated_mapping)
                
            return annotations
        else:
            print(f"Warning: No MUSCIMA++ annotation found for {image_path}")
            return np.zeros((0, 6))
    
    # Default: try to load YOLO format annotation
    return load_yolo_annotations(image_path)


class DummyConfusionMatrix:
    """Dummy class for when ConfusionMatrix fails to initialize"""
    def __init__(self, nc=1):
        self.nc = nc
        print("Using dummy confusion matrix.")
    
    def process_batch(self, *args, **kwargs):
        pass
    
    def matrix(self):
        return torch.zeros((self.nc, self.nc))
    
    def tp_fp_fn(self):
        # Return dummy values for tp, fp, fn, precision, recall, f1
        tp = torch.zeros(self.nc)
        fp = torch.zeros(self.nc)
        fn = torch.zeros(self.nc)
        p = torch.zeros(self.nc)
        r = torch.zeros(self.nc)
        f1 = torch.zeros(self.nc)
        return tp, fp, fn, p, r, f1


def run_evaluation(model, test_dataset, dataset_info, device, conf_threshold=0.3, iou_thresholds=None, use_class_conf_thresholds=True):
    """Evaluate model on test dataset with mAP calculation"""
    if model is None:
        print("Error: Model is None")
        return None
    
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    
    # Get class names
    class_names = dataset_info.get('class_names', {})
    num_classes = len(class_names)
    if num_classes == 0:
        print("Warning: No classes found in dataset info")
        # Try to infer from model
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'nc'):
                num_classes = model.model.nc
            elif hasattr(model, 'names'):
                # Some versions of YOLOv8 store class names directly
                class_names = model.names
                if isinstance(class_names, list):
                    class_names = {i: name for i, name in enumerate(class_names)}
                num_classes = len(class_names)
            else:
                print("Could not infer number of classes from model")
                num_classes = 1
                
            # If we only got the number but not the names
            if len(class_names) == 0:
                class_names = {i: f"class_{i}" for i in range(num_classes)}
        except Exception as e:
            print(f"Error inferring classes from model: {e}")
            num_classes = 1
            class_names = {0: "object"}
    
    print(f"Evaluating model with {num_classes} classes")
    
    # Define custom confidence thresholds for potentially problematic classes
    # Using a shared dictionary for common music notation classes that often need lower thresholds
    class_conf_thresholds = {}
    
    # Map class names to custom thresholds - these are examples based on common notation elements
    name_to_threshold = {
        # Generic symbols across datasets
        'stem': 0.05,
        'kStaffLine': 0.05,
        'barline': 0.07,
        'systemicBarline': 0.05,
        'augmentationDot': 0.10,
        'articTenutoBelow': 0.05,
        'articTenutoAbove': 0.05,
        'articStaccatoAbove': 0.07,
        'articStaccatoBelow': 0.07,
        'T3': 0.20,
        'tupletBracket': 0.10,
        'restHalf': 0.20,
        
        # DOREMI/Custom model classes
        'noteheadBlack': 0.15,
        'beam': 0.10,
        'accidentalFlat': 0.15,
        'accidentalSharp': 0.15,
        'restWhole': 0.15,
        'rest8th': 0.15,
        'slur': 0.08,
        'noteheadHalf': 0.15,
        'accidentalNatural': 0.15,
        'tie': 0.08,
        'gClef': 0.15,
        'restQuarter': 0.15,
        'rest16th': 0.15,
        'flag8thUp': 0.10,
        'flag8thDown': 0.10,
        'articStaccatoAbove': 0.10,
        'flag16thUp': 0.10,
        'fClef': 0.15,
        'noteheadWhole': 0.15,
        'tupletBracket': 0.10,
        'flag32ndUp': 0.10,
        'cClef': 0.15,
        'flag16thDown': 0.10,
        
        # MUSCIMA++ specific classes
        'notehead-full': 0.15,
        'ledger_line': 0.05,
        'staff_line': 0.05,
        'duration-dot': 0.10,
        'curved-line_(tie-or-slur)': 0.08,
        'thin_barline': 0.07,
        'thick_barline': 0.07,
        'measure_separator': 0.07,
        '8th_flag': 0.10,
        '16th_flag': 0.10,
        'quarter_rest': 0.15,
        '8th_rest': 0.15,
        '16th_rest': 0.15,
        'half_rest': 0.15,
        'whole_rest': 0.15,
        'tuple': 0.10,
        'tuple_bracketline': 0.08,
        'staccato-dot': 0.10
    }
    
    # Set up class-specific confidence thresholds if enabled
    if use_class_conf_thresholds:
        for class_id, class_name in class_names.items():
            # Check if the class name is in our predefined list
            if class_name in name_to_threshold:
                class_conf_thresholds[class_id] = name_to_threshold[class_name]
        
        # Print the custom thresholds if any were applied
        if class_conf_thresholds:
            print("\nUsing custom confidence thresholds for the following classes:")
            for cls_id, threshold in class_conf_thresholds.items():
                print(f"  Class {cls_id} ({class_names.get(cls_id, 'unknown')}): {threshold:.3f}")
    
    # Initialize per-class detection results storage for mAP calculation
    all_detections = defaultdict(list)  # class_id -> [detection1, detection2, ...]
    all_groundtruth = defaultdict(list)  # class_id -> [gt1, gt2, ...]
    
    # Initialize confusion matrix with error handling
    try:
        # Try different parameter names for compatibility
        try:
            confusion_matrix = ConfusionMatrix(nc=num_classes, conf=conf_threshold, iou_thres=iou_thresholds[0])
        except TypeError:
            try:
                # Try with different parameter names
                confusion_matrix = ConfusionMatrix(nc=num_classes, conf_thres=conf_threshold, iou_thres=iou_thresholds[0])
            except TypeError:
                # Last resort: minimal parameters
                confusion_matrix = ConfusionMatrix(nc=num_classes)
    except Exception as e:
        print(f"Warning: Could not initialize confusion matrix: {e}")
        # Create a dummy confusion matrix object to avoid errors
        confusion_matrix = DummyConfusionMatrix(nc=num_classes)
    
    # Initialize metrics storage
    stats = []
    
    # Basic metrics structure
    results = {
        'overall': {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        },
        'per_class': defaultdict(lambda: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0
        }),
        'mAP': {}
    }
    
    # Initialize mAP results
    for iou_threshold in iou_thresholds:
        results['mAP'][iou_threshold] = {
            'per_class': {},
            'overall': 0.0
        }
    
    # Process each image
    for idx, image_path in enumerate(tqdm(test_dataset, desc="Evaluating")):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Load ground truth annotations (using the appropriate format)
        gt_annotations = load_annotations(image_path, dataset_info)
        
        # Run inference with class-specific confidence thresholds if enabled
        # Use the minimum confidence threshold to catch all potential detections
        min_conf = min(class_conf_thresholds.values()) if class_conf_thresholds else conf_threshold
        results_list = model(image_path, conf=min_conf, verbose=False)
        
        # Extract predictions
        predictions = results_list[0]
        
        # Get bounding boxes (convert to proper format if needed)
        try:
            # Try new YOLOv8 format
            pred_boxes = predictions.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            pred_scores = predictions.boxes.conf.cpu().numpy()
            pred_labels = predictions.boxes.cls.cpu().numpy().astype(int)
        except AttributeError:
            try:
                # Try older YOLOv8 format
                pred_boxes = predictions.xyxy[0].cpu().numpy()[:, :4]  # [x1, y1, x2, y2]
                pred_scores = predictions.xyxy[0].cpu().numpy()[:, 4]
                pred_labels = predictions.xyxy[0].cpu().numpy()[:, 5].astype(int)
            except (AttributeError, IndexError):
                try:
                    # Try alternative format - direct numpy access
                    if hasattr(predictions, 'numpy'):
                        preds = predictions.numpy()
                        if len(preds) > 0:
                            pred_boxes = preds[:, :4]
                            pred_scores = preds[:, 4]
                            pred_labels = preds[:, 5].astype(int)
                        else:
                            pred_boxes = np.zeros((0, 4))
                            pred_scores = np.array([])
                            pred_labels = np.array([], dtype=int)
                    else:
                        # Last attempt - try different YOLOv8 attributes
                        pred_boxes = np.zeros((0, 4))
                        pred_scores = np.array([])
                        pred_labels = np.array([], dtype=int)
                        
                        # Check if any detections were made
                        if hasattr(predictions, 'boxes') and len(predictions.boxes) > 0:
                            pred_boxes = predictions.boxes.data.cpu().numpy()[:, :4]
                            pred_scores = predictions.boxes.data.cpu().numpy()[:, 4]
                            pred_labels = predictions.boxes.data.cpu().numpy()[:, 5].astype(int)
                except Exception as e:
                    print(f"Could not extract predictions for {image_path}: {e}")
                    continue
                    
        # Apply class-specific confidence thresholds if enabled
        if class_conf_thresholds:
            # Create mask to filter detections
            mask = np.zeros_like(pred_scores, dtype=bool)
            for i, (label, score) in enumerate(zip(pred_labels, pred_scores)):
                # Get class-specific threshold or use default
                threshold = class_conf_thresholds.get(int(label), conf_threshold)
                # Mark detection for keeping if score exceeds threshold
                if score >= threshold:
                    mask[i] = True
            
            # Apply mask to filter predictions
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
        else:
            # Apply global confidence threshold
            mask = pred_scores >= conf_threshold
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
        
        # Process ground truth
        if gt_annotations.size > 0:
            # Convert from YOLO format [class, x_center, y_center, width, height] to [x1, y1, x2, y2]
            gt_boxes_yolo = gt_annotations[:, 1:5]  # [x_center, y_center, width, height]
            gt_boxes = np.zeros_like(gt_boxes_yolo)
            
            # Convert to absolute pixel coordinates
            gt_boxes[:, 0] = (gt_boxes_yolo[:, 0] - gt_boxes_yolo[:, 2] / 2) * width  # x1
            gt_boxes[:, 1] = (gt_boxes_yolo[:, 1] - gt_boxes_yolo[:, 3] / 2) * height  # y1
            gt_boxes[:, 2] = (gt_boxes_yolo[:, 0] + gt_boxes_yolo[:, 2] / 2) * width  # x2
            gt_boxes[:, 3] = (gt_boxes_yolo[:, 1] + gt_boxes_yolo[:, 3] / 2) * height  # y2
            
            gt_labels = gt_annotations[:, 0].astype(int)
        else:
            gt_boxes = np.zeros((0, 4))
            gt_labels = np.zeros(0, dtype=int)
        
        # Get image ID for mAP calculation
        image_id = os.path.basename(image_path)
        
        # Store detections for mAP
        for i in range(len(pred_boxes)):
            class_id = pred_labels[i]
            class_name = class_names.get(class_id, f"class_{class_id}")
            all_detections[class_name].append({
                'image_id': image_id,
                'bbox': pred_boxes[i].tolist(),
                'score': pred_scores[i]
            })
        
        # Store ground truth for mAP
        for i in range(len(gt_boxes)):
            class_id = gt_labels[i]
            class_name = class_names.get(class_id, f"class_{class_id}")
            all_groundtruth[class_name].append({
                'image_id': image_id,
                'bbox': gt_boxes[i].tolist(),
                'used': False  # To track which ground truth boxes are matched during mAP calculation
            })
        
        # Update confusion matrix 
        # Older versions of Ultralytics used different parameter names
        try:
            # Try the newer API format
            confusion_matrix.process_batch(
                detections=torch.from_numpy(np.hstack((pred_boxes, pred_scores.reshape(-1, 1), pred_labels.reshape(-1, 1)))),
                gt=torch.from_numpy(np.hstack((gt_boxes, gt_labels.reshape(-1, 1))))
            )
        except TypeError:
            try:
                # Try with old API or different parameter names
                confusion_matrix.process_batch(
                    predn=torch.from_numpy(np.hstack((pred_boxes, pred_scores.reshape(-1, 1), pred_labels.reshape(-1, 1)))),
                    targets=torch.from_numpy(np.hstack((gt_boxes, gt_labels.reshape(-1, 1))))
                )
            except Exception as e:
                print(f"Warning: Failed to update confusion matrix: {e}")
                # Continue processing even if confusion matrix fails
    
    # Try to compute confusion matrix results using different API versions
    try:
        # Try newer API
        if hasattr(confusion_matrix, 'tp_fp_fn'):
            tp, fp, fn, p, r, f1 = confusion_matrix.tp_fp_fn()
        # Try older API
        elif hasattr(confusion_matrix, 'matrix'):
            # Extract metrics manually from confusion matrix
            matrix = confusion_matrix.matrix
            if isinstance(matrix, np.ndarray):
                matrix = torch.from_numpy(matrix)
            
            tp = torch.diag(matrix)  # True positives: diagonal elements
            fp = matrix.sum(dim=0) - tp  # False positives: column sum - true positives
            fn = matrix.sum(dim=1) - tp  # False negatives: row sum - true positives
            
            # Calculate precision, recall, and F1
            p = torch.zeros_like(tp)
            r = torch.zeros_like(tp)
            f1 = torch.zeros_like(tp)
            
            # Avoid division by zero
            valid = (tp + fp) > 0
            p[valid] = tp[valid] / (tp[valid] + fp[valid])
            
            valid = (tp + fn) > 0
            r[valid] = tp[valid] / (tp[valid] + fn[valid])
            
            valid = (p + r) > 0
            f1[valid] = 2 * (p[valid] * r[valid]) / (p[valid] + r[valid])
        else:
            # Fallback: derive metrics from all_detections and all_groundtruth
            print("Warning: Could not compute metrics from confusion matrix. Using manual calculation from detections.")
            
            # Initialize metrics
            num_classes = len(class_names)
            tp = np.zeros(num_classes)
            fp = np.zeros(num_classes)
            fn = np.zeros(num_classes)
            
            # Calculate TP, FP, FN per class from detections/ground truth
            for i, class_id in enumerate(sorted(class_names.keys())):
                class_name = class_names.get(class_id, f"class_{class_id}")
                
                # Calculate TPs and FPs for this class at IoU 0.5
                if class_name in all_detections and class_name in all_groundtruth:
                    class_tp = 0
                    class_fp = 0
                    
                    # Reset "used" flag for all ground truths
                    for gt in all_groundtruth[class_name]:
                        gt['used'] = False
                    
                    # Sort detections by confidence
                    sorted_dets = sorted(all_detections[class_name], key=lambda x: x['score'], reverse=True)
                    
                    # Get ground truth by image
                    gt_by_img = {}
                    for gt in all_groundtruth[class_name]:
                        img_id = gt['image_id']
                        if img_id not in gt_by_img:
                            gt_by_img[img_id] = []
                        gt_by_img[img_id].append(gt)
                    
                    # Process each detection
                    for det in sorted_dets:
                        img_id = det['image_id']
                        if img_id not in gt_by_img:
                            class_fp += 1
                            continue
                        
                        # Check against all ground truths in this image
                        max_iou = 0
                        best_gt = None
                        
                        for gt in gt_by_img[img_id]:
                            if gt['used']:
                                continue
                            
                            # Calculate IoU
                            iou = bbox_iou(det['bbox'], gt['bbox'])
                            if iou > max_iou:
                                max_iou = iou
                                best_gt = gt
                        
                        # Check if it's a match
                        if max_iou >= 0.5 and best_gt is not None:
                            class_tp += 1
                            best_gt['used'] = True
                        else:
                            class_fp += 1
                    
                    # Count false negatives (ground truths without a match)
                    class_fn = sum(1 for gt in all_groundtruth[class_name] if not gt.get('used', False))
                    
                    # Store metrics
                    tp[i] = class_tp
                    fp[i] = class_fp
                    fn[i] = class_fn
            
            # Convert to torch tensors
            tp = torch.tensor(tp)
            fp = torch.tensor(fp)
            fn = torch.tensor(fn)
            
            # Calculate precision, recall, and F1
            p = torch.zeros_like(tp, dtype=torch.float)
            r = torch.zeros_like(tp, dtype=torch.float)
            f1 = torch.zeros_like(tp, dtype=torch.float)
            
            # Avoid division by zero
            valid = (tp + fp) > 0
            p[valid] = tp[valid].float() / (tp[valid].float() + fp[valid].float())
            
            valid = (tp + fn) > 0
            r[valid] = tp[valid].float() / (tp[valid].float() + fn[valid].float())
            
            valid = (p + r) > 0
            f1[valid] = 2 * (p[valid] * r[valid]) / (p[valid] + r[valid])
    except Exception as e:
        print(f"Error computing metrics: {e}")
        # Initialize empty tensors as fallback
        num_classes = len(class_names)
        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)
        p = torch.zeros(num_classes)
        r = torch.zeros(num_classes)
        f1 = torch.zeros(num_classes)
    
    # Calculate metrics per class and overall
    for i, class_id in enumerate(sorted(class_names.keys())):
        if i < len(tp):  # Make sure we don't go out of bounds
            class_name = class_names.get(class_id, f"class_{class_id}")
            results['per_class'][class_name]['true_positives'] = tp[i].item()
            results['per_class'][class_name]['false_positives'] = fp[i].item()
            results['per_class'][class_name]['false_negatives'] = fn[i].item()
            results['per_class'][class_name]['precision'] = p[i].item()
            results['per_class'][class_name]['recall'] = r[i].item()
            results['per_class'][class_name]['f1_score'] = f1[i].item()
    
    # Calculate overall metrics
    results['overall']['true_positives'] = tp.sum().item()
    results['overall']['false_positives'] = fp.sum().item()
    results['overall']['false_negatives'] = fn.sum().item()
    results['overall']['precision'] = p.mean().item()
    results['overall']['recall'] = r.mean().item()
    results['overall']['f1_score'] = f1.mean().item()
    
    # Calculate mAP for different IoU thresholds
    for iou_threshold in iou_thresholds:
        # Calculate AP per class
        class_aps = {}
        
        for class_name in all_groundtruth.keys():
            try:
                ap = calculate_ap(
                    all_detections[class_name],
                    all_groundtruth[class_name],
                    iou_threshold
                )
                class_aps[class_name] = ap
            except Exception as e:
                print(f"Error calculating AP for class {class_name} at IoU {iou_threshold}: {e}")
                class_aps[class_name] = 0.0
        
        # Calculate mAP (mean of all class APs)
        if class_aps:
            map_score = sum(class_aps.values()) / len(class_aps)
        else:
            map_score = 0.0
        
        # Store results
        results['mAP'][iou_threshold]['per_class'] = class_aps
        results['mAP'][iou_threshold]['overall'] = map_score
    
    # Calculate mAP@[.5:.95] (COCO-style)
    if len(iou_thresholds) > 1:
        mAP_values = [results['mAP'][iou]['overall'] for iou in iou_thresholds]
        results['mAP']['average'] = sum(mAP_values) / len(mAP_values)
    
    return results


def calculate_ap(detections, ground_truths, iou_threshold):
    """
    Calculate Average Precision (AP) for a single class
    
    Args:
        detections: List of detection dictionaries with 'image_id', 'bbox', 'score'
        ground_truths: List of ground truth dictionaries with 'image_id', 'bbox'
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        AP: Average Precision score
    """
    if not detections or not ground_truths:
        return 0.0
    
    # Create a copy of ground truths to mark matches
    gt_by_img = defaultdict(list)
    for i, gt in enumerate(ground_truths):
        gt_by_img[gt['image_id']].append({
            'bbox': gt['bbox'],
            'matched': False,
            'index': i
        })
    
    # Sort detections by confidence score (descending)
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    # Initialize precision/recall arrays
    num_gt = len(ground_truths)
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    
    # Process each detection
    for i, detection in enumerate(detections):
        img_id = detection['image_id']
        pred_bbox = detection['bbox']
        
        # If no ground truth in this image, it's a false positive
        if img_id not in gt_by_img or not gt_by_img[img_id]:
            fp[i] = 1
            continue
        
        # Calculate IoU with all unmatched ground truths in this image
        max_iou = -float('inf')
        max_idx = -1
        
        for j, gt in enumerate(gt_by_img[img_id]):
            if gt['matched']:
                continue
                
            gt_bbox = gt['bbox']
            
            # Calculate IoU
            ix1 = max(pred_bbox[0], gt_bbox[0])
            iy1 = max(pred_bbox[1], gt_bbox[1])
            ix2 = min(pred_bbox[2], gt_bbox[2])
            iy2 = min(pred_bbox[3], gt_bbox[3])
            
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih
            
            pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
            gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            union = pred_area + gt_area - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou > max_iou:
                max_iou = iou
                max_idx = j
        
        # If IoU exceeds threshold, it's a true positive
        if max_iou >= iou_threshold:
            gt_by_img[img_id][max_idx]['matched'] = True
            tp[i] = 1
        else:
            fp[i] = 1
    
    # Compute cumulative precision and recall
    cumsum_tp = np.cumsum(tp)
    cumsum_fp = np.cumsum(fp)
    recalls = cumsum_tp / num_gt if num_gt > 0 else np.zeros_like(cumsum_tp)
    precisions = np.divide(cumsum_tp, (cumsum_tp + cumsum_fp), out=np.zeros_like(cumsum_tp), where=(cumsum_tp + cumsum_fp) > 0)
    
    # Add sentinel values for integral calculation
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))
    
    # Ensure precision decreases monotonically
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Compute AP as the area under the PR curve
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap


def save_results_to_csv(model_results, output_dir, include_per_class=False):
    """Save evaluation results to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare summary results (without per-class metrics)
    timestamp = int(time.time())
    summary_file = os.path.join(output_dir, f'yolo_evaluation_summary-{timestamp}.csv')
    
    # Define headers for summary file
    summary_headers = [
        'Model Name', 'Model Path', 'Test Dataset', 'Test Images', 
        'IoU Threshold', 'Conf Threshold', 'Precision', 'Recall', 
        'F1 Score', 'True Positives', 'False Positives', 'False Negatives', 
        'mAP@0.5', 'mAP@0.75', 'mAP@[.5:.95]', 'Num Classes'
    ]
    
    # Write summary results
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        
        for model_name, result in model_results.items():
            if not result.get('results'):
                continue
                
            # Get mAP values
            mAP_50 = result['results']['mAP'].get(0.5, {}).get('overall', 0.0)
            mAP_75 = result['results']['mAP'].get(0.75, {}).get('overall', 0.0)
            mAP_avg = result['results']['mAP'].get('average', 0.0)
                
            row = {
                'Model Name': model_name,
                'Model Path': result.get('model_path', ''),
                'Test Dataset': result.get('test_dir', ''),
                'Test Images': result.get('num_images', ''),
                'IoU Threshold': result.get('iou_threshold', ''),
                'Conf Threshold': result.get('conf_threshold', ''),
                'Precision': f"{result['results']['overall']['precision']:.4f}",
                'Recall': f"{result['results']['overall']['recall']:.4f}",
                'F1 Score': f"{result['results']['overall']['f1_score']:.4f}",
                'True Positives': result['results']['overall']['true_positives'],
                'False Positives': result['results']['overall']['false_positives'],
                'False Negatives': result['results']['overall']['false_negatives'],
                'mAP@0.5': f"{mAP_50:.4f}",
                'mAP@0.75': f"{mAP_75:.4f}",
                'mAP@[.5:.95]': f"{mAP_avg:.4f}",
                'Num Classes': len(result.get('class_names', {}))
            }
            writer.writerow(row)
    
    print(f"Summary results saved to: {summary_file}")
    
    # Save per-class results if requested
    if include_per_class:
        for model_name, result in model_results.items():
            if not result.get('results'):
                continue
                
            # Create per-class results file for this model
            per_class_file = os.path.join(output_dir, f'{model_name.replace("/", "-")}_per_class_results-{timestamp}.csv')
            
            # Define headers for per-class file
            per_class_headers = [
                'Class Name', 'Precision', 'Recall', 'F1 Score', 
                'True Positives', 'False Positives', 'False Negatives',
                'AP@0.5', 'AP@0.75'
            ]
            
            with open(per_class_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=per_class_headers)
                writer.writeheader()
                
                # Get per-class metrics
                per_class_metrics = result['results']['per_class']
                mAP_50_per_class = result['results']['mAP'].get(0.5, {}).get('per_class', {})
                mAP_75_per_class = result['results']['mAP'].get(0.75, {}).get('per_class', {})
                
                # Sort classes by F1 score (descending)
                sorted_classes = sorted(
                    per_class_metrics.items(),
                    key=lambda x: x[1]['f1_score'],
                    reverse=True
                )
                
                for class_name, metrics in sorted_classes:
                    row = {
                        'Class Name': class_name,
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1 Score': f"{metrics['f1_score']:.4f}",
                        'True Positives': metrics['true_positives'],
                        'False Positives': metrics['false_positives'],
                        'False Negatives': metrics['false_negatives'],
                        'AP@0.5': f"{mAP_50_per_class.get(class_name, 0.0):.4f}",
                        'AP@0.75': f"{mAP_75_per_class.get(class_name, 0.0):.4f}"
                    }
                    writer.writerow(row)
            
            print(f"Per-class results for {model_name} saved to: {per_class_file}")


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = select_device(args.device)
    print(f"Using device: {device}")
    
    # Parse mAP IoU thresholds
    map_iou_thresholds = [float(t) for t in args.map_iou_thresholds.split(',')]
    print(f"Using mAP IoU thresholds: {map_iou_thresholds}")
    
    # Process model directories
    model_dirs = []
    for model_dir in args.model_dirs:
        if os.path.isdir(model_dir):
            # Check if this is a weights directory
            if os.path.basename(model_dir) == "weights":
                # This is already a weights directory
                model_dirs.append(model_dir)
            else:
                # Look for weights directory
                weights_dir = os.path.join(model_dir, "weights")
                if os.path.exists(weights_dir):
                    model_dirs.append(weights_dir)
                else:
                    print(f"Warning: Could not find weights directory in {model_dir}")
        else:
            # This is a direct path to a model file
            model_dirs.append(model_dir)
    
    # Evaluate each model
    model_results = {}
    
    for model_dir in model_dirs:
        # Get model base name
        if '/' in model_dir:
            model_name = '/'.join(model_dir.split('/')[-3:-1])
        else:
            model_name = os.path.basename(model_dir)
        
        print(f"\n===== Evaluating model: {model_name} =====")
        
        # Load model info, including dataset yaml path
        model_info = load_model_info(
            model_dir, 
            args.default_data_yaml,
            args.special_data_yaml,
            args.special_models,
            args.muscima_models,
            args.muscima_data_dir,
            args.muscima_mapping,
            args.doremi_data_dir,
            args.doremi_mapping
        )
        
        if not model_info.get('model_path'):
            print(f"Skipping {model_name}: No model file found")
            continue
        
        # Load dataset info
        dataset_info = load_dataset_info(model_info, args.muscima_annotation_dir)
        
        # Load test dataset
        test_dir = dataset_info['test_dir']
        print(f"Loading test dataset from {test_dir}")
        test_dataset = load_test_dataset(test_dir, args.num_images, args.seed)
        
        if not test_dataset:
            print(f"Skipping {model_name}: No test images found")
            continue
        
        # Load model
        try:
            print(f"Loading model from {model_info['model_path']}")
            model = YOLO(model_info['model_path'])
            model.to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
        
        # Run evaluation
        results = run_evaluation(
            model, 
            test_dataset, 
            dataset_info,
            device,
            conf_threshold=args.conf_threshold,
            iou_thresholds=map_iou_thresholds,
            use_class_conf_thresholds=args.use_class_conf_thresholds
        )
        
        if results is None:
            print(f"Skipping {model_name}: Evaluation failed")
            continue
        
        # Store results
        model_results[model_name] = {
            'results': results,
            'model_info': model_info,
            'model_path': model_info['model_path'],
            'test_dir': test_dir,
            'num_images': len(test_dataset),
            'iou_threshold': args.iou_threshold,
            'conf_threshold': args.conf_threshold,
            'class_names': dataset_info.get('class_names', {})
        }
        
        # Print summary
        overall = results['overall']
        mAP_50 = results['mAP'].get(0.5, {}).get('overall', 0.0)
        mAP_75 = results['mAP'].get(0.75, {}).get('overall', 0.0)
        mAP_avg = results['mAP'].get('average', 0.0)
        
        print(f"\nResults for {model_name}:")
        print(f"Test dir: {test_dir}")
        print(f"Precision: {overall['precision']:.4f}")
        print(f"Recall: {overall['recall']:.4f}")
        print(f"F1 Score: {overall['f1_score']:.4f}")
        print(f"mAP@0.5: {mAP_50:.4f}")
        print(f"mAP@0.75: {mAP_75:.4f}")
        print(f"mAP@[.5:.95]: {mAP_avg:.4f}")
        print(f"True Positives: {overall['true_positives']}")
        print(f"False Positives: {overall['false_positives']}")
        print(f"False Negatives: {overall['false_negatives']}")
    
    # Save combined results
    output_dir = os.path.dirname(args.model_dirs[0]) if args.model_dirs else '.'
    save_results_to_csv(model_results, output_dir, include_per_class=True)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
    
    
    
    
    
"""    
# For Model B (MUSCIMA++)
python /homes/es314/omr-objdet-benchmark/scripts/yolo8/eval_muscima.py \
  --model_dirs "/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/detect/train11/weights/best.pt" \
  --muscima_mapping "/import/c4dm-05/elona/muscima-doremi-annotation/eval_mapping.json" \
  --muscima_data_dir "/import/c4dm-05/elona/muscima-doremi-annotation" 
  [other parameters]
"""