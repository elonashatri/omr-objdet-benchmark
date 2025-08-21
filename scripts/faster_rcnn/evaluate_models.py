#!/usr/bin/env python3
"""
Enhanced Evaluation Script for Multiple Faster R-CNN Models on OMR Dataset
This script evaluates multiple trained models on their respective test datasets based on args.json
and generates CSV reports, including Mean Average Precision (mAP) metrics.
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
import cv2
import random
import time
import csv
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# Import from torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
try:
    from torchvision.models.detection import fasterrcnn_resnet101_fpn
    HAS_RESNET101 = True
except ImportError:
    HAS_RESNET101 = False
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from torchvision.ops import box_iou

# Try to import dataset module
try:
    from omr_dataset import OMRDataset, get_transform
    HAS_DATASET = True
except ImportError:
    print("Warning: Could not import omr_dataset module. Will use basic dataset implementation.")
    HAS_DATASET = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate multiple Faster R-CNN models on their test data')
    
    # Model directories
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='List of directories containing model checkpoints to evaluate')
    
    # Default dataset parameters (used only if not found in args.json)
    parser.add_argument('--default_test_dir', type=str, 
                        default='/homes/es314/omr-objdet-benchmark/data/faster_rcnn_prepared_dataset/test',
                        help='Default directory for test data if not specified in args.json')
    parser.add_argument('--default_mapping_file', type=str, 
                        default='/homes/es314/omr-objdet-benchmark/data/faster_rcnn_prepared_dataset/mapping.txt',
                        help='Default path to class mapping file if not specified in args.json')
    
    # Evaluation parameters
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of test images to evaluate per model')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for considering a detection as correct')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for detections')
    parser.add_argument('--map_iou_thresholds', type=str, default='0.5,0.75',
                        help='IoU thresholds for mAP calculation, comma-separated')
    
    # Output parameters
    parser.add_argument('--save_visualization', action='store_true',
                        help='Save detection visualizations')
    parser.add_argument('--vis_dir', type=str, default='evaluation_visualizations',
                        help='Directory to save visualizations')
    
    # Hardware parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    
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


def load_model_info(model_dir, default_test_dir, default_mapping_file):
    """
    Load model information including args.json and model checkpoint
    Also determine the test directory and mapping file for this specific model
    """
    model_info = {}
    
    # Load args.json
    args_path = os.path.join(model_dir, 'args.json')
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            model_info['args'] = json.load(f)
    else:
        print(f"Warning: Could not find args.json in {model_dir}")
        model_info['args'] = {}
    
    # Find best checkpoint
    checkpoint_path = os.path.join(model_dir, 'best.pt')
    if os.path.exists(checkpoint_path):
        model_info['checkpoint_path'] = checkpoint_path
    else:
        # Try to find the latest checkpoint
        checkpoints = [f for f in os.listdir(model_dir) 
                      if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            model_info['checkpoint_path'] = os.path.join(model_dir, latest)
        else:
            print(f"Warning: Could not find checkpoint in {model_dir}")
            model_info['checkpoint_path'] = None
    
    # Extract model name from directory
    model_info['name'] = os.path.basename(model_dir)
    
    # Determine test directory path
    args = model_info['args']
    data_dir = args.get('data_dir', None)
    if data_dir and os.path.exists(os.path.join(data_dir, 'test')):
        test_dir = os.path.join(data_dir, 'test')
        print(f"Using test directory from args.json: {test_dir}")
    else:
        test_dir = default_test_dir
        print(f"Using default test directory: {test_dir}")
    model_info['test_dir'] = test_dir
    
    # Determine mapping file path
    mapping_file = None
    if data_dir and os.path.exists(os.path.join(data_dir, 'mapping.txt')):
        mapping_file = os.path.join(data_dir, 'mapping.txt')
        print(f"Using mapping file from args.json data directory: {mapping_file}")
    else:
        # Try to find mapping file in model directory
        model_mapping = os.path.join(model_dir, 'mapping.txt')
        if os.path.exists(model_mapping):
            mapping_file = model_mapping
            print(f"Using mapping file from model directory: {mapping_file}")
        else:
            mapping_file = default_mapping_file
            print(f"Using default mapping file: {mapping_file}")
    model_info['mapping_file'] = mapping_file
    
    return model_info


def load_model(model_info, device):
    """Load the model from checkpoint using configuration from args.json"""
    args = model_info['args']
    checkpoint_path = model_info['checkpoint_path']
    
    if checkpoint_path is None:
        return None
    
    print(f"Loading model from {checkpoint_path}")
    
    # Extract model parameters from args
    num_classes = args.get('num_classes', 217)
    backbone = args.get('backbone', 'resnet50')
    
    # Parse anchor parameters
    anchor_sizes_str = args.get('anchor_sizes', '16,32,64')
    aspect_ratios_str = args.get('aspect_ratios', '0.1,1.0,2.0,8.0')
    anchor_sizes = tuple([int(size) for size in anchor_sizes_str.split(',')])
    aspect_ratios = tuple([float(ratio) for ratio in aspect_ratios_str.split(',')])
    
    # Parse image size
    image_size_str = args.get('image_size', '500,1000')
    if ',' in image_size_str:
        min_size, max_size = map(int, image_size_str.split(','))
    else:
        min_size = max_size = int(image_size_str)
    
    # Override with explicit min_size and max_size if provided
    min_size = args.get('min_size', min_size)
    max_size = args.get('max_size', max_size)
    
    print(f"Using image size: min={min_size}, max={max_size}")
    print(f"Using anchor sizes: {anchor_sizes}")
    print(f"Using aspect ratios: {aspect_ratios}")
    
    # Create model with the correct backbone
    if backbone == 'resnet50':
        model = fasterrcnn_resnet50_fpn(
            pretrained=False,
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    elif backbone == 'resnet101' and HAS_RESNET101:
        model = fasterrcnn_resnet101_fpn(
            pretrained=False,
            min_size=min_size,
            max_size=max_size
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Modify anchor generator to match training settings
    model.rpn.anchor_generator.sizes = tuple((s,) for s in anchor_sizes)
    model.rpn.anchor_generator.aspect_ratios = tuple((aspect_ratios,) for _ in range(len(anchor_sizes)))
    
    # Set NMS thresholds
    first_stage_nms_iou = args.get('first_stage_nms_iou_threshold', 0.5)
    second_stage_nms_iou = args.get('second_stage_nms_iou_threshold', 0.5)
    
    model.rpn.nms_thresh = first_stage_nms_iou
    model.roi_heads.nms_thresh = second_stage_nms_iou
    model.roi_heads.score_thresh = args.get('second_stage_nms_score_threshold', 0.0)
    model.roi_heads.detections_per_img = args.get('second_stage_max_total_detections', 1600)
    
    # Replace box predictor with the correct number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if the checkpoint is the whole dictionary or just the model
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            epoch = checkpoint.get('epoch', -1)
            print(f"Loaded model from epoch {epoch}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")
        
        # Move model to device
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_class_mapping(mapping_file):
    """Load class mapping from file"""
    id_to_name = {}
    name_to_id = {}
    
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        class_id = int(parts[0])
                        class_name = ':'.join(parts[1:]).strip()
                        id_to_name[class_id] = class_name
                        name_to_id[class_name] = class_id
        
        # Add background class
        id_to_name[0] = 'background'
        name_to_id['background'] = 0
        
        print(f"Loaded {len(id_to_name)-1} classes from {mapping_file}")
        return id_to_name, name_to_id
    except Exception as e:
        print(f"Error loading class mapping: {e}")
        return {}, {}


def load_test_dataset(test_dir, mapping_file, num_images=100, seed=42):
    """Load test dataset compatible with your OMRDataset implementation"""
    global HAS_DATASET  # Add this line to use the global variable
    
    if HAS_DATASET:
        try:
            # Your OMRDataset uses root_dir, is_train parameters
            dataset = OMRDataset(
                root_dir=test_dir,
                transforms=get_transform(train=False),
                is_train=False
            )
            print(f"Created OMRDataset with {len(dataset)} images")
            
            # Select random subset of images
            if num_images and num_images < len(dataset):
                indices = random.sample(range(len(dataset)), num_images)
                dataset = torch.utils.data.Subset(dataset, indices)
                print(f"Selected random subset of {num_images} images")
            
            return dataset
            
        except Exception as e:
            print(f"Error creating OMRDataset: {e}")
            print("Falling back to basic dataset implementation")
            HAS_DATASET = False
    
    # Basic implementation if OMRDataset is not available or failed
    print("Using basic dataset implementation")
    images_dir = os.path.join(test_dir, 'images')
    annotations_file = os.path.join(test_dir, 'annotations.json')
    
    # Load annotations
    annotations = {}
    if os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
            
            # Convert from COCO format to our simplified format
            simplified_annotations = {}
            for ann in annotations.get('annotations', []):
                image_id = ann['image_id']
                image_info = next((img for img in annotations.get('images', []) if img['id'] == image_id), None)
                
                if image_info:
                    image_filename = image_info['file_name']
                    if image_filename not in simplified_annotations:
                        simplified_annotations[image_filename] = []
                    
                    # Convert annotation
                    simplified_annotations[image_filename].append({
                        'bbox': ann['bbox'],
                        'category': annotations.get('categories', [{}])[ann.get('category_id', 0) - 1].get('name', f"Class_{ann.get('category_id', 0)}")
                    })
            
            annotations = simplified_annotations
    
    # Get image paths
    image_paths = []
    if os.path.exists(images_dir):
        image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print(f"Warning: No images found in {images_dir}")
        return []
        
    # Select random subset of images
    if num_images and num_images < len(image_paths):
        random.seed(seed)
        image_paths = random.sample(image_paths, num_images)
    
    # Load class mapping
    id_to_name, name_to_id = load_class_mapping(mapping_file)
    
    # Create simplified dataset
    dataset = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        
        # Get annotations for this image
        image_anns = annotations.get(image_name, [])
        
        # Convert annotations to required format
        target = {
            'boxes': [],
            'labels': [],
            'image_id': image_name
        }
        
        for ann in image_anns:
            if 'bbox' in ann and 'category' in ann:
                bbox = ann['bbox']  # [x, y, width, height] format
                # Convert to [x1, y1, x2, y2] format
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                
                category = ann['category']
                if category in name_to_id:
                    target['boxes'].append(bbox)
                    target['labels'].append(name_to_id[category])
        
        # Convert to tensors if there are annotations
        if target['boxes']:
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros(0, dtype=torch.int64)
        
        dataset.append((image_path, target))
    
    print(f"Created basic dataset with {len(dataset)} images")
    return dataset

def prepare_image(image_path, model_args, device):
    """Prepare image for inference using model configuration"""
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return None, None
    
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    original_dims = (original_height, original_width)
    
    # Extract resize parameters from model args
    min_size = model_args.get('min_size', 500)
    max_size = model_args.get('max_size', 1000)
    
    # Convert to tensor and normalize
    img_tensor = F.to_tensor(image)
    img_tensor = F.normalize(
        img_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Resize to match model input size
    h, w = image.shape[:2]
    min_dim = min(h, w)
    max_dim = max(h, w)
    
    # Calculate scaling factor to maintain aspect ratio
    scale_factor = min(
        min_size / min_dim,
        max_size / max_dim
    )
    
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    if new_h != h or new_w != w:
        img_tensor = F.resize(img_tensor, [new_h, new_w])
    
    # Move to device
    img_tensor = img_tensor.to(device)
    
    # Calculate scaling factors for detection coordinates
    scaling_info = {
        'scaling_factors': (original_width / new_w, original_height / new_h),
        'original_dims': original_dims,
        'processed_dims': (new_h, new_w)
    }
    
    return img_tensor, scaling_info


def run_evaluation(model, dataset, id_to_name, device, conf_threshold=0.3, iou_thresholds=None, model_args=None):
    """Evaluate model on test dataset with mAP calculation, compatible with your OMRDataset format"""
    if model is None:
        print("Error: Model is None")
        return None
    
    if model_args is None:
        model_args = {}
    
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    
    # Initialize per-class detection results storage for mAP calculation
    all_detections = defaultdict(list)  # class_id -> [detection1, detection2, ...]
    all_groundtruth = defaultdict(list)  # class_id -> [gt1, gt2, ...]
    
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
    for idx, data_item in enumerate(tqdm(dataset, desc="Evaluating")):
        # Handle different dataset formats
        if isinstance(data_item, tuple) and len(data_item) == 2:
            # Basic dataset format: (image_path, target)
            image_path, target = data_item
            # Prepare image
            img_tensor, scaling_info = prepare_image(image_path, model_args, device)
        elif isinstance(data_item, dict) and 'image' in data_item:
            # Your OMRDataset format
            img_tensor = data_item['image'].to(device)
            target = {
                'boxes': data_item['boxes'],
                'labels': data_item['labels'],
                'image_id': data_item['image_id'] if 'image_id' in data_item else torch.tensor([idx])
            }
            # Get image shape for scaling
            original_height, original_width = img_tensor.shape[1:3]
            processed_height, processed_width = original_height, original_width
            scaling_info = {
                'scaling_factors': (1.0, 1.0),  # No scaling needed in this case
                'original_dims': (original_height, original_width),
                'processed_dims': (processed_height, processed_width)
            }
        else:
            print(f"Warning: Unknown dataset format for item {idx}")
            continue
            
        if img_tensor is None:
            continue
        
        # Run inference
        with torch.no_grad():
            # Handle single vs. multi-image formats
            if isinstance(img_tensor, list):
                predictions = model(img_tensor)[0]
            else:
                predictions = model([img_tensor])[0]
        
        # Extract predictions
        pred_boxes = predictions['boxes'].cpu()
        pred_scores = predictions['scores'].cpu()
        pred_labels = predictions['labels'].cpu()
        
        # Filter predictions by confidence threshold
        keep = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        pred_labels = pred_labels[keep]
        
        # Scale predictions to original image size if needed
        if scaling_info and scaling_info['scaling_factors'] != (1.0, 1.0):
            scale_x, scale_y = scaling_info['scaling_factors']
            scaled_boxes = []
            for box in pred_boxes:
                x1, y1, x2, y2 = box
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
                scaled_boxes.append([x1, y1, x2, y2])
            pred_boxes = torch.tensor(scaled_boxes)
        
        # Get ground truth
        gt_boxes = target['boxes'].cpu()
        gt_labels = target['labels'].cpu()
        
        # Store predictions and ground truth for mAP calculation
        if isinstance(data_item, tuple):
            image_id = os.path.basename(image_path)
        else:
            image_id = str(target['image_id'].item() if isinstance(target['image_id'], torch.Tensor) else target['image_id'])
        
        # Store detections for mAP
        for i in range(len(pred_boxes)):
            class_id = pred_labels[i].item()
            class_name = id_to_name.get(class_id, f"Unknown_{class_id}")
            all_detections[class_name].append({
                'image_id': image_id,
                'bbox': pred_boxes[i].tolist(),
                'score': pred_scores[i].item()
            })
        
        # Store ground truth for mAP
        for i in range(len(gt_boxes)):
            class_id = gt_labels[i].item()
            class_name = id_to_name.get(class_id, f"Unknown_{class_id}")
            all_groundtruth[class_name].append({
                'image_id': image_id,
                'bbox': gt_boxes[i].tolist(),
                'used': False  # To track which ground truth boxes are matched during mAP calculation
            })
        
        # Evaluate predictions against ground truth for basic metrics
        evaluate_image_predictions(
            pred_boxes, pred_labels, gt_boxes, gt_labels,
            results, id_to_name, iou_threshold=0.5
        )
    
    # Calculate basic metrics
    calculate_metrics(results['overall'])
    
    for class_id in results['per_class']:
        calculate_metrics(results['per_class'][class_id])
    
    # Calculate mAP for different IoU thresholds
    for iou_threshold in iou_thresholds:
        # Calculate AP per class
        class_aps = {}
        
        for class_name in all_groundtruth.keys():
            if class_name == 'background':
                continue
                
            ap = calculate_ap(
                all_detections[class_name],
                all_groundtruth[class_name],
                iou_threshold
            )
            class_aps[class_name] = ap
        
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


def evaluate_image_predictions(pred_boxes, pred_labels, gt_boxes, gt_labels, results, id_to_name, iou_threshold=0.5):
    """Evaluate predictions for a single image"""
    # Skip if no ground truth or predictions
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return
    
    # If no ground truth but have predictions, all are false positives
    if len(gt_boxes) == 0:
        results['overall']['false_positives'] += len(pred_boxes)
        for i, label in enumerate(pred_labels):
            class_id = label.item()
            class_name = id_to_name.get(class_id, f"Unknown_{class_id}")
            results['per_class'][class_name]['false_positives'] += 1
        return
    
    # If no predictions but have ground truth, all are false negatives
    if len(pred_boxes) == 0:
        results['overall']['false_negatives'] += len(gt_boxes)
        for i, label in enumerate(gt_labels):
            class_id = label.item()
            class_name = id_to_name.get(class_id, f"Unknown_{class_id}")
            results['per_class'][class_name]['false_negatives'] += 1
        return
    
    # Calculate IoU between all prediction and ground truth boxes
    iou_matrix = box_iou(pred_boxes, gt_boxes)
    
    # Track which ground truth boxes have been matched
    gt_matched = [False] * len(gt_boxes)
    
    # For each prediction, find the best matching ground truth
    for pred_idx, pred_label in enumerate(pred_labels):
        pred_class_id = pred_label.item()
        pred_class_name = id_to_name.get(pred_class_id, f"Unknown_{pred_class_id}")
        
        # Find best matching ground truth for this prediction
        best_gt_idx = -1
        best_iou = iou_threshold  # Must exceed threshold
        
        for gt_idx, gt_label in enumerate(gt_labels):
            gt_class_id = gt_label.item()
            
            # Only match same class and not already matched
            if gt_class_id == pred_class_id and not gt_matched[gt_idx]:
                iou = iou_matrix[pred_idx, gt_idx].item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            # True positive
            gt_matched[best_gt_idx] = True
            results['overall']['true_positives'] += 1
            results['per_class'][pred_class_name]['true_positives'] += 1
        else:
            # False positive
            results['overall']['false_positives'] += 1
            results['per_class'][pred_class_name]['false_positives'] += 1
    
    # Count unmatched ground truth as false negatives
    for gt_idx, matched in enumerate(gt_matched):
        if not matched:
            results['overall']['false_negatives'] += 1
            
            gt_class_id = gt_labels[gt_idx].item()
            gt_class_name = id_to_name.get(gt_class_id, f"Unknown_{gt_class_id}")
            results['per_class'][gt_class_name]['false_negatives'] += 1


def calculate_metrics(metrics):
    """Calculate precision, recall, and F1 score"""
    tp = metrics['true_positives']
    fp = metrics['false_positives']
    fn = metrics['false_negatives']
    
    # Calculate precision
    if tp + fp > 0:
        metrics['precision'] = tp / (tp + fp)
    else:
        metrics['precision'] = 0
    
    # Calculate recall
    if tp + fn > 0:
        metrics['recall'] = tp / (tp + fn)
    else:
        metrics['recall'] = 0
    
    # Calculate F1 score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0


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
    # Prepare summary results (without per-class metrics)
    summary_file = os.path.join(output_dir, 'evaluation_summary-1449.csv')
    
    # Define headers for summary file
    summary_headers = [
        'Model Name', 'Test Dataset', 'Mapping File', 'Test Images', 
        'IoU Threshold', 'Conf Threshold', 'Precision', 'Recall', 
        'F1 Score', 'True Positives', 'False Positives', 'False Negatives', 
        'mAP@0.5', 'mAP@0.75', 'mAP@[.5:.95]', 'Image Min Size', 
        'Image Max Size', 'Anchor Sizes', 'Aspect Ratios', 'NMS IoU Threshold'
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
                'Test Dataset': result.get('test_dir', ''),
                'Mapping File': result.get('mapping_file', ''),
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
                'Image Min Size': result.get('model_args', {}).get('min_size', ''),
                'Image Max Size': result.get('model_args', {}).get('max_size', ''),
                'Anchor Sizes': result.get('model_args', {}).get('anchor_sizes', ''),
                'Aspect Ratios': result.get('model_args', {}).get('aspect_ratios', ''),
                'NMS IoU Threshold': result.get('model_args', {}).get('second_stage_nms_iou_threshold', '')
            }
            writer.writerow(row)
    
    print(f"Summary results saved to: {summary_file}")
    
    # Save per-class results if requested
    if include_per_class:
        for model_name, result in model_results.items():
            if not result.get('results'):
                continue
                
            # Create per-class results file for this model
            per_class_file = os.path.join(output_dir, f'{model_name}_per_class_results-1449.csv')
            
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
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Parse mAP IoU thresholds
    map_iou_thresholds = [float(t) for t in args.map_iou_thresholds.split(',')]
    print(f"Using mAP IoU thresholds: {map_iou_thresholds}")
    
    # Evaluate each model
    model_results = {}
    
    for model_dir in args.model_dirs:
        model_name = os.path.basename(model_dir)
        print(f"\n===== Evaluating model: {model_name} =====")
        
        # Load model info, including test directory and mapping file paths
        model_info = load_model_info(model_dir, args.default_test_dir, args.default_mapping_file)
        if not model_info.get('checkpoint_path'):
            print(f"Skipping {model_name}: No checkpoint found")
            continue
        
        # Load class mapping for this model
        mapping_file = model_info['mapping_file']
        id_to_name, name_to_id = load_class_mapping(mapping_file)
        if not id_to_name:
            print(f"Error: Failed to load class mapping from {mapping_file}")
            continue
        
        # Load test dataset for this model
        test_dir = model_info['test_dir']
        print(f"Loading test dataset from {test_dir}")
        test_dataset = load_test_dataset(
            test_dir, mapping_file, args.num_images, args.seed
        )
        print(f"Loaded {len(test_dataset)} test images")
        
        # Load model
        model = load_model(model_info, device)
        if model is None:
            print(f"Skipping {model_name}: Failed to load model")
            continue
        
        # Run evaluation
        results = run_evaluation(
            model, test_dataset, id_to_name, device,
            conf_threshold=args.conf_threshold,
            iou_thresholds=map_iou_thresholds,
            model_args=model_info.get('args', {})
        )
        
        if results is None:
            print(f"Skipping {model_name}: Evaluation failed")
            continue
        
        # Store results
        model_results[model_name] = {
            'results': results,
            'model_info': model_info,
            'test_dir': test_dir,
            'mapping_file': mapping_file,
            'num_images': len(test_dataset),
            'iou_threshold': args.iou_threshold,
            'conf_threshold': args.conf_threshold,
            'model_args': model_info.get('args', {})
        }
        
        # Print summary
        overall = results['overall']
        mAP_50 = results['mAP'].get(0.5, {}).get('overall', 0.0)
        mAP_75 = results['mAP'].get(0.75, {}).get('overall', 0.0)
        mAP_avg = results['mAP'].get('average', 0.0)
        
        print(f"\nResults for {model_name}:")
        print(f"Test dir: {test_dir}")
        print(f"Mapping file: {mapping_file}")
        print(f"Precision: {overall['precision']:.4f}")
        print(f"Recall: {overall['recall']:.4f}")
        print(f"F1 Score: {overall['f1_score']:.4f}")
        print(f"mAP@0.5: {mAP_50:.4f}")
        print(f"mAP@0.75: {mAP_75:.4f}")
        print(f"mAP@[.5:.95]: {mAP_avg:.4f}")
        print(f"True Positives: {overall['true_positives']}")
        print(f"False Positives: {overall['false_positives']}")
        print(f"False Negatives: {overall['false_negatives']}")
        
        # Save per-class results to each model directory
        model_output_dir = model_dir
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Save summary and per-class results for this model
        single_model_results = {model_name: model_results[model_name]}
        save_results_to_csv(single_model_results, model_output_dir, include_per_class=True)
    
    # Save combined results to first model directory or current directory
    if model_results:
        output_dir = args.model_dirs[0] if args.model_dirs else '.'
        save_results_to_csv(model_results, output_dir, include_per_class=False)
        
        print("\nEvaluation complete!")
        print(f"Summary results saved to: {os.path.join(output_dir, 'evaluation_summary-1449.csv')}")
    else:
        print("No models were successfully evaluated")


if __name__ == "__main__":
    main()