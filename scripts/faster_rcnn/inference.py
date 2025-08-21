#!/usr/bin/env python3
"""
Inference Script for Faster R-CNN Model on OMR/MUSCIMA Dataset

This script loads a trained Faster R-CNN model and performs inference on input images,
using the same visualization and model configuration as the training pipeline.
"""
import os
import sys
import time
import json
import torch
import argparse
import numpy as np
import cv2
import glob
import traceback
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
from tqdm import tqdm
from collections import defaultdict

# Import from torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
try:
    from torchvision.models.detection import fasterrcnn_resnet101_fpn
    HAS_RESNET101 = True
except ImportError:
    HAS_RESNET101 = False
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Try to import from omr_dataset (same as in training)
try:
    from omr_dataset import get_transform, OMRDataset
    HAS_DATASET = True
except ImportError:
    print("Warning: Could not import omr_dataset module. Using basic transforms.")
    HAS_DATASET = False

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained Faster R-CNN model')
    
    # Input/output parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--mapping_file', type=str, default=None,
                        help='Path to class mapping file')
    parser.add_argument('--annotations_file', type=str, default=None,
                        help='Path to annotations file (for visualization)')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=218,  # Default to 218 based on error message
                        help='Number of classes in the model (including background)')
    parser.add_argument('--auto_detect_classes', action='store_true', default=True,
                        help='Automatically detect number of classes from checkpoint')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help='Backbone network for Faster R-CNN')
    
    # Image parameters - same as in training script
    parser.add_argument('--image_size', type=str, default='500,1000',
                        help='Image size (min_dimension,max_dimension) for resizing input images')
    parser.add_argument('--min_size', type=int, default=500,
                        help='Minimum size of the image to be rescaled before feeding it to the backbone')
    parser.add_argument('--max_size', type=int, default=1000,
                        help='Maximum size of the image to be rescaled before feeding it to the backbone')
    
    # Anchor parameters - same as in training script
    parser.add_argument('--anchor_sizes', type=str, default='16,32,64',
                        help='Comma-separated list of anchor sizes')
    parser.add_argument('--aspect_ratios', type=str, default='0.1,1.0,2.0,8.0',
                        help='Comma-separated list of aspect ratios')
    
    # Detection parameters
    parser.add_argument('--conf_threshold', type=float, default=0.4,
                        help='Confidence threshold for detections')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                        help='NMS threshold for detections')
    parser.add_argument('--max_detections', type=int, default=1000,
                        help='Maximum number of detections per image')
    
    # Visualization parameters
    parser.add_argument('--show_vis', action='store_true',
                        help='Show visualization during inference')
    parser.add_argument('--save_vis', action='store_true', default=True,
                        help='Save visualization results')
    parser.add_argument('--save_raw_predictions', action='store_true', default=True,
                        help='Save raw prediction data as JSON')
    parser.add_argument('--vis_size', type=str, default='1200,900',
                        help='Visualization size (width,height)')
    parser.add_argument('--staff_line_color', type=str, default='0.6,0.6,0.6',
                        help='RGB color for staff lines (values from 0-1)')
    
    # Hardware parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    
    return parser.parse_args()


def load_model(args, device):
    """
    Load the model from checkpoint using the same configuration as in training
    """
    print(f"Loading model from {args.model_path}")
    
    # First, try to detect the actual number of classes in the checkpoint
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Check the shape of classifier weight to determine number of classes
        for key in model_state.keys():
            if 'box_predictor.cls_score.weight' in key:
                actual_num_classes = model_state[key].shape[0]
                if actual_num_classes != args.num_classes:
                    print(f"Warning: Model checkpoint has {actual_num_classes} classes, but --num_classes={args.num_classes} was specified.")
                    print(f"Using {actual_num_classes} classes from the checkpoint instead.")
                    args.num_classes = actual_num_classes
                break
    except Exception as e:
        print(f"Could not automatically detect number of classes: {e}")
        print(f"Will try to use specified --num_classes={args.num_classes}")
    
    # Parse anchor parameters - same as in training
    anchor_sizes = tuple([int(size) for size in args.anchor_sizes.split(',')])
    aspect_ratios = tuple([float(ratio) for ratio in args.aspect_ratios.split(',')])
    
    # Parse image size - same as in training
    if ',' in args.image_size:
        min_dim, max_dim = map(int, args.image_size.split(','))
    else:
        min_dim = max_dim = int(args.image_size)
    
    # Override with explicit min_size and max_size if provided
    min_size = args.min_size if args.min_size else min_dim
    max_size = args.max_size if args.max_size else max_dim
    
    print(f"Using image size: min={min_size}, max={max_size}")
    print(f"Using anchor sizes: {anchor_sizes}")
    print(f"Using aspect ratios: {aspect_ratios}")
    
    # Create model with the correct backbone - same as in training
    if args.backbone == 'resnet50':
        model = fasterrcnn_resnet50_fpn(
            pretrained=False,
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
    elif args.backbone == 'resnet101' and HAS_RESNET101:
        model = fasterrcnn_resnet101_fpn(
            pretrained=False,
            min_size=min_size,
            max_size=max_size
        )
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    
    # Modify anchor generator to match training settings
    anchor_generator = model.rpn.anchor_generator
    anchor_generator.sizes = anchor_sizes
    anchor_generator.aspect_ratios = aspect_ratios
    
    # Modify inference parameters to match training
    model.rpn.nms_thresh = args.nms_threshold
    model.rpn.score_thresh = 0.0  # Keep all proposals at RPN stage
    model.rpn.post_nms_top_n_test = args.max_detections
    model.roi_heads.nms_thresh = args.nms_threshold
    model.roi_heads.score_thresh = args.conf_threshold
    model.roi_heads.detections_per_img = args.max_detections
    
    # Replace box predictor with the correct number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes)
    
    # Load checkpoint
    try:
        # If checkpoint was already loaded above, use it, otherwise load again
        if not locals().get('checkpoint'):
            checkpoint = torch.load(args.model_path, map_location=device)
            
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
        print("For detailed error information, check model architecture and ensure:")
        print("1. The model checkpoint matches the expected architecture")
        print("2. The number of classes matches (use --num_classes to specify the correct number)")
        print("3. The backbone and other parameters match the training configuration")
        raise


def load_class_names(mapping_file, num_classes):
    """
    Load class names from mapping file - similar to training script
    """
    class_names = {}
    
    if mapping_file and os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(':')
                        if len(parts) >= 2:
                            class_id = int(parts[0])
                            class_name = ':'.join(parts[1:]).strip()
                            class_names[class_id] = class_name
            print(f"Loaded {len(class_names)} class names from {mapping_file}")
            
            # If mapping file number doesn't match model's number of classes, warn user
            if len(class_names) + 1 != num_classes:  # +1 for background
                print(f"Warning: Mapping file has {len(class_names)} classes, but model has {num_classes} classes")
                print("Some class names may be incorrect or missing.")
        except Exception as e:
            print(f"Error loading class names: {e}")
            
    # If no mapping file or error, use default class names
    if not class_names:
        print(f"Using default class names for {num_classes} classes")
        for i in range(1, num_classes):  # Skip background class 0
            class_names[i] = f"Class_{i}"
            
    # Add background class
    class_names[0] = "background"
    
    # Make sure all class IDs have names (might be missing in mapping file)
    for i in range(1, num_classes):
        if i not in class_names:
            class_names[i] = f"Unknown_Class_{i}"
            
    return class_names


def load_image(image_path, device, args):
    """
    Load and prepare an image for inference using the same preprocessing as in training
    Returns: img_tensor, processed_image, processed_dims
    """
    # Load image based on type
    if isinstance(image_path, str):
        try:
            # Try using OpenCV first (more reliable for the transform pipeline)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Fallback to PIL
            try:
                pil_image = Image.open(image_path).convert('RGB')
                image = np.array(pil_image)
            except Exception as e2:
                raise ValueError(f"Failed to load image: {e2}")
    else:
        # Assume it's already a numpy array
        image = image_path
    
    # Get original dimensions for later use
    original_height, original_width = image.shape[:2]
    orig_image = image.copy()  # Keep a copy of original image
    
    # Track the processed dimensions
    processed_height = original_height
    processed_width = original_width
    
    # Use our custom dataset transform if available
    if HAS_DATASET:
        # Create transform like in validation
        transform = get_transform(
            train=False,
            min_size=args.min_size,
            max_size=args.max_size
        )
        
        # Create a dummy target with minimal required structure
        target = {"boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)}
        
        # Apply transform
        try:
            # Create a copy of image as numpy array
            image_np = image.copy()
            
            # Convert to PIL for testing
            image_pil = Image.fromarray(image)
            
            # Try with numpy array first (since error mentions 'shape')
            img_tensor, _ = transform(image_np, target)
            print("Transform function accepted numpy array")
            
            # Try to get the processed dimensions from the tensor
            if img_tensor.dim() == 3:
                processed_height = img_tensor.shape[1]
                processed_width = img_tensor.shape[2]
                
        except Exception as e1:
            try:
                # Try with PIL image
                img_tensor, _ = transform(image_pil, target)
                print("Transform function accepted PIL image")
                
                # Try to get the processed dimensions from the tensor
                if img_tensor.dim() == 3:
                    processed_height = img_tensor.shape[1]
                    processed_width = img_tensor.shape[2]
                    
            except Exception as e2:
                print(f"Both transform input types failed. Using fallback. Errors:")
                print(f"Numpy error: {e1}")
                print(f"PIL error: {e2}")
                # Use fallback method below
                img_tensor = None
    else:
        img_tensor = None
    
    # Fallback to basic transform if custom transform failed or not available
    if img_tensor is None:
        print("Using fallback transform pipeline")
        img_tensor = F.to_tensor(image)
        
        # Normalize like in training
        img_tensor = F.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Resize to match training dimensions
        h, w = image.shape[:2]
        min_dim = min(h, w)
        max_dim = max(h, w)
        
        # Calculate scaling factor to maintain aspect ratio
        scale_factor = min(
            args.min_size / min_dim,
            args.max_size / max_dim
        )
        
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        if new_h != h or new_w != w:
            img_tensor = F.resize(img_tensor, [new_h, new_w])
            processed_height = new_h
            processed_width = new_w
    
    # Move to device
    img_tensor = img_tensor.to(device)
    
    # Save the processed dimensions as a property of the tensor
    # Create a processed image for visualization if needed
    processed_image = F.to_pil_image(img_tensor.cpu())
    processed_image = np.array(processed_image)
    
    return img_tensor, processed_image, (processed_height, processed_width)


def run_inference(model, image_path, device, args):
    """
    Run inference on a single image and track both original and processed dimensions
    """
    # Load and prepare image
    try:
        # Read original image
        if isinstance(image_path, str):
            try:
                # Try using OpenCV first (more reliable for the transform pipeline)
                orig_image = cv2.imread(image_path)
                if orig_image is None:
                    raise ValueError(f"Could not load image from {image_path}")
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                # Fallback to PIL
                try:
                    pil_image = Image.open(image_path).convert('RGB')
                    orig_image = np.array(pil_image)
                except Exception as e2:
                    raise ValueError(f"Failed to load image: {e2}")
        else:
            # Assume it's already a numpy array
            orig_image = image_path
        
        # Store original dimensions
        original_height, original_width = orig_image.shape[:2]
        original_dims = (original_height, original_width)
        
        # Process the image for the model
        img_tensor, processed_image, processed_dims = load_image(image_path, device, args)
        
        # Get processed dimensions for scaling factor calculation
        processed_height, processed_width = processed_dims
        
        print(f"Original image dimensions: {original_width}x{original_height}")
        print(f"Processed image dimensions: {processed_width}x{processed_height}")
        
        # Calculate scaling factors
        scale_x = original_width / processed_width
        scale_y = original_height / processed_height
        
        print(f"Scaling factors: x={scale_x}, y={scale_y}")
        
    except Exception as e:
        print(f"Error preparing image {image_path}: {e}")
        traceback.print_exc()
        return None, None, None, None
    
    # Get model predictions
    with torch.no_grad():
        try:
            predictions = model([img_tensor])[0]
            
            # Don't try to attach attributes to the dictionary
            # Instead, create a separate scaling_info dictionary
            scaling_info = {
                'scaling_factors': (scale_x, scale_y),
                'original_dims': original_dims,
                'processed_dims': processed_dims
            }
            
        except Exception as e:
            print(f"Error during model inference: {e}")
            traceback.print_exc()
            return None, None, None, None
    
    return predictions, orig_image, original_dims, scaling_info

def visualize_detections(image, predictions, class_names, args, output_path=None, original_dims=None, scaling_info=None):
    """
    Create high-quality visualization of detections with dictionary-based scaling info
    """
    # Extract predictions
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    # Filter by confidence threshold
    keep = scores >= args.conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Get scaling factors from various possible sources
    scale_x = 1.0
    scale_y = 1.0
    
    # First priority: use scaling_info if provided
    if scaling_info and 'scaling_factors' in scaling_info:
        scale_x, scale_y = scaling_info['scaling_factors']
        print(f"Using scaling factors from scaling_info: x={scale_x}, y={scale_y}")
    # Second priority: calculate from original_dims
    elif original_dims is not None:
        orig_height, orig_width = original_dims
        curr_height, curr_width = image.shape[:2]
        
        if orig_height != curr_height or orig_width != curr_width:
            scale_x = orig_width / curr_width
            scale_y = orig_height / curr_height
            print(f"Calculated scaling factors: x={scale_x}, y={scale_y}")
    
    # Apply scaling if needed (if scale is not 1.0)
    if scale_x != 1.0 or scale_y != 1.0:
        print(f"Applying scaling to bounding boxes: x={scale_x}, y={scale_y}")
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            scaled_boxes.append([x1, y1, x2, y2])
        
        boxes = np.array(scaled_boxes)
    
    # If no detections, create a simple image saying so
    if len(boxes) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(image)
        ax.text(0.5, 0.5, f"No detections above threshold ({args.conf_threshold})",
                ha='center', va='center', fontsize=14, color='red',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.5'))
        ax.set_title("Detection Results")
        ax.axis('off')
        plt.tight_layout()
        
        if output_path and args.save_vis:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        if args.show_vis:
            plt.show()
        plt.close()
        return
    
    # Get image dimensions for the visualized output
    height, width = image.shape[:2]
    
    # Parse visualization size
    if ',' in args.vis_size:
        vis_width, vis_height = map(int, args.vis_size.split(','))
    else:
        vis_width = int(args.vis_size)
        vis_height = int(height * vis_width / width)
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(vis_width/100, vis_height/100))
    ax.imshow(image)
    
    # Create color map
    color_map = create_color_map(class_names, args)
    
    # Create class counters for statistics
    class_counts = defaultdict(int)
    
    # Draw boxes and labels
    for box, score, label_id in zip(boxes, scores, labels):
        # Update class counter
        class_counts[label_id] += 1
        
        # Get color for this class
        color_rgb = color_map.get(label_id, (1.0, 0.0, 0.0))  # Default to red
        
        # Get class name
        class_name = class_names.get(label_id, f"Class {label_id}")
        
        # Draw box
        x1, y1, x2, y2 = box
        width_box = x2 - x1
        height_box = y2 - y1
        
        rect = patches.Rectangle(
            (x1, y1), width_box, height_box,
            linewidth=1.5, edgecolor=color_rgb, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Create label text with class and confidence
        label_text = f"{class_name} ({score:.2f})"
        
        # Draw text with background that matches class color but is lighter
        bg_color = tuple(min(1.0, c * 1.5) for c in color_rgb)  # Lighten color
        
        # Position label based on available space
        if y1 < 30:  # Not enough space above, put below
            y_text = y1 + height_box + 5
            va = 'top'
        else:  # Put above
            y_text = y1 - 5
            va = 'bottom'
        
        ax.text(
            x1, y_text,
            label_text,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor=color_rgb, boxstyle='round,pad=0.2'),
            fontsize=5, color='black',
            verticalalignment=va
        )
    
    # Add detection statistics to image
    stat_text = f"Detected {len(boxes)} objects\n"
    
    # Sort classes by ID for consistent display
    sorted_counts = sorted(class_counts.items())
    
    # Limit displayed classes to prevent crowding
    MAX_CLASSES_DISPLAY = 10
    if len(sorted_counts) > MAX_CLASSES_DISPLAY:
        for label_id, count in sorted_counts[:MAX_CLASSES_DISPLAY-1]:
            class_name = class_names.get(label_id, f"Class {label_id}")
            stat_text += f"{class_name}: {count}\n"
        stat_text += f"+ {len(sorted_counts) - (MAX_CLASSES_DISPLAY-1)} more classes"
    else:
        for label_id, count in sorted_counts:
            class_name = class_names.get(label_id, f"Class {label_id}")
            stat_text += f"{class_name}: {count}\n"
    
    # Add dimensions info if available
    if scaling_info:
        if 'original_dims' in scaling_info and 'processed_dims' in scaling_info:
            orig_h, orig_w = scaling_info['original_dims']
            proc_h, proc_w = scaling_info['processed_dims']
            stat_text += f"\nOriginal: {orig_w}x{orig_h}\n"
            stat_text += f"Processed: {proc_w}x{proc_h}\n"
            stat_text += f"Scale: {scale_x:.2f}x, {scale_y:.2f}y"
    
    # Add statistics box in upper right
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
    ax.text(0.98, 0.02, stat_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Set title with detection count and confidence threshold
    ax.set_title(f"Detection Results (Confidence ≥ {args.conf_threshold:.2f})")
    
    # Turn off axis
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Save visualization if requested
    if output_path and args.save_vis:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Show visualization if requested
    if args.show_vis:
        plt.show()
    
    plt.close(fig)
    
def save_detections_to_json(predictions, image_path, output_path, class_names, args, original_dims=None, scaling_info=None):
    """
    Save detection results to a JSON file with dictionary-based scaling info
    """
    # Extract and filter predictions
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    # Filter by confidence threshold
    keep = scores >= args.conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    # Get scaling factors from various possible sources
    scale_x = 1.0
    scale_y = 1.0
    
    # First priority: use scaling_info if provided
    if scaling_info and 'scaling_factors' in scaling_info:
        scale_x, scale_y = scaling_info['scaling_factors']
        print(f"JSON: Using scaling factors from scaling_info: x={scale_x}, y={scale_y}")
    # Second priority: calculate from original_dims
    elif original_dims is not None:
        orig_height, orig_width = original_dims
        
        # Try to get processed dimensions from scaling_info
        if scaling_info and 'processed_dims' in scaling_info:
            proc_height, proc_width = scaling_info['processed_dims']
        else:
            # Fallback - try to infer from first box or use a default
            if len(boxes) > 0:
                max_x = max(boxes[:, 2])
                max_y = max(boxes[:, 3])
                proc_width = max_x
                proc_height = max_y
            else:
                # Default to min_size/max_size if we can't determine
                proc_width = args.min_size
                proc_height = args.max_size
        
        if orig_height != proc_height or orig_width != proc_width:
            scale_x = orig_width / proc_width
            scale_y = orig_height / proc_height
            print(f"JSON: Calculated scaling factors: x={scale_x}, y={scale_y}")
    
    # Apply scaling if needed (if scale is not 1.0)
    if scale_x != 1.0 or scale_y != 1.0:
        print(f"JSON: Applying scaling to bounding boxes: x={scale_x}, y={scale_y}")
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            scaled_boxes.append([x1, y1, x2, y2])
        
        boxes = np.array(scaled_boxes)
    
    # Convert numpy arrays to Python lists for JSON serialization
    boxes = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
    scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
    labels = labels.tolist() if isinstance(labels, np.ndarray) else labels
    
    # Create detection objects
    detections = []
    for box, score, label_id in zip(boxes, scores, labels):
        class_name = class_names.get(label_id, f"Class_{label_id}")
        detections.append({
            "box": box,  # [x1, y1, x2, y2] format
            "score": score,
            "label_id": int(label_id),
            "label_name": class_name
        })
    
    # Create results dictionary
    results = {
        "image_path": image_path,
        "detections": detections,
        "detection_count": len(detections),
        "confidence_threshold": args.conf_threshold,
        "nms_threshold": args.nms_threshold,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add dimensions information - prefer scaling_info if available
    if scaling_info and 'original_dims' in scaling_info:
        orig_h, orig_w = scaling_info['original_dims']
        results["image_dimensions"] = {
            "height": orig_h,
            "width": orig_w
        }
    elif original_dims is not None:
        results["image_dimensions"] = {
            "height": original_dims[0],
            "width": original_dims[1]
        }
    
    # Add scaling information
    if scaling_info and 'scaling_factors' in scaling_info:
        results["scaling_factors"] = {
            "x": scale_x,
            "y": scale_y
        }
    elif scale_x != 1.0 or scale_y != 1.0:
        results["scaling_factors"] = {
            "x": scale_x,
            "y": scale_y
        }
    
    # Add processed dimensions if available
    if scaling_info and 'processed_dims' in scaling_info:
        proc_h, proc_w = scaling_info['processed_dims']
        results["processed_dimensions"] = {
            "height": proc_h,
            "width": proc_w
        }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
        
def create_color_map(class_names, args):
    """
    Create a color map for the classes, with staff lines in gray
    """
    # Base colors (RGB)
    colors = [
        (1.0, 0.0, 0.0),    # Red
        (0.0, 0.8, 0.0),    # Green
        (0.0, 0.0, 1.0),    # Blue
        (1.0, 0.6, 0.0),    # Orange
        (0.5, 0.0, 0.5),    # Purple
        (0.0, 0.5, 0.5),    # Teal
        (1.0, 0.0, 0.5),    # Pink
        (0.5, 0.5, 0.0),    # Olive
        (0.0, 0.7, 1.0),    # Sky Blue
        (0.7, 0.0, 0.0),    # Dark Red
        (0.0, 0.5, 0.0),    # Dark Green
        (0.0, 0.0, 0.7),    # Dark Blue
        (0.7, 0.7, 0.0),    # Yellow
        (0.7, 0.0, 0.7),    # Magenta
        (0.0, 0.7, 0.7),    # Cyan
    ]
    
    # Parse staff line color
    if hasattr(args, 'staff_line_color') and args.staff_line_color:
        staff_line_color = tuple(map(float, args.staff_line_color.split(',')))
    else:
        staff_line_color = (0.6, 0.6, 0.6)  # Default gray for staff lines
    
    # Create color map
    color_map = {}
    for class_id, class_name in class_names.items():
        if class_id == 0:  # Background class
            color_map[class_id] = (0.1, 0.1, 0.1)  # Dark gray
        elif isinstance(class_name, str) and "staff" in class_name.lower():
            color_map[class_id] = staff_line_color
        else:
            color_map[class_id] = colors[class_id % len(colors)]
    
    return color_map        
        
def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model - this may update args.num_classes if auto_detect_classes is True
    try:
        model = load_model(args, device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Load class names after loading the model to ensure we use the correct num_classes
    class_names = load_class_names(args.mapping_file, args.num_classes)
    
    # Get list of input images
    if os.path.isdir(args.input):
        image_paths = glob.glob(os.path.join(args.input, '*.jpg')) + \
                     glob.glob(os.path.join(args.input, '*.jpeg')) + \
                     glob.glob(os.path.join(args.input, '*.png'))
        print(f"Found {len(image_paths)} images in {args.input}")
    else:
        image_paths = [args.input]
        print(f"Processing single image: {args.input}")
    
    # Process each image
    print("Running inference...")
    for image_path in tqdm(image_paths):
        try:
            # Extract image filename
            image_filename = os.path.basename(image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # Define output paths
            vis_output_path = os.path.join(args.output_dir, f"{image_name}_detection.png")
            json_output_path = os.path.join(args.output_dir, f"{image_name}_detection.json")
            
            # Run inference with auto-scaling - now returns scaling_info as well
            predictions, original_image, original_dims, scaling_info = run_inference(model, image_path, device, args)
            
            if predictions is None:
                print(f"Skipping {image_path} - inference failed")
                continue
            
            # Save detection results to JSON with scaling info
            if args.save_raw_predictions:
                save_detections_to_json(predictions, image_path, json_output_path, class_names, args, original_dims, scaling_info)
            
            # Visualize and save results with scaling info
            visualize_detections(original_image, predictions, class_names, args, vis_output_path, original_dims, scaling_info)
            
            print(f"Processed {image_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            traceback.print_exc()
    
    print("Inference complete!")
    print(f"Results saved to {args.output_dir}")
    

if __name__ == "__main__":
    main()