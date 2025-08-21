#!/usr/bin/env python3
"""
Train Faster R-CNN Model for OMR/MUSCIMA Dataset

This script trains a Faster R-CNN model on an Optical Music Recognition dataset,
with configuration parameters adapted from the TensorFlow Faster R-CNN with 
Inception Resnet v2 config for MUSCIMA.
"""
print("Testing imports...")
import os
import time
import datetime
import json
import glob
import torch
import traceback

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
try:
    from torchvision.models.detection import fasterrcnn_resnet101_fpn
    HAS_RESNET101 = True
except ImportError:
    HAS_RESNET101 = False
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR  # Changed to match TF config
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as F


print("Attempting to import omr_dataset...")
from omr_dataset import OMRDataset, get_transform
print("Successfully imported omr_dataset")


import argparse
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import itertools

# Add these lines at the very beginning of the script
print("Script starting...")
import sys
print(f"Python version: {sys.version}")
print(f"Arguments: {sys.argv}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN for OMR/MUSCIMA')
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, 
                        default='/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset',
                        help='Directory containing prepared dataset')
    parser.add_argument('--output_dir', type=str, default='./staffline_faster_rcnn_omr',
                        help='Directory to save model checkpoints')
    parser.add_argument('--test_images_dir', type=str, default='',
                        help='Directory containing test images for inference during training')
    # Add new argument for data subset
    parser.add_argument('--data_subset', type=float, default=0.5,
                      help='Fraction of data to use (e.g., 0.1 for 10%)')
    
    # Image parameters - updated to align with TF config
    parser.add_argument('--image_size', type=str, default='500,1000',
                        help='Image size (min_dimension,max_dimension) for resizing input images')
    parser.add_argument('--min_size', type=int, default=500,
                        help='Minimum size of the image to be rescaled before feeding it to the backbone')
    parser.add_argument('--max_size', type=int, default=1000,
                        help='Maximum size of the image to be rescaled before feeding it to the backbone')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=400,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1,  # Updated to match TF config
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Batch size for validation')
    parser.add_argument('--learning_rate', type=float, default=0.003,  # Updated to match TF config
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,  # Matches TF config
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0,  # Updated based on TF config (no L2 reg)
                        help='Weight decay')
    parser.add_argument('--decay_steps', type=int, default=80000,  # Added from TF config
                        help='Decay steps for exponential learning rate decay')
    parser.add_argument('--decay_factor', type=float, default=0.95,  # Added from TF config
                        help='Decay factor for exponential learning rate decay')
    parser.add_argument('--num_steps', type=int, default=80000,  # Added from TF config
                        help='Total number of training steps')
    
    # Hardware parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Use pin_memory in DataLoader for faster data transfer to GPU')
    
    # Logging and checkpointing
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Evaluation frequency in epochs')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Checkpoint saving frequency in epochs')
    parser.add_argument('--log_dir', type=str, default='./staffline-half-logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch')
    parser.add_argument('--num_visualizations', type=int, default=1,  # Added from TF config
                        help='Number of visualizations during evaluation')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=218,  # Updated from TF config
                        help='Number of classes to detect')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'inception_resnet_v2'],
                        help='Backbone network for Faster R-CNN')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone network')
    parser.add_argument('--frozen_layers', type=int, default=0,
                        help='Number of layers to freeze in the backbone')
    parser.add_argument('--features_stride', type=int, default=8,  # Added from TF config
                        help='First stage features stride')
    parser.add_argument('--initial_crop_size', type=int, default=17,  # Added from TF config
                        help='Initial crop size')
    parser.add_argument('--maxpool_kernel_size', type=int, default=1,  # Added from TF config
                        help='Max pooling kernel size')
    parser.add_argument('--maxpool_stride', type=int, default=1,  # Added from TF config
                        help='Max pooling stride')
    parser.add_argument('--atrous_rate', type=int, default=2,  # Added from TF config
                        help='First stage atrous rate')
    
    # Anchor parameters - updated to match TF config
    parser.add_argument('--anchor_sizes', type=str, default='16,32,64',
                        help='Comma-separated list of anchor sizes')
    parser.add_argument('--aspect_ratios', type=str, default='0.1,1.0,2.0,8.0',
                        help='Comma-separated list of aspect ratios')
    parser.add_argument('--height_stride', type=int, default=8,  # Added from TF config
                        help='Height stride for anchor generator')
    parser.add_argument('--width_stride', type=int, default=8,  # Added from TF config
                        help='Width stride for anchor generator')
    
    # NMS parameters - from TF config
    parser.add_argument('--first_stage_nms_score_threshold', type=float, default=0.0,
                        help='First stage NMS score threshold')
    parser.add_argument('--first_stage_nms_iou_threshold', type=float, default=0.5,
                        help='First stage NMS IoU threshold')
    parser.add_argument('--first_stage_max_proposals', type=int, default=1200,
                        help='First stage maximum proposals')
    parser.add_argument('--second_stage_nms_score_threshold', type=float, default=0.0,
                        help='Second stage NMS score threshold')
    parser.add_argument('--second_stage_nms_iou_threshold', type=float, default=0.5,
                        help='Second stage NMS IoU threshold')
    parser.add_argument('--second_stage_max_detections_per_class', type=int, default=1600,
                        help='Second stage maximum detections per class')
    parser.add_argument('--second_stage_max_total_detections', type=int, default=1600,
                        help='Second stage maximum total detections')
    
    # Loss weights - from TF config
    parser.add_argument('--first_stage_localization_loss_weight', type=float, default=2.0,
                        help='First stage localization loss weight')
    parser.add_argument('--first_stage_objectness_loss_weight', type=float, default=1.0,
                        help='First stage objectness loss weight')
    parser.add_argument('--second_stage_localization_loss_weight', type=float, default=2.0,
                        help='Second stage localization loss weight')
    parser.add_argument('--second_stage_classification_loss_weight', type=float, default=1.0,
                        help='Second stage classification loss weight')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gradient_clipping_by_norm', type=float, default=10.0,  # Added from TF config
                        help='Gradient clipping by norm value')
    
    return parser.parse_args()

def collate_fn(batch):
    """
    Custom collate function for the data loader to handle variable sized images and targets.
    """
    images = []
    boxes = []
    labels = []
    image_ids = []
    
    for sample in batch:
        images.append(sample['image'])
        boxes.append(sample['boxes'])
        labels.append(sample['labels'])
        image_ids.append(sample['image_id'])
    return images, boxes, labels, image_ids

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, clip_norm=None):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        optimizer: The optimizer
        data_loader: DataLoader for training data
        device: Device to train on
        epoch: Current epoch number
        print_freq: How often to print progress
        clip_norm: Value for gradient clipping (if None, no clipping is performed)
    """
    model.train()
    
    # Create tqdm progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    # Metrics
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    
    for i, data in enumerate(pbar):
        # Get sample batch
        try:
            # Handle data properly based on its actual structure
            images = []
            targets = []
            
            # Process the data based on the observed structure
            for batch_idx in range(len(data[0])):  # Iterate through batch
                # Get the elements for this batch item
                image = data[0][batch_idx]
                boxes = data[1][batch_idx]
                labels = data[2][batch_idx]
                image_id = data[3][batch_idx]
                
                # Skip if any element is a string (probably a file path instead of tensor)
                if isinstance(image, str) or isinstance(boxes, str) or isinstance(labels, str):
                    continue
                
                # Convert to tensors if needed and move to device
                image_tensor = image.to(device) if isinstance(image, torch.Tensor) else torch.tensor(image).to(device)
                images.append(image_tensor)
                
                # Create target dict
                try:
                    box_tensor = boxes.to(device) if isinstance(boxes, torch.Tensor) else torch.tensor(boxes).to(device)
                    label_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
                    img_id_tensor = image_id if isinstance(image_id, torch.Tensor) else torch.tensor([image_id]).to(device)
                    
                    target = {
                        'boxes': box_tensor,
                        'labels': label_tensor,
                        'image_id': img_id_tensor
                    }
                    targets.append(target)
                except Exception as e:
                    print(f"Error processing target for batch item {batch_idx}: {e}")
                    continue
            
            # Skip if no valid images
            if len(images) == 0:
                continue
                
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Backward pass
            losses.backward()
            
            # Apply gradient clipping if specified
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
            # Update weights
            optimizer.step()
            
            # Update running losses
            running_loss += losses.item()
            running_loss_classifier += loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
            running_loss_box_reg += loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
            running_loss_objectness += loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
            running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
            
            # Update progress bar
            if i % print_freq == 0 or i == len(data_loader) - 1:
                avg_loss = running_loss / (i + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'cls': f'{running_loss_classifier / (i + 1):.4f}',
                    'bbox': f'{running_loss_box_reg / (i + 1):.4f}',
                    'obj': f'{running_loss_objectness / (i + 1):.4f}',
                    'rpn': f'{running_loss_rpn_box_reg / (i + 1):.4f}'
                })
        except Exception as e:
            print(f"Error in training batch {i}: {e}")
            continue
    
    # Return average losses for the epoch
    metrics = {
        'loss': running_loss / len(data_loader),
        'loss_classifier': running_loss_classifier / len(data_loader),
        'loss_box_reg': running_loss_box_reg / len(data_loader),
        'loss_objectness': running_loss_objectness / len(data_loader),
        'loss_rpn_box_reg': running_loss_rpn_box_reg / len(data_loader)
    }
    
    return metrics

# Add this helper function at the top of your file or in the visualization function
def draw_boxes_with_labels(image_tensor, boxes, labels, colors="red", width=4, font_size=16):
    """
    Custom function to draw boxes with labels using PIL to ensure font rendering works.
    """
    # Convert tensor to PIL Image
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Convert tensor to numpy (assuming it's already scaled 0-255)
    if image_tensor.dtype != torch.uint8:
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    else:
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Convert to PIL
    pil_image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to find a suitable font
    try:
        import matplotlib.font_manager as fm
        system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        font = None
        
        if system_fonts:
            # Try some common fonts
            for font_name in ['DejaVuSans.ttf', 'Arial.ttf', 'FreeSans.ttf', 'LiberationSans-Regular.ttf']:
                matching_fonts = [f for f in system_fonts if font_name in f]
                if matching_fonts:
                    font = ImageFont.truetype(matching_fonts[0], size=font_size)
                    break
            
            # If no preferred font found, use first available
            if font is None and system_fonts:
                font = ImageFont.truetype(system_fonts[0], size=font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw boxes and labels
    for box, label in zip(boxes, labels):
        box_coords = box.tolist()
        
        # Convert color name to RGB
        if colors == "red":
            color = (255, 0, 0)
        elif colors == "green":
            color = (0, 255, 0)
        elif colors == "blue":
            color = (0, 0, 255)
        else:
            color = (255, 255, 255)  # Default to white
        
        # Draw box
        draw.rectangle(box_coords, outline=color, width=width)
        
        # Draw label background
        text_width, text_height = draw.textsize(label, font=font) if font else (len(label) * font_size // 2, font_size)
        draw.rectangle(
            [box_coords[0], box_coords[1] - text_height - 4, box_coords[0] + text_width, box_coords[1]],
            fill=color
        )
        
        # Draw text
        draw.text((box_coords[0], box_coords[1] - text_height - 2), label, fill=(0, 0, 0), font=font)
    
    # Convert back to tensor
    result_np = np.array(pil_image)
    result_tensor = torch.from_numpy(result_np).permute(2, 0, 1)
    
    if image_tensor.dtype != torch.uint8:
        result_tensor = result_tensor.float() / 255.0
        
    return result_tensor

from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as TF
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# This is the modified evaluate_map function to correctly use the improved visualization

# Fix for evaluate_map function
def evaluate_map(model, data_loader, device, writer=None, epoch=0, class_names=None):
    """
    Calculate mean Average Precision (mAP) metrics using torchmetrics.
    """
    model.eval()
    
    # Initialize metric
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_thresholds=[0.5, 0.75],
        rec_thresholds=None,
        max_detection_thresholds=[1, 10, 1000],
        class_metrics=True,
    )
    
    # Create progress bar
    pbar = tqdm(data_loader, desc="Calculating mAP")
    
    # Store images and targets for visualization
    vis_images = []
    vis_predictions = []
    vis_targets = []
    has_visualized = False
    
    # Process validation data
    with torch.no_grad():
        for i, data in enumerate(pbar):
            try:
                # Prepare images and targets
                images = []
                targets = []
                
                # Process the data based on the observed structure
                for batch_idx in range(len(data[0])):
                    # Get the elements for this batch item
                    image = data[0][batch_idx]
                    boxes = data[1][batch_idx]
                    labels = data[2][batch_idx]
                    image_id = data[3][batch_idx]
                    
                    # Skip if any element is a string
                    if isinstance(image, str) or isinstance(boxes, str) or isinstance(labels, str):
                        continue
                    
                    # Convert to tensors if needed and move to device
                    image_tensor = image.to(device) if isinstance(image, torch.Tensor) else torch.tensor(image).to(device)
                    images.append(image_tensor)
                    
                    # Create target dict - with simplified validation
                    try:
                        box_tensor = boxes.to(device) if isinstance(boxes, torch.Tensor) else torch.tensor(boxes).to(device)
                        label_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
                        img_id_tensor = image_id if isinstance(image_id, torch.Tensor) else torch.tensor([image_id])
                        
                        # Create target dict - WITH image_id preserved
                        target = {
                            'boxes': box_tensor,
                            'labels': label_tensor,
                            'image_id': img_id_tensor
                        }
                        targets.append(target)
                    except Exception as e:
                        print(f"Error processing target for batch item {batch_idx}: {e}")
                        continue
                
                # Skip if no valid images
                if len(images) == 0 or len(targets) == 0:
                    continue
                
                # Get predictions from model
                predictions = model(images)
                
                # Save first batch for visualization
                if i == 0 and not has_visualized and writer is not None and epoch % 5 == 0:
                    vis_images = images.copy()
                    vis_predictions = predictions.copy()
                    vis_targets = targets.copy()
                    has_visualized = True
                
                # Format predictions and targets for torchmetrics
                preds_formatted = []
                targets_formatted = []
                
                for pred, target in zip(predictions, targets):
                    # Format prediction
                    pred_dict = {
                        'boxes': pred['boxes'].cpu(),
                        'scores': pred['scores'].cpu(),
                        'labels': pred['labels'].cpu(),
                    }
                    preds_formatted.append(pred_dict)
                    
                    # Format target - no need for image_id for metrics
                    target_dict = {
                        'boxes': target['boxes'].cpu(),
                        'labels': target['labels'].cpu(),
                    }
                    targets_formatted.append(target_dict)
                
                # Update metric
                metric.update(preds_formatted, targets_formatted)
                
            except Exception as e:
                print(f"Error in mAP calculation batch {i}: {e}")
                continue
    
    # Visualize after all batches have been processed
    if has_visualized and writer is not None:
        # Limit to 5 images maximum
        max_images = min(5, len(vis_images))
        visualize_predictions(
            vis_images[:max_images], 
            vis_predictions[:max_images], 
            vis_targets[:max_images], 
            writer,  # Pass writer directly as expected
            epoch, 
            class_names=class_names
        )
    
    # Compute final metric
    result = metric.compute()
    
    # Log detailed results
    print("\nMean Average Precision Results:")
    print(f"mAP: {result['map'].item():.4f}")
    print(f"mAP@0.5: {result['map_50'].item():.4f}")
    print(f"mAP@0.75: {result['map_75'].item():.4f}")
    print(f"mAP small: {result['map_small'].item():.4f}")
    print(f"mAP medium: {result['map_medium'].item():.4f}")
    print(f"mAP large: {result['map_large'].item():.4f}")
    
    # Log to TensorBoard if writer is provided
    if writer is not None:
        writer.add_scalar('metrics/mAP', result['map'].item(), epoch)
        writer.add_scalar('metrics/mAP@0.5', result['map_50'].item(), epoch)
        writer.add_scalar('metrics/mAP@0.75', result['map_75'].item(), epoch)
        writer.add_scalar('metrics/mAP_small', result['map_small'].item(), epoch)
        writer.add_scalar('metrics/mAP_medium', result['map_medium'].item(), epoch)
        writer.add_scalar('metrics/mAP_large', result['map_large'].item(), epoch)
    
    return {
        'mAP': result['map'].item(),
        'mAP@0.5': result['map_50'].item(),
        'mAP@0.75': result['map_75'].item(),
        'mAP_small': result['map_small'].item(),
        'mAP_medium': result['map_medium'].item(),
        'mAP_large': result['map_large'].item(),
    }
def visualize_predictions(images, predictions, targets, writer, epoch, class_names=None, threshold=0.4):
    """
    Visualize model predictions using matplotlib with adjusted bounding boxes.
    
    Args:
        images: List of image tensors
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        writer: TensorBoard SummaryWriter
        epoch: Current epoch number
        class_names: Dictionary mapping class IDs to class names
        threshold: Confidence threshold for showing predictions
    """
    import os
    import cv2
    import json
    import traceback
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import urllib.parse
    from PIL import Image
    import io
    import torch
    
    print("\nStarting high-quality visualization with adjusted boxes...")
    
    # Path to validation images directory
    images_dir = '/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset/val/images'
    annotations_file = '/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset/val/annotations.json'
    
    # Define colors for different classes (RGB format for matplotlib)
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
    
    # Create a color map for unique class IDs
    def create_color_map(unique_class_ids):
        color_map = {}
        staff_line_color = (0.6, 0.6, 0.6)  # Gray for staff lines
        
        for i, class_id in enumerate(unique_class_ids):
            class_name = class_names.get(class_id, f"Class {class_id}").lower() if class_names else f"Class {class_id}"
            if isinstance(class_name, str) and "staff" in class_name:
                color_map[class_id] = staff_line_color
            else:
                color_map[class_id] = colors[i % len(colors)]
        
        return color_map
    
    # Load annotations to get image file names by ID
    id_to_filename = {}
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Create mapping from image_id to file_name
        for img in annotations['images']:
            id_to_filename[img['id']] = img['file_name']
            
        print(f"Loaded mapping for {len(id_to_filename)} images from annotations")
    except Exception as e:
        print(f"Error loading annotations: {e}")
        traceback.print_exc()
        return
    
    # Limit the number of images to visualize
    num_to_visualize = min(5, len(images))
    print(f"Will visualize {num_to_visualize} images")
    
    # Also plot ground truth for comparison
    show_ground_truth = True
    
    visualized_count = 0
    for i in range(num_to_visualize):
        try:
            # Get current sample
            image = images[i]
            pred = predictions[i]
            target = targets[i]
            
            # Get image ID with detailed debugging
            if 'image_id' not in target:
                print(f"No image_id in target for image {i}")
                continue
            
            # Get image ID and convert from tensor if needed
            image_id_tensor = target['image_id']
            if isinstance(image_id_tensor, torch.Tensor):
                if image_id_tensor.numel() == 1:
                    image_id = image_id_tensor.item()
                else:
                    image_id = image_id_tensor[0].item()
            else:
                image_id = image_id_tensor
            
            # Find original image file
            if image_id in id_to_filename:
                filename = id_to_filename[image_id]
                image_path = os.path.join(images_dir, filename)
                
                if os.path.exists(image_path):
                    try:
                        # Load original image
                        try:
                            # Try PIL first
                            original_img_pil = Image.open(image_path).convert('RGB')
                            original_img = np.array(original_img_pil)
                        except Exception as e:
                            # Fall back to OpenCV
                            original_img = cv2.imread(image_path)
                            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                        
                        # Get dimensions
                        orig_height, orig_width = original_img.shape[:2]
                        _, current_height, current_width = image.shape
                        
                        # Compute scaling factors
                        width_scale = orig_width / current_width
                        height_scale = orig_height / current_height
                        
                        # Debug scaling factors
                        print(f"Image dimensions: original={orig_width}x{orig_height}, model={current_width}x{current_height}")
                        print(f"Scaling factors: width={width_scale}, height={height_scale}")
                        
                        # Create a 1x2 subplot if showing ground truth, otherwise just one plot
                        if show_ground_truth:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40, orig_height / orig_width * 20))
                            ax1.imshow(original_img)
                            ax2.imshow(original_img)
                            ax1.set_title("Ground Truth", fontsize=20)
                            ax2.set_title("Predictions", fontsize=20)
                        else:
                            fig, ax = plt.subplots(figsize=(20, orig_height / orig_width * 20))
                            ax.imshow(original_img)
                        
                        # Plot ground truth boxes in left subplot
                        if show_ground_truth:
                            gt_boxes = target['boxes'].cpu().numpy()
                            gt_labels = target['labels'].cpu().numpy()
                            
                            # Get unique classes for color mapping
                            gt_unique_classes = set(gt_labels)
                            gt_color_map = create_color_map(gt_unique_classes)
                            
                            for box, label_id in zip(gt_boxes, gt_labels):
                                # Scale box to original image dimensions
                                x1 = box[0] * width_scale
                                y1 = box[1] * height_scale
                                x2 = box[2] * width_scale
                                y2 = box[3] * height_scale
                                
                                width = x2 - x1
                                height = y2 - y1
                                
                                # Get class name and color
                                class_name = class_names.get(label_id, f"Class {label_id}") if class_names else f"Class {label_id}"
                                color = gt_color_map.get(label_id, (0.0, 0.8, 0.0))  # Default to green
                                
                                # Draw bounding box
                                rect = patches.Rectangle(
                                    (x1, y1),
                                    width,
                                    height,
                                    linewidth=2,
                                    edgecolor=color,
                                    facecolor="none"
                                )
                                ax1.add_patch(rect)
                                
                                # Draw label
                                ax1.text(
                                    x1, 
                                    y1 - 5 if y1 > 10 else y2 + 5,
                                    class_name,
                                    color="black",
                                    fontsize=10,
                                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                                    ha="left",
                                    va="bottom" if y1 > 10 else "top"
                                )
                            
                            # Turn off axis for ground truth subplot
                            ax1.axis("off")
                        
                        # Process prediction boxes
                        pred_boxes = pred['boxes'].cpu().numpy()
                        pred_scores = pred['scores'].cpu().numpy()
                        pred_labels = pred['labels'].cpu().numpy()
                        
                        # Get unique classes for color mapping
                        pred_unique_classes = set(pred_labels)
                        pred_color_map = create_color_map(pred_unique_classes)
                        
                        # Filter predictions with score > threshold
                        keep = pred_scores > threshold
                        print(f"  - Predictions above threshold ({threshold}): {keep.sum()}/{len(pred_scores)}")
                        
                        if keep.sum() > 0:
                            pred_boxes = pred_boxes[keep]
                            pred_scores = pred_scores[keep]
                            pred_labels = pred_labels[keep]
                            
                            # Draw prediction boxes
                            for box, score, label_id in zip(pred_boxes, pred_scores, pred_labels):
                                # Scale box to original image dimensions
                                x1 = box[0] * width_scale
                                y1 = box[1] * height_scale
                                x2 = box[2] * width_scale
                                y2 = box[3] * height_scale
                                
                                width = x2 - x1
                                height = y2 - y1
                                
                                # Get class name and color
                                class_name = class_names.get(label_id, f"Class {label_id}") if class_names else f"Class {label_id}"
                                color = pred_color_map.get(label_id, (1.0, 0.0, 0.0))  # Default to red
                                
                                # Draw bounding box
                                rect = patches.Rectangle(
                                    (x1, y1),
                                    width,
                                    height,
                                    linewidth=2,
                                    edgecolor=color,
                                    facecolor="none"
                                )
                                
                                if show_ground_truth:
                                    ax2.add_patch(rect)
                                else:
                                    ax.add_patch(rect)
                                
                                # Prepare label text with class and confidence
                                label_text = f"{class_name} ({score:.2f})"
                                
                                # Calculate label position
                                y_offset = y1 - 5
                                if y_offset < 10:
                                    y_offset = y2 + 5  # Place below if not enough space above
                                
                                # Draw label with background box
                                if show_ground_truth:
                                    ax2.text(
                                        x1, 
                                        y_offset,
                                        label_text,
                                        color="black",
                                        fontsize=10,
                                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                                        ha="left",
                                        va="bottom" if y_offset == y1 - 5 else "top"
                                    )
                                else:
                                    ax.text(
                                        x1, 
                                        y_offset,
                                        label_text,
                                        color="black",
                                        fontsize=10,
                                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
                                        ha="left",
                                        va="bottom" if y_offset == y1 - 5 else "top"
                                    )
                        else:
                            # No predictions above threshold
                            message = f"No predictions above threshold ({threshold})"
                            if show_ground_truth:
                                ax2.text(
                                    0.5, 0.5,
                                    message,
                                    color="red",
                                    fontsize=20,
                                    ha="center",
                                    va="center",
                                    transform=ax2.transAxes,
                                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="red", pad=10)
                                )
                            else:
                                ax.text(
                                    0.5, 0.5,
                                    message,
                                    color="red",
                                    fontsize=20,
                                    ha="center",
                                    va="center",
                                    transform=ax.transAxes,
                                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="red", pad=10)
                                )
                        
                        # Set title and remove axes for the subplot(s)
                        if show_ground_truth:
                            ax2.axis("off")
                            fig.suptitle(f"Detection Results (Epoch {epoch}) - {filename}", fontsize=24)
                        else:
                            ax.set_title(f"Predictions (Epoch {epoch}) - {filename}", fontsize=20)
                            ax.axis("off")
                            
                        plt.tight_layout()
                        
                        # Save the plot to a buffer and convert to tensor for TensorBoard
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        
                        # Open as PIL Image
                        pil_img = Image.open(buf)
                        # Convert to tensor
                        img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1)
                        
                        # Add to TensorBoard
                        writer.add_image(f"detection/image_{i}", img_tensor, epoch)
                        
                        # Close buffer and figure to free memory
                        buf.close()
                        plt.close(fig)
                        
                        visualized_count += 1
                        print(f"Visualized image {i} successfully")
                        
                    except Exception as e:
                        print(f"Error during visualization: {e}")
                        traceback.print_exc()
                else:
                    print(f"Image file not found: {image_path}")
            else:
                print(f"Image ID {image_id} not found in mapping dictionary")
        except Exception as e:
            print(f"Error visualizing image {i}: {e}")
            traceback.print_exc()
    
    print(f"Visualization complete. Successfully visualized {visualized_count}/{num_to_visualize} images")
    return visualized_count
    
# Helper function to get image filename by ID
def get_image_filename_by_id(image_id, image_dir):
    """
    Get the image filename from its ID.
    This function needs to be adapted based on your dataset organization.
    """
    # Option 1: If you have a mapping between image IDs and filenames
    # For example, if you have a JSON file with this mapping
    mapping_file = os.path.join(image_dir, '../annotations.json')
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            annotations = json.load(f)
            for image_info in annotations.get('images', []):
                if image_info.get('id') == image_id:
                    return os.path.join(image_dir, image_info.get('file_name'))
    
    # Option 2: Search for image files that might contain the ID
    # This is less efficient but can work as a fallback
    for extension in ['.jpg', '.png', '.jpeg']:
        pattern = os.path.join(image_dir, f"*{image_id}*{extension}")
        matching_files = glob.glob(pattern)
        if matching_files:
            return matching_files[0]
    
    return None
        
def evaluate(model, data_loader, device):
    model.eval()
    
    # Create tqdm progress bar
    pbar = tqdm(data_loader, desc="Validation")
    
    # Metrics
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    num_processed_batches = 0
    
    with torch.no_grad():
        for i, data in enumerate(pbar):
            try:
                # Process one image at a time to avoid batch dimension issues
                for batch_idx in range(len(data[0])):
                    image = data[0][batch_idx]
                    boxes = data[1][batch_idx]
                    labels = data[2][batch_idx]
                    image_id = data[3][batch_idx]
                    
                    # Skip if any element is a string
                    if isinstance(image, str) or isinstance(boxes, str) or isinstance(labels, str):
                        continue
                    
                    # Convert to device
                    image_tensor = image.to(device) if isinstance(image, torch.Tensor) else torch.tensor(image).to(device)
                    
                    # Ensure image is properly shaped for model input [C, H, W]
                    if image_tensor.dim() == 4:  # [B, C, H, W] -> [C, H, W]
                        image_tensor = image_tensor.squeeze(0)
                    
                    # Process targets
                    try:
                        box_tensor = boxes.to(device) if isinstance(boxes, torch.Tensor) else torch.tensor(boxes).to(device)
                        label_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
                        img_id_tensor = image_id if isinstance(image_id, torch.Tensor) else torch.tensor([image_id]).to(device)
                        
                        # Validate box format
                        if box_tensor.dim() == 1 and box_tensor.size(0) == 4:  # Single box
                            box_tensor = box_tensor.unsqueeze(0)  # [4] -> [1, 4]
                        
                        # Validate that boxes are properly formed
                        if box_tensor.dim() != 2 or box_tensor.size(1) != 4:
                            print(f"Invalid box tensor shape: {box_tensor.shape}. Skipping...")
                            continue
                            
                        # Create target dict for a single image
                        target = {
                            'boxes': box_tensor,
                            'labels': label_tensor,
                            'image_id': img_id_tensor
                        }
                        
                        # Switch to training mode temporarily to get loss values
                        model.train()
                        loss_dict = model([image_tensor], [target])
                        model.eval()  # Switch back to eval mode
                        
                        # Handle the case where loss_dict is a list
                        if isinstance(loss_dict, list):
                            if len(loss_dict) > 0 and isinstance(loss_dict[0], dict):
                                loss_dict = loss_dict[0]  # Take the first dictionary
                            else:
                                # Skip if we can't get a proper loss_dict
                                print(f"Unexpected loss_dict format: {type(loss_dict)}")
                                continue
                        
                        # Sum losses
                        losses = sum(loss for loss in loss_dict.values())
                        
                        # Update running losses
                        running_loss += losses.item()
                        running_loss_classifier += loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
                        running_loss_box_reg += loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
                        running_loss_objectness += loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
                        running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
                        num_processed_batches += 1
                        
                    except Exception as e:
                        print(f"Error processing single image in batch {i}, item {batch_idx}: {e}")
                        continue
                        
                # Update progress bar
                pbar.set_postfix({
                    'val_loss': f'{running_loss / max(1, num_processed_batches):.4f}'
                })
                
            except Exception as e:
                print(f"Error in validation batch {i}: {e}")
                continue
    
    # Return average losses
    num_batches = max(1, num_processed_batches)  # Avoid division by zero
    metrics = {
        'val_loss': running_loss / num_batches,
        'val_loss_classifier': running_loss_classifier / num_batches,
        'val_loss_box_reg': running_loss_box_reg / num_batches,
        'val_loss_objectness': running_loss_objectness / num_batches,
        'val_loss_rpn_box_reg': running_loss_rpn_box_reg / num_batches
    }
    
    return metrics
# def evaluate(model, data_loader, device):
#     """
#     Evaluate the model on validation data.
#     """
#     model.eval()
    
#     # Create tqdm progress bar
#     pbar = tqdm(data_loader, desc="Validation")
    
#     # Metrics
#     running_loss = 0.0
#     running_loss_classifier = 0.0
#     running_loss_box_reg = 0.0
#     running_loss_objectness = 0.0
#     running_loss_rpn_box_reg = 0.0
    
#     with torch.no_grad():
#         for i, data in enumerate(pbar):
#             try:
#                 # Handle data properly based on its actual structure
#                 images = []
#                 targets = []
                
#                 # Process the data based on the observed structure
#                 for batch_idx in range(len(data[0])):  # Iterate through batch
#                     # Get the elements for this batch item
#                     image = data[0][batch_idx]
#                     boxes = data[1][batch_idx]
#                     labels = data[2][batch_idx]
#                     image_id = data[3][batch_idx]
                    
#                     # Skip if any element is a string
#                     if isinstance(image, str) or isinstance(boxes, str) or isinstance(labels, str):
#                         continue
                    
#                     # Convert to tensors if needed and move to device
#                     image_tensor = image.to(device) if isinstance(image, torch.Tensor) else torch.tensor(image).to(device)
#                     images.append(image_tensor)
                    
#                     try:
#                         # Create target dict - with additional validation
#                         box_tensor = boxes.to(device) if isinstance(boxes, torch.Tensor) else torch.tensor(boxes).to(device)
                        
#                         # Check box tensor shape
#                         if box_tensor.dim() == 1 and box_tensor.size(0) == 4:  # Single box with shape [4]
#                             box_tensor = box_tensor.unsqueeze(0)  # Convert to shape [1, 4]
                        
#                         # Validate box format
#                         if box_tensor.dim() != 2 or box_tensor.size(1) != 4:
#                             print(f"Invalid box tensor shape: {box_tensor.shape} in batch {i}, item {batch_idx}. Skipping...")
#                             continue
                            
#                         # Ensure box coordinates are valid (xmin < xmax, ymin < ymax)
#                         valid_boxes = (box_tensor[:, 2] > box_tensor[:, 0]) & (box_tensor[:, 3] > box_tensor[:, 1])
#                         if not valid_boxes.all():
#                             print(f"Invalid box coordinates in batch {i}, item {batch_idx}. Skipping...")
#                             continue
                            
#                         label_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
#                         img_id_tensor = image_id.to(device) if isinstance(image_id, torch.Tensor) else torch.tensor([image_id]).to(device)
                        
#                         target = {
#                             'boxes': box_tensor,
#                             'labels': label_tensor,
#                             'image_id': img_id_tensor
#                         }
#                         targets.append(target)
#                     except Exception as e:
#                         print(f"Error processing target for batch item {batch_idx}: {e}")
#                         continue
                
#                 # Skip if no valid images or targets
#                 if len(images) == 0 or len(targets) == 0:
#                     print(f"Skipping batch {i} - no valid images or targets")
#                     continue
                
#                 # Debug information for tensor dimensions
#                 if i % 500 == 0:  # Print every 500 batches to avoid too much output
#                     print(f"Validation batch {i}:")
#                     print(f"  Image shape: {images[0].shape}")
#                     print(f"  Number of boxes: {targets[0]['boxes'].size(0)}")
#                     print(f"  Box shape: {targets[0]['boxes'].shape}")
#                     if targets[0]['boxes'].size(0) > 0:
#                         print(f"  First box: {targets[0]['boxes'][0]}")
                
#                 # Forward pass
#                 # loss_dict = model(images, targets)
#                 # With:
#                 # Add this to the evaluate function before the model forward pass
#                 print(f"Image tensor shape before model: {images[0].shape}")
#                 for key, value in targets[0].items():
#                     if isinstance(value, torch.Tensor):
#                         print(f"Target {key} shape: {value.shape}")
#                 try:
#                     loss_dict = model(images, targets)
#                 except Exception as e:
#                     print(f"Error during model forward pass in batch {i}:")
#                     print(f"  Error: {e}")
#                     print(f"  Image shapes: {[img.shape for img in images]}")
#                     print(f"  Box shapes: {[t['boxes'].shape for t in targets]}")
#                     continue


                
#                 # Check that loss_dict is a dictionary, not a list
#                 if isinstance(loss_dict, list):
#                     # If it's a list, take the first item (which should be a dict)
#                     # This happens when the model returns results per image
#                     if len(loss_dict) > 0 and isinstance(loss_dict[0], dict):
#                         loss_dict = loss_dict[0]
#                     else:
#                         # Skip this batch if we can't get a proper loss_dict
#                         print(f"Skipping batch {i} - loss_dict is not in expected format")
#                         continue
                
#                 # Calculate total loss
#                 losses = sum(loss for loss in loss_dict.values())
                
#                 # Update running losses
#                 running_loss += losses.item()
#                 running_loss_classifier += loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
#                 running_loss_box_reg += loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
#                 running_loss_objectness += loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
#                 running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
                
#                 # Update progress bar
#                 pbar.set_postfix({
#                     'val_loss': f'{running_loss / (i + 1):.4f}'
#                 })
#             except Exception as e:
#                 print(f"Error in validation batch {i}: {e}")
#                 continue
    
#     # Return average losses for the epoch
#     num_batches = max(1, len(data_loader))  # Avoid division by zero
#     metrics = {
#         'val_loss': running_loss / num_batches,
#         'val_loss_classifier': running_loss_classifier / num_batches,
#         'val_loss_box_reg': running_loss_box_reg / num_batches,
#         'val_loss_objectness': running_loss_objectness / num_batches,
#         'val_loss_rpn_box_reg': running_loss_rpn_box_reg / num_batches
#     }
    
#     return metrics

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, args, is_best=False, global_step=0):
    """
    Save model checkpoint.
    """
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'metrics': metrics,
        'args': args,
        'global_step': global_step
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # If this is the best model, save as best.pt
    if is_best:
        best_path = os.path.join(args.output_dir, 'best.pt')
        torch.save(checkpoint, best_path)
    
    # Save latest.pt for easy resuming
    latest_path = os.path.join(args.output_dir, 'latest.pt')
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path

def plot_losses(train_metrics, val_metrics, args):
    """
    Plot training and validation losses.
    """
    epochs = range(1, len(train_metrics) + 1)
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot total loss
    ax1.plot(epochs, [m['loss'] for m in train_metrics], 'b-', label='Training Loss')
    ax1.plot(epochs, [m['val_loss'] for m in val_metrics], 'r-', label='Validation Loss')
    ax1.set_title('Total Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot component losses
    ax2.plot(epochs, [m['loss_classifier'] for m in train_metrics], 'b-', label='Classifier')
    ax2.plot(epochs, [m['loss_box_reg'] for m in train_metrics], 'g-', label='Box Reg')
    ax2.plot(epochs, [m['loss_objectness'] for m in train_metrics], 'r-', label='Objectness')
    ax2.plot(epochs, [m['loss_rpn_box_reg'] for m in train_metrics], 'y-', label='RPN Box Reg')
    ax2.set_title('Component Losses (Training)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(args.output_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    plt.close()

def visualize_batch(writer, images, targets, predictions=None, step=0, prefix="train"):
    """
    Visualize a batch of images with their ground truth boxes and predictions in TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        images: List of tensor images
        targets: List of target dictionaries with 'boxes' and 'labels'
        predictions: Optional list of prediction dictionaries with 'boxes', 'labels', and 'scores'
        step: Global step for TensorBoard
        prefix: Prefix for the tag in TensorBoard
    """
    max_images = min(5, len(images))  # Visualize at most 5 images to avoid clutter
    
    for i in range(max_images):
        # Convert tensor to PIL Image
        image = images[i].cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(image_pil)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype('/homes/es314/omr-objdet-benchmark/scripts/faster_rcnn/dejavu-sans/DejaVuSans.ttf', 12)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw ground truth boxes in green
        boxes = targets[i]['boxes'].cpu().numpy()
        labels = targets[i]['labels'].cpu().numpy()
        for box, label in zip(boxes, labels):
            draw.rectangle(box.tolist(), outline=(0, 255, 0), width=2)
            draw.text((box[0], box[1]), f"GT: {label}", fill=(0, 255, 0), font=font)
        
        # Draw predicted boxes in blue if available
        if predictions is not None:
            pred_boxes = predictions[i]['boxes'].cpu().numpy()
            pred_labels = predictions[i]['labels'].cpu().numpy()
            pred_scores = predictions[i]['scores'].cpu().numpy()
            
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                draw.rectangle(box.tolist(), outline=(0, 0, 255), width=2)
                draw.text((box[0], box[1] + 15), f"Pred: {label} ({score:.2f})", fill=(0, 0, 255), font=font)
        
        # Convert back to tensor for TensorBoard
        image_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
        
        # Add to TensorBoard
        writer.add_image(f"{prefix}/image_{i}", image_tensor, step)

def log_metrics(writer, metrics, step, prefix="train"):
    """
    Log metrics to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        metrics: Dictionary of metrics
        step: Global step for TensorBoard
        prefix: Prefix for the tag in TensorBoard
    """
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)

def run_inference_on_test_images(model, test_images_dir, class_names, device, writer, step, threshold=0.5):
    """
    Run inference on test images and log results to TensorBoard.
    
    Args:
        model: Trained model
        test_images_dir: Directory containing test images
        class_names: Dictionary mapping class IDs to class names
        device: Device to run inference on
        writer: TensorBoard SummaryWriter
        step: Global step for TensorBoard
        threshold: Detection threshold
    """
    if not os.path.exists(test_images_dir):
        return
    
    model.eval()
    
    # Get image files
    image_files = glob.glob(os.path.join(test_images_dir, '*.jpg')) + \
                 glob.glob(os.path.join(test_images_dir, '*.png'))
    
    if not image_files:
        return
    
    # Process 5 images at most
    image_files = image_files[:5]
    
    with torch.no_grad():
        for i, image_file in enumerate(image_files):
            # Load and preprocess image
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Prepare tensor
            image_tensor = F.to_tensor(image)
            image_tensor = image_tensor.to(device)
            
            # Run inference
            prediction = model([image_tensor])[0]
            
            # Filter predictions by threshold
            keep = prediction['scores'] > threshold
            boxes = prediction['boxes'][keep]
            labels = prediction['labels'][keep]
            scores = prediction['scores'][keep]
            
            # Create visualization
            image_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(image_pil)
            
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except IOError:
                font = ImageFont.load_default()
            
            # Generate colors
            colors = plt.cm.rainbow(np.linspace(0, 1, 100))
            colors = (colors[:, :3] * 255).astype(np.uint8)
            
            # Draw boxes
            for box, label, score in zip(boxes, labels, scores):
                box = box.cpu().numpy().astype(np.int32)
                label = label.cpu().item()
                score = score.cpu().item()
                
                # Get color for this class
                color = tuple(map(int, colors[label % len(colors)]))
                
                # Draw box
                draw.rectangle(box.tolist(), outline=color, width=3)
                
                # Get class name
                class_name = class_names.get(label, f"Class {label}")
                
                # Create label text
                label_text = f"{class_name}: {score:.2f}"
                
                # Draw text
                draw.text((box[0], box[1]), label_text, fill=color, font=font)
            
            # Convert back to tensor for TensorBoard
            image_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
            
            # Add to TensorBoard
            writer.add_image(f"inference/image_{i}", image_tensor, step)

def get_model_instance_segmentation(num_classes, args):
    """
    Get Faster R-CNN model with the specified backbone and adapted to match TensorFlow config.
    
    Args:
        num_classes: Number of classes to predict (including background)
        args: Command line arguments
        
    Returns:
        model: The Faster R-CNN model
    """
    # # Parse anchor parameters
    # anchor_sizes = tuple((float(size),) for size in args.anchor_sizes.split(','))
    # aspect_ratios = tuple((float(ratio),) for ratio in args.aspect_ratios.split(','))
    anchor_sizes = tuple([int(size) for size in args.anchor_sizes.split(',')])
    aspect_ratios = tuple([float(ratio) for ratio in args.aspect_ratios.split(',')])
    
    # Create more appropriate anchor sizes and aspect ratios for you
    # Select backbone
    if args.backbone == 'resnet50':
        model = fasterrcnn_resnet50_fpn(
            pretrained=args.pretrained,
            min_size=args.min_size,
            max_size=args.max_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
        if args.backbone != 'resnet50' and not HAS_RESNET101:
            print(f"Warning: {args.backbone} not available, using resnet50 instead")
    elif args.backbone == 'resnet101' and HAS_RESNET101:
        model = fasterrcnn_resnet101_fpn(
            pretrained=args.pretrained,
            min_size=args.min_size,
            max_size=args.max_size
        )
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    
    # Modify anchor generator to match TF config
    anchor_generator = model.rpn.anchor_generator
    anchor_generator.sizes = anchor_sizes
    anchor_generator.aspect_ratios = aspect_ratios
    
    # Modify RPN parameters to match TF config as closely as possible
    model.rpn.nms_thresh = args.first_stage_nms_iou_threshold
    model.rpn.score_thresh = args.first_stage_nms_score_threshold
    model.rpn.post_nms_top_n_train = args.first_stage_max_proposals
    model.rpn.post_nms_top_n_test = args.first_stage_max_proposals
    
    # Modify ROI parameters to match TF config
    # model.roi_heads.nms_thresh = args.second_stage_nms_iou_threshold
    # model.roi_heads.score_thresh = args.second_stage_nms_score_threshold
    # model.roi_heads.detections_per_img = args.second_stage_max_total_detections
    # Modify ROI parameters to match TF config, but with a reasonable detection limit
    model.roi_heads.nms_thresh = args.second_stage_nms_iou_threshold
    model.roi_heads.score_thresh = args.second_stage_nms_score_threshold
    model.roi_heads.detections_per_img = min(1000, args.second_stage_max_total_detections)  # Cap at 100
    # Freeze backbone layers if requested
    if args.frozen_layers > 0:
        for name, parameter in model.backbone.named_parameters():
            layer_num = int(name.split('.')[1]) if '.' in name and name.split('.')[1].isdigit() else -1
            if layer_num < args.frozen_layers:
                parameter.requires_grad_(False)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new head with the correct number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory and log directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Save args to a JSON file for future reference
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device based on gpu_id
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Parse image size
    if ',' in args.image_size:
        min_dim, max_dim = map(int, args.image_size.split(','))
    else:
        min_dim = max_dim = int(args.image_size)
    print(f"Using image size: min={min_dim}, max={max_dim}")
    
    # Load data
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    # Load mapping file to get number of classes and create class_names dict
    mapping_file = os.path.join(args.data_dir, 'mapping.txt')
    class_names = {}
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        class_id = int(parts[0])
                        class_name = ':'.join(parts[1:])
                        class_names[class_id] = class_name
    except FileNotFoundError:
        print(f"Warning: Mapping file {mapping_file} not found. Using default num_classes from args.")
    
    # If class_names is empty or we're using args.num_classes directly
    num_classes = len(class_names) + 1 if class_names else args.num_classes
    print(f"Training with {num_classes} classes (including background)")
    
    # Log all model arguments to TensorBoard
    writer.add_text('training/args', str(vars(args)), 0)
    
    # Initialize custom transforms that incorporate image resizing
    # def train_transform(image, target):
    #     # First apply standard transformations
    #     image, target = get_transform(
    #         train=True, 
    #         min_size=args.min_size,
    #         max_size=args.max_size
    #     )(image, target)
    #     return image, target
    
    # def val_transform(image, target):
    #     # First apply standard transformations
    #     image, target = get_transform(
    #         train=False,
    #         min_size=args.min_size,
    #         max_size=args.max_size
    #     )(image, target)
    #     return image, target
    # Initialize custom transforms that incorporate image resizing
    # def train_transform(image, target):
    #     # First apply standard transformations
    #     image, target = get_transform(train=True)(image, target)
    #     return image, target

    # def val_transform(image, target):
    #     # First apply standard transformations
    #     image, target = get_transform(train=False)(image, target)
    #     return image, target
    # Initialize custom transforms that incorporate image resizing
    def train_transform(image, target):
        # Now pass the min_size and max_size parameters to get_transform
        image, target = get_transform(
            train=True,
            min_size=args.min_size,
            max_size=args.max_size
        )(image, target)
        return image, target

    def val_transform(image, target):
        # Now pass the min_size and max_size parameters to get_transform
        image, target = get_transform(
            train=False,
            min_size=args.min_size,
            max_size=args.max_size
        )(image, target)
        return image, target
    
    print("Initializing datasets...")
    # Initialize datasets and data loaders
    train_dataset = OMRDataset(
        root_dir=train_dir,
        transforms=train_transform,
        is_train=True
    )
    
    val_dataset = OMRDataset(
        root_dir=val_dir,
        transforms=val_transform,
        is_train=False
    )
    
    # Apply subsetting if data_subset < 1.0
    if args.data_subset < 1.0:
        # Calculate subset sizes
        train_size = int(len(train_dataset) * args.data_subset)
        val_size = int(len(val_dataset) * args.data_subset)
        
        # Create subset indices
        train_indices = torch.randperm(len(train_dataset))[:train_size]
        val_indices = torch.randperm(len(val_dataset))[:val_size]
        
        # Create subset samplers
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        
        print(f"Using {train_size}/{len(train_dataset)} training samples and {val_size}/{len(val_dataset)} validation samples")
        
        # Create data loaders with samplers
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,  # Use sampler instead of shuffle
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=args.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            sampler=val_sampler,  # Use sampler instead of shuffle
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=args.pin_memory
        )
    else:
        # Original data loaders with all data
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=args.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=args.pin_memory
        )
    
    # ... rest of your main function ...
    print("Data loaders created")
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation images")
    
    print("Starting main training loop")
    print("Creating model...")
    # Initialize model
    model = get_model_instance_segmentation(num_classes, args)
    print("Moving model to device...")
    model.to(device)
    
    
    # Count and log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
    writer.add_text('model/params', f"Trainable: {trainable_params:,}, Total: {total_params:,}", 0)
    
    # Initialize optimizer - use RMSProp to match TF config
    print("Setting up optimizer...")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.RMSprop(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        alpha=0.9,  # decay in TF config
        eps=1.0     # epsilon in TF config
    )
    
    print("Setting up scheduler...")
    # Initialize exponential learning rate scheduler to match TF config
    lr_scheduler = ExponentialLR(
        optimizer,
        gamma=args.decay_factor**(1/args.decay_steps)  # Convert to per-step decay
    )
    
    print("Checking for resume checkpoint...")
    # Resume from checkpoint if specified
    start_epoch = args.start_epoch
    best_val_loss = float('inf')
    train_metrics_history = []
    val_metrics_history = []
    global_step = 0
    print("About to start epochs...")
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            try:
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch'] + 1
                
                if 'metrics' in checkpoint:
                    train_metrics_history = checkpoint['metrics'].get('train_history', [])
                    val_metrics_history = checkpoint['metrics'].get('val_history', [])
                    best_val_loss = min([m['val_loss'] for m in val_metrics_history]) if val_metrics_history else float('inf')
                
                if 'global_step' in checkpoint:
                    global_step = checkpoint['global_step']
                
                print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    print("Starting training")


    for epoch in range(start_epoch, args.num_epochs):
        # Train for one epoch
        epoch_start_time = time.time()
        
        # Use the train_one_epoch function with gradient clipping
        train_metrics = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            device, 
            epoch, 
            args.print_freq,
            clip_norm=args.gradient_clipping_by_norm
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Update global step
        global_step += len(train_loader)
        
        # Log metrics to TensorBoard
        log_metrics(writer, train_metrics, global_step)
        
        # Log current learning rate
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        
        # Evaluate on validation set (every eval_freq epochs)
        if epoch % args.eval_freq == 0 or epoch == args.num_epochs - 1:
            try:
                # Run evaluate function for losses
                val_metrics = evaluate(model, val_loader, device)
                
                # Run mAP evaluation less frequently (every 5 validation cycles or final epoch)
                if epoch % (args.eval_freq * 5) == 0 or epoch == args.num_epochs - 1:
                    print("Calculating mAP metrics...")
                    # map_metrics = evaluate_map(model, val_loader, device, writer, epoch)
                    # In your main function, where you call evaluate_map
                    # map_metrics = evaluate_map(model, val_loader, device, writer, epoch, class_names=class_names)
                    map_metrics = evaluate_map(
                        model, 
                        val_loader, 
                        device, 
                        writer, 
                        epoch, 
                        class_names=class_names
                        # image_dir = '/homes/es314/omr-objdet-benchmark/data/faster_rcnn_prepared_dataset/val/images'  # Path to original images
                    )
                    # Combine metrics
                    val_metrics.update(map_metrics)
                    
                    # Print mAP results
                    print("Mean Average Precision Results:")
                    for k, v in map_metrics.items():
                        print(f"  {k}: {v:.4f}")
                
                # Log validation metrics to TensorBoard
                log_metrics(writer, val_metrics, global_step, prefix="val")
                
                # Run inference on test images if provided
                if args.test_images_dir:
                    run_inference_on_test_images(
                        model, 
                        args.test_images_dir, 
                        class_names, 
                        device, 
                        writer, 
                        global_step
                    )
                
                # Print metrics
                print(f"Epoch {epoch} complete ({time.time() - epoch_start_time:.1f}s):")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Check if this is the best model so far
                is_best = val_metrics['val_loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                    print(f"  New best model! Val Loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Error during validation: {e}")
                traceback.print_exc()  # Print the full stack trace for debugging
                val_metrics = {'val_loss': float('nan')}
                is_best = False
        else:
            # If not evaluating, just use placeholders for validation metrics
            val_metrics = {
                'val_loss': float('nan'),
                'val_loss_classifier': float('nan'),
                'val_loss_box_reg': float('nan'),
                'val_loss_objectness': float('nan'),
                'val_loss_rpn_box_reg': float('nan')
            }
            is_best = False
            
            # Print metrics
            print(f"Epoch {epoch} complete ({time.time() - epoch_start_time:.1f}s):")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
        
        # Add metrics to history
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Save checkpoint (every save_freq epochs)
        if epoch % args.save_freq == 0 or is_best or epoch == args.num_epochs - 1:
            metrics = {
                'train_history': train_metrics_history,
                'val_history': val_metrics_history,
                'current_train': train_metrics,
                'current_val': val_metrics
            }
            
            checkpoint_path = save_checkpoint(
                model, optimizer, lr_scheduler,
                epoch, metrics, args, is_best,
                global_step=global_step
            )
            
            print(f"  Checkpoint saved to {checkpoint_path}")
        
        # Plot losses after each epoch
        try:
            plot_losses(train_metrics_history, val_metrics_history, args)
        except Exception as e:
            print(f"Error plotting losses: {e}")

    # Close TensorBoard writer
    writer.close()

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"All results saved to {args.output_dir}")
    print(f"TensorBoard logs saved to {args.log_dir}")
    
if __name__ == "__main__":
    main()