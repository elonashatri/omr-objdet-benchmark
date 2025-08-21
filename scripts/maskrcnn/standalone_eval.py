#!/usr/bin/env python3
"""
Standalone script for evaluating a trained Mask R-CNN model on OMR data
"""
import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import numpy as np
import time
import json
from tqdm import tqdm

# Import our custom modules
from omr_dataset_utils import OMRDataset, load_xml_dataset, visualize_sample
from evaluation_metrics import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OMR Mask R-CNN Model')
    
    # Model and data parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing XML and image files')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--img_ext', type=str, default='.png',
                        help='Image file extension (default: .png)')
    parser.add_argument('--img_size', type=str, default=None,
                        help='Resize images to this size (format: WIDTHxHEIGHT, e.g., 800x600)')
    
    # Evaluation parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda, cuda:0, cuda:1, cpu)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use if --device is not specified')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for evaluation')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for evaluation')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Confidence score threshold for evaluation')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions on sample images')
    parser.add_argument('--num_vis_samples', type=int, default=5,
                        help='Number of samples to visualize')
    
    return parser.parse_args()

def get_model_instance_segmentation(num_classes):
    """
    Get a Mask R-CNN model with a ResNet-50-FPN backbone
    """
    # Load model without pretrained weights
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Get number of input features for classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pretrained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get number of input features for mask classification
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model

def visualize_predictions(model, dataset, indices, device, output_dir):
    """
    Visualize model predictions on specific samples
    """
    model.eval()
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(image_tensor)[0]
        
        # Convert to numpy for visualization
        image_np = image.permute(1, 2, 0).numpy()
        
        # Get prediction components
        pred_boxes = prediction['boxes'].cpu().numpy()
        pred_scores = prediction['scores'].cpu().numpy()
        pred_labels = prediction['labels'].cpu().numpy()
        
        # Get ground truth
        gt_boxes = target['boxes'].numpy()
        gt_labels = target['labels'].numpy()
        
        # Apply score threshold
        score_threshold = 0.5
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]