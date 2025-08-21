#!/usr/bin/env python3
"""
Enhanced Multi-Scale Faster R-CNN Training Script for OMR/MUSCIMA Dataset

This script trains a Faster R-CNN model with advanced multi-scale techniques
for improved Optical Music Recognition, including:
1. Multi-scale training with variable input sizes
2. Enhanced Feature Pyramid Network 
3. Test-time augmentation
4. Image pyramid during inference

These improvements help detect musical notation elements of varying sizes.
"""
import os
import time
import datetime
import json
import glob
import torch
import torch.nn as nn
import traceback
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as F
import argparse
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# Import OMR dataset
from omr_dataset import OMRDataset, get_transform, FilteredOMRDataset



# Import multi-scale enhancements
from multi_scale_enhancements import (
    multi_scale_inference,
    multi_scale_transform,
    EnhancedFPN,
    test_time_augmentation,
    get_enhanced_model_instance_segmentation,
    adjust_training_pipeline,
    create_custom_transform_fn,
    enhance_evaluation_pipeline,
    filter_and_merge_classes  # Add this line
)

# Set up logging
print("Starting enhanced multi-scale OMR training...")
import sys
print(f"Python version: {sys.version}")
print(f"Arguments: {sys.argv}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multi-Scale Enhanced Faster R-CNN for OMR')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, 
                        default='/homes/es314/omr-objdet-benchmark/data/staff_faster_rcnn_prepared_dataset',
                        help='Directory containing prepared dataset')
    parser.add_argument('--output_dir', type=str, default='./staff_multiscale_faster_rcnn_omr',
                        help='Directory to save model checkpoints')
    parser.add_argument('--test_images_dir', type=str, default='',
                        help='Directory containing test images for inference during training')
    parser.add_argument('--data_subset', type=float, default=0.5,
                      help='Fraction of data to use (e.g., 0.1 for 10%)')
    
    # Image parameters with multi-scale support
    parser.add_argument('--image_size', type=str, default='600,1200',
                        help='Base image size (min_dimension,max_dimension) for resizing input images')
    parser.add_argument('--min_size', type=int, default=600,
                        help='Minimum size of the image to be rescaled before feeding it to the backbone')
    parser.add_argument('--max_size', type=int, default=1200,
                        help='Maximum size of the image to be rescaled before feeding it to the backbone')
    
    # Multi-scale parameters (new)
    parser.add_argument('--multi_scale_train', action='store_true', default=True,
                        help='Enable multi-scale training with variable input sizes')
    parser.add_argument('--multi_scale_min_sizes', type=str, default='400,500,600,700,800',
                        help='Comma-separated list of minimum sizes for multi-scale training')
    parser.add_argument('--multi_scale_inference', action='store_true', default=True,
                        help='Enable multi-scale inference with image pyramid')
    parser.add_argument('--inference_scales', type=str, default='0.5,0.75,1.0,1.25,1.5',
                        help='Comma-separated list of scales for multi-scale inference')
    parser.add_argument('--test_time_augmentation', action='store_true', default=True,
                        help='Enable test-time augmentation for improved results')
    parser.add_argument('--random_crop_prob', type=float, default=0.5,
                        help='Probability of applying random crop during training')
    
    # transformations
    parser.add_argument('--brightness_range', type=str, default='0.9,1.1',
                        help='Range of brightness variation during augmentation (min,max)')
    parser.add_argument('--contrast_range', type=str, default='1.0,1.2',
                        help='Range of contrast variation during augmentation (min,max)')
    parser.add_argument('--enable_sharpening', action='store_true', default=False,
                        help='Enable sharpening augmentation')
    parser.add_argument('--sharpening_prob', type=float, default=0.25,
                        help='Probability of applying sharpening during training')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=400,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Batch size for validation')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--decay_factor', type=float, default=0.95,
                        help='Decay factor for exponential learning rate decay')
    parser.add_argument('--num_steps', type=int, default=80000,
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
    parser.add_argument('--log_dir', type=str, default='./staff_multiscale_logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch')
    parser.add_argument('--num_visualizations', type=int, default=3,
                        help='Number of visualizations during evaluation')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=218,
                        help='Number of classes to detect')
    parser.add_argument('--backbone', type=str, default='resnet101-custom',
                        choices=['resnet50', 'resnet101', 'inception_resnet_v2', 'resnet101-custom'],
                        help='Backbone network for Faster R-CNN')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone network')
    parser.add_argument('--frozen_layers', type=int, default=0,
                        help='Number of layers to freeze in the backbone')
    parser.add_argument('--initial_crop_size', type=int, default=17,
                        help='Initial crop size')
    parser.add_argument('--maxpool_kernel_size', type=int, default=1,
                        help='Max pooling kernel size')
    parser.add_argument('--maxpool_stride', type=int, default=1,
                        help='Max pooling stride')
    parser.add_argument('--atrous_rate', type=int, default=2,
                        help='First stage atrous rate')
    
    # Anchor parameters
    parser.add_argument('--anchor_sizes', type=str, default='4,8,16,32,64,128',
                        help='Comma-separated list of anchor sizes')
    parser.add_argument('--aspect_ratios', type=str, default='0.05,0.1,0.25,1.0,2.0,4.0,10.0,20.0',
                        help='Comma-separated list of aspect ratios')
    parser.add_argument('--height_stride', type=int, default=4,
                        help='Height stride for anchor generator')
    parser.add_argument('--width_stride', type=int, default=4,
                        help='Width stride for anchor generator')
    parser.add_argument('--features_stride', type=int, default=4,
                        help='First stage features stride')
    
    # NMS parameters
    parser.add_argument('--first_stage_nms_score_threshold', type=float, default=0.05,
                        help='First stage NMS score threshold')
    parser.add_argument('--first_stage_nms_iou_threshold', type=float, default=0.5,
                        help='First stage NMS IoU threshold')
    parser.add_argument('--first_stage_max_proposals', type=int, default=2000,
                        help='First stage maximum proposals')
    parser.add_argument('--second_stage_nms_score_threshold', type=float, default=0.05,
                        help='Second stage NMS score threshold')
    parser.add_argument('--second_stage_nms_iou_threshold', type=float, default=0.4,
                        help='Second stage NMS IoU threshold')
    parser.add_argument('--second_stage_max_detections_per_class', type=int, default=2000,
                        help='Second stage maximum detections per class')
    parser.add_argument('--second_stage_max_total_detections', type=int, default=2500,
                    help='Second stage maximum total detections')
    
    # Loss weights
    parser.add_argument('--first_stage_localization_loss_weight', type=float, default=4.0,
                        help='First stage localization loss weight')
    parser.add_argument('--first_stage_objectness_loss_weight', type=float, default=2.0,
                        help='First stage objectness loss weight')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--decay_steps', type=int, default=40000,
                        help='Decay steps for exponential learning rate decay')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gradient_clipping_by_norm', type=float, default=10.0,
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
    Train the model for one epoch with multi-scale capabilities.
    
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
            traceback.print_exc()
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

def main():
    args = parse_args()
    
    # Process multi-scale parameters
    if args.multi_scale_min_sizes:
        args.multi_scale_min_sizes = [int(s) for s in args.multi_scale_min_sizes.split(',')]
    
    if args.inference_scales:
        args.inference_scales = [float(s) for s in args.inference_scales.split(',')]
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
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
    print(f"Using base image size: min={min_dim}, max={max_dim}")
    
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
    
    # Create custom transforms with multi-scale capabilities
    train_transform_fn, val_transform_fn = create_custom_transform_fn(args)
    
    print("Initializing datasets...")
    # Initialize datasets with enhanced transforms
    train_dataset = OMRDataset(
        root_dir=train_dir,
        transforms=train_transform_fn,
        is_train=True
    )
    
    val_dataset = OMRDataset(
        root_dir=val_dir,
        transforms=val_transform_fn,
        is_train=False
    )
    
    # Add the min_class_instances parameter to args
    args.min_class_instances = 24

    # Apply class filtering and merging
    print(f"Applying class filtering with min_instances={args.min_class_instances}...")
    class_map, inverse_map, keep_classes, merged_class_info, merge_map = filter_and_merge_classes(
        train_dataset, 
        args.min_class_instances,
        args  # Pass the complete args object
    )
    

    # Save the class maps and merge info for inference later
    with open(os.path.join(args.output_dir, 'class_map.json'), 'w') as f:
        json.dump({str(k): int(v) for k, v in class_map.items()}, f, indent=4)

    with open(os.path.join(args.output_dir, 'inverse_class_map.json'), 'w') as f:
        json.dump({str(k): int(v) for k, v in inverse_map.items()}, f, indent=4)

    with open(os.path.join(args.output_dir, 'class_info.json'), 'w') as f:
        json.dump(merged_class_info, f, indent=4)

    # Update number of classes
    args.num_classes = len(keep_classes) + 1  # +1 for background
    print(f"Updated num_classes to {args.num_classes} (including background)")

    # Wrap datasets with filtered versions
    train_dataset = FilteredOMRDataset(train_dataset, class_map, merge_map)
    val_dataset = FilteredOMRDataset(val_dataset, class_map, merge_map)
    
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
    
    print("Data loaders created")
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation images")
    
    print("Creating enhanced multi-scale model...")
    # Initialize enhanced model with multi-scale capabilities
    model = get_enhanced_model_instance_segmentation(num_classes, args)
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
    # Initialize exponential learning rate scheduler
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
    print("Starting training with multi-scale enhancements")
    
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
                # Use enhanced evaluation pipeline with multi-scale inference
                val_metrics = enhance_evaluation_pipeline(
                    model, 
                    val_loader, 
                    device, 
                    args,
                    writer,
                    epoch,
                    class_names
                )
                
                # Add standard loss metrics from the model
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                standard_val_metrics = {
                    'val_loss': val_metrics.get('val_loss', 0.0),
                    'val_loss_classifier': val_metrics.get('val_loss_classifier', 0.0),
                    'val_loss_box_reg': val_metrics.get('val_loss_box_reg', 0.0),
                    'val_loss_objectness': val_metrics.get('val_loss_objectness', 0.0),
                    'val_loss_rpn_box_reg': val_metrics.get('val_loss_rpn_box_reg', 0.0)
                }
                val_metrics.update(standard_val_metrics)
                
                # Log validation metrics to TensorBoard
                log_metrics(writer, val_metrics, global_step, prefix="val")
                
                # Print metrics
                print(f"Epoch {epoch} complete ({time.time() - epoch_start_time:.1f}s):")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Val MAP@0.5: {val_metrics['mAP@0.5']:.4f}")
                print(f"  Val MAP@0.75: {val_metrics['mAP@0.75']:.4f}")
                
                # Check if this is the best model so far
                is_best = val_metrics.get('mAP', 0.0) > best_val_loss if 'mAP' in val_metrics else False
                if is_best:
                    best_val_loss = val_metrics.get('mAP', 0.0)
                    print(f"  New best model! Val mAP: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Error during validation: {e}")
                traceback.print_exc()  # Print the full stack trace for debugging
                val_metrics = {'val_loss': float('nan'), 'mAP': 0.0}
                is_best = False
        else:
            # If not evaluating, just use placeholders for validation metrics
            val_metrics = {
                'val_loss': float('nan'),
                'val_loss_classifier': float('nan'),
                'val_loss_box_reg': float('nan'),
                'val_loss_objectness': float('nan'),
                'val_loss_rpn_box_reg': float('nan'),
                'mAP': 0.0,
                'mAP@0.5': 0.0,
                'mAP@0.75': 0.0
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

    print(f"Training complete. Best validation mAP: {best_val_loss:.4f}")
    print(f"All results saved to {args.output_dir}")
    print(f"TensorBoard logs saved to {args.log_dir}")

def plot_losses(train_metrics, val_metrics, args):
    """
    Plot training and validation losses and metrics.
    """
    epochs = range(1, len(train_metrics) + 1)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot total loss
    axes[0, 0].plot(epochs, [m['loss'] for m in train_metrics], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, [m.get('val_loss', float('nan')) for m in val_metrics], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot component losses
    axes[0, 1].plot(epochs, [m['loss_classifier'] for m in train_metrics], 'b-', label='Classifier')
    axes[0, 1].plot(epochs, [m['loss_box_reg'] for m in train_metrics], 'g-', label='Box Reg')
    axes[0, 1].plot(epochs, [m['loss_objectness'] for m in train_metrics], 'r-', label='Objectness')
    axes[0, 1].plot(epochs, [m['loss_rpn_box_reg'] for m in train_metrics], 'y-', label='RPN Box Reg')
    axes[0, 1].set_title('Component Losses (Training)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot mAP metrics
    valid_epochs = [i for i, m in enumerate(val_metrics, 1) if 'mAP' in m and not np.isnan(m['mAP'])]
    if valid_epochs:
        axes[1, 0].plot(valid_epochs, [val_metrics[i-1]['mAP'] for i in valid_epochs], 'b-', label='mAP')
        axes[1, 0].plot(valid_epochs, [val_metrics[i-1]['mAP@0.5'] for i in valid_epochs], 'g-', label='mAP@0.5')
        axes[1, 0].plot(valid_epochs, [val_metrics[i-1]['mAP@0.75'] for i in valid_epochs], 'r-', label='mAP@0.75')
        axes[1, 0].set_title('Mean Average Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot object size metrics
    if valid_epochs and 'mAP_small' in val_metrics[0]:
        axes[1, 1].plot(valid_epochs, [val_metrics[i-1]['mAP_small'] for i in valid_epochs], 'b-', label='Small Objects')
        axes[1, 1].plot(valid_epochs, [val_metrics[i-1]['mAP_medium'] for i in valid_epochs], 'g-', label='Medium Objects')
        axes[1, 1].plot(valid_epochs, [val_metrics[i-1]['mAP_large'] for i in valid_epochs], 'r-', label='Large Objects')
        axes[1, 1].set_title('Object Size Performance')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(args.output_dir, 'metrics_plot.png')
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    main()