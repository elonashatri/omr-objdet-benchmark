#!/usr/bin/env python3
"""
Train Faster R-CNN Model for OMR

This script trains a Faster R-CNN model on an Optical Music Recognition dataset.
"""

import os
import time
import datetime
import json
import glob
import torch
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
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as F

# Import the dataset loader
from omr_dataset import OMRDataset, get_transform

import argparse
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN for OMR')
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, 
                        default='/homes/es314/omr-objdet-benchmark/data/prepared_dataset',
                        help='Directory containing prepared dataset')
    parser.add_argument('--output_dir', type=str, default='./faster_rcnn_omr',
                        help='Directory to save model checkpoints')
    parser.add_argument('--test_images_dir', type=str, default='',
                        help='Directory containing test images for inference during training')
    
    # Image parameters
    parser.add_argument('--image_size', type=str, default='800,1200',
                        help='Image size (height,width) for resizing input images')
    parser.add_argument('--min_size', type=int, default=800,
                        help='Minimum size of the image to be rescaled before feeding it to the backbone')
    parser.add_argument('--max_size', type=int, default=1333,
                        help='Maximum size of the image to be rescaled before feeding it to the backbone')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=25,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Batch size for validation')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--lr_step_size', type=int, default=5,
                        help='Step size for learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Gamma for learning rate scheduler')
    
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
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Starting epoch')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnext101_32x8d'],
                        help='Backbone network for Faster R-CNN')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone network')
    parser.add_argument('--frozen_layers', type=int, default=0,
                        help='Number of layers to freeze in the backbone')
    
    # Anchor parameters
    parser.add_argument('--anchor_sizes', type=str, default='32,64,128,256,512',
                        help='Comma-separated list of anchor sizes')
    parser.add_argument('--aspect_ratios', type=str, default='0.5,1.0,2.0',
                        help='Comma-separated list of aspect ratios')
    parser.add_argument('--staff_line_anchors', action='store_true',
                        help='Add staff line specific anchors (very thin and wide)')
                        
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

# def collate_fn(batch):
#     """
#     Custom collate function for the data loader to handle variable sized images and targets.
#     """
#     return tuple(zip(*batch))

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

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """
    Train the model for one epoch.
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
        samples = [{k: v.to(device) for k, v in t.items()} for t in data]
        
        # Get images and targets
        images = [x['image'] for x in samples]
        targets = [{k: v for k, v in t.items() if k != 'image'} for t in samples]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Update running losses
        running_loss += losses.item()
        running_loss_classifier += loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
        running_loss_box_reg += loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
        running_loss_objectness += loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
        running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Backward pass
        losses.backward()
        
        # Update weights
        optimizer.step()
        
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
    
    # Return average losses for the epoch
    metrics = {
        'loss': running_loss / len(data_loader),
        'loss_classifier': running_loss_classifier / len(data_loader),
        'loss_box_reg': running_loss_box_reg / len(data_loader),
        'loss_objectness': running_loss_objectness / len(data_loader),
        'loss_rpn_box_reg': running_loss_rpn_box_reg / len(data_loader)
    }
    
    return metrics

def evaluate(model, data_loader, device):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    
    # Create tqdm progress bar
    pbar = tqdm(data_loader, desc="Validation")
    
    # Metrics
    running_loss = 0.0
    running_loss_classifier = 0.0
    running_loss_box_reg = 0.0
    running_loss_objectness = 0.0
    running_loss_rpn_box_reg = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(pbar):
            try:
                # Handle data properly based on its actual structure
                images = []
                targets = []
                
                # Process the data - this might need to be adjusted based on actual structure
                for img, box, label, img_id in zip(data[0], data[1], data[2], data[3]):
                    # Convert string image to tensor if needed
                    if isinstance(img, str):
                        # You'll need to load the image here
                        # For now, just skip this batch
                        continue
                    
                    img_tensor = img.to(device)
                    images.append(img_tensor)
                    
                    # Create target dict
                    target = {
                        'boxes': box.to(device) if isinstance(box, torch.Tensor) else torch.tensor(box).to(device),
                        'labels': label.to(device) if isinstance(label, torch.Tensor) else torch.tensor(label).to(device),
                        'image_id': img_id.to(device) if isinstance(img_id, torch.Tensor) else torch.tensor([img_id]).to(device)
                    }
                    targets.append(target)
                
                # Skip if no valid images
                if not images:
                    continue
                
                # Forward pass (including loss calculation)
                loss_dict = model(images, targets)
                
                # Calculate total loss
                losses = sum(loss for loss in loss_dict.values())
                
                # Update running losses
                running_loss += losses.item()
                running_loss_classifier += loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
                running_loss_box_reg += loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
                running_loss_objectness += loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
                running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
                
                # Update progress bar
                pbar.set_postfix({
                    'val_loss': f'{running_loss / (i + 1):.4f}'
                })
            except Exception as e:
                print(f"Error in validation batch {i}: {e}")
                continue
    
    # Return average losses for the epoch
    num_batches = max(1, len(data_loader))  # Avoid division by zero
    metrics = {
        'val_loss': running_loss / num_batches,
        'val_loss_classifier': running_loss_classifier / num_batches,
        'val_loss_box_reg': running_loss_box_reg / num_batches,
        'val_loss_objectness': running_loss_objectness / num_batches,
        'val_loss_rpn_box_reg': running_loss_rpn_box_reg / num_batches
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
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # If this is the best model, save as best.pth
    if is_best:
        best_path = os.path.join(args.output_dir, 'best.pth')
        torch.save(checkpoint, best_path)
    
    # Save latest.pth for easy resuming
    latest_path = os.path.join(args.output_dir, 'latest.pth')
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
            font = ImageFont.truetype("DejaVuSans.ttf", 12)
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
    Get Faster R-CNN model with the specified backbone.
    
    Args:
        num_classes: Number of classes to predict (including background)
        args: Command line arguments
        
    Returns:
        model: The Faster R-CNN model
    """
    # Parse anchor parameters
    anchor_sizes = [int(size) for size in args.anchor_sizes.split(',')]
    aspect_ratios = [float(ratio) for ratio in args.aspect_ratios.split(',')]
    
    # Add staff line specific anchors if requested
    if args.staff_line_anchors:
        # Add very thin and wide anchors for staff lines
        aspect_ratios += [0.01, 0.005]  # Very thin
    
    # Select backbone
    if args.backbone == 'resnet50' or not HAS_RESNET101:
        model = fasterrcnn_resnet50_fpn(
            pretrained=args.pretrained,
            min_size=args.min_size,
            max_size=args.max_size
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
    
    # Modify anchor generator if needed
    if args.staff_line_anchors or args.anchor_sizes != '32,64,128,256,512' or args.aspect_ratios != '0.5,1.0,2.0':
        anchor_generator = model.rpn.anchor_generator
        anchor_generator.sizes = tuple((s,) for s in anchor_sizes)
        anchor_generator.aspect_ratios = tuple((a,) for a in aspect_ratios)
    
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
    
    # Set device based on gpu_id
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Parse image size
    if ',' in args.image_size:
        height, width = map(int, args.image_size.split(','))
    else:
        height = width = int(args.image_size)
    print(f"Using image size: {height}x{width}")
    
    # Load data
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    # Load mapping file to get number of classes and create class_names dict
    mapping_file = os.path.join(args.data_dir, 'mapping.txt')
    class_names = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    class_id = int(parts[0])
                    class_name = ':'.join(parts[1:])
                    class_names[class_id] = class_name
    
    num_classes = len(class_names) + 1  # +1 for background class
    print(f"Training with {num_classes} classes (including background)")
    
    # Log all model arguments to TensorBoard
    writer.add_text('training/args', str(vars(args)), 0)
    
    # Initialize custom transforms that incorporate image resizing
    def train_transform(image, target):
        # First apply standard transformations
        image, target = get_transform(train=True)(image, target)
        return image, target
    
    def val_transform(image, target):
        # First apply standard transformations
        image, target = get_transform(train=False)(image, target)
        return image, target
    
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
    
    print("About to create data loaders")
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
    print("Initializing data loaders...")
    
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation images")
    
    # Log sample images to TensorBoard
    # sample_batch = next(iter(train_loader))
    # sample_images = [s['image'] for s in sample_batch]
    # sample_targets = [{k: v for k, v in s.items() if k != 'image'} for s in sample_batch]
    # visualize_batch(writer, sample_images, sample_targets, step=0, prefix="samples")
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
    
    # Initialize optimizer
    print("Setting up optimizer...")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    print("Setting up scheduler...")
    # Initialize learning rate scheduler
    lr_scheduler = StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
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
            
            
    # Training loop
    print("Starting training")

    for epoch in range(start_epoch, args.num_epochs):
        # Train for one epoch
        epoch_start_time = time.time()
        
        model.train()
        
        # Create tqdm progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        # Metrics
        running_loss = 0.0
        running_loss_classifier = 0.0
        running_loss_box_reg = 0.0
        running_loss_objectness = 0.0
        running_loss_rpn_box_reg = 0.0
        
        for i, data in enumerate(pbar):
            try:
                # Handle data properly based on its actual structure
                images = []
                targets = []
                
                # Process the data based on the observed structure
                # data is a tuple of 4 elements, each being a tuple of length batch_size
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
                        img_id_tensor = image_id.to(device) if isinstance(image_id, torch.Tensor) else torch.tensor([image_id]).to(device)
                        
                        target = {
                            'boxes': box_tensor,
                            'labels': label_tensor,
                            'image_id': img_id_tensor
                        }
                        targets.append(target)
                    except Exception as e:
                        print(f"Error processing target for batch item {batch_idx}: {e}")
                        continue
                
                # Skip this batch if no valid images were found
                if len(images) == 0:
                    print(f"Skipping batch {i} - no valid images")
                    continue
                
                # Forward pass
                loss_dict = model(images, targets)
                
                # Calculate total loss
                losses = sum(loss for loss in loss_dict.values())
                
                # Update running losses
                running_loss += losses.item()
                running_loss_classifier += loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
                running_loss_box_reg += loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
                running_loss_objectness += loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
                running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Backward pass
                losses.backward()
                
                # Update weights
                optimizer.step()
                
                # Update global step
                global_step += 1
                
                # Update progress bar
                if i % args.print_freq == 0 or i == len(train_loader) - 1:
                    avg_loss = running_loss / (i + 1)
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'cls': f'{running_loss_classifier / (i + 1):.4f}',
                        'bbox': f'{running_loss_box_reg / (i + 1):.4f}',
                        'obj': f'{running_loss_objectness / (i + 1):.4f}',
                        'rpn': f'{running_loss_rpn_box_reg / (i + 1):.4f}'
                    })
                
                # Log to TensorBoard (every 10 steps)
                if global_step % 10 == 0:
                    metrics = {
                        'loss/total': losses.item(),
                        'loss/classifier': loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0,
                        'loss/box_reg': loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0,
                        'loss/objectness': loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0,
                        'loss/rpn_box_reg': loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0,
                        'lr': optimizer.param_groups[0]['lr']
                    }
                    log_metrics(writer, metrics, global_step)
                
                # Visualize a batch in TensorBoard (every 100 steps)
                if global_step % 100 == 0:
                    # Skip visualization for now since data format is unexpected
                    pass
                    
            except Exception as e:
                print(f"Error in training batch {i}: {e}")
                continue
        
        # Calculate average losses for the epoch
        total_batches = max(1, len(train_loader))  # Avoid division by zero
        train_metrics = {
            'loss': running_loss / total_batches,
            'loss_classifier': running_loss_classifier / total_batches,
            'loss_box_reg': running_loss_box_reg / total_batches,
            'loss_objectness': running_loss_objectness / total_batches,
            'loss_rpn_box_reg': running_loss_rpn_box_reg / total_batches
        }
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set (every eval_freq epochs)
        if epoch % args.eval_freq == 0 or epoch == args.num_epochs - 1:
            try:
                val_metrics = {
                    'val_loss': float('nan'),
                    'val_loss_classifier': float('nan'),
                    'val_loss_box_reg': float('nan'),
                    'val_loss_objectness': float('nan'),
                    'val_loss_rpn_box_reg': float('nan')
                }
                
                # Try to run evaluation but be prepared for failures
                try:
                    model.eval()
                    
                    # Validation metrics
                    val_running_loss = 0.0
                    val_running_loss_classifier = 0.0
                    val_running_loss_box_reg = 0.0
                    val_running_loss_objectness = 0.0
                    val_running_loss_rpn_box_reg = 0.0
                    
                    val_batches = 0
                    
                    with torch.no_grad():
                        val_pbar = tqdm(val_loader, desc="Validation")
                        
                        for i, data in enumerate(val_pbar):
                            try:
                                # Handle data properly based on the observed structure
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
                                    
                                    # Create target dict
                                    box_tensor = boxes.to(device) if isinstance(boxes, torch.Tensor) else torch.tensor(boxes).to(device)
                                    label_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
                                    img_id_tensor = image_id.to(device) if isinstance(image_id, torch.Tensor) else torch.tensor([image_id]).to(device)
                                    
                                    target = {
                                        'boxes': box_tensor,
                                        'labels': label_tensor,
                                        'image_id': img_id_tensor
                                    }
                                    targets.append(target)
                                
                                # Skip if no valid images
                                if len(images) == 0:
                                    continue
                                
                                # Forward pass
                                loss_dict = model(images, targets)
                                
                                # Calculate total loss
                                losses = sum(loss for loss in loss_dict.values())
                                
                                # Update running losses
                                val_running_loss += losses.item()
                                val_running_loss_classifier += loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
                                val_running_loss_box_reg += loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
                                val_running_loss_objectness += loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
                                val_running_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
                                
                                val_batches += 1
                                
                                val_pbar.set_postfix({
                                    'val_loss': f'{val_running_loss / (val_batches):.4f}'
                                })
                                
                            except Exception as e:
                                print(f"Error in validation batch {i}: {e}")
                                continue
                    
                    # Calculate validation metrics
                    if val_batches > 0:
                        val_metrics = {
                            'val_loss': val_running_loss / val_batches,
                            'val_loss_classifier': val_running_loss_classifier / val_batches,
                            'val_loss_box_reg': val_running_loss_box_reg / val_batches,
                            'val_loss_objectness': val_running_loss_objectness / val_batches,
                            'val_loss_rpn_box_reg': val_running_loss_rpn_box_reg / val_batches
                        }
                except Exception as e:
                    print(f"Error during validation: {e}")
                
                # Log validation metrics to TensorBoard
                log_metrics(writer, val_metrics, global_step, prefix="val")
                
                # Run inference on test images if provided
                if args.test_images_dir:
                    # Skip for now until data format is resolved
                    pass
                
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
                val_metrics = {'val_loss': float('nan')}
                is_best = False
        else:
            # If not evaluating, just use the training metrics
            val_metrics = {'val_loss': float('nan')}
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
    else:
        print(f"No checkpoint found at '{args.resume}'")


if __name__ == "__main__":
    main()
