#!/usr/bin/env python3
"""
Script to run Mask R-CNN experiment with 10% of the dataset and 60/20/20 train/val/test split
"""
import os
import argparse
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import time
import json
import glob
from tqdm import tqdm
import sys

# Make sure our modules are importable
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import our custom modules
from omr_dataset_utils import OMRDataset, visualize_sample
from eval_metrics import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='Run OMR Mask R-CNN Experiment')
    
    # Main parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory containing annotations and images')
    parser.add_argument('--annotations_dir', type=str, default='annotations',
                        help='Directory name containing XML annotations under data_dir')
    parser.add_argument('--images_dir', type=str, default='images',
                        help='Directory name containing images under data_dir')
    parser.add_argument('--output_dir', type=str, default='./output/experiment_10percent',
                        help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--min_size', type=int, default=400,
                        help='Minimum size for Mask R-CNN image resizing (default: 400)')
    parser.add_argument('--max_size', type=int, default=667,
                        help='Maximum size for Mask R-CNN image resizing (default: 667)')
    parser.add_argument('--sample_percentage', type=float, default=10.0,
                        help='Percentage of data to use (1-100)')
    parser.add_argument('--train_split', type=float, default=0.6,
                        help='Fraction of sampled data for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of sampled data for validation')
    parser.add_argument('--test_split', type=float, default=0.2,
                        help='Fraction of sampled data for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def load_dataset(data_dir, annotations_dir, images_dir, img_ext=".png"):
    """
    Load dataset from directories
    """
    # Get all XML files
    annotations_path = os.path.join(data_dir, annotations_dir)
    xml_files = glob.glob(os.path.join(annotations_path, "**", "*.xml"), recursive=True)
    
    if not xml_files:
        raise ValueError(f"No XML files found in {annotations_path}")
    
    print(f"Found {len(xml_files)} XML annotation files")
    
    # Create mapping from XML to image files
    # Adjust paths to look for images in the images_dir
    for i, xml_path in enumerate(xml_files[:3]):  # Print just the first 3 for verification
        rel_path = os.path.relpath(xml_path, annotations_path)
        # Get corresponding image path
        img_path = os.path.join(data_dir, images_dir, os.path.splitext(rel_path)[0] + img_ext)
        
        # Print for verification
        print(f"XML: {xml_path}")
        print(f"IMG: {img_path}")
        print()
    
    # Create dataset
    dataset = OMRDataset(xml_files, img_ext=img_ext)
    
    return dataset, xml_files

def sample_and_split_dataset(xml_files, sample_percentage, train_split, val_split, test_split, seed=42):
    """
    Sample a percentage of the dataset and split into train/val/test
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Sample percentage of files
    num_files = len(xml_files)
    num_samples = int(num_files * sample_percentage / 100)
    
    # Randomly sample indices
    indices = list(range(num_files))
    sampled_indices = random.sample(indices, min(num_samples, num_files))
    
    # Ensure splits sum to 1.0
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        train_split /= total_split
        val_split /= total_split
        test_split /= total_split
    
    # Split sampled indices
    num_train = int(len(sampled_indices) * train_split)
    num_val = int(len(sampled_indices) * val_split)
    num_test = len(sampled_indices) - num_train - num_val
    
    # Shuffle sampled indices
    random.shuffle(sampled_indices)
    
    # Split into train/val/test
    train_indices = sampled_indices[:num_train]
    val_indices = sampled_indices[num_train:num_train+num_val]
    test_indices = sampled_indices[num_train+num_val:]
    
    print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    print(f"  (Total: {len(train_indices) + len(val_indices) + len(test_indices)} of {num_files} files)")
    
    return train_indices, val_indices, test_indices

def get_model_instance_segmentation(num_classes, min_size=400, max_size=667, pretrained=True):
    """
    Get a Mask R-CNN model with a ResNet-50-FPN backbone
    """
    # Load model with or without pretrained weights
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)
    
    # Override the default transform with smaller sizes
    model.transform = GeneralizedRCNNTransform(
        min_size=min_size,
        max_size=max_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    
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

def train_one_epoch(model, optimizer, data_loader, device):
    """
    Train for one epoch
    """
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    for images, targets in progress_bar:
        try:
            # Move data to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass and calculate loss
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and update
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += losses.item()
            progress_bar.set_postfix({"loss": losses.item()})
        except Exception as e:
            print(f"Error during training: {e}")
            # Continue with next batch
            continue
    
    return epoch_loss / max(1, len(data_loader))

def evaluate(model, data_loader, device):
    """
    Evaluate model on validation set
    """
    model.eval()
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validation")
        for images, targets in progress_bar:
            try:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                num_batches += 1
                
                progress_bar.set_postfix({"val_loss": losses.item()})
            except Exception as e:
                print(f"Error during validation: {e}")
                # Continue with next batch
                continue
    
    return val_loss / max(1, num_batches)

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename):
    """
    Save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, filename)

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, "models")
    eval_dir = os.path.join(args.output_dir, "evaluation")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Set device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load dataset
    dataset, xml_files = load_dataset(
        args.data_dir, 
        args.annotations_dir, 
        args.images_dir
    )
    
    # Sample and split dataset
    train_indices, val_indices, test_indices = sample_and_split_dataset(
        xml_files,
        args.sample_percentage,
        args.train_split,
        args.val_split,
        args.test_split,
        args.seed
    )
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders - use small batch size to avoid memory issues
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),  # Required for variable sized data
        num_workers=0  # Avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
        num_workers=0
    )
    
    # Initialize model with custom transform parameters
    num_classes = len(dataset.class_map) + 1  # +1 for background
    model = get_model_instance_segmentation(
        num_classes=num_classes, 
        min_size=args.min_size, 
        max_size=args.max_size,
        pretrained=True
    )
    
    model.to(device)
    print(f"Model created with min_size={args.min_size}, max_size={args.max_size}")
    
    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    print(f"Starting training for {args.num_epochs} epochs")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        try:
            # Train
            train_loss = train_one_epoch(model, optimizer, train_loader, device)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = evaluate(model, val_loader, device)
            val_losses.append(val_loss)
            
            # Update learning rate
            lr_scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(models_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(models_dir, "best_model.pth")
                save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_model_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            continue
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'))
    plt.close()
    
    # Save indices for reproducibility
    with open(os.path.join(args.output_dir, 'dataset_splits.json'), 'w') as f:
        json.dump({
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'sample_percentage': args.sample_percentage,
            'train_split': args.train_split,
            'val_split': args.val_split,
            'test_split': args.test_split,
            'seed': args.seed
        }, f, indent=2)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
        
        # Map class IDs to class names for better visualization
        class_names = {v: k for k, v in dataset.class_map.items()}
        # Add background class
        class_names[0] = "background"
        
        # Run evaluation
        test_eval_dir = os.path.join(eval_dir, "test_set")
        os.makedirs(test_eval_dir, exist_ok=True)
        
        metrics = evaluate_model(model, test_loader, device, test_eval_dir, class_names)
        
        # Display summary
        print("\nTest Set Evaluation Results:")
        print(f"mAP: {metrics['mAP']:.4f}")
        print(f"mIoU: {metrics['mIoU']:.4f}")
        print(f"Mean Precision: {metrics['mean_precision']:.4f}")
        print(f"Mean Recall: {metrics['mean_recall']:.4f}")
        print(f"Mean F1 Score: {metrics['mean_f1']:.4f}")
    except Exception as e:
        print(f"Error during final evaluation: {e}")
    
    print(f"\nExperiment completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()