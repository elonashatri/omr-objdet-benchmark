"""
Training and Evaluation Module for Staff Line Detection

This module handles the training and evaluation of the staff line detection model.
"""

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from model import StaffLineDetectionNet, staff_line_dice_loss
from data_loader import get_dataloader


def train_model(model, train_loader, val_loader, device, 
               criterion, optimizer, scheduler, num_epochs, 
               save_dir, log_dir=None, save_freq=5):
    """
    Train the staff line detection model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use for training
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs (int): Number of epochs to train
        save_dir (str): Directory to save model checkpoints
        log_dir (str, optional): Directory for tensorboard logs
        save_freq (int): Frequency of saving checkpoints
        
    Returns:
        dict: Training history
    """
    # Initialize tensorboard if log_dir provided
    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)
    
    # Create model directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize best validation loss
    best_val_loss = float('inf')
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc="Training") as t:
            for batch_idx, (images, masks) in enumerate(t):
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                t.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as t:
                for batch_idx, (images, masks) in enumerate(t):
                    # Move data to device
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    # Update statistics
                    val_loss += loss.item()
                    t.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        # Update learning rate
        if scheduler:
            scheduler.step(avg_val_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint at specified frequency
        if (epoch + 1) % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.1f}s")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print("Training completed!")
    
    # Close tensorboard writer
    if writer:
        writer.close()
    
    return history


def evaluate_model(model, test_loader, device, criterion=None):
    """
    Evaluate the trained model.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use for evaluation
        criterion: Loss function (optional)
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Initialize metrics
    metrics = {
        'loss': 0.0,
        'dice': [],
        'precision': [],
        'recall': []
    }
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as t:
            for images, masks in t:
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss if criterion provided
                if criterion:
                    loss = criterion(outputs, masks)
                    metrics['loss'] += loss.item()
                
                # Calculate metrics
                pred = torch.sigmoid(outputs) > 0.5
                
                # Compute metrics for each item in batch
                for i in range(pred.size(0)):
                    pred_flat = pred[i].view(-1).float()
                    mask_flat = masks[i].view(-1).float()
                    
                    # True positives, false positives, false negatives
                    tp = (pred_flat * mask_flat).sum().item()
                    fp = (pred_flat * (1 - mask_flat)).sum().item()
                    fn = ((1 - pred_flat) * mask_flat).sum().item()
                    
                    # Precision, recall, dice
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
                    
                    metrics['precision'].append(precision)
                    metrics['recall'].append(recall)
                    metrics['dice'].append(dice)
    
    # Average metrics
    if criterion:
        metrics['loss'] = metrics['loss'] / len(test_loader)
    metrics['avg_precision'] = np.mean(metrics['precision'])
    metrics['avg_recall'] = np.mean(metrics['recall'])
    metrics['avg_dice'] = np.mean(metrics['dice'])
    
    # Print results
    print(f"Test Results:")
    if criterion:
        print(f"Loss: {metrics['loss']:.4f}")
    print(f"Dice Score: {metrics['avg_dice']:.4f}")
    print(f"Precision: {metrics['avg_precision']:.4f}")
    print(f"Recall: {metrics['avg_recall']:.4f}")
    
    return metrics


def train_staff_line_model(img_dir, mask_dir=None, xml_dir=None, model_dir="models", 
                          log_dir="logs", batch_size=8, num_epochs=50, lr=0.001, 
                          device_id=0, num_workers=4, augment=True, weights=None,
                          target_size=(512, 512), subset_fraction=1.0):
    """
    Complete training function that sets up and trains the staff line detection model.
    
    Args:
        img_dir (str): Directory with input images
        mask_dir (str, optional): Directory with ground truth masks
        xml_dir (str, optional): Directory with XML annotations
        model_dir (str): Directory to save model checkpoints
        log_dir (str): Directory for tensorboard logs
        batch_size (int): Batch size for training
        num_epochs (int): Number of epochs to train
        lr (float): Initial learning rate
        device_id (int): GPU ID
        num_workers (int): Number of data loading workers
        augment (bool): Whether to apply data augmentation
        weights (str, optional): Path to pre-trained weights
        target_size (tuple): Target size for resizing images (height, width)
        subset_fraction (float): Fraction of dataset to use (0.0-1.0)
        
    Returns:
        dict: Training history
    """
    # Set device
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Get dataloaders
    train_loader, val_loader = get_dataloader(
        img_dir, mask_dir, xml_dir, 
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
        target_size=target_size,
        subset_fraction=subset_fraction
    )
    
    # Initialize model
    model = StaffLineDetectionNet(n_channels=1, n_classes=1)
    
    # Load pre-trained weights if specified
    if weights:
        print(f"Loading weights from {weights}")
        model.load_state_dict(torch.load(weights, map_location=device))
    
    # Move model to device
    model = model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=staff_line_dice_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        save_dir=model_dir,
        log_dir=log_dir,
        save_freq=5
    )
    
    return history


def evaluate_staff_line_model(model_path, img_dir, mask_dir=None, xml_dir=None, 
                             batch_size=8, device_id=0, num_workers=4,
                             target_size=(512, 512), subset_fraction=1.0):
    """
    Evaluate a trained staff line detection model.
    
    Args:
        model_path (str): Path to model weights
        img_dir (str): Directory with test images
        mask_dir (str, optional): Directory with ground truth masks
        xml_dir (str, optional): Directory with XML annotations
        batch_size (int): Batch size for evaluation
        device_id (int): GPU ID
        num_workers (int): Number of data loading workers
        target_size (tuple): Target size for resizing images (height, width)
        subset_fraction (float): Fraction of dataset to use (0.0-1.0)
        
    Returns:
        dict: Evaluation metrics
    """
    # Set device
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get test dataloader
    test_loader, _ = get_dataloader(
        img_dir, mask_dir, xml_dir, 
        batch_size=batch_size,
        train_val_split=0.0,  # Use all data for testing
        num_workers=num_workers,
        augment=False,
        target_size=target_size,
        subset_fraction=subset_fraction
    )
    
    # Initialize model
    model = StaffLineDetectionNet(n_channels=1, n_classes=1)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        criterion=staff_line_dice_loss
    )
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate staff line detection model")
    
    # Mode selection
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train or evaluate")
    
    # Data paths
    parser.add_argument("--img_dir", type=str, required=True, 
                        help="Directory with input images")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Directory with ground truth masks (optional)")
    parser.add_argument("--xml_dir", type=str, default=None,
                        help="Directory with XML annotations (optional)")
    
    # Output paths
    parser.add_argument("--model_dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for tensorboard logs")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pre-trained weights (optional)")
    
    # Hardware parameters
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Check that either mask_dir or xml_dir is provided
    if not args.mask_dir and not args.xml_dir:
        parser.error("Either --mask_dir or --xml_dir must be provided")
    
    if args.mode == "train":
        # Train model
        train_staff_line_model(
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            xml_dir=args.xml_dir,
            model_dir=args.model_dir,
            log_dir=args.log_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            lr=args.lr,
            device_id=args.gpu_id,
            num_workers=args.num_workers,
            augment=not args.no_augment,
            weights=args.weights
        )
    
    elif args.mode == "eval":
        # Determine model path
        if args.weights:
            model_path = args.weights
        else:
            # Try to find best model in model_dir
            model_path = os.path.join(args.model_dir, "best_model.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(args.model_dir, "final_model.pth")
        
        if not os.path.exists(model_path):
            parser.error(f"Model not found at {model_path}. Please specify --weights")
        
        # Evaluate model
        evaluate_staff_line_model(
            model_path=model_path,
            img_dir=args.img_dir,
            mask_dir=args.mask_dir,
            xml_dir=args.xml_dir,
            batch_size=args.batch_size,
            device_id=args.gpu_id,
            num_workers=args.num_workers
        )