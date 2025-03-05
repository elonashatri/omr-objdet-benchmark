import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from pathlib import Path

# Import the Doremi dataset class
from doremi_loader import DoremiDataset

def collate_fn(batch):
    """Custom collate function for the DataLoader"""
    return tuple(zip(*batch))

def get_model(num_classes, pretrained=True):
    """
    Create a Faster R-CNN model
    """
    # Create Faster R-CNN ResNet-50 FPN model
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, clip_value=1.0):
    """
    Train the model for one epoch
    """
    model.train()
    
    # Setup metrics tracking
    epoch_loss = 0.0
    running_loss = 0.0
    n_valid_batches = 0
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for i, (images, targets) in enumerate(pbar):
        try:
            # Convert numpy arrays to tensors if needed
            images = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 if isinstance(img, np.ndarray) else img for img in images]
            images = [img.to(device) for img in images]
            
            # Move targets to device
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Check if loss_dict is a list or dictionary
            if isinstance(loss_dict, list):
                # Take only the first batch item for now
                loss_dict = loss_dict[0]
            
            # Filter out NaN losses
            valid_losses = {k: v for k, v in loss_dict.items() if not torch.isnan(v)}
            
            if not valid_losses:
                print(f"Warning: All losses are NaN in batch {i}, skipping")
                continue
                
            losses = sum(valid_losses.values())
            
            # Skip if loss is NaN
            if torch.isnan(losses):
                print(f"Warning: NaN loss in batch {i}, skipping")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            
            # Clip gradients
            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
            optimizer.step()
            
            # Update metrics
            running_loss += losses.item()
            epoch_loss += losses.item()
            n_valid_batches += 1
            
            # Update progress bar every print_freq
            if i % print_freq == 0:
                if n_valid_batches > 0:
                    avg_loss = running_loss / n_valid_batches
                else:
                    avg_loss = float('nan')
                pbar.set_postfix({"loss": avg_loss})
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue
    
    # Final metrics
    if n_valid_batches > 0:
        avg_epoch_loss = epoch_loss / n_valid_batches
    else:
        avg_epoch_loss = float('nan')
        
    end_time = time.time()
    
    print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds, avg loss: {avg_epoch_loss:.4f}")
    print(f"Valid batches: {n_valid_batches}/{len(data_loader)}")
    
    return avg_epoch_loss

@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluate the model on the validation set
    """
    model.eval()
    
    # Setup metrics tracking
    val_loss = 0.0
    n_valid_batches = 0
    
    # Progress bar
    pbar = tqdm(data_loader, desc="Validating")
    
    for images, targets in pbar:
        try:
            # Convert numpy arrays to tensors if needed
            images = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 if isinstance(img, np.ndarray) else img for img in images]
            images = [img.to(device) for img in images]
            
            # Move targets to device
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Handle both list and dictionary outputs
            if isinstance(loss_dict, list):
                # If loss_dict is a list, sum all batch items
                batch_loss = 0
                for item in loss_dict:
                    if isinstance(item, dict):
                        item_loss = sum(v for v in item.values() if not torch.isnan(v))
                        batch_loss += item_loss
                losses = batch_loss
            else:
                # If loss_dict is a dictionary, sum as before
                valid_losses = {k: v for k, v in loss_dict.items() if not torch.isnan(v)}
                losses = sum(valid_losses.values()) if valid_losses else 0
            
            # Skip if loss is NaN
            if torch.isnan(losses):
                continue
                
            # Update metrics
            val_loss += losses.item()
            n_valid_batches += 1
            
        except Exception as e:
            print(f"Error during validation: {e}")
            continue
    
    # Final metrics
    if n_valid_batches > 0:
        avg_val_loss = val_loss / n_valid_batches
    else:
        avg_val_loss = float('inf')  # Return infinity if no valid batches
    
    print(f"Validation loss: {avg_val_loss:.4f} (from {n_valid_batches} valid batches)")
    
    return avg_val_loss

def train_model(args):
    """
    Train the model
    """
    # Set up device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load class mapping
    with open(args.class_mapping, 'r') as f:
        class_to_idx = json.load(f)
    
    num_classes = len(class_to_idx) + 1  # +1 for background
    print(f"Using {num_classes} classes (including background)")
    
    # Create dataset
    dataset = DoremiDataset(
        args.data_dir,
        args.annotation_dir,
        args.class_mapping,
        min_box_size=args.min_box_size,
        max_classes=None
    )
    
    print(f"Dataset contains {len(dataset)} samples")
    
    # Split dataset into train and validation
    dataset_size = len(dataset)
    if args.val_ratio > 0 and args.val_ratio < 1:
        indices = list(range(dataset_size))
        split = int(np.floor(args.val_ratio * dataset_size))
        
        # Shuffle indices
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]
        
        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        val_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        print(f"Split dataset into {len(train_indices)} training and {len(val_indices)} validation samples")
    else:
        # No validation set
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        val_dataloader = None
        
        print(f"Using all {dataset_size} samples for training")
    
    # Create model
    model = get_model(num_classes, args.pretrained)
    model.to(device)
    
    # Create optimizer 
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Use a smaller learning rate to avoid NaN losses
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=0.0005)
    
    # Create learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Loaded checkpoint {args.resume} (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Train the model
    print("Start training")
    
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch with gradient clipping
        train_loss = train_one_epoch(model, optimizer, train_dataloader, device, epoch, clip_value=args.clip_grad_norm)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_loss': best_loss,
        }, os.path.join(args.output_dir, f"checkpoint_latest.pth"))
        
        # Evaluate on validation set
        if val_dataloader is not None:
            val_loss = evaluate(model, val_dataloader, device)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_loss': best_loss,
                }, os.path.join(args.output_dir, "best_model.pth"))
                print(f"New best model saved with loss: {val_loss:.4f}")
        else:
            # If no validation set, use training loss
            if train_loss < best_loss and not np.isnan(train_loss):
                best_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_loss': best_loss,
                }, os.path.join(args.output_dir, "best_model.pth"))
                print(f"New best model saved with loss: {train_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"))
    
    # Save final model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'best_loss': best_loss,
    }, os.path.join(args.output_dir, "final_model.pth"))
    
    print("Training completed")

def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on Doremi dataset")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory containing XML annotations")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    
    # Model parameters
    parser.add_argument("--pretrained", action="store_true", help="Use pre-trained model")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"], help="Optimizer to use")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--lr_step_size", type=int, default=10, help="Learning rate step size")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Learning rate gamma")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Data processing parameters
    parser.add_argument("--min_box_size", type=int, default=5, help="Minimum size for bounding boxes")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Device parameters
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command-line arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Train the model
    train_model(args)

if __name__ == "__main__":
    main()