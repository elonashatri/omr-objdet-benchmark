#!/usr/bin/env python3
"""
Staff Line-specific Dataset Enhancements

This module provides specialized dataset handling for OMR with emphasis on
improving staff line detection.
"""

import os
import json
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import cv2
import numpy as np
from PIL import Image
import random
from omr_dataset import OMRDataset

class StaffLineEnhancedDataset(OMRDataset):
    """
    Enhanced dataset class for Optical Music Recognition (OMR) with specialized
    handling for staff lines.
    """
    
    def __init__(self, 
                 root_dir, 
                 transforms=None, 
                 is_train=True,
                 staff_line_class='kStaffLine',
                 staff_line_width_range=(0, 1),
                 patch_size=None,
                 patch_overlap=0.25):
        """
        Args:
            root_dir (string): Directory with images and annotations
            transforms (callable, optional): Optional transforms to be applied on a sample
            is_train (bool): Whether this is training data (affects data augmentation)
            staff_line_class (string): Class name for staff lines
            staff_line_width_range (tuple): Range for random width adjustment of staff lines
            patch_size (tuple): Size of patches to extract (height, width); if None, use full images
            patch_overlap (float): Overlap between patches (as a fraction of patch size)
        """
        super().__init__(root_dir, transforms, is_train)
        self.staff_line_class = staff_line_class
        self.staff_line_width_range = staff_line_width_range
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        
        # Map class names to IDs
        self.class_name_to_id = {}
        for cat in self.annotations['categories']:
            self.class_name_to_id[cat['name']] = cat['id']
        
        # If using patches, create patch indices
        self.patches = []
        if self.patch_size is not None:
            self._create_patches()
    
    def _create_patches(self):
        """
        Create indices for patch-based training.
        """
        patch_height, patch_width = self.patch_size
        stride_h = int(patch_height * (1 - self.patch_overlap))
        stride_w = int(patch_width * (1 - self.patch_overlap))
        
        for idx in range(len(self.image_ids)):
            image_id = self.image_ids[idx]
            img_info = self.image_info[image_id]
            img_height, img_width = img_info['height'], img_info['width']
            
            # Calculate number of patches
            for y in range(0, img_height - patch_height + 1, stride_h):
                for x in range(0, img_width - patch_width + 1, stride_w):
                    self.patches.append({
                        'image_idx': idx,
                        'x': x,
                        'y': y,
                        'width': patch_width,
                        'height': patch_height
                    })
    
    def __len__(self):
        """
        Return the number of patches if patch-based, otherwise number of images.
        """
        if self.patch_size is not None:
            return len(self.patches)
        return len(self.image_ids)
    
    def _crop_annotations(self, annotations, x, y, width, height):
        """
        Crop annotations to a specific patch.
        
        Args:
            annotations: List of annotation dictionaries
            x, y: Top-left corner of the patch
            width, height: Dimensions of the patch
            
        Returns:
            List of cropped annotations
        """
        cropped_anns = []
        
        for ann in annotations:
            # Get the original box coordinates (COCO format: [x, y, width, height])
            bbox = ann['bbox']
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            
            # Check if box intersects with the patch
            if x2 > x and x1 < x + width and y2 > y and y1 < y + height:
                # Crop box to patch
                new_x1 = max(x1, x)
                new_y1 = max(y1, y)
                new_x2 = min(x2, x + width)
                new_y2 = min(y2, y + height)
                
                # Calculate new width and height
                new_w = new_x2 - new_x1
                new_h = new_y2 - new_y1
                
                # Only include if the cropped box has sufficient area
                if new_w > 0 and new_h > 0:
                    # Adjust coordinates relative to patch
                    new_x1 -= x
                    new_y1 -= y
                    
                    # Create new annotation
                    new_ann = ann.copy()
                    new_ann['bbox'] = [new_x1, new_y1, new_w, new_h]
                    cropped_anns.append(new_ann)
        
        return cropped_anns
    
    def _enhance_staff_lines(self, image, boxes, labels):
        """
        Apply staff line-specific enhancements.
        
        Args:
            image: The image tensor or array
            boxes: Bounding boxes
            labels: Class labels
            
        Returns:
            Enhanced image, boxes, and labels
        """
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            is_tensor = True
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            is_tensor = False
            image_np = image.copy()
        
        # Find staff line boxes
        staff_line_class_id = self.class_name_to_id.get(self.staff_line_class)
        if staff_line_class_id is None:
            # If staff line class not found, return unchanged
            return image, boxes, labels
        
        staff_line_indices = [i for i, label in enumerate(labels) if label == staff_line_class_id]
        if not staff_line_indices:
            return image, boxes, labels
        
        # Process each staff line
        for idx in staff_line_indices:
            # Get the box
            x1, y1, x2, y2 = boxes[idx]
            
            # Randomly adjust width if in training mode
            if self.is_train and random.random() < 0.5:
                # Get current height (width of the staff line)
                height = y2 - y1
                
                # Random adjustment within range
                width_change = random.uniform(self.staff_line_width_range[0], self.staff_line_width_range[1])
                
                # Apply width change
                new_y1 = max(0, y1 - width_change/2)
                new_y2 = min(image_np.shape[0], y2 + width_change/2)
                
                # Update the box
                boxes[idx] = torch.tensor([x1, new_y1, x2, new_y2])
                
                # Adjust the line in the image
                if width_change > 0:
                    # Thicken the line
                    cv2.rectangle(image_np, 
                                 (int(x1), int(new_y1)), 
                                 (int(x2), int(new_y2)), 
                                 (0, 0, 0), 
                                 -1)  # -1 for filled rectangle
        
        # Convert back to tensor if needed
        if is_tensor:
            image = torch.from_numpy(image_np).permute(2, 0, 1)
        else:
            image = image_np
        
        return image, boxes, labels
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with image and annotation data
        """
        if self.patch_size is not None:
            # Patch-based retrieval
            patch_info = self.patches[idx]
            image_idx = patch_info['image_idx']
            x, y = patch_info['x'], patch_info['y']
            width, height = patch_info['width'], patch_info['height']
            
            # Get the original image
            image_id = self.image_ids[image_idx]
            img_info = self.image_info[image_id]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Load and crop image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[y:y+height, x:x+width]
            
            # Get and crop annotations
            anns = self.image_annotations[image_id]
            cropped_anns = self._crop_annotations(anns, x, y, width, height)
            
            # Extract boxes and labels
            boxes = []
            labels = []
            
            for ann in cropped_anns:
                x1, y1, w, h = ann['bbox']
                boxes.append([x1, y1, x1 + w, y1 + h])
                labels.append(ann['category_id'])
            
            # Handle case with no valid boxes
            if not boxes:
                # Return a dummy box that will be filtered out
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0), dtype=torch.int64)
                
                # Convert image to tensor if needed
                if self.transforms:
                    image = F.to_tensor(image)
                
                return {
                    'image': image,
                    'boxes': boxes,
                    'labels': labels,
                    'image_id': torch.tensor([image_id])
                }
            
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
        else:
            # Standard image retrieval (use parent class implementation)
            result = super().__getitem__(idx)
            image = result['image']
            boxes = result['boxes']
            labels = result['labels']
            image_id = result['image_id']
            
            # Apply staff line enhancements
            image, boxes, labels = self._enhance_staff_lines(image, boxes, labels)
            
            # Apply transforms if needed
            if self.transforms and not self.patch_size:
                target = {'boxes': boxes, 'labels': labels, 'image_id': image_id}
                image, target = self.transforms(image, target)
                boxes = target['boxes']
                labels = target['labels']
            
            return {
                'image': image,
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id
            }

# Function to create data loaders with staff line enhancements
def create_staff_line_data_loaders(args):
    """
    Create data loaders with staff line-specific enhancements.
    
    Args:
        args: Command line arguments
        
    Returns:
        train_loader, val_loader, class_names
    """
    # Parse image size
    if ',' in args.image_size:
        height, width = map(int, args.image_size.split(','))
    else:
        height = width = int(args.image_size)
    
    # Parse patch size if specified
    patch_size = None
    if args.patch_based:
        if ',' in args.patch_size:
            patch_height, patch_width = map(int, args.patch_size.split(','))
            patch_size = (patch_height, patch_width)
        else:
            patch_size = (int(args.patch_size), int(args.patch_size))
    
    # Load mapping file to get class names
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
    
    # Paths to dataset directories
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    # Initialize custom transforms
    from omr_dataset import get_transform
    
    # Create datasets
    train_dataset = StaffLineEnhancedDataset(
        root_dir=train_dir,
        transforms=get_transform(train=True),
        is_train=True,
        staff_line_class='kStaffLine',
        staff_line_width_range=(-1, 1),
        patch_size=patch_size,
        patch_overlap=args.patch_overlap
    )
    
    val_dataset = StaffLineEnhancedDataset(
        root_dir=val_dir,
        transforms=get_transform(train=False),
        is_train=False,
        staff_line_class='kStaffLine',
        patch_size=patch_size,
        patch_overlap=args.patch_overlap
    )
    
    print(f"Enhanced datasets created. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Function to collate batches
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory
    )
    
    return train_loader, val_loader, class_names

# Add these arguments to your parser
def add_staff_line_args(parser):
    """
    Add staff line-specific arguments to argument parser.
    
    Args:
        parser: ArgumentParser instance
    """
    # Patch-based training
    parser.add_argument('--patch_based', action='store_true',
                      help='Use patch-based training')
    parser.add_argument('--patch_size', type=str, default='512,512',
                      help='Size of patches (height,width) for patch-based training')
    parser.add_argument('--patch_overlap', type=float, default=0.25,
                      help='Overlap between patches (as a fraction of patch size)')
    
    # Staff line enhancements
    parser.add_argument('--staff_line_class', type=str, default='kStaffLine',
                      help='Class name for staff lines')
    parser.add_argument('--staff_line_width_range', type=str, default='-1,1',
                      help='Range for random width adjustment of staff lines (min,max)')
    
    return parser