#!/usr/bin/env python3
"""
PyTorch Dataset Loader for OMR

This module provides a custom PyTorch Dataset class for loading OMR data 
formatted in COCO style.
"""

import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import cv2
import numpy as np
from PIL import Image
import random

class OMRDataset(Dataset):
    """
    Dataset class for Optical Music Recognition (OMR) data.
    Expects data in COCO format.
    """
    
    def __init__(self, 
                 root_dir, 
                 transforms=None, 
                 is_train=True):
        """
        Args:
            root_dir (string): Directory with images and annotations
            transforms (callable, optional): Optional transforms to be applied on a sample
            is_train (bool): Whether this is training data (affects data augmentation)
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.transforms = transforms
        self.is_train = is_train
        
        # Load the annotations file
        with open(os.path.join(root_dir, 'annotations.json'), 'r') as f:
            self.annotations = json.load(f)
        
        # Create mappings for easy access
        self.image_info = {img['id']: img for img in self.annotations['images']}
        self.categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}
        
        # Group annotations by image_id
        self.image_annotations = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)
        
        # Get list of all valid image IDs (that have annotations)
        self.image_ids = list(self.image_annotations.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data point
            
        Returns:
            dict: A dictionary containing:
                - 'image': The image tensor
                - 'boxes': The bounding boxes in [x_min, y_min, x_max, y_max] format
                - 'labels': The category labels for each box
                - 'image_id': The ID of the image
        """
        # Get image ID and image info
        image_id = self.image_ids[idx]
        img_info = self.image_info[image_id]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Get annotations for this image
        anns = self.image_annotations[image_id]
        
        # Extract bounding boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            # COCO format: [x, y, width, height]
            # PyTorch expects: [xmin, ymin, xmax, ymax]
            x, y, w, h = ann['bbox']
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            
            # Add box if it has valid dimensions
            if w > 0 and h > 0:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(ann['category_id'])
        
        # Handle case with no valid boxes
        if not boxes:
            # Return a dummy box that will be filtered out
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0), dtype=torch.int64)
            return {
                'image': F.to_tensor(image),
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([image_id])
            }
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Prepare target dict
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }
        
        # Apply transforms if provided
        if self.transforms:
            image, target = self.transforms(image, target)
        else:
            # Basic conversion to tensor if no transforms
            image = F.to_tensor(image)
        
        return {
            'image': image,
            'boxes': target['boxes'],
            'labels': target['labels'],
            'image_id': target['image_id']
        }
    
    def get_img_info(self, idx):
        """Get image info based on index"""
        image_id = self.image_ids[idx]
        return self.image_info[image_id]
    
    def get_height_and_width(self, idx):
        """Get image dimensions based on index"""
        img_info = self.get_img_info(idx)
        return img_info['height'], img_info['width']

# Custom transforms that work with both images and targets
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        # Convert image to tensor
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Convert tensor to numpy if it's a tensor
            if isinstance(image, torch.Tensor):
                # Convert tensor to numpy
                image_np = image.permute(1, 2, 0).numpy()
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            else:
                # It's already a numpy array
                image_pil = Image.fromarray(image)
                
            image_flipped = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Convert back to the original format
            if isinstance(image, torch.Tensor):
                # Convert back to tensor
                image = torch.from_numpy(np.array(image_flipped)).permute(2, 0, 1).float() / 255.0
            else:
                # Convert back to numpy
                image = np.array(image_flipped)
            
            # Get image width for flipping boxes
            width = image.shape[1] if isinstance(image, torch.Tensor) else image.shape[1]
            
            # Flip boxes
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target["boxes"] = boxes
            
        return image, target

class RandomScale:
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.min_scale = min_scale
        self.max_scale = max_scale
        
    def __call__(self, image, target):
        scale = random.uniform(self.min_scale, self.max_scale)
        
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale), int(width * scale)
        
        # Resize image
        image_resized = cv2.resize(image, (new_width, new_height))
        
        # Scale boxes
        boxes = target["boxes"].clone()
        boxes = boxes * scale
        target["boxes"] = boxes
        
        return image_resized, target
    
class SquareResize:
    def __init__(self, size=800):
        self.size = size
        
    def __call__(self, image, target):
        # If image is already a tensor, convert to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy()
        else:
            image_np = image
            
        # Get original dimensions
        h, w = image_np.shape[:2]
        
        # Resize to fixed square size
        image_resized = cv2.resize(image_np, (self.size, self.size))
        
        # Scale boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (self.size / w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (self.size / h)
            target["boxes"] = boxes
            
        # Convert back to tensor if needed
        if isinstance(image, torch.Tensor):
            image_resized = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            
        return image_resized, target
    
class Resize:
    def __init__(self, min_size=500, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size
        
    def __call__(self, image, target):
        # If image is already a tensor, convert to numpy
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy()
        else:
            image_np = image
            
        # Get original dimensions
        h, w = image_np.shape[:2]
        
        # Calculate scale based on min dimension
        min_dim = min(h, w)
        scale = self.min_size / min_dim
        
        # Check if max dimension exceeds max_size after scaling
        max_dim = max(h, w)
        if max_dim * scale > self.max_size:
            # If so, scale based on max dimension instead
            scale = self.max_size / max_dim
            
        # Calculate new dimensions while preserving aspect ratio
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        image_resized = cv2.resize(image_np, (new_w, new_h))
        
        # Scale boxes
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / h)
            target["boxes"] = boxes
            
        # Convert back to tensor if needed
        if isinstance(image, torch.Tensor):
            image_resized = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            
        return image_resized, target


# def get_transform(train, min_size=500, max_size=1000):
#     transforms = []
#     # Add resizing transform
#     transforms.append(Resize(min_size, max_size))
    
#     # Add the basic ToTensor transform
#     transforms.append(ToTensor())
    
#     # Add data augmentation for training
#     if train:
#         transforms.append(RandomHorizontalFlip(0.5))
    
#     # Add normalization
#     transforms.append(Normalize())
    
#     return Compose(transforms)
def get_transform(train, min_size=500, max_size=1000):
    transforms = []
    
    # Use SquareResize instead of Resize
    # transforms.append(SquareResize(size=800))  # or use min_size
    transforms.append(Resize(min_size=min_size, max_size=max_size))
    
    # Add the basic ToTensor transform
    transforms.append(ToTensor())
    
    # Add data augmentation for training
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    
    # Add normalization
    transforms.append(Normalize())
    
    return Compose(transforms)

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, target):
        # Normalize only affects the image, not the target
        if isinstance(image, torch.Tensor):
            image = F.normalize(image, mean=self.mean, std=self.std)
        else:
            # Convert numpy image to tensor first
            image = F.to_tensor(image)
            image = F.normalize(image, mean=self.mean, std=self.std)
        
        return image, target

# # Function to get transforms for train and validation
# def get_transform(train):
#     transforms = []
#     # Add the basic ToTensor transform
#     transforms.append(ToTensor())
    
#     # Add data augmentation for training
#     if train:
#         transforms.append(RandomHorizontalFlip(0.5))
#         # Uncomment this for scale augmentation if needed
#         # transforms.append(RandomScale(0.8, 1.2))
        
#     # Add normalization
#     transforms.append(Normalize())
    
#     return Compose(transforms)

# Example usage
if __name__ == "__main__":
    # Example usage of the dataset
    dataset_path = "/homes/es314/omr-objdet-benchmark/data/prepared_dataset/train"
    dataset = OMRDataset(
        root_dir=dataset_path,
        transforms=get_transform(train=True)
    )
    
    print(f"Dataset contains {len(dataset)} images")
    
    # Display a sample
    sample = dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample has {len(sample['boxes'])} bounding boxes")
    print(f"Labels: {sample['labels']}")