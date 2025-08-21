"""
Data Loader for Staff Line Detection

This module contains dataset classes and data loaders
for handling musical score images and staff line masks.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm



class StaffLineDataset(Dataset):
    """
    Dataset for staff line detection in musical scores.
    
    Args:
        img_dir (str): Directory with input images
        mask_dir (str): Directory with ground truth masks (staff line binary segmentation)
        transform (callable, optional): Optional transform to be applied on samples
        train (bool): Whether this is for training (enables augmentation)
        target_size (tuple): Target size for resizing images (height, width)
        subset_fraction (float): Fraction of dataset to use (0.0-1.0)
        subset_seed (int): Random seed for subset selection
    """
    def __init__(self, img_dir, mask_dir, transform=None, train=True, 
                 target_size=(512, 512), subset_fraction=1.0, subset_seed=42):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.train = train
        self.target_size = target_size
        
        # Get all image files
        all_img_files = [f for f in os.listdir(img_dir) 
                         if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.tif')]
        
        # Take a subset of the data if requested
        if subset_fraction < 1.0:
            # Set random seed for reproducibility
            random.seed(subset_seed)
            subset_size = max(1, int(len(all_img_files) * subset_fraction))
            self.img_filenames = random.sample(all_img_files, subset_size)
            print(f"Using {subset_size} images ({subset_fraction:.1%} of {len(all_img_files)})")
        else:
            self.img_filenames = all_img_files
        
    def __len__(self):
        return len(self.img_filenames)
    
    def __getitem__(self, idx):
        img_name = self.img_filenames[idx]
        mask_name = os.path.splitext(img_name)[0] + '.png'  # Assuming masks are PNG files
        
        # Load image and mask
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        try:
            # Read as grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
                
            # Check if mask exists
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to load mask: {mask_path}")
            else:
                # If mask is missing, create an empty mask with the same dimensions as the image
                mask = np.zeros_like(image)
            
            # ALWAYS resize to target size
            if self.target_size:
                image = cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                                  interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            
            # Convert to tensors
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            
            # Apply transforms if provided
            if self.transform and self.train:
                image, mask = self.transform(image, mask)
                
            # Double-check sizes match target size after transform
            if image.shape[1] != self.target_size[0] or image.shape[2] != self.target_size[1]:
                image = TF.resize(image, [self.target_size[0], self.target_size[1]])
                
            if mask.shape[1] != self.target_size[0] or mask.shape[2] != self.target_size[1]:
                mask = TF.resize(mask, [self.target_size[0], self.target_size[1]], 
                                interpolation=TF.InterpolationMode.NEAREST)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            # Return a blank image and mask of the correct size as fallback
            blank_image = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            blank_mask = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return blank_image, blank_mask



class XMLStaffLineDataset(Dataset):
    """
    Dataset for staff line detection that parses XML annotations directly.
    
    Args:
        img_dir (str): Directory with input images
        xml_dir (str): Directory with XML annotations
        transform (callable, optional): Optional transform to be applied on samples
        train (bool): Whether this is for training (enables augmentation)
        target_size (tuple): Target size for resizing images (height, width)
        subset_fraction (float): Fraction of dataset to use (0.0-1.0)
        subset_seed (int): Random seed for subset selection
    """
    def __init__(self, img_dir, xml_dir, transform=None, train=True, 
                target_size=(512, 512), subset_fraction=1.0, subset_seed=42):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.train = train
        self.target_size = target_size
        
        # Get all image files
        all_img_files = [f for f in os.listdir(img_dir) 
                        if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.tif')]
        
        # Take a subset of the data if requested
        if subset_fraction < 1.0:
            # Set random seed for reproducibility
            random.seed(subset_seed)
            subset_size = max(1, int(len(all_img_files) * subset_fraction))
            self.img_filenames = random.sample(all_img_files, subset_size)
            print(f"Using {subset_size} images ({subset_fraction:.1%} of {len(all_img_files)})")
        else:
            self.img_filenames = all_img_files
        
    def __len__(self):
        return len(self.img_filenames)
    
    def __getitem__(self, idx):
        img_name = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            # Get corresponding XML file
            xml_name = os.path.splitext(img_name)[0] + '.xml'
            xml_path = os.path.join(self.xml_dir, xml_name)
            
            # Load image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            # Generate mask from XML if it exists
            if os.path.exists(xml_path):
                mask = self.parse_xml_to_mask(xml_path, image.shape)
            else:
                print(f"Warning: XML not found for {img_name}. Creating empty mask.")
                mask = np.zeros_like(image)
            
            # ALWAYS resize to target size
            if self.target_size:
                image = cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                                  interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            
            # Convert to tensors
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
            mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
            
            # Apply transforms if provided
            if self.transform and self.train:
                image, mask = self.transform(image, mask)
                
            # Double-check sizes match target size after transform
            if image.shape[1] != self.target_size[0] or image.shape[2] != self.target_size[1]:
                image = TF.resize(image, [self.target_size[0], self.target_size[1]])
                
            if mask.shape[1] != self.target_size[0] or mask.shape[2] != self.target_size[1]:
                mask = TF.resize(mask, [self.target_size[0], self.target_size[1]], 
                                interpolation=TF.InterpolationMode.NEAREST)
            
            return image, mask
            
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            # Return a blank image and mask of the correct size as fallback
            blank_image = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            blank_mask = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return blank_image, blank_mask

class StaffLineAugmentation:
    """
    Custom augmentation for staff line detection.
    Takes into account the special properties of musical scores.
    """
    def __init__(self, p=0.5, target_size=(512, 512)):
        self.p = p
        self.target_size = target_size
        
    def __call__(self, image, mask):
        # Don't rotate too much to preserve staff line orientation
        if random.random() < self.p:
            angle = random.uniform(-5, 5)  # Small rotation
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Random crop
        if random.random() < self.p:
            # Ensure we don't crop too much
            min_height = int(self.target_size[0] * 0.9)  # At least 90% of target height
            min_width = int(self.target_size[1] * 0.9)   # At least 90% of target width
            
            # Ensure crop size isn't larger than image
            crop_height = min(min_height, image.shape[1])
            crop_width = min(min_width, image.shape[2])
            
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(crop_height, crop_width))
            
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
        
        # IMPORTANT: Always ensure final size matches target_size
        image = TF.resize(image, [self.target_size[0], self.target_size[1]])
        mask = TF.resize(mask, [self.target_size[0], self.target_size[1]], interpolation=TF.InterpolationMode.NEAREST)
        
        # Add noise (simulate degradation, stains, etc.)
        if random.random() < self.p:
            noise = torch.randn_like(image) * 0.05
            image = torch.clamp(image + noise, 0, 1)
            
        # Adjust brightness and contrast
        if random.random() < self.p:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            image = TF.adjust_contrast(image, contrast_factor)
            
        # Simulate staff line discontinuities (interrupted by notes)
        if random.random() < self.p:
            mask = self.add_discontinuities(mask)
        
        return image, mask
    
    def add_discontinuities(self, mask, max_gaps=10, gap_size_range=(5, 20)):
        """Add random gaps to staff lines to simulate note interruptions."""
        c, h, w = mask.shape
        mask_np = mask.squeeze(0).numpy()
        
        # Find non-zero rows (rows with staff lines)
        staff_line_rows = np.where(np.sum(mask_np, axis=1) > 0)[0]
        
        # For each row with a staff line
        for row in staff_line_rows:
            # Get indices where the line exists in this row
            line_indices = np.where(mask_np[row, :] > 0)[0]
            
            if len(line_indices) > 0:
                # Calculate number of gaps to add - ensure at least 1
                max_possible_gaps = max(1, len(line_indices) // 50)
                num_gaps = random.randint(1, min(max_gaps, max_possible_gaps))
                
                for _ in range(num_gaps):
                    # Get start index for the gap
                    if len(line_indices) > 0:
                        start_idx = random.choice(line_indices)
                        
                        # Determine gap size
                        gap_size = random.randint(gap_size_range[0], gap_size_range[1])
                        
                        # Create gap (set pixels to 0)
                        end_idx = min(start_idx + gap_size, w)
                        mask_np[row, start_idx:end_idx] = 0
        
        # Convert back to tensor
        return torch.from_numpy(mask_np).float().unsqueeze(0)


def get_dataloader(img_dir, mask_dir=None, xml_dir=None, batch_size=8, 
                   train_val_split=0.8, num_workers=4, augment=True, 
                   target_size=(512, 512), subset_fraction=1.0):
    """
    Creates DataLoaders for training and validation.
    
    Args:
        img_dir (str): Directory with input images
        mask_dir (str, optional): Directory with ground truth masks
        xml_dir (str, optional): Directory with XML annotations
        batch_size (int): Batch size
        train_val_split (float): Proportion of data to use for training
        num_workers (int): Number of workers for data loading
        augment (bool): Whether to apply data augmentation
        target_size (tuple): Target size for resizing images (height, width)
        subset_fraction (float): Fraction of dataset to use (0.0-1.0)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create transformer for data augmentation
    transform = StaffLineAugmentation() if augment else None
    
    # Create datasets based on available data
    if xml_dir and not mask_dir:
        # Using XML annotations directly
        dataset = XMLStaffLineDataset(
            img_dir, xml_dir, transform=None, train=False, 
            target_size=target_size, subset_fraction=subset_fraction
        )
    elif mask_dir:
        # Using pre-generated mask images
        dataset = StaffLineDataset(
            img_dir, mask_dir, transform=None, train=False, 
            target_size=target_size, subset_fraction=subset_fraction
        )
    else:
        raise ValueError("Either mask_dir or xml_dir must be provided")
    
    # Get the filenames
    img_files = dataset.img_filenames
    
    # Shuffle files
    random.seed(42)  # For reproducibility
    random.shuffle(img_files)
    
    # Split into training and validation sets
    split_idx = int(len(img_files) * train_val_split)
    train_files = img_files[:split_idx]
    val_files = img_files[split_idx:]
    
    print(f"Training set: {len(train_files)} images, Validation set: {len(val_files)} images")
    
    # Create datasets for train and validation
    if xml_dir and not mask_dir:
        train_dataset = XMLStaffLineDataset(
            img_dir, xml_dir, transform=transform, train=True,
            target_size=target_size, subset_fraction=1.0  # We've already taken subset
        )
        train_dataset.img_filenames = train_files
        
        val_dataset = XMLStaffLineDataset(
            img_dir, xml_dir, transform=None, train=False,
            target_size=target_size, subset_fraction=1.0  # We've already taken subset
        )
        val_dataset.img_filenames = val_files
    
    elif mask_dir:
        train_dataset = StaffLineDataset(
            img_dir, mask_dir, transform=transform, train=True,
            target_size=target_size, subset_fraction=1.0  # We've already taken subset
        )
        train_dataset.img_filenames = train_files
        
        val_dataset = StaffLineDataset(
            img_dir, mask_dir, transform=None, train=False,
            target_size=target_size, subset_fraction=1.0  # We've already taken subset
        )
        val_dataset.img_filenames = val_files
    
    # Create data loaders
    train_loader = None
    if len(train_files) > 0:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
    )
    
    val_loader = None
    if len(val_files) > 0:
        val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test the dataset and dataloader
    import matplotlib.pyplot as plt
    
    # Example paths
    img_dir = "path/to/images"
    mask_dir = "path/to/masks"
    xml_dir = "path/to/xml"
    
    # Only run this if directories exist
    if os.path.exists(img_dir) and (os.path.exists(mask_dir) or os.path.exists(xml_dir)):
        # Create test dataset
        if os.path.exists(mask_dir):
            dataset = StaffLineDataset(img_dir, mask_dir)
            print(f"Created dataset with {len(dataset)} samples from masks")
        else:
            dataset = XMLStaffLineDataset(img_dir, xml_dir)
            print(f"Created dataset with {len(dataset)} samples from XML")
            
        # Test loading an item
        if len(dataset) > 0:
            img, mask = dataset[0]
            print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
            
            # Visualize
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img.squeeze().numpy(), cmap='gray')
            plt.title("Image")
            plt.subplot(1, 2, 2)
            plt.imshow(mask.squeeze().numpy(), cmap='gray')
            plt.title("Mask")
            plt.tight_layout()
            plt.show()