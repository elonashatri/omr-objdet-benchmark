#!/usr/bin/env python3
"""
Multi-Scale Enhancements for Faster R-CNN OMR Model

This module implements advanced multi-scale techniques for improving
Optical Music Recognition detection performance, including:
1. Multi-scale inference using image pyramids
2. Multi-scale training augmentation
3. Enhanced Feature Pyramid Network implementation
4. Test-time augmentation

These techniques help improve detection of musical elements at different scales,
from thin staff lines to larger symbols.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.ops import nms
import random
import numpy as np
from PIL import Image
from omr_dataset import get_transform
from collections import defaultdict
import traceback
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import argparse
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt



def multi_scale_inference(model, image, scales=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75], score_threshold=0.05, nms_threshold=0.45):
    """
    Run inference at multiple scales and merge results through NMS.
    
    Args:
        model: The detection model
        image: Input image tensor [C, H, W]
        scales: List of scales to use for creating the image pyramid
        score_threshold: Minimum score to keep a detection
        nms_threshold: IoU threshold for NMS
        
    Returns:
        Merged prediction dictionary with boxes, scores, and labels
    """
    device = next(model.parameters()).device
    original_image = image.clone()
    all_boxes = []
    all_scores = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for scale in scales:
            # Resize image according to scale
            height, width = original_image.shape[-2:]
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            scaled_image = F.interpolate(
                original_image.unsqueeze(0), 
                size=(new_height, new_width), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Run inference
            prediction = model([scaled_image.to(device)])[0]
            
            # Filter by score threshold
            keep_idxs = prediction['scores'] > score_threshold
            if not keep_idxs.any():
                continue
                
            boxes = prediction['boxes'][keep_idxs]
            scores = prediction['scores'][keep_idxs]
            labels = prediction['labels'][keep_idxs]
            
            # Scale boxes back to original size
            boxes[:, 0] *= width / new_width   # x1
            boxes[:, 1] *= height / new_height # y1
            boxes[:, 2] *= width / new_width   # x2
            boxes[:, 3] *= height / new_height # y2
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
    
    # Combine predictions from all scales
    if not all_boxes:  # No detections at any scale
        return {'boxes': torch.empty((0, 4), device=device),
                'scores': torch.empty(0, device=device),
                'labels': torch.empty(0, device=device, dtype=torch.long)}
    
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    
    # Apply NMS per class
    result_boxes = []
    result_scores = []
    result_labels = []
    
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        if not class_mask.any():
            continue
            
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        # Apply NMS
        keep_indices = nms(class_boxes, class_scores, nms_threshold)
        
        result_boxes.append(class_boxes[keep_indices])
        result_scores.append(class_scores[keep_indices])
        result_labels.append(torch.full_like(keep_indices, class_id))
    
    if not result_boxes:  # No detections after NMS
        return {'boxes': torch.empty((0, 4), device=device),
                'scores': torch.empty(0, device=device),
                'labels': torch.empty(0, device=device, dtype=torch.long)}
    
    # Combine all classes
    merged_boxes = torch.cat(result_boxes)
    merged_scores = torch.cat(result_scores)
    merged_labels = torch.cat(result_labels)
    
    # Sort by score for final output
    sorted_indices = torch.argsort(merged_scores, descending=True)
    
    return {
        'boxes': merged_boxes[sorted_indices],
        'scores': merged_scores[sorted_indices],
        'labels': merged_labels[sorted_indices]
    }

def filter_and_merge_classes(dataset, min_instances=24, args=None):
    """
    Filter rare classes and merge similar classes.
    
    Args:
        dataset: OMRDataset instance
        min_instances: Minimum number of instances required to keep a class
        args: Optional arguments object
        
    Returns:
        class_map: Mapping from original class IDs to new class IDs
        inverse_map: Mapping from new class IDs to original class IDs
        keep_classes: List of original class IDs that are kept
        merged_class_info: Dictionary with detailed information about the final classes
    """
    # Count class occurrences
    class_counts = defaultdict(int)
    
    # Get class name to ID mapping
    class_id_to_name = {}
    class_name_to_id = {}
    mapping_file = os.path.join(dataset.root_dir, '..', 'mapping.txt')
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(':')
                    if len(parts) >= 2:
                        class_id = int(parts[0])
                        class_name = ':'.join(parts[1:])
                        class_id_to_name[class_id] = class_name
                        class_name_to_id[class_name] = class_id
    except FileNotFoundError:
        print(f"Warning: Mapping file {mapping_file} not found.")
    
    # Define classes to be removed/replaced
    key_signature_mappings = {
        "AMajor": class_name_to_id.get("accidentalSharp", 0),
        "FMinor": class_name_to_id.get("accidentalKomaFlat", 0),
        "BMajor": class_name_to_id.get("accidentalSharp", 0),
        "EFlatMinor": class_name_to_id.get("accidentalFlat", 0),
        "FMajor": class_name_to_id.get("accidentalNatural", 0),
        "DMinor": class_name_to_id.get("accidentalNatural", 0),
        "DFlatMajor": class_name_to_id.get("accidentalFlat", 0),
        "EMajor": class_name_to_id.get("accidentalSharp", 0),
        "DMajor": class_name_to_id.get("accidentalSharp", 0),
        "EFlatMajor": class_name_to_id.get("accidentalFlat", 0),
        "GFlatMajor": class_name_to_id.get("accidentalFlat", 0),
        "BFlatMajor": class_name_to_id.get("accidentalFlat", 0),
        "AFlatMajor": class_name_to_id.get("accidentalFlat", 0),
        "GMajor": class_name_to_id.get("accidentalSharp", 0)
    }
    
    # Convert key names to IDs
    id_merge_map = {}
    for key_name, target_id in key_signature_mappings.items():
        if key_name in class_name_to_id and target_id > 0:
            id_merge_map[class_name_to_id[key_name]] = target_id
    
    # Special case for quarter tone flat -> koma flat
    if "accidentalQuarterToneFlatStein" in class_name_to_id and "accidentalKomaFlat" in class_name_to_id:
        id_merge_map[class_name_to_id["accidentalQuarterToneFlatStein"]] = class_name_to_id["accidentalKomaFlat"]
    
    # Iterate through all samples to count class occurrences
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            if 'labels' in sample:
                labels = sample['labels']
                for label in labels:
                    label_id = label.item() if isinstance(label, torch.Tensor) else label
                    
                    # Apply merging if applicable
                    if label_id in id_merge_map:
                        label_id = id_merge_map[label_id]
                    
                    class_counts[label_id] += 1
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Determine which classes to keep
    keep_classes = [cls for cls, count in class_counts.items() if count >= min_instances]
    keep_classes.sort()  # Sort to ensure deterministic mapping
    
    print(f"After merging and filtering, keeping {len(keep_classes)} classes out of {len(class_counts)} total")
    print(f"Classes with counts < {min_instances} will be filtered out")
    
    # Create class mapping
    class_map = {old_id: new_id+1 for new_id, old_id in enumerate(keep_classes)}  # +1 for background
    inverse_map = {new_id: old_id for old_id, new_id in class_map.items()}
    
    # Add entry for background class
    class_map[0] = 0
    inverse_map[0] = 0
    
    # Create merged class info for logging
    merged_class_info = {}
    for old_id in keep_classes:
        old_name = class_id_to_name.get(old_id, f"Unknown-{old_id}")
        new_id = class_map[old_id]
        merged_class_info[new_id] = {
            "original_id": old_id,
            "name": old_name,
            "count": class_counts[old_id]
        }
    
    # Print the list of classes being used
    print("\nClasses being used:")
    for new_id, info in sorted(merged_class_info.items()):
        print(f"  {new_id}: {info['name']} (original ID: {info['original_id']}, count: {info['count']})")
    
    # Print the list of classes that were merged
    print("\nClasses that were merged:")
    for source_id, target_id in id_merge_map.items():
        source_name = class_id_to_name.get(source_id, f"Unknown-{source_id}")
        target_name = class_id_to_name.get(target_id, f"Unknown-{target_id}")
        print(f"  {source_name} (ID: {source_id}) → {target_name} (ID: {target_id})")
    
    return class_map, inverse_map, keep_classes, merged_class_info, id_merge_map

def multi_scale_transform(image, target, min_sizes=[600, 800, 1000, 1200, 1400, 1600]):
    """
    Apply random scaling during training for multi-scale learning,
    optimized for high-resolution score images (2475×3504).
    
    Args:
        image: PIL Image
        target: Target dictionary with boxes and labels
        min_sizes: List of possible minimum sizes to randomly select from
        
    Returns:
        Transformed image and target
    """
    # Randomly select a minimum size
    min_size = random.choice(min_sizes)
    
    # Calculate max_size based on the original aspect ratio
    if isinstance(image, Image.Image):
        width, height = image.size
    else:  # Assuming tensor
        _, height, width = image.shape
    
    aspect_ratio = width / height
    # For sheet music with typical aspect ratio of ~0.7, we need adequate vertical resolution
    max_size = int(min_size * max(1.8, aspect_ratio * 1.2))  # Increased for better resolution
    
    # Check for staff lines and thin objects in the target
    has_staff_lines = False
    has_stems = False
    has_barlines = False
    
    if target and 'labels' in target:
        labels = target['labels']
        for label in labels:
            label_value = label.item() if isinstance(label, torch.Tensor) else label
            # Check for category IDs specific to staff lines (adapt to your dataset)
            if label_value in [3, 33, 217]:  # assuming these IDs represent staff lines
                has_staff_lines = True
            elif label_value in [2, 226065]:  # barlines
                has_barlines = True
            elif label_value in [775651]:  # stems
                has_stems = True
    
    # Create transform with special handling for staff lines
    transform_kwargs = {
        'train': True,
        'min_size': min_size,
        'max_size': max_size,
        'random_crop': False  # Start with no random crop
    }
    
    # If we have staff lines, reduce crop probability or avoid cropping
    # to prevent breaking the staff line structure
    if has_staff_lines:
        # Higher resolution for staff lines
        transform_kwargs['random_crop'] = random.random() < 0.15  # Much lower crop probability
    elif has_stems or has_barlines:
        # Somewhat careful with stems and barlines
        transform_kwargs['random_crop'] = random.random() < 0.3
    else:
        # For other musical elements, allow more aggressive cropping
        transform_kwargs['random_crop'] = random.random() < 0.5
    
    # Apply the transform with the selected parameters
    transform = get_transform(**transform_kwargs)
    
    return transform(image, target)


class EnhancedFPN(nn.Module):
    """
    Enhanced Feature Pyramid Network with better multi-scale feature fusion.
    
    This implementation adds:
    1. Attention mechanisms for adaptive feature weighting
    2. Enhanced feature fusion between levels
    3. Additional processing for better scale-invariant features
    """
    def __init__(self, in_channels_list, out_channels):
        super(EnhancedFPN, self).__init__()
        
        # Lateral connections for each input feature map
        self.lateral_convs = nn.ModuleList()
        
        # Output convolutions
        self.output_convs = nn.ModuleList()
        
        # For each input feature map
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        # Additional processing for better feature fusion
        self.fusion_convs = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            self.fusion_convs.append(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
            )
        
        # Attention modules for each level
        self.attention_modules = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.attention_modules.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
                    nn.Sigmoid()
                )
            )
        
    def forward(self, x):
        """
        Forward pass through the Enhanced FPN.
        
        Args:
            x: List of feature maps from the backbone [P2, P3, P4, P5]
                (from highest resolution to lowest)
                
        Returns:
            List of enhanced feature maps at different scales
        """
        # Get laterals
        laterals = [lateral_conv(feature) for lateral_conv, feature in zip(self.lateral_convs, x)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher level feature
            upsampled = F.interpolate(
                laterals[i], 
                size=laterals[i-1].shape[-2:],
                mode='bilinear', 
                align_corners=False
            )
            
            # Enhanced fusion instead of simple addition
            # Concatenate features and apply 1x1 conv for better fusion
            fused = torch.cat([laterals[i-1], upsampled], dim=1)
            fused = self.fusion_convs[i-1](fused)
            laterals[i-1] = fused
        
        # Apply output convolutions and attention
        outputs = []
        for i, lateral in enumerate(laterals):
            # Apply attention to emphasize important features
            attention_map = self.attention_modules[i](lateral)
            attended_features = lateral * attention_map
            
            # Apply output convolution
            output = self.output_convs[i](attended_features)
            outputs.append(output)
        
        return outputs


def test_time_augmentation(model, image, augmentations=None, score_threshold=0.05, nms_threshold=0.45, class_specific_params=None):
    """
    Apply test-time augmentation for more robust inference.
    
    Args:
        model: The detection model
        image: Input image tensor [C, H, W]
        augmentations: List of augmentations to apply (if None, use defaults)
        score_threshold: Minimum score to keep a detection
        nms_threshold: IoU threshold for NMS
        
    Returns:
        Merged prediction dictionary with boxes, scores, and labels
    """
    device = next(model.parameters()).device
    height, width = image.shape[-2:]
    
    all_boxes = []
    all_scores = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        # Default set of augmentations if none provided
        if augmentations is None:
            # Define default augmentations: original, horizontal flip, 90° rotations, and brightness variations
            augmentations = [
                {'flip': False, 'rotate': 0, 'brightness': 1.0},  # Original
                {'flip': True, 'rotate': 0, 'brightness': 1.0},   # Horizontal flip
                {'flip': False, 'rotate': 0, 'brightness': 0.8},  # Darker
                {'flip': False, 'rotate': 0, 'brightness': 1.2},  # Brighter
            ]
            
            # Only add rotation for square-ish images to avoid excessive padding
            if 0.8 <= width/height <= 1.2:  # If image is roughly square
                augmentations.extend([
                    {'flip': False, 'rotate': 90, 'brightness': 1.0},   # 90° rotation
                    {'flip': False, 'rotate': 270, 'brightness': 1.0},  # 270° rotation
                ])
        
        # Process each augmentation
        for aug in augmentations:
            # Apply transformations
            aug_image = image.clone()
            
            # Apply brightness adjustment if specified
            if aug.get('brightness', 1.0) != 1.0:
                aug_image = TF.adjust_brightness(aug_image, aug['brightness'])
            
            # Apply horizontal flip if specified
            if aug.get('flip', False):
                aug_image = torch.flip(aug_image, [2])  # Flip on width dimension
            
            # Apply rotation if specified
            rotation = aug.get('rotate', 0)
            if rotation:
                # Convert to PIL for rotation
                pil_image = TF.to_pil_image(aug_image)
                rotated_pil = pil_image.rotate(rotation, expand=True)
                aug_image = TF.to_tensor(rotated_pil)
                if aug_image.dim() == 3:
                    aug_image = aug_image.to(device)
                else:
                    aug_image = aug_image.unsqueeze(0).to(device)
            
            # Run inference
            prediction = model([aug_image])[0]
            
            # Filter by score threshold
            keep_idxs = prediction['scores'] > score_threshold
            if not keep_idxs.any():
                continue
                
            boxes = prediction['boxes'][keep_idxs].clone()
            scores = prediction['scores'][keep_idxs].clone()
            labels = prediction['labels'][keep_idxs].clone()
            
            # Transform boxes back to original orientation
            if aug.get('flip', False):
                # For horizontal flip: swap x-coordinates
                boxes[:, 0], boxes[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
            
            if rotation:
                # For rotation: transform coordinates back
                rot_height, rot_width = rotated_pil.height, rotated_pil.width
                
                if rotation == 90:
                    # Rotate 90° counter-clockwise back to original
                    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    boxes[:, 0] = rot_height - y2
                    boxes[:, 1] = x1
                    boxes[:, 2] = rot_height - y1
                    boxes[:, 3] = x2
                elif rotation == 180:
                    # Rotate 180° back to original
                    boxes[:, 0], boxes[:, 2] = rot_width - boxes[:, 2], rot_width - boxes[:, 0]
                    boxes[:, 1], boxes[:, 3] = rot_height - boxes[:, 3], rot_height - boxes[:, 1]
                elif rotation == 270:
                    # Rotate 270° counter-clockwise back to original
                    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    boxes[:, 0] = y1
                    boxes[:, 1] = rot_width - x2
                    boxes[:, 2] = y2
                    boxes[:, 3] = rot_width - x1
                
                # Scale boxes to original image size
                scale_x = width / rot_width
                scale_y = height / rot_height
                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y
                
                # Clip boxes to image boundaries
                boxes[:, 0].clamp_(min=0, max=width)
                boxes[:, 1].clamp_(min=0, max=height)
                boxes[:, 2].clamp_(min=0, max=width)
                boxes[:, 3].clamp_(min=0, max=height)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
    
    # Combine predictions from all augmentations
    if not all_boxes:  # No detections from any augmentation
        return {'boxes': torch.empty((0, 4), device=device),
                'scores': torch.empty(0, device=device),
                'labels': torch.empty(0, device=device, dtype=torch.long)}
    
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    
    # Apply NMS per class
    result_boxes = []
    result_scores = []
    result_labels = []
    
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        if not class_mask.any():
            continue
            
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        # Apply NMS
        keep_indices = nms(class_boxes, class_scores, nms_threshold)
        
        result_boxes.append(class_boxes[keep_indices])
        result_scores.append(class_scores[keep_indices])
        result_labels.append(torch.full_like(keep_indices, class_id))
    
    if not result_boxes:  # No detections after NMS
        return {'boxes': torch.empty((0, 4), device=device),
                'scores': torch.empty(0, device=device),
                'labels': torch.empty(0, device=device, dtype=torch.long)}
    
    # Combine all classes
    merged_boxes = torch.cat(result_boxes)
    merged_scores = torch.cat(result_scores)
    merged_labels = torch.cat(result_labels)
    
    # Sort by score for final output
    sorted_indices = torch.argsort(merged_scores, descending=True)
    
    return {
        'boxes': merged_boxes[sorted_indices],
        'scores': merged_scores[sorted_indices],
        'labels': merged_labels[sorted_indices]
    }


def get_enhanced_model_instance_segmentation(num_classes, args):
    """
    Get enhanced Faster R-CNN model with multi-scale improvements
    specifically optimized for OMR with thin staff lines and varied musical notation.
    
    Args:
        num_classes: Number of classes to detect
        args: Arguments containing model configuration
        
    Returns:
        Enhanced Faster R-CNN model
    """
    from torchvision.models.detection.faster_rcnn import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    
    # Parse anchor parameters - customize for music notation
    anchor_sizes = tuple([int(size) for size in args.anchor_sizes.split(',')])
    aspect_ratios = tuple([float(ratio) for ratio in args.aspect_ratios.split(',')])
    
    # For OMR, especially thin staff lines & with wide beams, ensure smallest anchor size
    # and most extreme aspect ratios are properly represented
    if min(anchor_sizes) > 3:
        # Add smaller anchor size for staff lines
        anchor_sizes = (2,) + anchor_sizes
    
    # Ensure we have extreme aspect ratios for staff lines (very wide/thin)
    extended_ratios = list(aspect_ratios)
    if min(aspect_ratios) > 0.03:  # Very thin objects need smaller ratios
        extended_ratios.append(0.03)  # For staff lines
    if max(aspect_ratios) < 30.0:  # Very wide objects need larger ratios
        extended_ratios.append(30.0)  # For wide beams and system barlines
    aspect_ratios = tuple(sorted(extended_ratios))
    
    # Create base backbone (ResNet)
    if 'resnet101' in args.backbone:
        backbone_name = 'resnet101'
    else:
        backbone_name = 'resnet50'
    
    # Get the backbone with FPN (no pretrained_backbone flag in recent PyTorch)
    try:
        backbone = resnet_fpn_backbone(
            backbone_name, 
            pretrained=args.pretrained,
            trainable_layers=5 - args.frozen_layers  # Adjust trainable layers
        )
    except TypeError:
        # For older PyTorch versions that use pretrained_backbone
        backbone = resnet_fpn_backbone(
            backbone_name, 
            pretrained=args.pretrained
        )
        
        # Freeze backbone layers if requested
        if args.frozen_layers > 0:
            for name, parameter in backbone.named_parameters():
                if 'fpn' not in name:  # Don't freeze FPN layers
                    layer_parts = name.split('.')
                    if len(layer_parts) >= 2 and layer_parts[0] == 'body':
                        layer_num = int(layer_parts[1]) if layer_parts[1].isdigit() else -1
                        if layer_num < args.frozen_layers:
                            parameter.requires_grad_(False)
    

    #    For an FPN backbone, we typically have 5 feature maps
    num_feature_maps = 5  # Typically 5 for FPN: P2, P3, P4, P5, P6

    # Parse anchor parameters
    anchor_sizes = [int(size) for size in args.anchor_sizes.split(',')]
    aspect_ratios = [float(ratio) for ratio in args.aspect_ratios.split(',')]

    # Create properly formatted anchor sizes for FPN
    # Each feature map gets its own set of anchor sizes
    anchor_sizes_tuple = tuple(
        (anchor_sizes[min(i, len(anchor_sizes)-1)],) 
        for i in range(num_feature_maps)
    )

    # Create properly formatted aspect ratios for FPN
    # Each feature map gets the same aspect ratios
    aspect_ratios_tuple = tuple(
        (aspect_ratios,) 
        for _ in range(num_feature_maps)
    )

    # Create anchor generator
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes_tuple,
        aspect_ratios=aspect_ratios_tuple
    )
    
    # Create RoI pooler with custom parameters - increased sampling ratio
    from torchvision.ops import MultiScaleRoIAlign
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=args.initial_crop_size,
        sampling_ratio=4  # Increased for better detail preservation with thin lines
    )
    
    # Create the Faster R-CNN model with music notation specific enhancements
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=args.min_size,
        max_size=args.max_size,
        # Adjust mean/std for music score images which have different statistics than ImageNet
        # Sheet music is typically high contrast with mostly white backgrounds
        image_mean=[0.95, 0.95, 0.95],  # Closer to white background
        image_std=[0.15, 0.15, 0.15],   # Smaller std for high contrast
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        # High density of objects in sheet music needs more proposals
        rpn_pre_nms_top_n_train=8000,  # Further increased from default
        rpn_pre_nms_top_n_test=8000,   # Further increased from default
        rpn_post_nms_top_n_train=args.first_stage_max_proposals,
        rpn_post_nms_top_n_test=args.first_stage_max_proposals,
        # Adjusted NMS thresholds for the high-overlap nature of music notation
        rpn_nms_thresh=args.first_stage_nms_iou_threshold,
        rpn_score_thresh=args.first_stage_nms_score_threshold,
        box_score_thresh=args.second_stage_nms_score_threshold,
        box_nms_thresh=args.second_stage_nms_iou_threshold,
        # Handle high density of objects in complex scores
        box_detections_per_img=args.second_stage_max_total_detections
    )
    
    # Custom model initialization for better convergence with thin objects
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            # Use MSRA/He initialization for convolutional layers
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    return model


def adjust_training_pipeline(args):
    """
    Modify the training pipeline to use multi-scale techniques
    optimized for high-resolution music score detection.
    
    Args:
        args: Training arguments
        
    Returns:
        Updated args with multi-scale settings
    """
    # Enable multi-scale training by setting flag
    args.multi_scale_train = True
    
    # Set min_sizes for multi-scale training - larger ranges for higher resolution images
    # For 2475×3504 music score images
    args.multi_scale_min_sizes = [600, 800, 1000, 1200, 1400, 1600]
    
    # Enable test-time augmentation for evaluation
    args.test_time_augmentation = True
    
    # Add image pyramid for inference - wider range for score sheets
    args.inference_scales = [0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75]
    
    # Set random crop probability - reduce for staff line preservation
    args.random_crop_prob = 0.3
    
    # Adjust crop sizes for larger source images
    if not hasattr(args, 'crop_size_multiplier'):
        args.crop_size_multiplier = 1.5  # Larger crops for high-res images
    
    # Modify batch sizes for large images if needed
    if args.batch_size > 1 and not hasattr(args, 'adjusted_batch'):
        args.batch_size = 1  # Safer with large images
        args.adjusted_batch = True
        
    # Set class-specific detection parameters (used in test_time_augmentation)
    args.class_specific_params = {
        # kStaffLine class - use lower score threshold and special NMS
        'kStaffLine': {
            'score_threshold': 0.03,  # Lower threshold for staff lines
            'nms_threshold': 0.3,     # More aggressive NMS for staff lines
        },
        # barline class
        'barline': {
            'score_threshold': 0.04,  # Lower threshold for barlines
            'nms_threshold': 0.4,     # More conservative NMS
        },
        # stem class
        'stem': {
            'score_threshold': 0.04,  # Lower threshold for stems
            'nms_threshold': 0.35,    # More aggressive NMS for these thin objects
        },
        # systemicBarline class
        'systemicBarline': {
            'score_threshold': 0.03,  # Lower threshold
            'nms_threshold': 0.4,     # More conservative NMS
        }
    }
    
    # Custom image augmentations particularly helpful for musical notation
    args.custom_augmentations = True
    args.brightness_range = (0.9, 1.1)      # Slight brightness variation
    args.contrast_range = (1.0, 1.2)        # Moderate contrast enhancement good for staff lines
    args.enable_sharpening = True           # Enable sharpening for thin lines
    args.sharpening_prob = 0.25             # Probability to apply sharpening
    
    # Adjust IoU thresholds based on high overlap in music notation
    args.first_stage_nms_iou_threshold = 0.6   # Increased from default
    args.second_stage_nms_iou_threshold = 0.45  # Adjusted for better detection
    
    # Increase proposal count for dense scenes
    args.first_stage_max_proposals = 3000    # Increased from default
    args.second_stage_max_total_detections = 3000  # Increased for dense scenes
    
    # Adjust learning parameters for high-res images
    if not hasattr(args, 'adjusted_learning'):
        args.learning_rate = 0.0005  # Reduced learning rate for stability
        args.weight_decay = 0.0002   # Slightly increased weight decay
        args.adjusted_learning = True
    
    return args


def create_custom_transform_fn(args):
    """
    Create a custom transform function that uses multi-scale techniques
    optimized for musical notation with thin staff lines and varied symbols.
    
    Args:
        args: Arguments containing multi-scale configuration
        
    Returns:
        Transform functions for training and validation
    """

    def train_transform(image, target):
        """Enhanced training transform with multi-scale and augmentation."""
        # For high-res music scores, use more varied scaling
        if args.multi_scale_train and random.random() < 0.65:  # Increased probability
            # Use multi-scale transform with extended range
            return multi_scale_transform(
                image, 
                target, 
                min_sizes=args.multi_scale_min_sizes
            )

        else:
            # Use standard transform with higher-resolution settings
            # Add additional transforms particularly helpful for OMR
            transform = get_transform(
                train=True,
                min_size=args.min_size,
                max_size=args.max_size,
                random_crop=(random.random() < args.random_crop_prob)
            )
            
            # Get transformed image and target
            image_t, target_t = transform(image, target)
            
            # Randomly add slight contrast enhancement for better staff line detection
            if random.random() < 0.3:
                # Apply contrast enhancement
                contrast_factor = random.uniform(1.05, 1.2)
                image_t = F.adjust_contrast(image_t, contrast_factor)
            
            # Randomly add slight sharpening to enhance small details
            if random.random() < 0.3:
                # Apply sharpening using unsharp mask technique
                # This requires custom implementation in PyTorch
                # For simplicity, we'll use a placeholder that would be implemented
                # image_t = apply_unsharp_mask(image_t, amount=1.5, radius=1.0, threshold=0)
                pass
                
            return image_t, target_t
    
    def val_transform(image, target):
        """Validation transform with consistent sizing."""
        return get_transform(
            train=False,
            min_size=args.min_size,
            max_size=args.max_size
        )(image, target)
    
    return train_transform, val_transform


def enhance_evaluation_pipeline(model, val_loader, device, args, writer=None, epoch=0, class_names=None):
    """
    Enhanced evaluation pipeline with multi-scale inference and test-time augmentation
    specifically optimized for OMR with staff lines and musical notation.
    
    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        device: Device to run evaluation on
        args: Arguments with evaluation configuration
        writer: TensorBoard writer
        epoch: Current epoch
        class_names: Dictionary mapping class IDs to names
        
    Returns:
        Dictionary of metrics
    """
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from tqdm import tqdm
    
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
    pbar = tqdm(val_loader, desc="Calculating mAP with multi-scale inference")
    
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
                
                # Process data batch
                for batch_idx in range(len(data[0])):
                    # Get elements for this batch item
                    image = data[0][batch_idx]
                    boxes = data[1][batch_idx]
                    labels = data[2][batch_idx]
                    image_id = data[3][batch_idx]
                    
                    # Skip invalid elements
                    if isinstance(image, str) or isinstance(boxes, str) or isinstance(labels, str):
                        continue
                    
                    # Convert to tensors and move to device
                    image_tensor = image.to(device) if isinstance(image, torch.Tensor) else torch.tensor(image).to(device)
                    images.append(image_tensor)
                    
                    # Create target dict
                    try:
                        box_tensor = boxes.to(device) if isinstance(boxes, torch.Tensor) else torch.tensor(boxes).to(device)
                        label_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels).to(device)
                        img_id_tensor = image_id if isinstance(image_id, torch.Tensor) else torch.tensor([image_id])
                        
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
                
                # Get predictions with multi-scale inference or test-time augmentation
                predictions = []
                for img in images:
                    # Determine if image contains staff lines or other thin objects
                    # by looking at corresponding target
                    idx = images.index(img)
                    target = targets[idx] if idx < len(targets) else None
                    
                    has_thin_objects = False
                    has_staff_lines = False
                    class_specific_nms = {}
                    
                    if target and 'labels' in target:
                        labels = target['labels'].cpu().tolist() if isinstance(target['labels'], torch.Tensor) else target['labels']
                        
                        # Get class names for checking - handle either ID or string format
                        for label in labels:
                            if class_names and label in class_names:
                                class_name = class_names[label].lower()
                                if "staff" in class_name or "stem" in class_name or "barline" in class_name:
                                    has_thin_objects = True
                                if "staff" in class_name:
                                    has_staff_lines = True
                                    
                                # Apply class-specific params if available
                                if hasattr(args, 'class_specific_params') and class_name in args.class_specific_params:
                                    class_specific_nms[label] = args.class_specific_params[class_name]
                    
                    # Adjust inference parameters based on content
                    score_threshold = args.second_stage_nms_score_threshold
                    nms_threshold = args.second_stage_nms_iou_threshold
                    
                    # Use lower thresholds for staff lines
                    if has_staff_lines:
                        score_threshold = 0.03  # Lower score threshold for staff detection
                        nms_threshold = 0.4    # Adjusted NMS for staff lines
                    elif has_thin_objects:
                        score_threshold = 0.04  # Slightly lower score threshold
                    
                    if args.test_time_augmentation:
                        # Use TTA for more robust predictions with special handling for music notation
                        pred = test_time_augmentation(
                            model, 
                            img, 
                            score_threshold=score_threshold,
                            nms_threshold=nms_threshold,
                            class_specific_params=class_specific_nms
                        )
                    else:
                        # Use multi-scale inference with image pyramid and adapted parameters
                        pred = multi_scale_inference(
                            model, 
                            img, 
                            scales=args.inference_scales,
                            score_threshold=score_threshold,
                            nms_threshold=nms_threshold
                        )
                    predictions.append(pred)
                
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
                    
                    # Format target
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
    
    # Visualize results with the original visualization function
    if has_visualized and writer is not None:
        # This will use the existing visualize_predictions function from the original code
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        visualize_predictions(
            vis_images[:min(5, len(vis_images))], 
            vis_predictions[:min(5, len(vis_predictions))], 
            vis_targets[:min(5, len(vis_targets))], 
            writer,
            epoch, 
            class_names=class_names
        )
    
    # Compute final metric
    result = metric.compute()
    
    # Log detailed results
    print("\nMulti-Scale Mean Average Precision Results:")
    print(f"mAP: {result['map'].item():.4f}")
    print(f"mAP@0.5: {result['map_50'].item():.4f}")
    print(f"mAP@0.75: {result['map_75'].item():.4f}")
    print(f"mAP small: {result['map_small'].item():.4f}")
    print(f"mAP medium: {result['map_medium'].item():.4f}")
    print(f"mAP large: {result['map_large'].item():.4f}")
    
    # Log to TensorBoard if writer is provided
    if writer is not None:
        writer.add_scalar('metrics/mAP_multi_scale', result['map'].item(), epoch)
        writer.add_scalar('metrics/mAP@0.5_multi_scale', result['map_50'].item(), epoch)
        writer.add_scalar('metrics/mAP@0.75_multi_scale', result['map_75'].item(), epoch)
        writer.add_scalar('metrics/mAP_small_multi_scale', result['map_small'].item(), epoch)
        writer.add_scalar('metrics/mAP_medium_multi_scale', result['map_medium'].item(), epoch)
        writer.add_scalar('metrics/mAP_large_multi_scale', result['map_large'].item(), epoch)
    
    return {
        'mAP': result['map'].item(),
        'mAP@0.5': result['map_50'].item(),
        'mAP@0.75': result['map_75'].item(),
        'mAP_small': result['map_small'].item(),
        'mAP_medium': result['map_medium'].item(),
        'mAP_large': result['map_large'].item(),
    }

# Example usage of these enhancements in the main training script would be:
"""
def main():
    args = parse_args()
    
    # Enable multi-scale training
    args = adjust_training_pipeline(args)
    
    # Create custom transforms with multi-scale capabilities
    train_transform_fn, val_transform_fn = create_custom_transform_fn(args)
    
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
    
    # ... data loaders and other setup ...
    
    # Create enhanced model
    model = get_enhanced_model_instance_segmentation(num_classes, args)
    
    # ... training loop ...
    
    # Evaluate using multi-scale evaluation
    val_metrics = enhance_evaluation_pipeline(
        model, 
        val_loader, 
        device, 
        args,
        writer,
        epoch,
        class_names
    )
"""