#!/usr/bin/env python
"""
Mask R-CNN
Multi-scale inference and evaluation for the DOREMI Dataset with musical scores

Usage:
    # Run evaluation comparing detections with XML annotations
    python inference_maskrcnn.py --weights=/path/to/weights.h5 --images-dir=/path/to/images --annot-dir=/path/to/annotations
    
module load miniconda/4.7.12
conda activate maskrcnn_cpu
"""

import os
import sys
sys.path.append('/homes/es314/DOREMI_version_2/MRCNN_DOREMI_20210503')  # Adjust this to your Mask R-CNN location
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import datetime
import time
import cv2
import skimage.io
import skimage.color
import glob
import csv
import xml.etree.ElementTree as ET
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from keras import backend as K
from collections import defaultdict
import pandas as pd
from xml.dom import minidom

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to classnames file
CLASSNAMES_PATH = "/import/c4dm-05/elona/doremi_v5_half/train_validation_test_records/mapping.json"

############################################################
#  Configurations
############################################################

class DoremiConfig(Config):
    """
    Configuration for inference on the DOREMI dataset with musical scores.
    """
    # Give the configuration a recognizable name
    NAME = "doremi"

    # We use a GPU with 12GB memory
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 71  # Background + 71 classes

    # Our image size 
    IMAGE_MAX_DIM = 1024
    
    BACKBONE = "resnet101"
    
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # Increase max instances for dense musical scores
    DETECTION_MAX_INSTANCES = 1000
    # Set NMS threshold higher to avoid removing valid musical notation
    DETECTION_NMS_THRESHOLD = 0.8
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    
    # Lower confidence threshold for musical notation which often has similar patterns
    DETECTION_MIN_CONFIDENCE = 0.05
    
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    
    # Increase these for better detection on musical scores
    POST_NMS_ROIS_INFERENCE = 2500
    #  A higher value (0.8) means more region proposals will be considered
    RPN_NMS_THRESHOLD = 0.9

############################################################
#  Load Annotations
############################################################

def load_annotations(xml_path, class_name_to_id):
    """
    Load annotations from an XML file in DOREMI format.
    
    Returns:
    --------
    dict: Contains ground truth boxes, class_ids, and masks
    """
    try:
        xmldoc = minidom.parse(xml_path)
        
        # Initialize ground truth containers
        gt_boxes = []
        gt_class_ids = []
        gt_masks = []  # Not used for evaluation but kept for compatibility
        
        # Get image dimensions
        # In DOREMI format, we can get this from the XML or use a default
        # Hardcoded for this example, but you can adjust for your XML format
        img_height = 3504  # Default value
        img_width = 2474   # Default value
        
        # Parse nodes (annotations)
        nodes = xmldoc.getElementsByTagName('Node')
        
        for node in nodes:
            try:
                # Get class name
                node_classname_el = node.getElementsByTagName('ClassName')[0]
                class_name = node_classname_el.firstChild.data
                
                # Skip if class name not in mapping
                if class_name not in class_name_to_id:
                    continue
                
                class_id = class_name_to_id[class_name]
                
                # Get bounding box coordinates
                node_top = int(node.getElementsByTagName('Top')[0].firstChild.data)
                node_left = int(node.getElementsByTagName('Left')[0].firstChild.data)
                node_width = int(node.getElementsByTagName('Width')[0].firstChild.data)
                node_height = int(node.getElementsByTagName('Height')[0].firstChild.data)
                
                # Convert to [y1, x1, y2, x2] format
                y1 = node_top
                x1 = node_left
                y2 = node_top + node_height
                x2 = node_left + node_width
                
                # Add to lists
                gt_boxes.append([y1, x1, y2, x2])
                gt_class_ids.append(class_id)
                
                # Create a simple rectangular mask (actual mask could be more complex)
                # Not used for evaluation but kept for compatibility
                mask = np.zeros([img_height, img_width], dtype=np.bool)
                mask[y1:y2, x1:x2] = True
                gt_masks.append(mask)
                
            except Exception as e:
                print(f"Error parsing node in {xml_path}: {e}")
                continue
        
        # Convert lists to numpy arrays
        if gt_boxes:
            gt_boxes = np.array(gt_boxes)
            gt_class_ids = np.array(gt_class_ids)
            gt_masks = np.array(gt_masks).transpose(1, 2, 0)  # [height, width, num_instances]
        else:
            gt_boxes = np.zeros((0, 4))
            gt_class_ids = np.array([])
            gt_masks = np.zeros([img_height, img_width, 0])
        
        return {
            "boxes": gt_boxes,
            "class_ids": gt_class_ids,
            "masks": gt_masks,
            "image_shape": (img_height, img_width)
        }
        
    except Exception as e:
        print(f"Error loading annotations from {xml_path}: {e}")
        return None


def visualize_masks(image, detections, class_names, output_path):
    """
    Create a visualization of masks for each detection.
    
    Parameters:
    -----------
    image : numpy array
        The original image
    detections : dict
        Dictionary containing detection results
    class_names : list
        List of class names
    output_path : str
        Path to save the visualization
    """
    # Make a copy of the image for blending
    img_display = image.copy()
    
    # Get masks and class IDs
    masks = detections['masks']
    class_ids = detections['class_ids']
    scores = detections['scores']
    
    # Create an overlay for the masks
    masks_overlay = np.zeros((*image.shape[:2], 4), dtype=np.float32)  # RGBA
    
    # Assign a different color to each mask
    for i, mask in enumerate(masks.transpose(2, 0, 1)):
        class_id = class_ids[i]
        score = scores[i]
        
        # Create a color based on class ID for consistency
        hue = (class_id * 0.15) % 1.0
        rgb = plt.cm.hsv(hue)[:3]
        
        # Set alpha based on confidence
        alpha = min(0.7, score * 0.7)  # Cap at 0.7 for visibility
        
        # Create RGBA color
        rgba = np.array([*rgb, alpha])
        
        # Apply the mask with this color
        # For each pixel where mask is True, set the color
        for c in range(4):
            masks_overlay[:, :, c] = np.where(mask, rgba[c], masks_overlay[:, :, c])
    
    # Create a figure
    plt.figure(figsize=(16, 8))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot masks overlay on image
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    # Use alpha blending for the masks
    plt.imshow(masks_overlay, alpha=0.7)
    plt.title("Masks Overlay")
    plt.axis('off')
    
    # Add a legend for classes
    # Get unique class IDs
    unique_classes = np.unique(class_ids)
    handles = []
    labels = []
    for cls_id in unique_classes:
        hue = (cls_id * 0.15) % 1.0
        rgb = plt.cm.hsv(hue)[:3]
        patch = plt.Rectangle((0, 0), 1, 1, color=rgb)
        handles.append(patch)
        labels.append(class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}")
    
    # Place legend outside the plot
    plt.figlegend(handles, labels, loc='lower center', ncol=min(5, len(unique_classes)))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual mask images for each detection
    masks_dir = os.path.join(os.path.dirname(output_path), "individual_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    for i, (mask, class_id, score) in enumerate(zip(masks.transpose(2, 0, 1), class_ids, scores)):
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        mask_filename = f"{os.path.splitext(os.path.basename(output_path))[0]}_mask_{i}_{class_name}_{score:.2f}.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        
        # Save binary mask
        plt.figure(figsize=(8, 8))
        plt.imshow(mask, cmap='gray')
        plt.title(f"{class_name} (Score: {score:.2f})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(mask_path, dpi=150)
        plt.close()
    
    print(f"Saved mask visualization to {output_path}")
    print(f"Saved individual masks to {masks_dir}")


    
def export_binary_masks(detections, image_name, output_dir):
    """
    Export each mask as a binary image.
    
    Parameters:
    -----------
    detections : dict
        Dictionary containing detection results
    image_name : str
        Name of the image (without path and extension)
    output_dir : str
        Directory to save the binary masks
    """
    # Create output directory
    masks_dir = os.path.join(output_dir, f"{image_name}_binary_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # Get masks, class IDs, and scores
    masks = detections['masks']
    class_ids = detections['class_ids']
    scores = detections['scores']
    
    # Process each mask
    for i, (mask, class_id, score) in enumerate(zip(masks.transpose(2, 0, 1), class_ids, scores)):
        # Create filename with detection information
        mask_filename = f"mask_{i}_class_{class_id}_score_{score:.4f}.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        
        # Convert boolean mask to uint8 (0 or 255)
        binary_mask = mask.astype(np.uint8) * 255
        
        # Save as PNG
        cv2.imwrite(mask_path, binary_mask)
    
    # Create a metadata file
    metadata_path = os.path.join(masks_dir, "metadata.json")
    metadata = {
        "image_name": image_name,
        "num_masks": len(class_ids),
        "masks": []
    }
    
    for i, (class_id, score) in enumerate(zip(class_ids, scores)):
        metadata["masks"].append({
            "id": i,
            "class_id": int(class_id),
            "score": float(score),
            "filename": f"mask_{i}_class_{class_id}_score_{score:.4f}.png"
        })
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Exported {len(class_ids)} binary masks to {masks_dir}")
    return masks_dir
    
############################################################
#  Multi-Scale Detection with IoU-based Merging
############################################################

def detect_with_optimized_scales(model, image, config, 
                               scales=[0.8, 1.0, 1.2, 1.5],  # Optimized for musical scores
                               confidence_threshold=0.05,   # Lowered from 0.1 to 0.05
                               iou_threshold=0.4,           # Reduced from 0.5 to 0.4 for more lenient merging
                               verbose=1):
    """
    Perform detection at multiple scales optimized for musical scores and
    merge the results using sophisticated IoU-based filtering with more lenient thresholds.
    """
    import numpy as np
    from skimage import transform
    
    if verbose > 0:
        print("\n===== STARTING MULTI-SCALE DETECTION OPTIMIZED FOR MUSICAL SCORES =====")
        print(f"Image shape: {image.shape}")
        print(f"Using scales: {scales}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"IoU threshold for merging: {iou_threshold}")
        print("===============================================================\n")
    
    # Initialize empty arrays for all results
    combined_rois = []
    combined_masks = []
    combined_class_ids = []
    combined_scores = []
    
    original_shape = image.shape[:2]
    
    # Store per-scale stats for reporting
    per_scale_stats = {}
    
    # Run detection at each scale
    for i, scale in enumerate(scales):
        if verbose > 0:
            print(f"\n----- Processing scale {i+1}/{len(scales)}: {scale} -----")
        
        # Resize image according to scale
        if scale != 1.0:
            # Compute new dimensions
            new_h = int(original_shape[0] * scale)
            new_w = int(original_shape[1] * scale)
            
            if verbose > 0:
                print(f"Resizing image to {new_h}x{new_w} pixels")
            
            # Resize the image
            resized_img = transform.resize(
                image, 
                (new_h, new_w), 
                order=1, 
                preserve_range=True
            ).astype(np.uint8)
        else:
            if verbose > 0:
                print("Using original image size")
            resized_img = image
        
        # Run detection
        if verbose > 0:
            print("Running detection on this scale...")
        start_time = time.time()
        results = model.detect([resized_img], verbose=0)[0]
        elapsed = time.time() - start_time
        
        if verbose > 0:
            print(f"Detection at scale {scale} took {elapsed:.2f} seconds")
            print(f"Found {len(results['rois'])} raw detections at scale {scale}")
        
        # Track class counts at this scale
        class_counts = {}
        for class_id in results['class_ids']:
            if class_id in class_counts:
                class_counts[class_id] += 1
            else:
                class_counts[class_id] = 1
                
        per_scale_stats[scale] = {
            "detections": len(results['rois']),
            "time_seconds": elapsed,
            "class_counts": class_counts
        }
        
        if len(results['rois']) == 0:
            if verbose > 0:
                print("No detections at this scale, continuing to next scale")
            continue
            
        # Adjust ROIs back to original image scale
        if scale != 1.0:
            if verbose > 0:
                print("Adjusting detections back to original scale...")
            # Adjust ROIs (y1, x1, y2, x2)
            adjusted_rois = results['rois'] / scale
            
            # Resize masks back to original dimensions
            if verbose > 0:
                print("Resizing masks to original dimensions...")
            adjusted_masks = np.zeros((original_shape[0], original_shape[1], results['masks'].shape[2]), 
                                     dtype=np.bool)
            
            for i in range(results['masks'].shape[2]):
                mask = results['masks'][:, :, i]
                adjusted_mask = transform.resize(
                    mask.astype(float), 
                    original_shape, 
                    order=0,  # Nearest-neighbor to preserve binary values
                    preserve_range=True,
                    anti_aliasing=False
                ) > 0.5  # Threshold to get binary mask
                adjusted_masks[:, :, i] = adjusted_mask
        else:
            adjusted_rois = results['rois']
            adjusted_masks = results['masks']
        
        # Filter by confidence
        if verbose > 0:
            print(f"Filtering detections by confidence threshold {confidence_threshold}...")
        keep_indices = np.where(results['scores'] >= confidence_threshold)[0]
        
        if verbose > 0:
            print(f"Keeping {len(keep_indices)}/{len(results['scores'])} detections above threshold")
        
        if len(keep_indices) == 0:
            if verbose > 0:
                print("No detections above confidence threshold, continuing to next scale")
            continue
            
        # Add to combined results
        combined_rois.append(adjusted_rois[keep_indices])
        combined_masks.append(adjusted_masks[:, :, keep_indices])
        combined_class_ids.append(results['class_ids'][keep_indices])
        combined_scores.append(results['scores'][keep_indices])
    
    if verbose > 0:
        print("\n----- Combining results from all scales -----")
    
    if len(combined_rois) == 0:
        if verbose > 0:
            print("No detections found at any scale!")
        # No detections at any scale
        return {
            'rois': np.array([]),
            'masks': np.zeros((original_shape[0], original_shape[1], 0), dtype=np.bool),
            'class_ids': np.array([], dtype=np.int32),
            'scores': np.array([]),
            'per_scale_stats': per_scale_stats
        }
    
    # Concatenate results from all scales
    if verbose > 0:
        print("Concatenating results...")
    all_rois = np.vstack(combined_rois)
    all_class_ids = np.concatenate(combined_class_ids)
    all_scores = np.concatenate(combined_scores)
    
    if verbose > 0:
        print(f"Total combined detections: {len(all_scores)}")
    
    # Concatenate masks from all scales
    total_instances = sum(m.shape[2] for m in combined_masks)
    if verbose > 0:
        print(f"Creating combined mask array of shape ({original_shape[0]}, {original_shape[1]}, {total_instances})")
    
    all_masks = np.zeros((original_shape[0], original_shape[1], total_instances), dtype=np.bool)
    
    instance_count = 0
    for masks in combined_masks:
        for i in range(masks.shape[2]):
            all_masks[:, :, instance_count] = masks[:, :, i]
            instance_count += 1
    
    # Apply optimized merging algorithm to remove duplicates
    if verbose > 0:
        print("\n----- Applying Optimized Detection Merging (Lenient Version) -----")
    
    # Group by class first
    class_indices = defaultdict(list)
    for i, class_id in enumerate(all_class_ids):
        class_indices[class_id].append(i)
    
    if verbose > 0:
        print(f"Found {len(class_indices)} unique classes in detections")
    
    # Final results after merging
    final_rois = []
    final_masks = []
    final_class_ids = []
    final_scores = []
    
    # Process each class separately for more control over merging
    for class_id, indices in class_indices.items():
        if verbose > 1:
            print(f"Processing class ID {class_id} with {len(indices)} detections")
        
        # Skip if only one detection for this class
        if len(indices) <= 1:
            for idx in indices:
                final_rois.append(all_rois[idx])
                final_masks.append(all_masks[:, :, idx])
                final_class_ids.append(all_class_ids[idx])
                final_scores.append(all_scores[idx])
            continue
        
        # NEW: Sort indices by score in descending order
        sorted_indices = sorted(indices, key=lambda i: all_scores[i], reverse=True)
        
        # MODIFIED: Cluster detections with adjusted strategy
        # Use a more adaptive approach where we adjust IoU threshold based on object class and size
        class_rois = all_rois[indices]
        class_scores = all_scores[indices]
        
        # For each detection
        keep_indices = []
        mask_indices = set(range(len(indices)))
        
        # MODIFIED MERGING APPROACH:
        # 1. We'll first keep all high confidence detections regardless of overlap
        high_conf_threshold = 0.3  # Keep all detections above this threshold
        for i, idx in enumerate(sorted_indices):
            if all_scores[idx] >= high_conf_threshold:
                keep_indices.append(i)
                if i in mask_indices:
                    mask_indices.remove(i)
        
        # 2. Then process remaining detections with adaptive IoU threshold
        while mask_indices:
            # Get highest scoring detection
            remaining_scores = [class_scores[i] for i in mask_indices]
            highest_idx = list(mask_indices)[np.argmax(remaining_scores)]
            keep_indices.append(highest_idx)
            mask_indices.remove(highest_idx)
            
            # Get IoU with all remaining detections
            base_roi = class_rois[highest_idx].reshape(1, 4)
            if mask_indices:
                remaining_rois = np.array([class_rois[i] for i in mask_indices])
                ious = utils.compute_overlaps(base_roi, remaining_rois)[0]
                
                # MODIFIED: Adjust IoU threshold based on confidence
                # Allow more overlap (lower IoU threshold) for higher confidence detections
                current_confidence = class_scores[highest_idx]
                # Adjust threshold - more lenient for higher confidence
                adjusted_iou_threshold = iou_threshold + 0.1 if current_confidence > 0.7 else iou_threshold
                
                # Remove overlapping detections based on adjusted IoU threshold
                to_remove = []
                for i, idx in enumerate(mask_indices):
                    if ious[i] > adjusted_iou_threshold:
                        to_remove.append(idx)
                
                for idx in to_remove:
                    mask_indices.remove(idx)
        
        # Add kept detections to finals
        for idx in keep_indices:
            real_idx = indices[idx]
            final_rois.append(all_rois[real_idx])
            final_masks.append(all_masks[:, :, real_idx])
            final_class_ids.append(all_class_ids[real_idx])
            final_scores.append(all_scores[real_idx])
    
    # Convert lists to arrays
    if verbose > 0:
        print("\n----- Finalizing results -----")
    
    if len(final_rois) > 0:
        if verbose > 0:
            print(f"Final detection count after merging: {len(final_rois)}")
        final_rois = np.array(final_rois)
        final_class_ids = np.array(final_class_ids)
        final_scores = np.array(final_scores)
        
        # Stack masks
        if verbose > 0:
            print(f"Creating final mask array of shape ({original_shape[0]}, {original_shape[1]}, {len(final_masks)})")
        final_masks_array = np.zeros((original_shape[0], original_shape[1], len(final_masks)), dtype=np.bool)
        for i, mask in enumerate(final_masks):
            final_masks_array[:, :, i] = mask
    else:
        if verbose > 0:
            print("No detections remain after merging!")
        final_rois = np.array([])
        final_masks_array = np.zeros((original_shape[0], original_shape[1], 0), dtype=np.bool)
        final_class_ids = np.array([], dtype=np.int32)
        final_scores = np.array([])
    
    # Return final results with stats
    if verbose > 0:
        print("\n===== MULTI-SCALE DETECTION COMPLETE =====\n")
        print(f"Initial confidence threshold: {confidence_threshold}")
        print(f"IoU threshold for merging: {iou_threshold}")
        print(f"Final detection count: {len(final_scores)}")
        
        # Print confidence score distribution
        if len(final_scores) > 0:
            print("\nConfidence score distribution:")
            bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            hist, _ = np.histogram(final_scores, bins=bins)
            for i in range(len(bins)-1):
                print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]} detections")
    
    return {
        'rois': final_rois,
        'masks': final_masks_array,
        'class_ids': final_class_ids,
        'scores': final_scores,
        'per_scale_stats': per_scale_stats
    }

def save_masks_data(detections, image_name, output_dir, class_names):
    """
    Save detection data including masks to JSON and CSV files for post-processing
    
    Args:
        detections: dict containing detection results from Mask R-CNN with 'rois', 'masks', 'class_ids', 'scores'
        image_name: name of the image (without path and extension)
        output_dir: directory to save the detection data
        class_names: list of class names for mapping class IDs to names
    
    Returns:
        tuple: (json_output_path, csv_output_path, contour_csv_path)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from detections
    boxes = detections['rois']
    masks = detections['masks']
    class_ids = detections['class_ids']
    scores = detections['scores']
    
    # Create a list of detections
    detections_list = []
    mask_data_list = []
    
    for i, (box, mask, class_id, score) in enumerate(zip(boxes, masks.transpose(2, 0, 1), class_ids, scores)):
        # Get box coordinates in [y1, x1, y2, x2] format (Mask R-CNN format)
        y1, x1, y2, x2 = map(float, box)
        
        # Convert to [x1, y1, x2, y2] format (standard format)
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        
        # Get class name from class_id
        class_name = class_names[class_id] if class_id < len(class_names) else f"cls_{class_id}"
        
        # Compute mask statistics
        if mask.shape[0] > 0 and mask.shape[1] > 0:
            mask_area = float(np.sum(mask))
            mask_percentage = float(mask_area / (mask.shape[0] * mask.shape[1]))
            
            # Compute mask contour for serialization
            # We'll save a downsampled version of the contour to keep file sizes reasonable
            mask_binary = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                # Downsample contour - store fewer points
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                # Convert to list for JSON serialization
                contour_list = approx_contour.flatten().tolist()
            else:
                contour_list = []
        else:
            mask_area = 0.0
            mask_percentage = 0.0
            contour_list = []
        
        # Create detection data
        detection = {
            "class_id": int(class_id),
            "class_name": class_name,
            "confidence": float(score),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "center_x": center_x,
                "center_y": center_y
            },
            "mask": {
                "area": mask_area,
                "percentage": mask_percentage,
                "contour_points": len(contour_list) // 2,
                "contour_index": i
            }
        }
        
        detections_list.append(detection)
        
        # Add mask data for CSV
        mask_data = {
            "detection_id": i,
            "class_id": int(class_id),
            "class_name": class_name,
            "confidence": float(score),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "center_x": center_x,
            "center_y": center_y,
            "mask_area": mask_area,
            "mask_percentage": mask_percentage,
            "contour": contour_list
        }
        
        mask_data_list.append(mask_data)
    
    # Save detections to JSON file
    json_output_path = os.path.join(output_dir, f"{image_name}_mask_detections.json")
    with open(json_output_path, 'w') as f:
        json.dump({
            "detections": detections_list,
            "masks": [{
                "detection_id": i,
                "contour": data["contour"]
            } for i, data in enumerate(mask_data_list)]
        }, f, indent=2)
    
    # Also save as CSV for easier data analysis
    csv_output_path = os.path.join(output_dir, f"{image_name}_mask_detections.csv")
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            "detection_id", "class_id", "class_name", "confidence", 
            "x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y",
            "mask_area", "mask_percentage"
        ])
        # Write data (excluding contours which are too complex for CSV)
        for i, data in enumerate(mask_data_list):
            writer.writerow([
                i,
                data["class_id"],
                data["class_name"],
                data["confidence"],
                data["x1"],
                data["y1"],
                data["x2"],
                data["y2"],
                data["width"],
                data["height"],
                data["center_x"],
                data["center_y"],
                data["mask_area"],
                data["mask_percentage"]
            ])
    
    # Save contours to a separate CSV file
    contour_csv_path = os.path.join(output_dir, f"{image_name}_mask_contours.csv")
    with open(contour_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["detection_id", "point_index", "x", "y"])
        # Write contour data
        for i, data in enumerate(mask_data_list):
            contour = data["contour"]
            for j in range(0, len(contour), 2):
                if j + 1 < len(contour):
                    writer.writerow([i, j // 2, contour[j], contour[j + 1]])
    
    print(f"Saved mask detection data to {json_output_path} and {csv_output_path}")
    print(f"Saved mask contour data to {contour_csv_path}")
    
    return json_output_path, csv_output_path, contour_csv_path


############################################################
#  Evaluation Functions
############################################################

def compute_evaluation_metrics(detections, ground_truth, iou_threshold=0.5):
    """
    Compute precision, recall, F1, and IoU metrics comparing detections to ground truth.
    
    Parameters:
    -----------
    detections : dict
        Dictionary containing detection results
    ground_truth : dict
        Dictionary containing ground truth annotations
    iou_threshold : float
        IoU threshold for considering a detection as correct
        
    Returns:
    --------
    dict : Evaluation metrics
    """
    # Extract components
    det_boxes = detections['rois']
    det_class_ids = detections['class_ids']
    det_scores = detections['scores']
    
    gt_boxes = ground_truth['boxes']
    gt_class_ids = ground_truth['class_ids']
    
    # Initialize metrics
    metrics = {
        "overall": {
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "mAP": 0,
            "IoU": 0
        },
        "per_class": {}
    }
    
    # If either ground truth or detections are empty, return early
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        if len(gt_boxes) > 0:
            metrics["overall"]["precision"] = 0
            metrics["overall"]["recall"] = 0
            metrics["overall"]["f1"] = 0
            metrics["overall"]["mAP"] = 0
            metrics["overall"]["IoU"] = 0
            
            # Initialize per-class metrics for all ground truth classes
            for class_id in np.unique(gt_class_ids):
                metrics["per_class"][int(class_id)] = {
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "gt_count": int(np.sum(gt_class_ids == class_id)),
                    "det_count": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": int(np.sum(gt_class_ids == class_id))
                }
        elif len(det_boxes) > 0:
            metrics["overall"]["precision"] = 0
            metrics["overall"]["recall"] = 0
            metrics["overall"]["f1"] = 0
            metrics["overall"]["mAP"] = 0
            metrics["overall"]["IoU"] = 0
            
            # Initialize per-class metrics for all detection classes
            for class_id in np.unique(det_class_ids):
                metrics["per_class"][int(class_id)] = {
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "gt_count": 0,
                    "det_count": int(np.sum(det_class_ids == class_id)),
                    "tp": 0,
                    "fp": int(np.sum(det_class_ids == class_id)),
                    "fn": 0
                }
        return metrics
    
    # Compute IoUs between all detections and ground truths
    # First for all boxes, regardless of class
    ious = utils.compute_overlaps(det_boxes, gt_boxes)
    
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Mark which ground truths have been matched
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    
    # Initialize per-class counters
    class_metrics = {}
    for class_id in np.unique(np.concatenate([gt_class_ids, det_class_ids])):
        class_metrics[class_id] = {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
    
    # For each detection, find the best matching ground truth
    for i in range(len(det_boxes)):
        # Only consider ground truths of the same class
        class_id = det_class_ids[i]
        class_gt_indices = np.where(gt_class_ids == class_id)[0]
        
        # Skip if no ground truths of this class
        if len(class_gt_indices) == 0:
            false_positives += 1
            class_metrics[class_id]["fp"] += 1
            continue
        
        # Get IoUs with ground truths of the same class
        class_ious = ious[i, class_gt_indices]
        
        # Find the best matching ground truth
        best_iou_idx = np.argmax(class_ious)
        best_iou = class_ious[best_iou_idx]
        best_gt_idx = class_gt_indices[best_iou_idx]
        
        # Check if the match is good enough and the ground truth hasn't been matched yet
        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            true_positives += 1
            class_metrics[class_id]["tp"] += 1
            gt_matched[best_gt_idx] = True
        else:
            false_positives += 1
            class_metrics[class_id]["fp"] += 1
    
    # Count unmatched ground truths as false negatives
    for i in range(len(gt_boxes)):
        if not gt_matched[i]:
            false_negatives += 1
            class_id = gt_class_ids[i]
            class_metrics[class_id]["fn"] += 1
    
    # Calculate overall metrics
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
    
    # Calculate average IoU for true positives
    average_iou = 0
    if true_positives > 0:
        # Recalculate IoUs for matched detections and ground truths
        matched_ious = []
        for i in range(len(det_boxes)):
            class_id = det_class_ids[i]
            class_gt_indices = np.where(gt_class_ids == class_id)[0]
            if len(class_gt_indices) == 0:
                continue
                
            class_ious = ious[i, class_gt_indices]
            best_iou_idx = np.argmax(class_ious)
            best_iou = class_ious[best_iou_idx]
            best_gt_idx = class_gt_indices[best_iou_idx]
            
            if best_iou >= iou_threshold and gt_matched[best_gt_idx]:
                matched_ious.append(best_iou)
        
        if matched_ious:
            average_iou = np.mean(matched_ious)
    
    # Add to metrics
    metrics["overall"]["precision"] = float(precision)
    metrics["overall"]["recall"] = float(recall)
    metrics["overall"]["f1"] = float(f1)
    metrics["overall"]["mAP"] = float(precision)  # Simplified mAP
    metrics["overall"]["IoU"] = float(average_iou)
    
    # Calculate per-class metrics
    for class_id, counts in class_metrics.items():
        # Count total ground truths and detections for this class
        gt_count = np.sum(gt_class_ids == class_id)
        det_count = np.sum(det_class_ids == class_id)
        
        # Calculate metrics
        class_precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        class_recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        class_f1 = 2 * (class_precision * class_recall) / max(class_precision + class_recall, 1e-6)
        
        metrics["per_class"][int(class_id)] = {
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "gt_count": int(gt_count),
            "det_count": int(det_count),
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"])
        }
    
    return metrics

def evaluate_images(model, image_dir, annot_dir, class_names, class_name_to_id, 
                 scales=[0.8, 1.0, 1.2, 1.5], confidence_threshold=0.1, 
                 iou_threshold=0.3, output_dir=None, max_images=None, save_masks=True):
    """
    Evaluate model on images and annotations.
    
    Parameters:
    -----------
    model : Mask R-CNN model
        The model to evaluate
    image_dir : str
        Directory containing images
    annot_dir : str
        Directory containing XML annotations
    class_names : list
        List of class names
    class_name_to_id : dict
        Mapping from class names to IDs
    scales : list
        Scales to use for multi-scale detection
    confidence_threshold : float
        Confidence threshold for detections
    iou_threshold : float
        IoU threshold for evaluation
    output_dir : str
        Directory to save visualizations and results
    max_images : int
        Maximum number of images to evaluate (None for all)
    save_masks : bool
        Whether to save mask data to JSON/CSV files
        
    Returns:
    --------
    dict : Evaluation results
    """
    # Get image files
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    # If no PNG files found, try other common image extensions
    if not image_files:
        for ext in [".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
            image_files = sorted(glob.glob(os.path.join(image_dir, f"*{ext}")))
            if image_files:
                print(f"Found {len(image_files)} images with extension {ext}")
                break
    
    if max_images is not None and max_images > 0:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images for evaluation")
    
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Initialize metrics
    all_metrics = []
    overall_metrics = {
        "precision": [],
        "recall": [],
        "f1": [],
        "mAP": [],
        "IoU": []
    }
    per_class_metrics = defaultdict(lambda: {
        "precision": [],
        "recall": [],
        "f1": [],
        "gt_count": 0,
        "det_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0
    })
    
    # Process each image
    for i, image_file in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # Get corresponding XML file
        image_basename = os.path.splitext(os.path.basename(image_file))[0]
        xml_file = os.path.join(annot_dir, f"{image_basename}.xml")
        
        if not os.path.exists(xml_file):
            print(f"WARNING: No annotation file found for {image_basename}")
            continue
        
        # Load image
        try:
            image = skimage.io.imread(image_file)
            
            # Make sure image is RGB
            if len(image.shape) == 2:
                image = skimage.color.gray2rgb(image)
            elif image.shape[2] == 4:
                image = image[:,:,:3]
                
            print(f"Image shape: {image.shape}")
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
            continue
        
        # Load annotations
        ground_truth = load_annotations(xml_file, class_name_to_id)
        if ground_truth is None:
            print(f"Error loading annotations from {xml_file}")
            continue
            
        print(f"Loaded {len(ground_truth['class_ids'])} annotations")
        
        # Run detection
        detections = detect_with_optimized_scales(
            model, image, model.config, 
            scales=scales, 
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            verbose=1
        )
        
        print(f"Found {len(detections['class_ids'])} detections")
        
        # Save mask data to JSON and CSV if requested
        if save_masks and output_dir:
            # Create masks directory
            masks_dir = os.path.join(output_dir, "masks_data")
            os.makedirs(masks_dir, exist_ok=True)
            
            try:
                # Call the save_masks_data function
                json_path, csv_path, contour_path = save_masks_data(
                    detections, 
                    image_basename, 
                    masks_dir,
                    class_names
                )
                print(f"Saved mask data to {masks_dir}")
                
                # Also export binary masks
                binary_masks_dir = export_binary_masks(
                    detections,
                    image_basename,
                    masks_dir
                )
                
                # Create mask visualization
                vis_dir = os.path.join(output_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                mask_vis_path = os.path.join(vis_dir, f"{image_basename}_masks_vis.png")
                visualize_masks(
                    image,
                    detections,
                    class_names,
                    mask_vis_path
                )
            except Exception as e:
                print(f"Error saving mask data: {e}")
                import traceback
                traceback.print_exc()
        
        # Compute metrics
        metrics = compute_evaluation_metrics(
            detections, ground_truth, iou_threshold=iou_threshold
        )
        
        # Add to overall metrics
        overall_metrics["precision"].append(metrics["overall"]["precision"])
        overall_metrics["recall"].append(metrics["overall"]["recall"])
        overall_metrics["f1"].append(metrics["overall"]["f1"])
        overall_metrics["mAP"].append(metrics["overall"]["mAP"])
        overall_metrics["IoU"].append(metrics["overall"]["IoU"])
        
        # Add to per-class metrics
        for class_id, class_metrics in metrics["per_class"].items():
            per_class_metrics[class_id]["precision"].append(class_metrics["precision"])
            per_class_metrics[class_id]["recall"].append(class_metrics["recall"])
            per_class_metrics[class_id]["f1"].append(class_metrics["f1"])
            per_class_metrics[class_id]["gt_count"] += class_metrics["gt_count"]
            per_class_metrics[class_id]["det_count"] += class_metrics["det_count"]
            per_class_metrics[class_id]["tp"] += class_metrics["tp"]
            per_class_metrics[class_id]["fp"] += class_metrics["fp"]
            per_class_metrics[class_id]["fn"] += class_metrics["fn"]
        
        # Save image metrics
        all_metrics.append({
            "image": image_basename,
            "metrics": metrics
        })
        
        # Save visualization if output directory is provided
        if output_dir:
            # Create directories
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Create visualization showing ground truth and detections
            plt.figure(figsize=(16, 8))
            
            # Ground truth subplot
            ax1 = plt.subplot(1, 2, 1)
            ax1.imshow(image)
            ax1.set_title("Ground Truth")
            
            # Draw ground truth boxes
            for i in range(len(ground_truth["boxes"])):
                y1, x1, y2, x2 = ground_truth["boxes"][i]
                class_id = ground_truth["class_ids"][i]
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                
                # Draw box
                color = (0, 1, 0)  # Green for ground truth
                h = y2 - y1
                w = x2 - x1
                patch = plt.Rectangle((x1, y1), w, h, fill=False, color=color, linewidth=2)
                ax1.add_patch(patch)
                
                # Add label
                label = class_name
                ax1.text(x1, y1-5, label, color=color, size=10, backgroundcolor="white")
            
            # Detections subplot
            ax2 = plt.subplot(1, 2, 2)
            visualize.display_instances(
                image,
                detections["rois"],
                detections["masks"],
                detections["class_ids"],
                class_names,
                detections["scores"],
                ax=ax2,
                title="Detections"
            )
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{image_basename}_comparison.png"))
            plt.close()
    
    # Calculate average metrics
    avg_metrics = {
        "overall": {
            "precision": float(np.mean(overall_metrics["precision"])) if overall_metrics["precision"] else 0.0,
            "recall": float(np.mean(overall_metrics["recall"])) if overall_metrics["recall"] else 0.0,
            "f1": float(np.mean(overall_metrics["f1"])) if overall_metrics["f1"] else 0.0,
            "mAP": float(np.mean(overall_metrics["mAP"])) if overall_metrics["mAP"] else 0.0,
            "IoU": float(np.mean(overall_metrics["IoU"])) if overall_metrics["IoU"] else 0.0
        },
        "per_class": {}
    }
    
    # Calculate average per-class metrics
    for class_id, metrics in per_class_metrics.items():
        if len(metrics["precision"]) > 0:
            avg_metrics["per_class"][int(class_id)] = {
                "name": class_names[class_id] if class_id < len(class_names) else f"Class {class_id}",
                "precision": float(np.mean(metrics["precision"])),
                "recall": float(np.mean(metrics["recall"])),
                "f1": float(np.mean(metrics["f1"])),
                "gt_count": int(metrics["gt_count"]),
                "det_count": int(metrics["det_count"]),
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"])
            }
    
    # Save results if output directory is provided
    if output_dir:
        # Save individual image metrics
        with open(os.path.join(output_dir, "image_metrics.json"), "w") as f:
            json.dump(all_metrics, f, indent=4)
        
        # Save average metrics
        with open(os.path.join(output_dir, "average_metrics.json"), "w") as f:
            json.dump(avg_metrics, f, indent=4)
        
        # Create CSV with per-class metrics
        csv_data = []
        for class_id, metrics in avg_metrics["per_class"].items():
            csv_data.append({
                "class_id": class_id,
                "class_name": metrics["name"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "gt_count": metrics["gt_count"],
                "det_count": metrics["det_count"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"]
            })
        
        df = pd.DataFrame(csv_data)
        if not df.empty:
            df = df.sort_values("f1", ascending=False)
            df.to_csv(os.path.join(output_dir, "per_class_metrics.csv"), index=False)
            
            # Generate summary plots
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Only generate plots if we have data
            if len(df) > 0:
                # Plot precision, recall, F1 by class
                plt.figure(figsize=(15, 10))
                top_classes = df.head(min(30, len(df)))
                x = np.arange(len(top_classes))
                width = 0.25
                
                plt.bar(x - width, top_classes["precision"], width, label="Precision")
                plt.bar(x, top_classes["recall"], width, label="Recall")
                plt.bar(x + width, top_classes["f1"], width, label="F1")
                
                plt.xlabel("Class")
                plt.ylabel("Score")
                plt.title("Precision, Recall, and F1 Score by Class")
                plt.xticks(x, top_classes["class_name"], rotation=90)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "class_metrics.png"))
                plt.close()
                
                # Plot ground truth vs detections counts
                plt.figure(figsize=(15, 10))
                plt.bar(x - width/2, top_classes["gt_count"], width, label="Ground Truth")
                plt.bar(x + width/2, top_classes["det_count"], width, label="Detections")
                
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title("Ground Truth vs Detections by Class")
                plt.xticks(x, top_classes["class_name"], rotation=90)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "count_comparison.png"))
                plt.close()
                
                # Plot true positives, false positives, false negatives
                plt.figure(figsize=(15, 10))
                plt.bar(x - width, top_classes["tp"], width, label="True Positives")
                plt.bar(x, top_classes["fp"], width, label="False Positives")
                plt.bar(x + width, top_classes["fn"], width, label="False Negatives")
                
                plt.xlabel("Class")
                plt.ylabel("Count")
                plt.title("Detection Quality by Class")
                plt.xticks(x, top_classes["class_name"], rotation=90)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "detection_quality.png"))
                plt.close()
        else:
            print("Warning: No class metrics data available to generate plots")
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Precision: {avg_metrics['overall']['precision']:.4f}")
    print(f"Recall: {avg_metrics['overall']['recall']:.4f}")
    print(f"F1 Score: {avg_metrics['overall']['f1']:.4f}")
    print(f"mAP: {avg_metrics['overall']['mAP']:.4f}")
    print(f"IoU: {avg_metrics['overall']['IoU']:.4f}")
    
    # Print top classes by F1 score if available
    top_classes = sorted(avg_metrics["per_class"].items(), 
                       key=lambda x: x[1]["f1"], reverse=True)[:5]
    if top_classes:
        print("\nTop 5 classes by F1 score:")
        for class_id, metrics in top_classes:
            print(f"  {metrics['name']}: F1={metrics['f1']:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f})")
    
    return avg_metrics


def run_inference(model, image_path, class_names, scales=[0.8, 1.0, 1.2, 1.5], 
                confidence_threshold=0.1, iou_threshold=0.3, output_dir=None, save_masks=True):
    """
    Run inference on a single image and save results.
    
    Parameters:
    -----------
    model : Mask R-CNN model
        The model to use for inference
    image_path : str
        Path to the image file
    class_names : list
        List of class names
    scales : list
        Scales to use for multi-scale detection
    confidence_threshold : float
        Confidence threshold for detections
    iou_threshold : float
        IoU threshold for NMS
    output_dir : str
        Directory to save results (if None, no output will be saved)
    save_masks : bool
        Whether to save mask data to JSON/CSV files
        
    Returns:
    --------
    dict : Detection results
    """
    # Start timing
    start_time = time.time()
    
    # Load image
    try:
        image = skimage.io.imread(image_path)
        
        # Make sure image is RGB
        if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)
        elif image.shape[2] == 4:
            image = image[:,:,:3]
            
        print(f"Image shape: {image.shape}")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    # Run detection
    detections = detect_with_optimized_scales(
        model, image, model.config, 
        scales=scales, 
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        verbose=1
    )
    
    # Get image basename for saving results
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"Found {len(detections['class_ids'])} detections")
    
    # Summarize detection classes
    class_counts = {}
    for class_id in detections['class_ids']:
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    print("\nDetections by class:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count}")
    
    # Save results if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations directory
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save detection visualization
        plt.figure(figsize=(16, 16))
        visualize.display_instances(
            image,
            detections["rois"],
            detections["masks"],
            detections["class_ids"],
            class_names,
            detections["scores"],
            title=f"Detections for {image_basename}"
        )
        detection_vis_path = os.path.join(vis_dir, f"{image_basename}_detections.png")
        plt.savefig(detection_vis_path)
        plt.close()
        print(f"Saved detection visualization to {detection_vis_path}")
        
        # Save masks if requested
        if save_masks:
            # Create masks directory
            masks_dir = os.path.join(output_dir, "masks_data")
            os.makedirs(masks_dir, exist_ok=True)
            
            try:
                # Save mask data to JSON and CSV
                json_path, csv_path, contour_path = save_masks_data(
                    detections, 
                    image_basename, 
                    masks_dir,
                    class_names
                )
                print(f"Saved mask data to:")
                print(f"  - JSON: {json_path}")
                print(f"  - CSV: {csv_path}")
                print(f"  - Contours: {contour_path}")
                
                # Export binary masks
                binary_masks_dir = export_binary_masks(
                    detections,
                    image_basename,
                    masks_dir
                )
                
                # Create mask visualization
                mask_vis_path = os.path.join(vis_dir, f"{image_basename}_masks_vis.png")
                visualize_masks(
                    image,
                    detections,
                    class_names,
                    mask_vis_path
                )
                print(f"Saved mask visualization to {mask_vis_path}")
                
                # Create color-coded object id visualization
                plt.figure(figsize=(16, 16))
                
                # Show original image
                plt.imshow(image)
                
                # Create an overlay with object IDs
                for i, (roi, class_id, score) in enumerate(zip(detections["rois"], detections["class_ids"], detections["scores"])):
                    y1, x1, y2, x2 = roi
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                    
                    # Draw bounding box
                    color = plt.cm.hsv((class_id * 0.15) % 1.0)[:3]
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color, linewidth=2)
                    plt.gca().add_patch(rect)
                    
                    # Add object ID label
                    label = f"ID: {i} ({class_name[:10]}... {score:.2f})" if len(class_name) > 10 else f"ID: {i} ({class_name} {score:.2f})"
                    plt.text(x1, y1-5, label, color=color, backgroundcolor="white", fontsize=9)
                
                # Save figure
                object_id_vis_path = os.path.join(vis_dir, f"{image_basename}_object_ids.png")
                plt.title("Object IDs")
                plt.tight_layout()
                plt.savefig(object_id_vis_path, dpi=200)
                plt.close()
                print(f"Saved object ID visualization to {object_id_vis_path}")
                
            except Exception as e:
                print(f"Error saving mask data: {e}")
                import traceback
                traceback.print_exc()
        
        # Save detection details as JSON
        detection_data = {
            "image": os.path.basename(image_path),
            "shape": {
                "height": image.shape[0],
                "width": image.shape[1],
                "channels": image.shape[2]
            },
            "num_detections": len(detections["class_ids"]),
            "class_counts": {str(k): v for k, v in class_counts.items()},
            "detection_time_seconds": time.time() - start_time
        }
        
        detection_json_path = os.path.join(output_dir, f"{image_basename}_detection_summary.json")
        with open(detection_json_path, 'w') as f:
            json.dump(detection_data, f, indent=4)
        print(f"Saved detection summary to {detection_json_path}")
    
    # Report processing time
    elapsed = time.time() - start_time
    print(f"\nInference completed in {elapsed:.2f} seconds")
    
    return detections

############################################################
#  Main Function
############################################################

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate or run inference with Mask R-CNN on musical scores.')
    
    # Required arguments
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    
    # Operation mode selection
    parser.add_argument('--mode', required=False, default='evaluate',
                        choices=['evaluate', 'inference'],
                        help="Operation mode: 'evaluate' for evaluation against ground truth or 'inference' for running detection only")
    
    # Arguments required for evaluation mode
    parser.add_argument('--images-dir', required=False,
                        metavar="/path/to/images/",
                        help='Directory containing images to evaluate/process (required for evaluate mode, optional for inference mode)')
    parser.add_argument('--annot-dir', required=False,
                        metavar="/path/to/annotations/",
                        help='Directory containing XML annotations (required for evaluate mode only)')
    
    # Arguments required for inference mode
    parser.add_argument('--image', required=False,
                        metavar="/path/to/image.png",
                        help='Path to a single image for inference (required for inference mode if --images-dir not provided)')
    
    # Optional arguments for both modes
    parser.add_argument('--scales', required=False,
                        default="0.3, 0.8, 1.5, 1.8",
                        help='Comma-separated list of scales to use for detection')
    parser.add_argument('--confidence', required=False,
                        default=0.1, type=float,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', required=False,
                        default=0.25, type=float,
                        help='IoU threshold for evaluation/NMS')
    parser.add_argument('--output-dir', required=False,
                        default="./results",
                        help='Directory to save evaluation results or detections')
    parser.add_argument('--max-images', required=False,
                        default=None, type=int,
                        help='Maximum number of images to evaluate (default: all) - applies to evaluate mode only')
    parser.add_argument('--save-masks', required=False,
                        default=True, type=bool,
                        help='Whether to save mask data to JSON/CSV files')
    
    args = parser.parse_args()
    
    # Parse scales
    scales = [float(s) for s in args.scales.split(',')]
    
    print("\n===== DOREMI MASK R-CNN FOR MUSICAL SCORES =====")
    print(f"Operation mode: {args.mode}")
    print(f"Weights: {os.path.abspath(args.weights)}")
    print(f"Scales: {scales}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"IoU threshold: {args.iou}")
    
    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Save parameters to file
    with open(os.path.join(output_dir, "parameters.json"), "w") as f:
        json.dump({
            "mode": args.mode,
            "weights": args.weights,
            "images_dir": args.images_dir if hasattr(args, 'images_dir') else None,
            "image": args.image if hasattr(args, 'image') else None,
            "annot_dir": args.annot_dir if hasattr(args, 'annot_dir') else None,
            "scales": scales,
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou,
            "max_images": args.max_images if hasattr(args, 'max_images') else None,
            "save_masks": args.save_masks
        }, f, indent=4)
    
    # Create inference configuration
    config = DoremiConfig()
    config.DETECTION_MIN_CONFIDENCE = args.confidence
    print("Model configuration:")
    config.display()
    
    # Create model
    print("\nCreating model...")
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="./")
    
    # Load weights
    print(f"Loading weights from: {os.path.abspath(args.weights)}")
    try:
        model.load_weights(args.weights, by_name=True)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load weights: {e}")
        sys.exit(1)
    
    # Load class names and create mapping
    print("\nLoading class names...")
    class_names = ["BG"]  # Background first
    class_name_to_id = {}
    try:
        with open(CLASSNAMES_PATH) as json_file:
            data = json.load(json_file)
            # Sort by ID to ensure correct ordering
            data_sorted = sorted(data, key=lambda x: x["id"])
            for id_class in data_sorted:
                while len(class_names) <= id_class["id"]:
                    class_names.append(f"Class {len(class_names)}")
                class_names[id_class["id"]] = id_class["name"]
                class_name_to_id[id_class["name"]] = id_class["id"]
        print(f"Loaded {len(class_names)-1} class names")
    except Exception as e:
        print(f"Warning: Could not load class names: {e}")
        print("Creating default class names...")
        # Create default class names
        for i in range(1, 72):
            class_names.append(f"Class {i}")
            class_name_to_id[f"Class {i}"] = i
    
    # Run based on selected mode
    if args.mode == 'evaluate':
        # Validate required arguments for evaluation mode
        if not args.images_dir:
            print("ERROR: --images-dir is required for evaluation mode")
            sys.exit(1)
        if not args.annot_dir:
            print("ERROR: --annot-dir is required for evaluation mode")
            sys.exit(1)
            
        # Run evaluation
        start_time = time.time()
        try:
            metrics = evaluate_images(
                model=model,
                image_dir=args.images_dir,
                annot_dir=args.annot_dir,
                class_names=class_names,
                class_name_to_id=class_name_to_id,
                scales=scales,
                confidence_threshold=args.confidence,
                iou_threshold=args.iou,
                output_dir=output_dir,
                max_images=args.max_images,
                save_masks=args.save_masks
            )
            
            elapsed = time.time() - start_time
            print(f"\nEvaluation completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"ERROR during evaluation: {e}")
            import traceback
            traceback.print_exc()
            
    elif args.mode == 'inference':
        # Get the image path(s) for inference
        image_paths = []
        
        if args.image:
            # Single image inference
            if os.path.exists(args.image):
                image_paths = [args.image]
            else:
                print(f"ERROR: Image file not found at {args.image}")
                sys.exit(1)
        elif args.images_dir:
            # Directory inference
            if os.path.exists(args.images_dir):
                image_paths = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
                if not image_paths:
                    # Try other common image extensions
                    for ext in [".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
                        image_paths = sorted(glob.glob(os.path.join(args.images_dir, f"*{ext}")))
                        if image_paths:
                            break
                            
                if args.max_images and len(image_paths) > args.max_images:
                    image_paths = image_paths[:args.max_images]
                    
                if not image_paths:
                    print(f"ERROR: No image files found in {args.images_dir}")
                    sys.exit(1)
            else:
                print(f"ERROR: Images directory not found at {args.images_dir}")
                sys.exit(1)
        else:
            print("ERROR: Either --image or --images-dir must be provided for inference mode")
            sys.exit(1)
            
        print(f"Running inference on {len(image_paths)} image(s)")
        
        # Create detections directory
        detections_dir = os.path.join(output_dir, "detections")
        os.makedirs(detections_dir, exist_ok=True)
        
        # Process each image
        start_time = time.time()
        for i, image_path in enumerate(image_paths):
            try:
                print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
                
                # Get image basename
                image_basename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Load image
                try:
                    image = skimage.io.imread(image_path)
                    
                    # Make sure image is RGB
                    if len(image.shape) == 2:
                        image = skimage.color.gray2rgb(image)
                    elif image.shape[2] == 4:
                        image = image[:,:,:3]
                        
                    print(f"Image shape: {image.shape}")
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
                
                # Run detection
                detections = detect_with_optimized_scales(
                    model, image, model.config, 
                    scales=scales, 
                    confidence_threshold=args.confidence,
                    iou_threshold=args.iou,
                    verbose=1
                )
                
                print(f"Found {len(detections['class_ids'])} detections")
                
                # Create visualization directory
                vis_dir = os.path.join(detections_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                
                # Create visualization showing detections
                plt.figure(figsize=(16, 16))
                
                # Display detections
                visualize.display_instances(
                    image,
                    detections["rois"],
                    detections["masks"],
                    detections["class_ids"],
                    class_names,
                    detections["scores"],
                    title=f"Detections for {image_basename}"
                )
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"{image_basename}_detections.png"))
                plt.close()
                
                # Save mask data if requested
                if args.save_masks:
                    # Create masks directory
                    masks_dir = os.path.join(detections_dir, "masks_data")
                    os.makedirs(masks_dir, exist_ok=True)
                    
                    try:
                        # Call the save_masks_data function
                        json_path, csv_path, contour_path = save_masks_data(
                            detections, 
                            image_basename, 
                            masks_dir,
                            class_names
                        )
                        print(f"Saved mask data to {masks_dir}")
                        
                        # Also export binary masks
                        binary_masks_dir = export_binary_masks(
                            detections,
                            image_basename,
                            masks_dir
                        )
                        
                        # Create mask visualization
                        mask_vis_path = os.path.join(vis_dir, f"{image_basename}_masks_vis.png")
                        visualize_masks(
                            image,
                            detections,
                            class_names,
                            mask_vis_path
                        )
                    except Exception as e:
                        print(f"Error saving mask data: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                import traceback
                traceback.print_exc()
                
        elapsed = time.time() - start_time
        print(f"\nInference completed in {elapsed:.2f} seconds")
    
    # Clean up
    K.clear_session()

############################################################
#  Main
############################################################

if __name__ == "__main__":
    main()
    
    
    # python /homes/es314/DOREMI_version_2/MRCNN_DOREMI_20210503/multi_scaling_testing_mrcnn.py --weights=/import/c4dm-05/elona/maskrcnn-logs/1685_dataset_resnet101_20210901/doremi20210504T0325/mask_rcnn_doremi_0079.h5 --images-dir=/homes/es314/music_detection_results/examples/images --annot-dir=/homes/es314/music_detection_results/examples/annotations
    
    
    # Enhanced Mask R-CNN Script for Musical Score Detection Examples

# # 1. Run inference on a single image
# python inference_maskrcnn.py \
#   --mode inference \
#   --weights /import/c4dm-05/elona/maskrcnn-logs/1685_dataset_resnet101_20210901/doremi20210504T0325/mask_rcnn_doremi_0079.h5 \
#   --image /homes/es314/omr-objdet-benchmark/data/images/1-2-Kirschner_-_Chissà_che_cosa_pensa-001.png \
#   --confidence 0.1 \
#   --iou 0.25 \
#   --scales "0.6, 1.0, 2.5" \
#   --output-dir /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/results/mask_rccn_inference \
#   --save-masks True

# # 2. Run inference on all images in a directory
# python inference_maskrcnn.py \
#   --mode inference \
#   --weights /path/to/weights.h5 \
#   --images-dir /path/to/image/directory \
#   --confidence 0.1 \
#   --max-images 10 \
#   --output-dir ./batch_inference_results

# # 3. Run evaluation with ground truth annotations
# python inference_maskrcnn.py \
#   --mode evaluate \
#   --weights /path/to/weights.h5 \
#   --images-dir /path/to/images \
#   --annot-dir /path/to/annotations \
#   --confidence 0.1 \
#   --output-dir ./evaluation_results

# # 4. Run quick inference without saving mask data (faster)
# python inference_maskrcnn.py \
#   --mode inference \
#   --weights /path/to/weights.h5 \
#   --image /path/to/single/image.png \
#   --output-dir ./quick_results \
#   --save-masks False

# # 5. Example with specific parameters for musical notation
# python inference_maskrcnn.py \
#   --mode inference \
#   --weights /import/c4dm-05/elona/maskrcnn-logs/1685_dataset_resnet101_20210901/doremi20210504T0325/mask_rcnn_doremi_0079.h5 \
#   --images-dir /homes/es314/music_detection_results/examples/images \
#   --scales "0.3, 0.8, 1.5, 1.8" \
#   --confidence 0.05 \
#   --iou 0.35 \
#   --output-dir ./music_notation_results