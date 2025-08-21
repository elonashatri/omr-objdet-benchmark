#!/usr/bin/env python3
"""
Multi-Scale Inference Script for Optical Music Recognition

This script performs inference using a trained Faster R-CNN model
with advanced multi-scale techniques:
1. Image pyramid inference across multiple scales
2. Test-time augmentation with flips and rotations
3. Fine-tuned NMS for staff line and music symbol detection

Usage:
    python multi_scale_inference.py --model_path=/path/to/model.pt --input_dir=/path/to/images
"""

import os
import sys
import glob
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
import cv2
from tqdm import tqdm

# Import multi-scale enhancements
from multi_scale_enhancements import (
    multi_scale_inference,
    test_time_augmentation,
    get_enhanced_model_instance_segmentation
)

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Scale Inference for OMR Detection')
    
    # Required parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save detection results')
    
    # Optional parameters
    parser.add_argument('--mapping_file', type=str, default='',
                        help='Path to class mapping file')
    parser.add_argument('--scales', type=str, default='0.5,0.75,1.0,1.25,1.5',
                        help='Comma-separated list of scales for multi-scale inference')
    parser.add_argument('--test_time_augmentation', action='store_true',
                        help='Enable test-time augmentation')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                        help='NMS threshold for detections')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--image_size', type=str, default='600,1200',
                        help='Image size (min_dimension,max_dimension) for resizing input images')
    parser.add_argument('--draw_scores', action='store_true', default=True,
                        help='Draw confidence scores on output images')
    parser.add_argument('--save_json', action='store_true', default=True,
                        help='Save detections as JSON files')
    
    return parser.parse_args()

def load_model(model_path, device):
    """
    Load a trained Faster R-CNN model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model onto
        
    Returns:
        model: Loaded model
        args: Original training arguments
    """
    print(f"Loading model from {model_path}")
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get original args
        if 'args' in checkpoint:
            args = checkpoint['args']
            print(f"Loaded model was trained for {args.num_classes} classes")
        else:
            # Create default args
            args = argparse.Namespace()
            args.num_classes = 219  # Default (218 + background)
            args.backbone = 'resnet101-custom'
            args.pretrained = False
            args.min_size = 600
            args.max_size = 1200
            args.anchor_sizes = '4,8,16,32,64,128'
            args.aspect_ratios = '0.05,0.1,0.25,1.0,2.0,4.0,10.0,20.0'
            args.height_stride = 4
            args.width_stride = 4
            args.features_stride = 4
            args.initial_crop_size = 17
            args.maxpool_kernel_size = 1
            args.maxpool_stride = 1
            args.atrous_rate = 2
            args.first_stage_nms_score_threshold = 0.0
            args.first_stage_nms_iou_threshold = 0.5
            args.first_stage_max_proposals = 2000
            args.second_stage_nms_score_threshold = 0.05
            args.second_stage_nms_iou_threshold = 0.4
            args.second_stage_max_detections_per_class = 2000
            args.second_stage_max_total_detections = 2500
            args.frozen_layers = 0
            print("Warning: No args found in checkpoint, using defaults")
        
        # Create model with the same architecture
        model = get_enhanced_model_instance_segmentation(args.num_classes, args)
        
        # Load weights
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully")
        return model, args
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model checkpoint contains both weights and training args")
        sys.exit(1)

def load_class_mapping(mapping_file):
    """
    Load class mapping from file.
    
    Args:
        mapping_file: Path to the mapping file
        
    Returns:
        Dictionary mapping class IDs to class names
    """
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
        print(f"Loaded {len(class_names)} class mappings")
    except FileNotFoundError:
        print(f"Warning: Mapping file {mapping_file} not found. Using generic class names.")
    
    return class_names

def get_color_for_class(class_id, class_names=None):
    """
    Generate a consistent color for each class, optimized for music notation visualization.
    
    Args:
        class_id: Class ID
        class_names: Dictionary mapping class IDs to names
        
    Returns:
        RGB color tuple
    """
    # Define colors for specific musical symbol categories
    staff_colors = (80, 80, 80)     # Darker gray for staff lines
    note_colors = (0, 0, 220)       # Blue for notes
    rest_colors = (220, 0, 0)       # Red for rests
    clef_colors = (0, 180, 0)       # Green for clefs
    accidental_colors = (200, 100, 0)  # Orange for accidentals
    time_sig_colors = (128, 0, 128)    # Purple for time signatures
    barline_colors = (50, 50, 50)      # Dark gray for barlines
    beam_colors = (180, 100, 0)        # Brown for beams
    slur_colors = (0, 150, 150)        # Teal for slurs/ties
    dynamic_colors = (150, 0, 150)     # Purple for dynamics
    stem_colors = (100, 100, 100)      # Gray for stems
    flag_colors = (200, 50, 50)        # Reddish for flags
    
    # Specific ID mapping from the dataset statistics
    # Using the most common class IDs from the provided dataset stats
    id_to_category = {
        # These are example mappings based on your dataset stats
        # You should adjust these to match your actual class IDs
        'noteheadBlack': note_colors,
        'stem': stem_colors,
        'kStaffLine': staff_colors,
        'barline': barline_colors,
        'beam': beam_colors,
        'accidentalFlat': accidental_colors,
        'accidentalSharp': accidental_colors,
        'restWhole': rest_colors,
        'rest8th': rest_colors,
        'systemicBarline': barline_colors,
        'slur': slur_colors,
        'noteheadHalf': note_colors,
        'accidentalNatural': accidental_colors,
        'tie': slur_colors,
        'gClef': clef_colors,
        'restQuarter': rest_colors,
        'rest16th': rest_colors,
        'flag8thUp': flag_colors,
        'flag8thDown': flag_colors,
        'articStaccatoAbove': (150, 100, 50),
        'flag16thUp': flag_colors,
        'fClef': clef_colors
    }
    
    # Check class name if available
    class_name = None
    if class_names and class_id in class_names:
        class_name = class_names[class_id].lower()
        
        # First check if the exact class name is in our mapping
        if class_name in id_to_category:
            return id_to_category[class_name]
            
        # Then do substring matching
        if 'staff' in class_name or 'stave' in class_name:
            return staff_colors
        elif 'note' in class_name or 'head' in class_name:
            return note_colors
        elif 'rest' in class_name:
            return rest_colors
        elif 'clef' in class_name:
            return clef_colors
        elif 'sharp' in class_name or 'flat' in class_name or 'natural' in class_name:
            return accidental_colors
        elif 'time' in class_name:
            return time_sig_colors
        elif 'bar' in class_name:
            return barline_colors
        elif 'beam' in class_name:
            return beam_colors
        elif 'slur' in class_name or 'tie' in class_name:
            return slur_colors
        elif 'dynamic' in class_name:
            return dynamic_colors
        elif 'stem' in class_name:
            return stem_colors
        elif 'flag' in class_name:
            return flag_colors
    
    # Check for specific IDs based on your dataset statistics
    known_ids = {
        865918: note_colors,      # noteheadBlack
        775651: stem_colors,      # stem
        322401: staff_colors,     # kStaffLine
        226065: barline_colors,   # barline
        159264: beam_colors,      # beam
        95119: accidental_colors, # accidentalFlat
        91672: accidental_colors, # accidentalSharp
        66567: rest_colors,       # restWhole
        61476: rest_colors,       # rest8th
        58540: barline_colors,    # systemicBarline
        49642: slur_colors,       # slur
        46624: note_colors,       # noteheadHalf
        41753: accidental_colors, # accidentalNatural
        38811: slur_colors,       # tie
        37586: clef_colors,       # gClef
        34691: rest_colors,       # restQuarter
        34591: rest_colors,       # rest16th
        32384: flag_colors,       # flag8thUp
        30668: flag_colors,       # flag8thDown
        24192: (150, 100, 50),    # articStaccatoAbove
        20855: flag_colors,       # flag16thUp
        18831: clef_colors,       # fClef
    }
    
    if class_id in known_ids:
        return known_ids[class_id]
    
    # If not a special category or no class names available, generate color from ID
    color_map = plt.cm.get_cmap('hsv', 218)  # Using the number of unique classes from your dataset
    color = color_map(class_id % 218)[:3]  # Take RGB from RGBA
    return tuple(int(c * 255) for c in color)

def draw_detections(image, boxes, labels, scores, class_names=None, threshold=0.5, draw_scores=True, line_width_factor=1.0):
    """
    Draw detection boxes on an image.
    
    Args:
        image: PIL Image
        boxes: Tensor of boxes [N, 4]
        labels: Tensor of labels [N]
        scores: Tensor of scores [N]
        class_names: Dictionary mapping class IDs to class names
        threshold: Confidence threshold
        draw_scores: Whether to draw confidence scores
        
    Returns:
        PIL Image with detections drawn
    """
    draw = ImageDraw.Draw(image)
    
    # Try to find a suitable font
    try:
        # Try common system fonts
        system_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/usr/share/fonts/TTF/DejaVuSans.ttf',
            '/Library/Fonts/Arial.ttf',
            'C:\\Windows\\Fonts\\Arial.ttf'
        ]
        
        font = None
        for font_path in system_fonts:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 14)
                break
                
        if font is None:
            # Fall back to default
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Convert tensors to numpy if needed
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Draw each detection
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
            
        # Get color for this class
        color = get_color_for_class(label, class_names)
        
        # Adjust line width based on class - thinner for staff lines, thicker for noteheads
        if class_names and label in class_names:
            class_name = class_names[label].lower()
            
            # Base line width on object type and size
            if 'staff' in class_name or 'stem' in class_name or 'barline' in class_name:
                # Thinner lines for staff objects
                box_width = max(1, min(3, int((box[2] - box[0]) / 200)))
            elif 'note' in class_name or 'head' in class_name or 'rest' in class_name:
                # Thicker lines for noteheads and rests
                box_width = max(2, min(5, int((box[2] - box[0]) / 80)))
            else:
                # Default adaptive line width
                box_width = max(1, min(4, int((box[2] - box[0]) / 100)))
        else:
            # Default adaptive line width
            box_width = max(1, min(4, int((box[2] - box[0]) / 100)))
        
        # Apply line width factor and ensure minimum width of 1
        box_width = max(1, int(box_width * line_width_factor))
        
        # Draw rectangle with appropriate width
        draw.rectangle(box.tolist(), outline=color, width=box_width)
        
        # Prepare label text
        if class_names and label in class_names:
            label_text = class_names[label]
        else:
            label_text = f"Class {label}"
            
        if draw_scores:
            label_text = f"{label_text} ({score:.2f})"
        
        # Draw label with background
        text_width, text_height = draw.textsize(label_text, font=font) if font else (len(label_text) * 7, 14)
        
        # Position label above box if there's room, otherwise inside box
        if box[1] > text_height + 5:
            # Above box
            text_box = [box[0], box[1] - text_height - 5, box[0] + text_width, box[1] - 1]
        else:
            # Inside box at top
            text_box = [box[0], box[1], box[0] + text_width, box[1] + text_height]
        
        # Draw text background (slightly transparent)
        draw.rectangle(text_box, fill=color + (200,))  # Add alpha channel for transparency
        
        # Draw text
        draw.text((text_box[0], text_box[1]), label_text, fill=(0, 0, 0), font=font)
    
    return image

def process_image(model, image_path, output_path, class_names, args):
    """
    Process a single image with multi-scale inference.
    
    Args:
        model: Trained Faster R-CNN model
        image_path: Path to input image
        output_path: Path to save output image
        class_names: Dictionary mapping class IDs to class names
        args: Command-line arguments
    
    Returns:
        Dictionary with detection results
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    
    # Get device
    device = next(model.parameters()).device
    
    # Prepare class-specific parameters
    class_specific_params = {}
    
    # Define special detection parameters for staff lines, barlines, etc.
    staff_line_ids = []
    barline_ids = []
    stem_ids = []
    
    # Map class names to IDs for specialized handling
    for class_id, name in class_names.items():
        lower_name = name.lower()
        if 'staff' in lower_name or 'stave' in lower_name:
            staff_line_ids.append(class_id)
            class_specific_params[class_id] = {
                'score_threshold': 0.03,  # Lower threshold for staff lines
                'nms_threshold': 0.3      # More aggressive NMS
            }
        elif 'barline' in lower_name:
            barline_ids.append(class_id)
            class_specific_params[class_id] = {
                'score_threshold': 0.04,  # Lower threshold for barlines
                'nms_threshold': 0.4      # More conservative NMS
            }
        elif 'stem' in lower_name:
            stem_ids.append(class_id)
            class_specific_params[class_id] = {
                'score_threshold': 0.04,  # Lower threshold for stems
                'nms_threshold': 0.35     # More aggressive NMS
            }
    
    # Use known class IDs from dataset stats if no matches found
    if not staff_line_ids:
        staff_line_ids = [322401]  # kStaffLine from dataset stats
        class_specific_params[322401] = {
            'score_threshold': 0.03,
            'nms_threshold': 0.3
        }
    if not barline_ids:
        barline_ids = [226065, 58540]  # barline, systemicBarline from dataset stats
        for id in barline_ids:
            class_specific_params[id] = {
                'score_threshold': 0.04,
                'nms_threshold': 0.4
            }
    if not stem_ids:
        stem_ids = [775651]  # stem from dataset stats
        class_specific_params[775651] = {
            'score_threshold': 0.04,
            'nms_threshold': 0.35
        }
    
    # For high-resolution music scores, use wider range of scales
    scales = [float(s) for s in args.scales.split(',')]
    if not scales:
        scales = [0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75]
    
    # Image analysis - check if it looks like a music score
    is_music_score = True
    avg_color = np.mean(np.array(image))
    if avg_color < 100:  # Very dark image unlikely to be a score
        is_music_score = False
    
    if args.test_time_augmentation:
        # Define custom augmentations optimized for music scores
        if is_music_score:
            augmentations = [
                {'flip': False, 'rotate': 0, 'brightness': 1.0},    # Original
                {'flip': False, 'rotate': 0, 'brightness': 0.9},    # Darker
                {'flip': False, 'rotate': 0, 'brightness': 1.1},    # Brighter
                {'flip': False, 'rotate': 0, 'contrast': 1.1},      # More contrast for staff lines
                {'flip': True, 'rotate': 0, 'brightness': 1.0},     # Horizontal flip
            ]
        else:
            # Default augmentations for non-music images
            augmentations = None
            
        # Use test-time augmentation with class-specific parameters
        predictions = test_time_augmentation(
            model, 
            image_tensor.to(device), 
            augmentations=augmentations,
            score_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            class_specific_params=class_specific_params
        )
    else:
        # Use image pyramid inference with optimized scales for music scores
        predictions = multi_scale_inference(
            model, 
            image_tensor.to(device), 
            scales=scales,
            score_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold
        )
    
    # Draw detections on image with special handling for music notation
    # Use larger line width factor for high-res music scores
    output_image = draw_detections(
        image.copy(), 
        predictions['boxes'], 
        predictions['labels'], 
        predictions['scores'], 
        class_names,
        threshold=args.confidence_threshold,
        draw_scores=args.draw_scores,
        line_width_factor=1.5  # Increased for better visibility in high-res scores
    )
    
    # Save output image
    output_image.save(output_path)
    
    # Create detection results dictionary
    detection_results = {
        'file_name': os.path.basename(image_path),
        'width': image.width,
        'height': image.height,
        'detections': []
    }
    
    # Convert tensors to lists for JSON serialization
    boxes = predictions['boxes'].cpu().numpy().tolist()
    scores = predictions['scores'].cpu().numpy().tolist()
    labels = predictions['labels'].cpu().numpy().tolist()
    
    # Add each detection to results
    for box, score, label in zip(boxes, scores, labels):
        if score >= args.confidence_threshold:
            class_name = class_names.get(label, f"Class {label}")
            detection = {
                'bbox': box,  # [x1, y1, x2, y2] format
                'score': score,
                'category_id': label,
                'category_name': class_name
            }
            detection_results['detections'].append(detection)
    
    return detection_results

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Optimize parameters for high-res music scores (2475×3504)
    if not args.scales:
        args.scales = "0.25,0.4,0.6,0.8,1.0,1.25,1.5,1.75"
    
    # Adjust confidence threshold for staff lines
    if args.confidence_threshold > 0.3:
        print("Note: Using lower confidence threshold (0.04) for better staff line detection")
        args.confidence_threshold = 0.04
        
    # Use more specialized NMS for overlapping music notation
    if args.nms_threshold > 0.4:
        print("Note: Using adjusted NMS threshold (0.45) for better overlapping symbol detection")
        args.nms_threshold = 0.45
    
    # Set device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, model_args = load_model(args.model_path, device)
    
    # Load class mapping
    class_names = load_class_mapping(args.mapping_file) if args.mapping_file else {}
    
    # Count classes by category for reporting
    class_categories = {
        'Staff Lines': 0,
        'Notes': 0,
        'Rests': 0,
        'Clefs': 0,
        'Accidentals': 0,
        'Time Signatures': 0,
        'Other': 0
    }
    
    for class_id, name in class_names.items():
        lower_name = name.lower()
        if 'staff' in lower_name or 'stave' in lower_name:
            class_categories['Staff Lines'] += 1
        elif 'note' in lower_name or 'head' in lower_name:
            class_categories['Notes'] += 1
        elif 'rest' in lower_name:
            class_categories['Rests'] += 1
        elif 'clef' in lower_name:
            class_categories['Clefs'] += 1
        elif 'flat' in lower_name or 'sharp' in lower_name or 'natural' in lower_name:
            class_categories['Accidentals'] += 1
        elif 'time' in lower_name or 'sig' in lower_name:
            class_categories['Time Signatures'] += 1
        else:
            class_categories['Other'] += 1
    
    print(f"Model loaded with {len(class_names)} musical symbol classes:")
    for category, count in class_categories.items():
        if count > 0:
            print(f"  - {category}: {count} classes")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images to process")
    
    # Create progress bar
    pbar = tqdm(image_files)
    
    # Process each image
    all_results = {}
    for image_path in pbar:
        try:
            # Set output path
            filename = os.path.basename(image_path)
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output_dir, f"{base_name}_detections{ext}")
            
            # Set status in progress bar
            pbar.set_description(f"Processing {filename}")
            
            # Process image
            result = process_image(model, image_path, output_path, class_names, args)
            
            # Save detection results as JSON if requested
            if args.save_json:
                json_path = os.path.join(args.output_dir, f"{base_name}_detections.json")
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2)
            
            # Collect results
            all_results[filename] = result
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Save combined results
    if args.save_json:
        combined_path = os.path.join(args.output_dir, "all_detections.json")
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    print(f"Finished processing {len(image_files)} images")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()