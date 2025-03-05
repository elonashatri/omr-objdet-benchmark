import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
import matplotlib.pyplot as plt
import random

def load_class_names(class_mapping_file):
    """Load class names from class mapping file"""
    with open(class_mapping_file, 'r') as f:
        class_mapping = json.load(f)
    
    # Convert from 1-indexed to 0-indexed
    return {class_id - 1: class_name for class_name, class_id in class_mapping.items()}

def plot_detections(image, results, class_names, conf_threshold=0.25, output_path=None, 
                    max_detections=100, plot_labels=True, random_colors=False):
    """Plot detections on image"""
    # If image is a file path, load it
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a copy of the image
    img_display = image.copy()
    h, w = img_display.shape[:2]
    
    # Get detection results
    if results.boxes.xyxy.shape[0] > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Filter by confidence
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # Limit number of detections to display
        if len(boxes) > max_detections:
            # Sort by confidence and take top max_detections
            indices = np.argsort(scores)[::-1][:max_detections]
            boxes = boxes[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]
        
        # Create random colors for classes if requested
        if random_colors:
            np.random.seed(42)  # For reproducibility
            colors = {cls_id: tuple(np.random.randint(0, 255, 3).tolist()) for cls_id in np.unique(class_ids)}
        else:
            # Use fixed color scheme based on class ID
            colors = {}
            for cls_id in np.unique(class_ids):
                hue = (cls_id * 0.15) % 1.0  # Cycle through hues
                rgb = plt.cm.hsv(hue)[:3]  # Convert to RGB
                colors[cls_id] = tuple((np.array(rgb) * 255).astype(int).tolist())
        
        # Draw bounding boxes
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Get class name
            if cls_id in class_names:
                class_name = class_names[cls_id]
            else:
                class_name = f"Class {cls_id}"
            
            # Get color for this class
            color = colors.get(cls_id, (0, 255, 0))
            
            # Draw rectangle with increased thickness
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)  # Increased thickness from 1 to 2
            
            # Draw label if requested
            if plot_labels:
                label = f"{class_name}: {score:.2f}"
                # Calculate text size with increased font scale
                font_scale = 0.6  # Increased from 0.3
                thickness = 2    # Increased from 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Ensure label is inside the image
                y_text = max(y1 - 2, text_height + 2)
                
                # Draw label background
                cv2.rectangle(img_display, 
                              (x1, y_text - text_height - baseline - 2), 
                              (x1 + text_width, y_text + baseline), 
                              color, -1)
                
                # Draw label text
                cv2.putText(img_display, label, (x1, y_text), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    # Add detection stats with increased font size
    if hasattr(results.boxes, 'xyxy') and len(results.boxes.xyxy) > 0:
        num_detections = len(results.boxes.xyxy)
        stats_text = f"Total detections: {num_detections}"
        if len(boxes) < num_detections:
            stats_text += f" (showing top {len(boxes)} with conf>{conf_threshold:.2f})"
        
        cv2.putText(img_display, stats_text, (10, 30),  # Y position increased from 20 to 30
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Font scale increased from 0.5 to 0.8, thickness from 1 to 2
    else:
        cv2.putText(img_display, "No detections", (10, 30),  # Y position increased from 20 to 30
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  # Font scale increased from 0.5 to 0.8, thickness from 1 to 2
    
    # Save or display result with higher DPI
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Increase figure size and DPI for higher resolution
        plt.figure(figsize=(20, 20 * h / w))  # Increased from 15 to 20
        plt.imshow(img_display)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)  # Added dpi=300 for higher resolution
        plt.close()
        print(f"Saved high-resolution detection result to {output_path}")
    
    return img_display

def detect_music_notation(model_path, image_path, class_mapping_file, 
                         output_dir="results", conf_threshold=0.25, max_detections=100):
    """Detect music notation in image"""
    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load class names
    try:
        class_names = load_class_names(class_mapping_file)
    except Exception as e:
        print(f"Error loading class names: {e}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if image_path is a directory or a file
    if os.path.isdir(image_path):
        # Process all images in directory
        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(list(Path(image_path).glob(f"*{ext}")))
        
        print(f"Found {len(image_files)} images to process")
        
        for img_file in image_files:
            # Generate output path
            output_path = os.path.join(output_dir, f"{img_file.stem}_detection.jpg")
            
            # Run detection
            try:
                results = model(str(img_file), conf=conf_threshold)[0]
                
                # Plot and save results
                plot_detections(
                    str(img_file), 
                    results, 
                    class_names, 
                    conf_threshold, 
                    output_path,
                    max_detections
                )
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    else:
        # Process single image
        try:
            # Generate output path
            img_name = os.path.basename(image_path)
            img_stem = os.path.splitext(img_name)[0]
            output_path = os.path.join(output_dir, f"{img_stem}_detection.jpg")
            
            # Run detection
            results = model(image_path, conf=conf_threshold)[0]
            
            # Plot and save results
            plot_detections(
                image_path, 
                results, 
                class_names, 
                conf_threshold, 
                output_path,
                max_detections
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Detect music notation in images")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 model")
    parser.add_argument("--image", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--output_dir", type=str, default="results/detections", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max_detections", type=int, default=100, help="Maximum number of detections to display")
    
    args = parser.parse_args()
    
    detect_music_notation(
        args.model, 
        args.image, 
        args.class_mapping, 
        args.output_dir,
        args.conf,
        args.max_detections
    )

if __name__ == "__main__":
    main()