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
import csv

def load_class_names(class_mapping_file):
    """Load class names from class mapping file"""
    with open(class_mapping_file, 'r') as f:
        class_mapping = json.load(f)
    
    # Convert from 1-indexed to 0-indexed
    return {class_id - 1: class_name for class_name, class_id in class_mapping.items()}

def save_detection_data(boxes, scores, class_ids, class_names, image_name, output_dir):
    """
    Save detection data to a JSON file for post-processing
    
    Args:
        boxes: numpy array of bounding boxes in format [x1, y1, x2, y2]
        scores: numpy array of confidence scores
        class_ids: numpy array of class IDs
        class_names: dictionary mapping class IDs to class names
        image_name: name of the image (without path and extension)
        output_dir: directory to save the detection data
    """
    # Create a list of detections
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(float, box)
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        
        detection = {
            "class_id": int(class_id),
            "class_name": class_names.get(class_id, f"cls_{class_id}"),
            "confidence": float(score),
            "bbox": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "width": float(width),
                "height": float(height),
                "center_x": float(center_x),
                "center_y": float(center_y)
            }
        }
        detections.append(detection)
    
    # Save detections to JSON file
    json_output_path = os.path.join(output_dir, f"{image_name}_detections.json")
    with open(json_output_path, 'w') as f:
        json.dump({"detections": detections}, f, indent=2)
    
    # Also save as CSV for easier data analysis
    csv_output_path = os.path.join(output_dir, f"{image_name}_detections.csv")
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            "class_id", "class_name", "confidence", 
            "x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y"
        ])
        # Write data
        for det in detections:
            writer.writerow([
                det["class_id"],
                det["class_name"],
                det["confidence"],
                det["bbox"]["x1"],
                det["bbox"]["y1"],
                det["bbox"]["x2"],
                det["bbox"]["y2"],
                det["bbox"]["width"],
                det["bbox"]["height"],
                det["bbox"]["center_x"],
                det["bbox"]["center_y"]
            ])
    
    print(f"Saved detection data to {json_output_path} and {csv_output_path}")
    return json_output_path, csv_output_path

def plot_detections(
    image,
    results,
    class_names,
    conf_threshold=0.25,
    output_path=None,
    max_detections=200,
    plot_labels=True,
    random_colors=False,
    alpha_box=0.4,     # Opacity for bounding box outlines
    alpha_label=0.7    # Opacity for label box outlines
):
    """
    Plot detections on image such that both the detection boxes and label boxes
    have only a border (no filled color), and can be partially transparent.
    The text itself is drawn fully opaque on top.
    """

    # If image is a file path, load it
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a copy for the final display
    img_display = image.copy()
    h, w = img_display.shape[:2]
    
    # Overlays for box outlines and label box outlines
    box_overlay = img_display.copy()
    label_overlay = img_display.copy()
    
    # Initialize empty arrays for boxes, scores, and class_ids
    boxes = np.array([])
    scores = np.array([])
    class_ids = np.array([])
    
    # Extract boxes, scores, and class IDs from the results
    if results.boxes.xyxy.shape[0] > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Filter by confidence
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # Limit the number of detections
        if len(boxes) > max_detections:
            idxs_top = np.argsort(scores)[::-1][:max_detections]
            boxes = boxes[idxs_top]
            scores = scores[idxs_top]
            class_ids = class_ids[idxs_top]
        
        # Pick colors for each class
        if random_colors:
            np.random.seed(42)
            unique_cls = np.unique(class_ids)
            colors = {
                cls_id: tuple(np.random.randint(0, 255, 3).tolist()) for cls_id in unique_cls
            }
        else:
            # Use a hue-based approach
            colors = {}
            for cls_id in np.unique(class_ids):
                hue = (cls_id * 0.15) % 1.0
                rgb = plt.cm.hsv(hue)[:3]  # tuple in [0,1]
                colors[cls_id] = tuple((np.array(rgb) * 255).astype(int).tolist())
        
        # 1) Draw bounding box outlines (no fill) on box_overlay
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(cls_id, (0, 255, 0))
            # Outline only, thickness=2
            cv2.rectangle(box_overlay, (x1, y1), (x2, y2), color, 2)
        
        # Blend the bounding box outlines onto img_display with partial opacity
        cv2.addWeighted(box_overlay, alpha_box, img_display, 1 - alpha_box, 0, img_display)
        
        # 2) Draw label boxes as outlines on label_overlay, then blend
        if plot_labels:
            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                color = colors.get(cls_id, (0, 255, 0))
                
                label_str = f"{class_names.get(cls_id, f'cls_{cls_id}')} {score:.2f}"
                
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Position the label box above the detection box if possible
                label_ymin = y1 - text_height - baseline - 4
                # If there's no space above, put it below
                if label_ymin < 0:
                    label_ymin = y1 + text_height + baseline + 4
                
                # Draw only an outline (no fill) for the label
                cv2.rectangle(
                    label_overlay,
                    (x1, label_ymin - (text_height + baseline)),
                    (x1 + text_width, label_ymin),
                    color,
                    2
                )
            
            # Blend label box outlines onto img_display
            cv2.addWeighted(label_overlay, alpha_label, img_display, 1 - alpha_label, 0, img_display)
            
            # 3) Finally, draw text fully opaque on img_display
            for box, score, cls_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                color_text = (0, 0, 0)  # black text
                label_str = f"{class_names.get(cls_id, f'cls_{cls_id}')} {score:.2f}"
                
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                label_ymin = y1 - text_height - baseline - 4
                if label_ymin < 0:
                    label_ymin = y1 + text_height + baseline + 4
                
                cv2.putText(
                    img_display,
                    label_str,
                    (x1, label_ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color_text,
                    thickness
                )
    
    # Detection stats
    if hasattr(results.boxes, 'xyxy') and len(results.boxes.xyxy) > 0:
        num_detections = len(results.boxes.xyxy)
        stats_text = f"Total detections: {num_detections}"
        if len(boxes) < num_detections:
            stats_text += f" (showing top {len(boxes)} with conf>{conf_threshold:.2f})"
        cv2.putText(img_display, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        cv2.putText(img_display, "No detections", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Optionally save
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(20, 20 * h / w))
        plt.imshow(img_display)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved high-resolution detection result to {output_path}")
    
    return img_display, boxes, scores, class_ids


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
                
                # Plot results and get detection data
                _, boxes, scores, class_ids = plot_detections(
                    str(img_file), 
                    results, 
                    class_names, 
                    conf_threshold, 
                    output_path,
                    max_detections
                )
                
                # Save detection data to file
                save_detection_data(boxes, scores, class_ids, class_names, img_file.stem, output_dir)
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
            
            # Plot results and get detection data
            _, boxes, scores, class_ids = plot_detections(
                image_path, 
                results, 
                class_names, 
                conf_threshold, 
                output_path,
                max_detections
            )
            
            # Save detection data to file
            save_detection_data(boxes, scores, class_ids, class_names, img_stem, output_dir)
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