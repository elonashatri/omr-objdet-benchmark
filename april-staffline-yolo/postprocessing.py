import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import json
import yaml
from ultralytics import YOLO
from tqdm import tqdm

def refine_staffline_detections(image_path, detections, staffline_class_id, staff_height=None):
    """
    Refine staffline detections using image processing techniques
    
    Args:
        image_path: Path to input image
        detections: YOLO detection results
        staffline_class_id: Class ID for stafflines
        staff_height: Optional known staff height
        
    Returns:
        Updated detections with refined stafflines
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error reading image: {image_path}")
        return detections
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Filter out staffline detections
    staffline_dets = [det for det in detections if det['class'] == staffline_class_id]
    other_dets = [det for det in detections if det['class'] != staffline_class_id]
    
    # If no staffline detections, return original
    if not staffline_dets:
        return detections
    
    # Process image to enhance stafflines
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    
    # Create horizontal kernel for staffline detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    
    # Detect horizontal lines (stafflines)
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Find connected components (potential stafflines)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(detected_lines)
    
    # Group stafflines into staves (5 consecutive lines)
    refined_stafflines = []
    
    # Sort components by y-coordinate (top to bottom)
    sorted_comps = sorted([(i, stats[i]) for i in range(1, num_labels)], 
                           key=lambda x: x[1][1])  # Sort by y position
    
    # Identify valid stafflines
    staffline_groups = []
    current_group = []
    
    for i, (label_id, stat) in enumerate(sorted_comps):
        # Filter by aspect ratio - stafflines are very wide compared to height
        x, y, w, h, area = stat
        aspect_ratio = w / h if h > 0 else 0
        
        # Only consider horizontal lines with sufficient width
        if aspect_ratio > 10 and w > width * 0.5:
            # If current_group is empty or this line is close to the last line in the group
            if not current_group or (y - current_group[-1][1] - current_group[-1][3] < height * 0.03):
                current_group.append((x, y, w, h))
            else:
                # Check if we have a complete staff (5 lines)
                if len(current_group) >= 3:  # At least 3 lines to be considered a staff
                    staffline_groups.append(current_group)
                current_group = [(x, y, w, h)]
    
    # Add the last group if it exists
    if len(current_group) >= 3:
        staffline_groups.append(current_group)
    
    # Convert staffline groups to detections
    for group in staffline_groups:
        for x, y, w, h in group:
            # Convert to YOLO format (normalized)
            center_x = (x + w/2) / width
            center_y = (y + h/2) / height
            norm_width = w / width
            norm_height = h / height
            
            # Create detection dictionary
            refined_stafflines.append({
                'class': staffline_class_id,
                'confidence': 0.95,  # High confidence for refined detections
                'x': center_x,
                'y': center_y,
                'width': norm_width,
                'height': norm_height
            })
    
    # Combine refined stafflines with other detections
    combined_detections = other_dets + refined_stafflines
    
    return combined_detections

def process_model_predictions(model_path, image_dir, output_path, class_mapping, staffline_class_name="kStaffLine", conf_threshold=0.25):
    """
    Process YOLO predictions and refine staffline detections
    
    Args:
        model_path: Path to trained YOLO model
        image_dir: Directory containing images
        output_path: Path to save results
        class_mapping: Path to class mapping file
        staffline_class_name: Name of staffline class
        conf_threshold: Confidence threshold for detections
    """
    # Load model
    model = YOLO(model_path)
    
    # Load class mapping
    with open(class_mapping, 'r') as f:
        class_map = json.load(f)
    
    # Invert class mapping (name -> id)
    name_to_idx = {name: idx for idx, name in class_map.items()}
    
    # Get staffline class ID
    staffline_class_id = None
    for name, idx in name_to_idx.items():
        if staffline_class_name.lower() in name.lower():
            staffline_class_id = int(idx) - 1  # Convert to 0-indexed for YOLO
            print(f"Found staffline class: {name} (ID: {staffline_class_id})")
            break
    
    if staffline_class_id is None:
        print(f"Warning: Could not find staffline class '{staffline_class_name}' in class mapping")
        return
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get image files
    image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    print(f"Processing {len(image_files)} images")
    
    # Process each image
    results = []
    for img_path in tqdm(image_files, desc="Processing images"):
        # Debug info moved to tqdm progress bar
        
        # Run inference with confidence threshold
        detections = model(img_path, conf=conf_threshold)[0]
        
        # Convert to list of dictionaries
        detection_list = []
        for det in detections.boxes.data:
            x1, y1, x2, y2, conf, cls = det.tolist()
            
            # Convert to center coordinates
            width = x2 - x1
            height = y2 - y1
            center_x = x1 + width/2
            center_y = y1 + height/2
            
            # Normalize
            img_width, img_height = detections.orig_shape[1], detections.orig_shape[0]
            norm_center_x = center_x / img_width
            norm_center_y = center_y / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            
            detection_list.append({
                'class': int(cls),
                'confidence': conf,
                'x': norm_center_x,
                'y': norm_center_y,
                'width': norm_width,
                'height': norm_height
            })
        
        # Refine staffline detections
        refined_detections = refine_staffline_detections(
            img_path, detection_list, staffline_class_id
        )
        
        # Save to results
        results.append({
            'image': img_path.name,
            'detections': refined_detections
        })
    
    # Save results to JSON
    results_file = output_dir / 'refined_detections.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Create visualization of the refined detections
    os.makedirs(output_dir / 'visualizations', exist_ok=True)
    
    for result in tqdm(results, desc="Creating visualizations"):
        img_path = Path(image_dir) / result['image']
        img = cv2.imread(str(img_path))
        
        if img is None:
            continue
            
        img_height, img_width = img.shape[:2]
        
        # Draw detections
        for det in result['detections']:
            cls = det['class']
            conf = det['confidence']
            
            # Denormalize coordinates
            center_x = int(det['x'] * img_width)
            center_y = int(det['y'] * img_height)
            width = int(det['width'] * img_width)
            height = int(det['height'] * img_height)
            
            # Calculate top-left corner
            x1 = int(center_x - width/2)
            y1 = int(center_y - height/2)
            
            # Draw rectangle
            if cls == staffline_class_id:
                # Use green for stafflines
                color = (0, 255, 0)
            else:
                # Use blue for other detections
                color = (255, 0, 0)
                
            cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), color, 1)
            
            # Add label if not a staffline (too many stafflines makes visualization cluttered)
            if cls != staffline_class_id:
                # Find class name
                class_name = "Unknown"
                for name, idx in name_to_idx.items():
                    if int(idx) - 1 == cls:
                        class_name = name
                        break
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save visualization
        vis_path = output_dir / 'visualizations' / result['image']
        cv2.imwrite(str(vis_path), img)
    
    print(f"Visualizations saved to {output_dir / 'visualizations'}")

def main():
    parser = argparse.ArgumentParser(description="Process and refine music notation detections")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument("--images", type=str, required=True, help="Directory containing images to process")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--class-mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--staffline-class", type=str, default="kStaffLine", help="Name of staffline class")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    # Process predictions
    process_model_predictions(
        args.model,
        args.images,
        args.output,
        args.class_mapping,
        args.staffline_class,
        args.conf
    )

if __name__ == "__main__":
    main()