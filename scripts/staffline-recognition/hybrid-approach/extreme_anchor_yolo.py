#!/usr/bin/env python3
import os
import argparse
import yaml
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def train_staffline_focused(data_yaml, model_path, epochs, batch_size, device, 
                           staffline_class_id, output_dir="runs/train/staffline_focused"):
    """
    Train with settings optimized for extreme aspect ratio stafflines
    """
    print(f"Training with staffline optimization (class ID: {staffline_class_id})")
    
    # Create model
    model = YOLO(model_path)
    
    # Create custom class weights to prioritize stafflines
    # Create a list of weights with higher value for staffline class
    # NOTE: This is a workaround since we can't use custom anchors directly
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a config file with staffline info
    config = {
        'staffline_class_id': staffline_class_id,
        'model': model_path,
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'device': device
    }
    
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Train with parameters optimized for thin objects
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=1280,  # Higher resolution helps with thin objects
        device=device,
        project=os.path.dirname(output_dir),
        name=os.path.basename(output_dir),
        exist_ok=True,
        
        # Core training parameters
        patience=100,     # More patience for convergence
        cos_lr=True,      # Cosine LR schedule
        lr0=0.01,         # Higher initial learning rate
        lrf=0.001,        # Final learning rate factor
        
        # Loss weights
        box=10.0,         # Higher box loss weight for better localization
        cls=1.0,          # Base classification loss weight
        dfl=1.5,          # Distribution focal loss weight
        
        # Augmentation optimized for stafflines
        mosaic=0.3,       # Reduce mosaic as it can break lines
        degrees=1,        # Minimal rotation to preserve horizontality
        shear=0.0,        # No shearing (preserves horizontal lines)
        perspective=0.0,  # No perspective (preserves horizontal lines)
        scale=0.2,        # Scale jitter
        fliplr=0.1,       # Minimal horizontal flip (stafflines are symmetric)
        flipud=0.0,       # No vertical flip (would confuse stafflines)
        hsv_h=0.01,       # Minimal hue shift
        hsv_s=0.1,        # Minimal saturation shift
        hsv_v=0.1,        # Minimal brightness shift
        
        # Save more frequent checkpoints
        save_period=10,
        
        # Optimization for thin objects
        overlap_mask=True,
        
        # Enable YOLO to output more information during training
        verbose=True
    )
    
    print(f"\nTraining complete. Results saved to {output_dir}")
    return results

def create_post_training_optimization(model_path, data_yaml, staffline_class_id, output_dir):
    """
    Create a post-training optimization script for stafflines
    """
    script_path = os.path.join(output_dir, "optimize_staffline_detection.py")
    
    script_content = f'''
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def optimize_staffline_detection(image_path, model, staffline_class_id, conf=0.25, output_path=None):
    """
    Apply post-processing to enhance staffline detection
    """
    # Run standard YOLO detection
    results = model.predict(image_path, conf=conf, verbose=False)[0]
    
    # Get original image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {{image_path}}")
        return results
    
    # Extract staffline predictions
    staffline_boxes = []
    for i, box in enumerate(results.boxes):
        if int(box.cls.item()) == staffline_class_id:
            staffline_boxes.append({{
                'xyxy': box.xyxy[0].cpu().numpy(),  # x1, y1, x2, y2
                'conf': float(box.conf.item())
            }})
    
    # If few stafflines detected, try to find more with Hough transform
    if len(staffline_boxes) < 5:  # Typical staff has 5 lines
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # Remove small noise
        kernel = np.ones((1, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Detect lines using probabilistic Hough transform
        lines = cv2.HoughLinesP(
            binary, 
            rho=1, 
            theta=np.pi/180,
            threshold=100,
            minLineLength=img.shape[1]//4,
            maxLineGap=20
        )
        
        if lines is not None:
            h, w = img.shape[:2]
            additional_stafflines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Skip if not horizontal-ish
                if abs(y2 - y1) > 5:
                    continue
                
                # Skip if too short
                if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < w * 0.2:
                    continue
                
                # Compute line properties
                y_center = (y1 + y2) / 2
                height = max(3, abs(y2 - y1))
                
                # Check if this line overlaps with existing stafflines
                is_new_line = True
                for box in staffline_boxes:
                    y_min, y_max = box['xyxy'][1], box['xyxy'][3]
                    if y_min <= y_center <= y_max or abs(y_center - (y_min + y_max)/2) < height:
                        is_new_line = False
                        break
                
                if is_new_line:
                    # Add to additional stafflines
                    xyxy = np.array([x1, y_center - height/2, x2, y_center + height/2])
                    additional_stafflines.append({{
                        'xyxy': xyxy,
                        'conf': 0.7  # Reasonable confidence
                    }})
            
            # Add new stafflines to the results
            if additional_stafflines:
                print(f"Found {{len(additional_stafflines)}} additional stafflines")
                staffline_boxes.extend(additional_stafflines)
        
    # Visualize if output path is provided
    if output_path:
        # Draw original predictions
        vis_img = img.copy()
        
        # Draw non-staffline boxes
        for i, box in enumerate(results.boxes):
            if int(box.cls.item()) != staffline_class_id:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(vis_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        
        # Draw staffline boxes
        for box in staffline_boxes:
            xyxy = box['xyxy'].astype(int)
            cv2.rectangle(vis_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 1)
        
        # Save visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_img)
        print(f"Saved visualization to {{output_path}}")
    
    return results, staffline_boxes

def main():
    parser = argparse.ArgumentParser(description="Optimize Staffline Detection")
    parser.add_argument("--model", type=str, default="{model_path}", help="Path to YOLO model")
    parser.add_argument("--data", type=str, default="{data_yaml}", help="Path to dataset YAML")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="results/optimized", help="Output directory")
    parser.add_argument("--staffline_class_id", type=int, default={staffline_class_id}, 
                        help="Class ID for stafflines (0-indexed)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Load model
    model = YOLO(args.model)
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single image
        output_path = os.path.join(args.output, f"{{input_path.stem}}_optimized.jpg")
        results, boxes = optimize_staffline_detection(
            str(input_path), model, args.staffline_class_id, args.conf, output_path
        )
        print(f"Processed {{input_path.name}}: Found {{len(boxes)}} stafflines")
    else:
        # Process directory
        os.makedirs(args.output, exist_ok=True)
        image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        
        for img_path in image_paths:
            output_path = os.path.join(args.output, f"{{img_path.stem}}_optimized.jpg")
            results, boxes = optimize_staffline_detection(
                str(img_path), model, args.staffline_class_id, args.conf, output_path
            )
            print(f"Processed {{img_path.name}}: Found {{len(boxes)}} stafflines")

if __name__ == "__main__":
    main()
'''
    
    # Write script to file
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created post-processing script: {script_path}")
    print(f"Run after training with: python {script_path} --input <image_or_directory>")
    
    return script_path

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 with staffline focus")
    parser.add_argument("--yaml", type=str, required=True, help="Path to dataset.yaml file")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="YOLO model to use")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--staffline_class_id", type=int, required=True,
                        help="Class ID for stafflines in your dataset (0-indexed)")
    parser.add_argument("--output", type=str, default="runs/train/staffline_focused",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    print(f"Training {args.model} with staffline optimization")
    print(f"Dataset: {args.yaml}")
    print(f"Staffline class ID: {args.staffline_class_id}")
    
    # Run training
    results = train_staffline_focused(
        args.yaml,
        args.model,
        args.epochs,
        args.batch,
        args.device,
        args.staffline_class_id,
        args.output
    )
    
    # Create post-training optimization script
    best_model = os.path.join(args.output, "weights/best.pt")
    create_post_training_optimization(best_model, args.yaml, args.staffline_class_id, args.output)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()