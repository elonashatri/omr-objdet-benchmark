from ultralytics import YOLO
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes in [x_center, y_center, width, height] format
    """
    # Convert to [x1, y1, x2, y2] format
    box1_x1 = box1[0] - box1[2]/2
    box1_y1 = box1[1] - box1[3]/2
    box1_x2 = box1[0] + box1[2]/2
    box1_y2 = box1[1] + box1[3]/2
    
    box2_x1 = box2[0] - box2[2]/2
    box2_y1 = box2[1] - box2[3]/2
    box2_x2 = box2[0] + box2[2]/2
    box2_y2 = box2[1] + box2[3]/2
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def calculate_map(detections, ground_truth, iou_threshold=0.5, num_classes=216):
    """
    Calculate mAP50 for the given detections against ground truth
    
    Args:
        detections: List of [image_id, class_id, x_center, y_center, width, height, confidence]
        ground_truth: List of [image_id, class_id, x_center, y_center, width, height]
        iou_threshold: IoU threshold (default 0.5 for mAP50)
        num_classes: Number of classes
        
    Returns:
        Dictionary with mAP50 for each class and overall
    """
    # Organize ground truth by image and class
    gt_by_image = defaultdict(list)
    for gt in ground_truth:
        img_id, cls_id = gt[0], int(gt[1])
        # Create a list of [x, y, w, h, matched_flag]
        gt_box = [float(gt[2]), float(gt[3]), float(gt[4]), float(gt[5]), False]
        gt_by_image[(img_id, cls_id)].append(gt_box)
    
    # Count ground truth per class
    gt_count_per_class = defaultdict(int)
    for gt in ground_truth:
        gt_count_per_class[int(gt[1])] += 1
    
    # Sort detections by confidence (descending)
    detections = sorted(detections, key=lambda x: x[6], reverse=True)
    
    # Initialize true positives and false positives arrays
    tp = defaultdict(list)
    fp = defaultdict(list)
    
    # Process each detection
    for det in detections:
        img_id, cls_id = det[0], int(det[1])
        box = [float(det[2]), float(det[3]), float(det[4]), float(det[5])]  # x, y, w, h
        
        # Skip if no ground truth for this image-class combination
        if (img_id, cls_id) not in gt_by_image:
            fp[cls_id].append(1)
            tp[cls_id].append(0)
            continue
        
        # Find best matching ground truth box
        gt_boxes = gt_by_image[(img_id, cls_id)]
        max_iou = -1
        max_idx = -1
        
        for i, gt_box in enumerate(gt_boxes):
            # Skip already matched ground truth boxes
            if gt_box[4]:  # Matched flag
                continue
            
            iou = calculate_iou(box, gt_box[:4])
            if iou > max_iou:
                max_iou = iou
                max_idx = i
        
        # Check if match is good enough
        if max_iou >= iou_threshold:
            # Mark ground truth as matched
            gt_boxes[max_idx][4] = True
            tp[cls_id].append(1)
            fp[cls_id].append(0)
        else:
            tp[cls_id].append(0)
            fp[cls_id].append(1)
    
    # Calculate precision and recall for each class
    ap_per_class = {}
    precision_per_class = {}
    recall_per_class = {}
    
    for cls_id in range(num_classes):
        if cls_id not in tp or gt_count_per_class[cls_id] == 0:
            ap_per_class[cls_id] = 0.0
            precision_per_class[cls_id] = 0.0
            recall_per_class[cls_id] = 0.0
            continue
        
        # Convert to numpy arrays for cumulative sum
        tp_cls = np.array(tp[cls_id])
        fp_cls = np.array(fp[cls_id])
        
        # Cumulative sum to get cumulative TP and FP
        tp_cumsum = np.cumsum(tp_cls)
        fp_cumsum = np.cumsum(fp_cls)
        
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / (gt_count_per_class[cls_id] + 1e-10)
        
        # Store last values for summary
        if len(precision) > 0:
            precision_per_class[cls_id] = precision[-1]
        else:
            precision_per_class[cls_id] = 0.0
            
        if len(recall) > 0:
            recall_per_class[cls_id] = recall[-1]
        else:
            recall_per_class[cls_id] = 0.0
        
        # Calculate AP (area under PR curve)
        # Add point at beginning and end of curve
        precision = np.concatenate(([0.0], precision, [0.0]))
        recall = np.concatenate(([0.0], recall, [1.0]))
        
        # Make precision monotonically decreasing
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        
        # Find indices where recall changes
        indices = np.where(recall[1:] != recall[:-1])[0]
        
        # Sum incremental areas under curve
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
        ap_per_class[cls_id] = ap
    
    # Calculate mean AP across all classes
    mAP = sum(ap_per_class.values()) / len(ap_per_class)
    
    return {
        'mAP50': mAP,
        'ap_per_class': ap_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class
    }

def predict_with_custom_thresholds(model_path, data_yaml, img_size=1280, device=0):
    """
    Run YOLOv8 validation with custom confidence thresholds for specific classes.
    """
    # Load the model
    model = YOLO(model_path)
    
    # Get class names from the model
    class_names = model.names
    
    # Define custom confidence thresholds for problematic classes
    class_conf_thresholds = {
        1: 0.05,    # stem - very low threshold (0.001 recall, 0.983 precision)
        2: 0.03,    # kStaffLine - extremely low threshold (0.000 recall)
        3: 0.07,    # barline - very low threshold (0.031 recall, 0.997 precision)
        9: 0.05,    # systemicBarline - very low threshold (0.005 recall, 1.000 precision)
        32: 0.10,   # augmentationDot - low threshold (0.100 recall, 1.000 precision)
        43: 0.05,   # articTenutoBelow - very low threshold (0.000 recall)
        45: 0.05,   # articTenutoAbove - very low threshold (0.000 recall)
        19: 0.07,   # articStaccatoAbove - low threshold (0.043 recall, 0.973 precision)
        23: 0.07,   # articStaccatoBelow - low threshold (0.045 recall, 0.977 precision)
        22: 0.20,   # T3 - reduced threshold (0.450 recall, 0.965 precision)
        28: 0.10,   # tupletBracket - reduced threshold (0.283 recall, 0.892 precision)
        29: 0.20,   # restHalf - reduced threshold (0.476 recall, 0.986 precision)
    }
    
    # Print the custom thresholds
    print("Using custom confidence thresholds for the following classes:")
    for cls_id, threshold in class_conf_thresholds.items():
        print(f"  Class {cls_id} ({class_names[cls_id]}): {threshold:.3f}")
    
    # Load the dataset YAML file
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    val_path = data_cfg['val']
    # Handle relative paths
    if not os.path.isabs(val_path):
        base_dir = os.path.dirname(data_yaml)
        val_path = os.path.join(base_dir, val_path)
    
    # Get list of validation images
    if os.path.isdir(val_path):
        val_images = [os.path.join(val_path, img) for img in os.listdir(val_path) 
                     if img.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        # If it's a text file listing the images
        with open(val_path, 'r') as f:
            val_images = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(val_images)} validation images")
    
    # Run standard validation
    print("\n=== Running baseline validation (standard thresholds) ===")
    baseline_results = model.val(
        data=data_yaml,
        imgsz=img_size,
        device=device,
        verbose=True
    )
    
    # Extract baseline metrics by parsing validation output
    # We'll need to extract this from the printed validation results
    import io
    import sys
    from contextlib import redirect_stdout
    
    baseline_metrics = {}
    
    try:
        # Run the validation again to capture the output
        f = io.StringIO()
        with redirect_stdout(f):
            model.val(data=data_yaml, imgsz=img_size, device=device, verbose=True)
        
        output = f.getvalue()
        
        # Parse output to extract per-class metrics
        lines = output.strip().split('\n')
        for line in lines:
            for cls_id, threshold in class_conf_thresholds.items():
                class_name = class_names[cls_id]
                if line.strip().startswith(class_name) and len(line.split()) >= 7:
                    parts = line.split()
                    try:
                        # Extract metrics from line
                        images = int(parts[-6])
                        instances = int(parts[-5])
                        precision = float(parts[-4])
                        recall = float(parts[-3])
                        map50 = float(parts[-2])
                        
                        baseline_metrics[cls_id] = {
                            'precision': precision,
                            'recall': recall,
                            'mAP50': map50
                        }
                    except (ValueError, IndexError):
                        pass
    except Exception as e:
        print(f"Error extracting baseline metrics: {e}")
    
    # Debug: Print extracted baseline metrics
    print(f"\nExtracted baseline metrics for {len(baseline_metrics)} classes")
    for cls_id, metrics in baseline_metrics.items():
        print(f"  Class {cls_id} ({class_names[cls_id]}): Recall: {metrics['recall']:.4f}, mAP50: {metrics['mAP50']:.4f}")
    # Run custom inference with adjusted thresholds
    print("\n=== Running custom inference with adjusted thresholds ===")
    
    # Set up containers for detections and ground truth
    all_detections = []
    all_labels = []
    
    # Get default confidence threshold
    default_conf = 0.25  # Default in YOLOv8
    
    # Process each validation image
    for img_path in tqdm(val_images, desc="Processing images"):
        # Get minimum threshold for inference
        min_conf = min(class_conf_thresholds.values()) if class_conf_thresholds else default_conf
        results = model(img_path, conf=min_conf, verbose=False)
        
        # Get corresponding ground truth
        label_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        
        # Load ground truth if available
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            all_labels.append([os.path.basename(img_path), cls_id, x, y, w, h])
        
        # Process detection results with custom thresholds
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            # Apply custom thresholds by class
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = boxes.conf[i].item()
                
                # Get threshold for this class
                threshold = class_conf_thresholds.get(cls_id, default_conf)
                
                # Keep detection if confidence exceeds class-specific threshold
                if conf >= threshold:
                    # Get coordinates (normalized format for compatibility with YOLO)
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    img_w, img_h = results[0].orig_shape[1], results[0].orig_shape[0]
                    x_center = (x1 + x2) / (2 * img_w)
                    y_center = (y1 + y2) / (2 * img_h)
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    # Store detection
                    all_detections.append([
                        os.path.basename(img_path),
                        cls_id,
                        x_center,
                        y_center,
                        width,
                        height,
                        conf
                    ])
    
    # Print detection statistics
    print(f"\nTotal detections with custom thresholds: {len(all_detections)}")
    
    # Count detections by class
    class_counts = {}
    for det in all_detections:
        cls_id = det[1]
        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
    
    # Print detection counts for problematic classes
    print("\nDetection counts for classes with custom thresholds:")
    for cls_id, count in sorted(class_counts.items()):
        if cls_id in class_conf_thresholds:
            cls_name = class_names[cls_id]
            threshold = class_conf_thresholds[cls_id]
            print(f"  Class {cls_id} ({cls_name}): {count} detections with threshold {threshold:.3f}")
    
    # Save detections to file
    output_dir = os.path.join(os.path.dirname(model_path), "custom_threshold_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    detections_path = os.path.join(output_dir, "custom_detections.txt")
    with open(detections_path, 'w') as f:
        for det in all_detections:
            f.write(f"{det[0]} {det[1]} {det[2]:.6f} {det[3]:.6f} {det[4]:.6f} {det[5]:.6f} {det[6]:.6f}\n")
    
    print(f"\nSaved custom detections to {detections_path}")
    
    # Debug: Check ground truth labels
    print(f"\nLoaded {len(all_labels)} ground truth labels")
    label_class_counts = {}
    for label in all_labels:
        cls_id = int(label[1])
        label_class_counts[cls_id] = label_class_counts.get(cls_id, 0) + 1

    print("Ground truth counts for classes with custom thresholds:")
    for cls_id, threshold in class_conf_thresholds.items():
        count = label_class_counts.get(cls_id, 0)
        print(f"  Class {cls_id} ({class_names[cls_id]}): {count} ground truth instances")
    # Calculate mAP with custom thresholds
    if all_labels and all_detections:
        print("\n=== Calculating mAP with custom thresholds ===")
        custom_metrics = calculate_map(
            all_detections,
            all_labels,
            iou_threshold=0.5,
            num_classes=len(class_names)
        )
        # Debug: Print custom metrics
        print(f"\nCalculated custom metrics for {len(custom_metrics['recall_per_class'])} classes")
        for cls_id in class_conf_thresholds.keys():
            if cls_id in custom_metrics['recall_per_class']:
                print(f"  Class {cls_id} ({class_names[cls_id]}): Recall: {custom_metrics['recall_per_class'][cls_id]:.4f}, mAP50: {custom_metrics['ap_per_class'][cls_id]:.4f}")
            else:
                print(f"  Class {cls_id} ({class_names[cls_id]}): No custom metrics calculated")
        
        # Compare baseline vs custom thresholds
        print("\n=== Comparison of baseline vs. custom thresholds ===")
        print("Class                    Baseline Recall    Custom Recall    Baseline mAP50    Custom mAP50")
        print("-----------------------------------------------------------------------------------------")
        
        improved_classes = 0
        total_assessed_classes = 0
        
        for cls_id, threshold in class_conf_thresholds.items():
            if cls_id in baseline_metrics:
                total_assessed_classes += 1
                cls_name = class_names[cls_id]
                baseline_recall = baseline_metrics[cls_id]['recall']
                baseline_map = baseline_metrics[cls_id]['mAP50']
                
                custom_recall = custom_metrics['recall_per_class'].get(cls_id, 0)
                custom_map = custom_metrics['ap_per_class'].get(cls_id, 0)
                
                recall_change = custom_recall - baseline_recall
                map_change = custom_map - baseline_map
                
                recall_symbol = "↑" if recall_change > 0 else "↓" if recall_change < 0 else "="
                map_symbol = "↑" if map_change > 0 else "↓" if map_change < 0 else "="
                
                print(f"{cls_name.ljust(25)}: {baseline_recall:.4f}         {custom_recall:.4f} {recall_symbol}      {baseline_map:.4f}         {custom_map:.4f} {map_symbol}")
                
                if recall_change > 0:
                    improved_classes += 1
        
        if total_assessed_classes > 0:
            print(f"\nImproved recall for {improved_classes} out of {total_assessed_classes} assessed classes")
            print(f"Overall baseline mAP50: {sum(m['mAP50'] for m in baseline_metrics.values()) / len(baseline_metrics):.4f}")
            print(f"Overall custom mAP50 for assessed classes: {sum(custom_metrics['ap_per_class'].get(cls_id, 0) for cls_id in baseline_metrics) / len(baseline_metrics):.4f}")
    
    return baseline_results

if __name__ == "__main__":
    # Example usage
    model_path = "/homes/es314/runs/detect/train2/weights/best.pt"
    data_yaml = "/homes/es314/omr-objdet-benchmark/DOREMI_v3/dataset.yaml"
    
    results = predict_with_custom_thresholds(
        model_path=model_path,
        data_yaml=data_yaml,
        img_size=1280,
        device=0
    )
    
    print("\nCustom threshold analysis complete!")