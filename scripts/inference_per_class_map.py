# from ultralytics import YOLO

# def get_map_per_class(model_path, data_yaml, img_size=1280, device=0):
#     # Load the model
#     model = YOLO(model_path)
    
#     # Run validation with verbose option to get per-class metrics
#     results = model.val(
#         data=data_yaml,
#         imgsz=img_size,
#         device=device,
#         verbose=True  # This ensures detailed metrics are printed
#     )
    
#     # The metrics are available in the results object
#     print("\n--- mAP Per Class ---")
#     for i, class_name in enumerate(results.names):
#         # Check if class exists in validation set
#         if i in results.metrics.class_ids:
#             idx = results.metrics.class_ids.index(i)
#             ap50 = results.metrics.ap50_per_class[idx]
#             ap = results.metrics.ap_per_class[idx]
#             print(f"Class {i} ({class_name}): mAP50 = {ap50:.4f}, mAP50-95 = {ap:.4f}")
    
#     return results

# # Example usage
# get_map_per_class(
#     "/homes/es314/omr-objdet-benchmark/runs/staffline_extreme/weights/best.pt",
#     "/homes/es314/omr-objdet-benchmark/results/yolo_fixed/dataset.yaml",
#     img_size=1280,
#     device=0
# )


from ultralytics import YOLO
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

def predict_with_custom_thresholds(model_path, data_yaml, img_size=1280, device=0):
    """
    Run YOLOv8 validation with custom confidence thresholds for specific classes.
    
    Args:
        model_path: Path to the YOLOv8 model weights
        data_yaml: Path to the dataset YAML file
        img_size: Image size for inference
        device: GPU device ID (0 by default)
        
    Returns:
        The validation results object
    """
    # Load the model
    model = YOLO(model_path)
    
    # Get class names from the model
    class_names = model.names
    
    # Define custom confidence thresholds for problematic classes
    # Based on the analysis, we'll use lower thresholds for classes with high precision but low recall
    class_conf_thresholds = {
        1: 0.005,    # stem - very low threshold (0.001 recall, 0.983 precision)
        2: 0.003,    # kStaffLine - extremely low threshold (0.000 recall)
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
    
    # Load the dataset YAML file to get validation images
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
    
    # First, run standard validation to get baseline metrics
    print("\n=== Running baseline validation (standard thresholds) ===")
    baseline_results = model.val(
        data=data_yaml,
        imgsz=img_size,
        device=device,
        verbose=True
    )
    
    # Store baseline metrics for problematic classes
    baseline_metrics = {}
    for cls_id in class_conf_thresholds.keys():
        if cls_id in baseline_results.names:
            cls_name = baseline_results.names[cls_id]
            if cls_id in [c.item() for c in baseline_results.metrics.class_ids]:
                idx = [c.item() for c in baseline_results.metrics.class_ids].index(cls_id)
                precision = baseline_results.metrics.precision[idx]
                recall = baseline_results.metrics.recall[idx]
                map50 = baseline_results.metrics.ap50_per_class[idx]
                baseline_metrics[cls_id] = {
                    'precision': precision,
                    'recall': recall,
                    'mAP50': map50
                }
    
    # Now, let's run custom inference with adjusted thresholds
    print("\n=== Running custom inference with adjusted thresholds ===")
    
    # Set up containers for custom detections
    all_detections = []
    all_labels = []
    
    # Get default confidence threshold
    default_conf = 0.25  # This is usually the default in YOLOv8
    
    # Process each validation image
    for img_path in tqdm(val_images, desc="Processing images"):
        # Run prediction with lowest threshold to catch all potential detections
        min_conf = min(class_conf_thresholds.values()) if class_conf_thresholds else default_conf
        results = model(img_path, conf=min_conf, verbose=False)
        
        # Get the corresponding ground truth file (assuming YOLO format)
        label_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        
        # Load ground truth if available
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f if line.strip()]
                labels = [[int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in labels]
                all_labels.extend([(os.path.basename(img_path), *l) for l in labels])
        
        # Process results with custom thresholds
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
                    
                    # Store detection in YOLO format for compatibility
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
    
    # Save detections to file for analysis
    if all_detections:
        output_dir = os.path.join(os.path.dirname(model_path), "custom_threshold_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detections in a format that can be used for evaluation
        detections_path = os.path.join(output_dir, "custom_detections.txt")
        with open(detections_path, 'w') as f:
            for det in all_detections:
                f.write(f"{det[0]} {det[1]} {det[2]:.6f} {det[3]:.6f} {det[4]:.6f} {det[5]:.6f} {det[6]:.6f}\n")
        
        print(f"\nSaved custom detections to {detections_path}")
    
    # Run evaluation comparing baseline to custom thresholds
    # (This would require implementing mAP calculation with the custom detections)
    # For simplicity, we're just returning the baseline results here
    
    print("\n=== Comparison of baseline vs. custom thresholds ===")
    print("Class                    Baseline Recall    Custom Recall    Baseline mAP50")
    print("-------------------------------------------------------------------------")
    for cls_id, threshold in class_conf_thresholds.items():
        if cls_id in baseline_metrics:
            cls_name = class_names[cls_id]
            baseline_recall = baseline_metrics[cls_id]['recall']
            # Custom recall would need to be calculated from the detections and ground truth
            # For now, we'll just show expected improvement based on threshold reduction
            expected_recall_gain = f"(↑ with {threshold:.3f} threshold)"
            print(f"{cls_name.ljust(25)}: {baseline_recall:.4f}         {expected_recall_gain}")
    
    return baseline_results

if __name__ == "__main__":
    # Example usage
    model_path = "/homes/es314/omr-objdet-benchmark/runs/staffline_extreme/weights/best.pt"
    data_yaml = "/homes/es314/omr-objdet-benchmark/results/yolo_fixed/dataset.yaml"
    
    results = predict_with_custom_thresholds(
        model_path=model_path,
        data_yaml=data_yaml,
        img_size=1280,
        device=0
    )
    
    print("\nCustom threshold analysis complete!")