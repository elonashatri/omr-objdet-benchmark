import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# Import the Doremi dataset class
from doremi_loader import DoremiDataset


def collate_fn(batch):
    """Custom collate function for the DataLoader"""
    return tuple(zip(*batch))


def get_model(model_name, num_classes, device):
    """Load a pre-trained object detection model"""
    if model_name == 'faster_rcnn':
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        # Modify the classifier to match our number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes)
    
    elif model_name == 'retinanet':
        model = retinanet_resnet50_fpn(pretrained=True)
        # Modify the classifier
        num_anchors = model.head.classification_head.num_anchors
        in_features = model.head.classification_head.cls_logits.in_channels
        model.head.classification_head.num_classes = num_classes
        model.head.classification_head.cls_logits = nn.Conv2d(
            in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    
    elif model_name == 'yolov8':
        # For YOLOv8, we'll use the Ultralytics implementation
        model = YOLO('yolov8x.pt')  # Load pre-trained YOLOv8 model
        # We'll handle the class count differently for YOLO
        return model
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model.to(device)


def evaluate_model(model, data_loader, device, model_name, idx_to_class):
    """Evaluate model performance on the dataset"""
    model.eval()
    results = []
    
    # Track metrics
    total_time = 0
    detection_count = 0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"Evaluating {model_name}"):
            # Move images to device
            if model_name != 'yolov8':
                # Convert numpy arrays to tensors if needed
                images = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 if isinstance(img, np.ndarray) else img for img in images]
                images = [img.to(device) for img in images]
                
                # Measure inference time
                start_time = time.time()
                outputs = model(images)
                end_time = time.time()

                
                # Process predictions
                for i, output in enumerate(outputs):
                    image_id = targets[i]["image_id"].item()
                    gt_boxes = targets[i]["boxes"].cpu().numpy()
                    gt_labels = targets[i]["labels"].cpu().numpy()
                    
                    pred_boxes = output["boxes"].cpu().numpy()
                    pred_scores = output["scores"].cpu().numpy()
                    pred_labels = output["labels"].cpu().numpy()
                    
                    # Only keep predictions with confidence > 0.5
                    keep = pred_scores > 0.5
                    pred_boxes = pred_boxes[keep]
                    pred_scores = pred_scores[keep]
                    pred_labels = pred_labels[keep]
                    
                    detection_count += len(pred_boxes)
                    
                    # Store results for this image
                    results.append({
                        "image_id": image_id,
                        "gt_boxes": gt_boxes.tolist(),
                        "gt_labels": gt_labels.tolist(),
                        "pred_boxes": pred_boxes.tolist(),
                        "pred_scores": pred_scores.tolist(),
                        "pred_labels": pred_labels.tolist()
                    })
            else:
                # Handle YOLOv8 separately
                for i, (image, target) in enumerate(zip(images, targets)):
                    # Image already in the right format for YOLO if it's a numpy array
                    if isinstance(image, torch.Tensor):
                        image_np = image.permute(1, 2, 0).cpu().numpy()
                        # If normalized, unnormalize
                        image_np = (image_np * 255).astype(np.uint8)
                    else:
                        image_np = image
                    
                    image_id = target["image_id"].item()
                    gt_boxes = target["boxes"].numpy()
                    gt_labels = target["labels"].numpy()
                    
                    # Measure inference time
                    start_time = time.time()
                    yolo_results = model(image_np)
                    end_time = time.time()
                    
                    # Extract predictions
                    pred_boxes = []
                    pred_scores = []
                    pred_labels = []
                    
                    if len(yolo_results[0].boxes) > 0:
                        pred_boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                        pred_scores = yolo_results[0].boxes.conf.cpu().numpy()
                        pred_labels = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
                    
                    detection_count += len(pred_boxes)
                    
                    # Store results for this image
                    results.append({
                        "image_id": image_id,
                        "gt_boxes": gt_boxes.tolist(),
                        "gt_labels": gt_labels.tolist(),
                        "pred_boxes": pred_boxes.tolist(),
                        "pred_scores": pred_scores.tolist(),
                        "pred_labels": pred_labels.tolist()
                    })
            
            total_time += (end_time - start_time)
    
    # Calculate average inference time
    avg_time = total_time / len(data_loader)
    fps = len(data_loader) / total_time
    
    print(f"Model: {model_name}")
    print(f"Average inference time: {avg_time:.4f} seconds per batch")
    print(f"FPS: {fps:.2f}")
    print(f"Total detections: {detection_count}")
    
    return results, {"avg_time": avg_time, "fps": fps, "total_detections": detection_count}


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    # Get the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection rectangle
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    inter_area = width * height
    
    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou


def calculate_map(results, idx_to_class, iou_threshold=0.5):
    """Calculate mean Average Precision for object detection results"""
    # Group predictions by class
    class_predictions = {}
    class_ground_truths = {}
    
    # Initialize AP for each class
    average_precisions = {}
    
    # Count total number of ground truth objects for each class
    for result in results:
        gt_labels = result["gt_labels"]
        gt_boxes = result["gt_boxes"]
        
        for label, box in zip(gt_labels, gt_boxes):
            if label not in class_ground_truths:
                class_ground_truths[label] = []
            
            class_ground_truths[label].append({
                "image_id": result["image_id"],
                "box": box,
                "matched": False
            })
    
    # Collect all predictions sorted by confidence
    for result in results:
        pred_labels = result["pred_labels"]
        pred_boxes = result["pred_boxes"]
        pred_scores = result["pred_scores"]
        
        for label, box, score in zip(pred_labels, pred_boxes, pred_scores):
            if label not in class_predictions:
                class_predictions[label] = []
            
            class_predictions[label].append({
                "image_id": result["image_id"],
                "box": box,
                "score": score
            })
    
    # Calculate AP for each class
    for label in class_ground_truths:
        if label not in class_predictions or len(class_predictions[label]) == 0:
            # No predictions for this class
            average_precisions[label] = 0
            continue
        
        # Sort predictions by confidence score (descending)
        predictions = sorted(class_predictions[label], key=lambda x: x["score"], reverse=True)
        
        # Total number of ground truths for this class
        n_gt = len(class_ground_truths[label])
        
        # Initialize arrays for precision and recall calculation
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        # Match predictions to ground truths
        for i, pred in enumerate(predictions):
            # Find ground truths in the same image
            gt_in_img = [gt for gt in class_ground_truths[label] 
                        if gt["image_id"] == pred["image_id"] and not gt["matched"]]
            
            if not gt_in_img:
                # False positive (no ground truth in image)
                fp[i] = 1
                continue
            
            # Find the best matching ground truth (highest IoU)
            max_iou = -float('inf')
            max_idx = -1
            
            for j, gt in enumerate(gt_in_img):
                iou = calculate_iou(pred["box"], gt["box"])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            # Check if the match meets the IoU threshold
            if max_iou >= iou_threshold:
                # Mark the ground truth as matched
                gt_idx = [idx for idx, gt in enumerate(class_ground_truths[label]) 
                         if gt["image_id"] == pred["image_id"]][max_idx]
                
                if not class_ground_truths[label][gt_idx]["matched"]:
                    tp[i] = 1
                    class_ground_truths[label][gt_idx]["matched"] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        # Calculate cumulative precision and recall
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        recall = cumsum_tp / n_gt if n_gt > 0 else np.zeros_like(cumsum_tp)
        precision = cumsum_tp / (cumsum_tp + cumsum_fp)
        
        # Calculate average precision using all points interpolation
        ap = 0
        for r in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= r) == 0:
                p_at_r = 0
            else:
                p_at_r = np.max(precision[recall >= r])
            ap += p_at_r / 11
        
        average_precisions[label] = ap
    
    # Calculate mean AP across all classes
    mAP = np.mean([ap for ap in average_precisions.values()])
    
    return mAP, average_precisions


def visualize_detections(image, predictions, ground_truths, idx_to_class, threshold=0.5, output_path=None):
    """
    Visualize detections on an image at higher resolution with clearer labels.
    Green = Ground Truth, Blue = Predictions.
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Draw ground truth boxes
    for box, label_id in zip(ground_truths["boxes"], ground_truths["labels"]):
        x1, y1, x2, y2 = map(int, box)
        label_id = int(label_id)
        
        if label_id in idx_to_class:
            label = idx_to_class[label_id]
        else:
            label = f"Class {label_id}"
        
        # Thicker lines for better visibility
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Larger font scale
        cv2.putText(vis_image, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw predicted boxes
    for box, score, label_id in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
        if score < threshold:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        label_id = int(label_id)
        
        if label_id in idx_to_class:
            label = idx_to_class[label_id]
        else:
            label = f"Class {label_id}"
        
        # Thicker lines for better visibility
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # Larger font scale
        cv2.putText(vis_image, f"{label}: {score:.2f}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Display with matplotlib at higher DPI
    plt.figure(figsize=(20, 12), dpi=300)  # Larger, higher-resolution canvas
    plt.imshow(vis_image)
    plt.axis('off')
    plt.title("Object Detections (Green: GT, Blue: Predictions)")
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)  
        # 'bbox_inches="tight"' removes extra whitespace
    plt.close()
    
    return vis_image



def run_benchmark(args):
    """Run the benchmark with specified models and parameters"""
    # Set up device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load class mapping
    with open(args.class_mapping, 'r') as f:
        class_to_idx = json.load(f)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx) + 1  # +1 for background
    print(f"Using {num_classes} classes (including background)")
    
    # Set up dataset and dataloader
    dataset = DoremiDataset(
        args.data_dir,
        args.annotation_dir,
        args.class_mapping,
        min_box_size=args.min_box_size,
        max_classes=args.max_classes
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Run benchmarks for each specified model
    all_results = {}
    all_metrics = {}
    
    for model_name in args.models:
        print(f"\nBenchmarking {model_name}...")
        
        # Load model
        model = get_model(model_name, num_classes, device)
        
        # Evaluate model
        results, metrics = evaluate_model(model, data_loader, device, model_name, idx_to_class)
        
        # Calculate mAP
        mAP, class_aps = calculate_map(results, idx_to_class)
        
        # Add mAP to metrics
        metrics["mAP"] = mAP
        metrics["class_AP"] = class_aps
        
        # Store results
        all_results[model_name] = results
        all_metrics[model_name] = metrics
        
        print(f"mAP: {mAP:.4f}")
        
        # Save detailed class AP
        print("Class APs:")
        top_classes = sorted(class_aps.items(), key=lambda x: x[1], reverse=True)[:20]  # Top 20 classes
        for label_id, ap in top_classes:
            class_name = idx_to_class.get(label_id, f"Class {label_id}")
            print(f"  {class_name}: {ap:.4f}")
            
        # Visualize some examples if requested
        if args.visualize:
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Select a few samples for visualization
            num_vis = min(args.num_visualizations, len(dataset))
            for i in range(num_vis):
                image, target = dataset[i]
                
                # Get corresponding result
                result = next((r for r in results if r["image_id"] == target["image_id"].item()), None)
                
                if result:
                    # Prepare ground truth and predictions
                    gt = {
                        "boxes": target["boxes"].numpy(),
                        "labels": target["labels"].numpy()
                    }
                    
                    pred = {
                        "boxes": np.array(result["pred_boxes"]),
                        "scores": np.array(result["pred_scores"]),
                        "labels": np.array(result["pred_labels"])
                    }
                    
                    # Visualize
                    out_path = os.path.join(args.output_dir, f"{model_name}_vis_{i}.png")
                    visualize_detections(image, pred, gt, idx_to_class, output_path=out_path)
    
    # Compare all models
    compare_models(all_metrics, idx_to_class, args.output_dir)
    
    # Save results to file
    save_results(all_results, all_metrics, args.output_dir)
    
    return all_results, all_metrics


def compare_models(metrics, idx_to_class, output_dir):
    """Compare metrics across different models"""
    # Extract key metrics
    model_names = list(metrics.keys())
    
    # Create comparison dataframe
    comparison = {
        "Model": model_names,
        "mAP": [metrics[model]["mAP"] for model in model_names],
        "FPS": [metrics[model]["fps"] for model in model_names],
        "Inference Time (ms)": [metrics[model]["avg_time"] * 1000 for model in model_names]
    }
    
    df = pd.DataFrame(comparison)
    print("\nModel Comparison:")
    print(df)
    
    # Create comparison plots
    os.makedirs(output_dir, exist_ok=True)
    
    # mAP comparison
    plt.figure(figsize=(10, 6))
    ax = plt.bar(df["Model"], df["mAP"])
    plt.ylabel("mAP")
    plt.title("Mean Average Precision Comparison")
    plt.ylim(0, 1)
    
    # Add values on top of bars
    for i, v in enumerate(df["mAP"]):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.savefig(os.path.join(output_dir, "map_comparison.png"), bbox_inches='tight')
    plt.close()
    
    # FPS comparison
    plt.figure(figsize=(10, 6))
    ax = plt.bar(df["Model"], df["FPS"])
    plt.ylabel("Frames Per Second")
    plt.title("Speed Comparison")
    
    # Add values on top of bars
    for i, v in enumerate(df["FPS"]):
        plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
    
    plt.savefig(os.path.join(output_dir, "fps_comparison.png"), bbox_inches='tight')
    plt.close()
    
    # Class AP comparison (top N classes)
    top_n = 20  # Top N classes to compare
    all_class_aps = {}
    
    # Collect APs for all classes across models
    for model in model_names:
        if "class_AP" in metrics[model]:
            for label_id, ap in metrics[model]["class_AP"].items():
                if int(label_id) in idx_to_class:
                    class_name = idx_to_class[int(label_id)]
                else:
                    class_name = f"Class {label_id}"
                
                if class_name not in all_class_aps:
                    all_class_aps[class_name] = {}
                
                all_class_aps[class_name][model] = ap
    
    # Calculate average AP across models for each class
    class_avg_aps = {cls: sum(aps.values()) / len(aps) for cls, aps in all_class_aps.items()}
    
    # Select top N classes by average AP
    top_classes = sorted(class_avg_aps.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_class_names = [cls for cls, _ in top_classes]
    
    # Create per-class comparison
    plt.figure(figsize=(15, 10))
    x = np.arange(len(top_class_names))
    width = 0.8 / len(model_names)
    
    for i, model in enumerate(model_names):
        values = [all_class_aps[cls].get(model, 0) for cls in top_class_names]
        plt.bar(x + i * width - 0.4 + width/2, values, width, label=model)
    
    plt.ylabel('AP')
    plt.title('Average Precision for Top Classes')
    plt.xticks(x, top_class_names, rotation=90)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, "top_class_ap_comparison.png"), bbox_inches='tight')
    plt.close()


def save_results(results, metrics, output_dir):
    """Save benchmark results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        # Convert NumPy values to Python native types for JSON serialization
        serializable_metrics = {}
        for model, model_metrics in metrics.items():
            serializable_metrics[model] = {}
            for k, v in model_metrics.items():
                if k == "class_AP":
                    # Handle class AP dictionary
                    class_ap_dict = {}
                    for class_id, ap in v.items():
                        class_ap_dict[int(class_id)] = float(ap)
                    serializable_metrics[model][k] = class_ap_dict
                elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                    serializable_metrics[model][k] = float(v)
                else:
                    serializable_metrics[model][k] = v
        
        json.dump(serializable_metrics, f, indent=4)
    
    # Save results
    for model, model_results in results.items():
        with open(os.path.join(output_dir, f"{model}_results.json"), "w") as f:
            # Convert NumPy arrays to lists for JSON serialization
            serializable_results = []
            for result in model_results:
                # Already converted to lists when storing in results
                serializable_results.append(result)
            
            json.dump(serializable_results, f, indent=4)
    
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark object detection models for Optical Music Recognition")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset images")
    parser.add_argument("--annotation_dir", type=str, required=True, help="Directory containing XML annotations")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping JSON file")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Directory to save results")
    parser.add_argument("--models", nargs="+", default=["faster_rcnn", "retinanet", "yolov8"], 
                        help="Models to benchmark")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--min_box_size", type=int, default=5, help="Minimum size for bounding boxes")
    parser.add_argument("--max_classes", type=int, default=None, help="Maximum number of classes to use")
    parser.add_argument("--visualize", action="store_true", help="Visualize some predictions")
    parser.add_argument("--num_visualizations", type=int, default=10, help="Number of images to visualize")
    
    args = parser.parse_args()
    
    # Run the benchmark
    run_benchmark(args)


if __name__ == "__main__":
    main()