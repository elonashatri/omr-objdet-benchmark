import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union between two binary masks
    
    Args:
        pred_mask: Predicted mask (binary numpy array)
        gt_mask: Ground truth mask (binary numpy array)
        
    Returns:
        iou: IoU score between 0 and 1
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 0
    
    return intersection / union

def calculate_iou_bbox(box1, box2):
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        iou: IoU score between 0 and 1
    """
    # Determine coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0
    
    return intersection / union

def evaluate_detections(predictions, ground_truth, iou_threshold=0.5, score_threshold=0.5):
    """
    Evaluate detection performance
    
    Args:
        predictions: List of dictionaries containing 'boxes', 'scores', 'labels', 'masks'
        ground_truth: List of dictionaries containing 'boxes', 'labels', 'masks'
        iou_threshold: IoU threshold for considering a detection correct
        score_threshold: Score threshold for filtering detections
        
    Returns:
        metrics: Dictionary containing precision, recall, F1, AP for each class
    """
    # Initialize counters for each class
    class_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'AP': 0, 'precisions': [], 'recalls': [], 'iou': []})
    
    # For calculating mAP
    all_detections = []
    
    # Process each image
    for i, (preds, gt) in enumerate(zip(predictions, ground_truth)):
        # Filter predictions by score
        keep_indices = np.where(preds['scores'] > score_threshold)[0]
        if len(keep_indices) == 0:
            # If no predictions above threshold, count all GT as false negatives
            for gt_idx in range(len(gt['labels'])):
                gt_label = gt['labels'][gt_idx]
                class_metrics[gt_label]['FN'] += 1
            continue
            
        filtered_boxes = preds['boxes'][keep_indices]
        filtered_scores = preds['scores'][keep_indices]
        filtered_labels = preds['labels'][keep_indices]
        filtered_masks = preds['masks'][keep_indices] if 'masks' in preds else None
        
        # Track which ground truth objects have been matched
        gt_matched = [False] * len(gt['labels'])
        
        # For each prediction, find best matching ground truth
        for pred_idx in range(len(filtered_labels)):
            pred_label = filtered_labels[pred_idx]
            pred_box = filtered_boxes[pred_idx]
            pred_score = filtered_scores[pred_idx]
            pred_mask = filtered_masks[pred_idx] if filtered_masks is not None else None
            
            # Store detection for mAP calculation
            all_detections.append({
                'image_id': i,
                'label': pred_label,
                'score': pred_score,
                'matched': False
            })
            
            # Find best matching ground truth for this prediction
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(gt['labels'])):
                # Skip if GT is already matched or labels don't match
                if gt_matched[gt_idx] or gt['labels'][gt_idx] != pred_label:
                    continue
                
                # Calculate IoU (either for masks or boxes)
                if pred_mask is not None and 'masks' in gt:
                    gt_mask = gt['masks'][gt_idx]
                    iou = calculate_iou(pred_mask[0] > 0.5, gt_mask)
                else:
                    gt_box = gt['boxes'][gt_idx]
                    iou = calculate_iou_bbox(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Store IoU for this class
            class_metrics[pred_label]['iou'].append(best_iou)
            
            # If IoU exceeds threshold, count as true positive
            if best_iou > iou_threshold and best_gt_idx >= 0:
                class_metrics[pred_label]['TP'] += 1
                gt_matched[best_gt_idx] = True
                all_detections[-1]['matched'] = True
            else:
                class_metrics[pred_label]['FP'] += 1
        
        # Count unmatched ground truth as false negatives
        for gt_idx in range(len(gt['labels'])):
            if not gt_matched[gt_idx]:
                gt_label = gt['labels'][gt_idx]
                class_metrics[gt_label]['FN'] += 1
    
    # Calculate precision and recall for each class
    for class_id, metrics in class_metrics.items():
        tp = metrics['TP']
        fp = metrics['FP']
        fn = metrics['FN']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou_avg = np.mean(metrics['iou']) if metrics['iou'] else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        metrics['iou_avg'] = iou_avg
    
    # Calculate AP for each class
    class_detections = defaultdict(list)
    for det in all_detections:
        class_detections[det['label']].append((det['score'], det['matched']))
    
    for class_id, detections in class_detections.items():
        # Sort by decreasing score
        detections.sort(key=lambda x: x[0], reverse=True)
        
        # Calculate cumulative precision and recall
        tp_cumsum = 0
        precisions = []
        recalls = []
        
        total_positives = class_metrics[class_id]['TP'] + class_metrics[class_id]['FN']
        if total_positives == 0:
            continue
            
        for i, (_, matched) in enumerate(detections):
            if matched:
                tp_cumsum += 1
            
            # Calculate precision and recall at this point
            cum_precision = tp_cumsum / (i + 1)
            cum_recall = tp_cumsum / total_positives
            
            precisions.append(cum_precision)
            recalls.append(cum_recall)
        
        # Store precision and recall curves
        class_metrics[class_id]['precisions'] = precisions
        class_metrics[class_id]['recalls'] = recalls
        
        # Calculate AP (area under the PR curve using 11-point interpolation)
        if len(precisions) > 0:
            # Initialize AP with precision at different recall thresholds
            ap = 0
            for recall_threshold in np.arange(0, 1.1, 0.1):
                # Find precision at recall >= recall_threshold
                precision_at_recall = [p for r, p in zip(recalls, precisions) if r >= recall_threshold]
                max_precision = max(precision_at_recall) if precision_at_recall else 0
                ap += max_precision / 11
            
            class_metrics[class_id]['AP'] = ap
    
    # Calculate mean metrics across classes
    class_count = len(class_metrics)
    mean_precision = sum(m['precision'] for m in class_metrics.values()) / class_count if class_count > 0 else 0
    mean_recall = sum(m['recall'] for m in class_metrics.values()) / class_count if class_count > 0 else 0
    mean_f1 = sum(m['f1'] for m in class_metrics.values()) / class_count if class_count > 0 else 0
    mean_iou = sum(m['iou_avg'] for m in class_metrics.values()) / class_count if class_count > 0 else 0
    mean_ap = sum(m['AP'] for m in class_metrics.values()) / class_count if class_count > 0 else 0
    
    # Return all metrics
    return {
        'class_metrics': dict(class_metrics),
        'mAP': mean_ap,
        'mIoU': mean_iou,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1
    }

def plot_precision_recall_curves(metrics, output_dir, class_names=None):
    """
    Plot precision-recall curves for each class
    
    Args:
        metrics: Metrics dictionary from evaluate_detections
        output_dir: Directory to save plots
        class_names: Dictionary mapping class IDs to names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve for each class
    for class_id, class_metrics in metrics['class_metrics'].items():
        if 'precisions' in class_metrics and 'recalls' in class_metrics:
            precisions = class_metrics['precisions']
            recalls = class_metrics['recalls']
            
            if len(precisions) > 0:
                class_name = class_names[class_id] if class_names and class_id in class_names else f"Class {class_id}"
                ap = class_metrics['AP']
                plt.plot(recalls, precisions, label=f"{class_name} (AP={ap:.3f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
    plt.close()

def plot_metrics_summary(metrics, output_dir, class_names=None):
    """
    Plot summary of metrics for each class
    
    Args:
        metrics: Metrics dictionary from evaluate_detections
        output_dir: Directory to save plots
        class_names: Dictionary mapping class IDs to names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract class metrics
    class_ids = list(metrics['class_metrics'].keys())
    if not class_ids:
        return
        
    # Convert class IDs to names if provided
    if class_names:
        class_labels = [class_names.get(cid, f"Class {cid}") for cid in class_ids]
    else:
        class_labels = [f"Class {cid}" for cid in class_ids]
    
    # Extract metrics for each class
    precisions = [metrics['class_metrics'][cid]['precision'] for cid in class_ids]
    recalls = [metrics['class_metrics'][cid]['recall'] for cid in class_ids]
    f1_scores = [metrics['class_metrics'][cid]['f1'] for cid in class_ids]
    ious = [metrics['class_metrics'][cid]['iou_avg'] for cid in class_ids]
    aps = [metrics['class_metrics'][cid]['AP'] for cid in class_ids]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(class_labels))
    width = 0.15
    
    ax.bar(x - width*2, precisions, width, label='Precision')
    ax.bar(x - width, recalls, width, label='Recall')
    ax.bar(x, f1_scores, width, label='F1')
    ax.bar(x + width, ious, width, label='IoU')
    ax.bar(x + width*2, aps, width, label='AP')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_class.png'))
    plt.close()
    
    # Create summary metrics table
    plt.figure(figsize=(10, 6))
    plt.axis('tight')
    plt.axis('off')
    
    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['mAP', f"{metrics['mAP']:.4f}"],
        ['mIoU', f"{metrics['mIoU']:.4f}"],
        ['Mean Precision', f"{metrics['mean_precision']:.4f}"],
        ['Mean Recall', f"{metrics['mean_recall']:.4f}"],
        ['Mean F1', f"{metrics['mean_f1']:.4f}"]
    ]
    
    table = plt.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.title('Summary Metrics', fontsize=16, pad=20)
    plt.savefig(os.path.join(output_dir, 'summary_metrics.png'))
    plt.close()

def evaluate_model(model, data_loader, device, output_dir, class_names=None):
    """
    Evaluate model on a dataset and compute metrics
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        output_dir: Directory to save results
        class_names: Dictionary mapping class IDs to names
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    model.eval()
    
    # Store all predictions and ground truth
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Move images to device
            images = list(img.to(device) for img in images)
            
            # Get predictions
            outputs = model(images)
            
            # Convert outputs to CPU for evaluation
            for i, output in enumerate(outputs):
                # Create prediction dictionary
                prediction = {
                    'boxes': output['boxes'].cpu().numpy(),
                    'scores': output['scores'].cpu().numpy(),
                    'labels': output['labels'].cpu().numpy(),
                }
                
                if 'masks' in output:
                    prediction['masks'] = output['masks'].cpu().numpy()
                
                all_predictions.append(prediction)
                
                # Create ground truth dictionary
                target = {
                    'boxes': targets[i]['boxes'].cpu().numpy(),
                    'labels': targets[i]['labels'].cpu().numpy(),
                }
                
                if 'masks' in targets[i]:
                    target['masks'] = targets[i]['masks'].cpu().numpy()
                
                all_targets.append(target)
    
    # Compute metrics
    metrics = evaluate_detections(all_predictions, all_targets)
    
    # Plot results
    os.makedirs(output_dir, exist_ok=True)
    plot_precision_recall_curves(metrics, output_dir, class_names)
    plot_metrics_summary(metrics, output_dir, class_names)
    
    # Save detailed metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Mean Average Precision (mAP): {metrics['mAP']:.4f}\n")
        f.write(f"Mean IoU (mIoU): {metrics['mIoU']:.4f}\n")
        f.write(f"Mean Precision: {metrics['mean_precision']:.4f}\n")
        f.write(f"Mean Recall: {metrics['mean_recall']:.4f}\n")
        f.write(f"Mean F1 Score: {metrics['mean_f1']:.4f}\n\n")
        
        f.write("Per-class metrics:\n")
        for class_id, class_metrics in metrics['class_metrics'].items():
            class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {class_metrics['f1']:.4f}\n")
            f.write(f"  IoU: {class_metrics['iou_avg']:.4f}\n")
            f.write(f"  AP: {class_metrics['AP']:.4f}\n")
            f.write(f"  True Positives: {class_metrics['TP']}\n")
            f.write(f"  False Positives: {class_metrics['FP']}\n")
            f.write(f"  False Negatives: {class_metrics['FN']}\n")
    
    return metrics