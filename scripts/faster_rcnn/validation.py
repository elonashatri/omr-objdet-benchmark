import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omr_dataset import OMRDataset, get_transform
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
import json
import argparse

def load_class_names(mapping_file):
    """Load class names from mapping file"""
    class_names = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) >= 2:
                class_id = int(parts[0])
                class_name = ':'.join(parts[1:])
                class_names[class_id] = class_name
    return class_names

def calculate_map(pred_boxes, pred_labels, pred_scores, 
                  gt_boxes, gt_labels, num_classes, iou_threshold=0.5):
    """
    Calculate mAP for a single image
    """
    # Initialize precision array for each class
    ap_per_class = np.zeros(num_classes)
    
    # Process each class
    for class_id in range(1, num_classes):  # Skip background class (0)
        # Find predictions for this class
        class_pred_indices = np.where(pred_labels == class_id)[0]
        if len(class_pred_indices) == 0:
            continue  # No predictions for this class
            
        # Get predictions for this class
        class_pred_boxes = pred_boxes[class_pred_indices]
        class_pred_scores = pred_scores[class_pred_indices]
        
        # Sort by confidence
        sorted_indices = np.argsort(-class_pred_scores)
        class_pred_boxes = class_pred_boxes[sorted_indices]
        
        # Find ground truth for this class
        class_gt_indices = np.where(gt_labels == class_id)[0]
        if len(class_gt_indices) == 0:
            continue  # No ground truth for this class
            
        # Get ground truth for this class
        class_gt_boxes = gt_boxes[class_gt_indices]
        
        # Convert boxes to tensors for IoU calculation
        if len(class_pred_boxes) > 0 and len(class_gt_boxes) > 0:
            class_pred_boxes_tensor = torch.tensor(class_pred_boxes)
            class_gt_boxes_tensor = torch.tensor(class_gt_boxes)
            
            # Calculate IoU between predicted and ground truth boxes
            iou_matrix = box_iou(class_pred_boxes_tensor, class_gt_boxes_tensor)
            
            # For each prediction, find best matching ground truth
            gt_matched = np.zeros(len(class_gt_boxes))
            
            # Calculate precision and recall points
            tp = np.zeros(len(class_pred_boxes))
            fp = np.zeros(len(class_pred_boxes))
            
            for pred_idx in range(len(class_pred_boxes)):
                # Find best matching ground truth
                if len(class_gt_boxes) > 0:
                    max_iou = np.max(iou_matrix[pred_idx].numpy())
                    max_idx = np.argmax(iou_matrix[pred_idx].numpy())
                    
                    if max_iou >= iou_threshold and gt_matched[max_idx] == 0:
                        tp[pred_idx] = 1
                        gt_matched[max_idx] = 1
                    else:
                        fp[pred_idx] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / (len(class_gt_boxes) + 1e-6)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            # Append sentinel values
            precisions = np.concatenate(([0], precisions, [0]))
            recalls = np.concatenate(([0], recalls, [1]))
            
            # Compute AP using 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11
            
            ap_per_class[class_id] = ap
    
    # Calculate mAP (mean of AP for all classes)
    valid_classes = ap_per_class > 0
    if np.sum(valid_classes) > 0:
        mAP = np.mean(ap_per_class[valid_classes])
    else:
        mAP = 0
        
    return mAP, ap_per_class

def validate_checkpoint(checkpoint_path, val_loader, device, num_classes, class_names):
    """
    Validate a single checkpoint with comprehensive metrics
    """
    # Load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Validation metrics
    val_loss = 0
    val_loss_classifier = 0
    val_loss_box_reg = 0
    val_loss_objectness = 0
    val_loss_rpn_box_reg = 0
    
    # For mAP calculation
    all_predictions = []
    all_ground_truths = []
    
    # For confusion matrix (predicted vs actual class)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # For object size analysis
    size_categories = {
        'small': {'count': 0, 'correct': 0},  # area < 32²
        'medium': {'count': 0, 'correct': 0}, # 32² <= area < 96²
        'large': {'count': 0, 'correct': 0}   # area >= 96²
    }
    
    # Run validation
    num_samples = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader, desc=f"Validating {os.path.basename(checkpoint_path)}")):
            try:
                # Handle data format 
                images = []
                targets = []
                
                # Assuming data is in format returned by DataLoader with collate_fn
                for img, boxes, labels, img_id in zip(data[0], data[1], data[2], data[3]):
                    if isinstance(img, str):
                        continue
                        
                    img_tensor = img.to(device)
                    images.append(img_tensor)
                    
                    target = {
                        'boxes': boxes.to(device),
                        'labels': labels.to(device),
                        'image_id': img_id.to(device)
                    }
                    targets.append(target)
                
                if not images:
                    continue
                    
                # Calculate loss
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                
                # Update component losses
                val_loss_classifier += loss_dict['loss_classifier'].item() if 'loss_classifier' in loss_dict else 0
                val_loss_box_reg += loss_dict['loss_box_reg'].item() if 'loss_box_reg' in loss_dict else 0
                val_loss_objectness += loss_dict['loss_objectness'].item() if 'loss_objectness' in loss_dict else 0
                val_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item() if 'loss_rpn_box_reg' in loss_dict else 0
                
                # Get predictions
                predictions = model(images)
                
                # Process each image
                for img_idx, (prediction, target) in enumerate(zip(predictions, targets)):
                    # Store prediction and ground truth for mAP calculation
                    pred_info = {
                        'boxes': prediction['boxes'].cpu().numpy(),
                        'labels': prediction['labels'].cpu().numpy(),
                        'scores': prediction['scores'].cpu().numpy()
                    }
                    
                    gt_info = {
                        'boxes': target['boxes'].cpu().numpy(),
                        'labels': target['labels'].cpu().numpy()
                    }
                    
                    all_predictions.append(pred_info)
                    all_ground_truths.append(gt_info)
                    
                    # Update confusion matrix for high-confidence predictions
                    for pred_idx, (box, label, score) in enumerate(zip(
                            prediction['boxes'].cpu().numpy(),
                            prediction['labels'].cpu().numpy(),
                            prediction['scores'].cpu().numpy())):
                        
                        if score >= 0.25:  # Consider only confident predictions
                            # Find best matching ground truth
                            best_match_idx = -1
                            best_match_iou = 0.5  # IoU threshold
                            
                            for gt_idx, gt_box in enumerate(target['boxes'].cpu().numpy()):
                                # Calculate IoU
                                iou = box_iou(
                                    torch.tensor([box]), 
                                    torch.tensor([gt_box])
                                )[0, 0].item()
                                
                                if iou > best_match_iou:
                                    best_match_iou = iou
                                    best_match_idx = gt_idx
                            
                            # Update confusion matrix
                            if best_match_idx != -1:
                                gt_label = target['labels'][best_match_idx].item()
                                confusion_matrix[gt_label, label] += 1
                    
                    # Object size analysis for ground truth objects
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(
                            target['boxes'].cpu().numpy(),
                            target['labels'].cpu().numpy())):
                        
                        # Calculate box area
                        x1, y1, x2, y2 = gt_box
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Determine size category
                        if area < 32*32:
                            size_category = 'small'
                        elif area < 96*96:
                            size_category = 'medium'
                        else:
                            size_category = 'large'
                        
                        size_categories[size_category]['count'] += 1
                        
                        # Check if this ground truth was correctly detected
                        for pred_idx, (box, label, score) in enumerate(zip(
                                prediction['boxes'].cpu().numpy(),
                                prediction['labels'].cpu().numpy(),
                                prediction['scores'].cpu().numpy())):
                            
                            if score >= 0.25 and label == gt_label:
                                # Calculate IoU
                                iou = box_iou(
                                    torch.tensor([box]), 
                                    torch.tensor([gt_box])
                                )[0, 0].item()
                                
                                if iou >= 0.5:
                                    size_categories[size_category]['correct'] += 1
                                    break
                
                num_samples += len(images)
                
            except Exception as e:
                print(f"Error in validation batch {i}: {e}")
                continue
    
    # Calculate average losses
    if num_samples > 0:
        val_loss /= len(val_loader)
        val_loss_classifier /= len(val_loader)
        val_loss_box_reg /= len(val_loader)
        val_loss_objectness /= len(val_loader)
        val_loss_rpn_box_reg /= len(val_loader)
    
    # Calculate mAP using all validation samples
    iou_thresholds = [0.5, 0.75]
    map_metrics = calculate_map_coco(all_predictions, all_ground_truths, num_classes, iou_thresholds)
    
    # Calculate class-specific AP
    class_ap = map_metrics['AP_per_class'][0.5]  # Use IoU=0.5 for class AP
    
    # Calculate accuracy by object size
    size_accuracy = {}
    for size, stats in size_categories.items():
        if stats['count'] > 0:
            size_accuracy[size] = stats['correct'] / stats['count']
        else:
            size_accuracy[size] = 0
    
    # Return all metrics
    return {
        'loss': val_loss,
        'loss_classifier': val_loss_classifier,
        'loss_box_reg': val_loss_box_reg,
        'loss_objectness': val_loss_objectness,
        'loss_rpn_box_reg': val_loss_rpn_box_reg,
        'mAP': map_metrics['mAP'],
        'precision': map_metrics['precision'],
        'recall': map_metrics['recall'],
        'f1': map_metrics['f1'],
        'class_ap': class_ap,
        'size_accuracy': size_accuracy,
        'confusion_matrix': confusion_matrix
    }

def calculate_map_coco(predictions, ground_truths, num_classes, iou_thresholds=[0.5]):
    """
    Calculate mAP and other metrics following COCO evaluation protocol
    """
    metrics = {
        'mAP': {},
        'precision': {},
        'recall': {},
        'f1': {},
        'AP_per_class': {}
    }
    
    # Initialize metrics for each IoU threshold
    for threshold in iou_thresholds:
        metrics['mAP'][threshold] = 0
        metrics['precision'][threshold] = 0
        metrics['recall'][threshold] = 0
        metrics['f1'][threshold] = 0
        metrics['AP_per_class'][threshold] = np.zeros(num_classes)
    
    # Process all images
    all_tps = {threshold: 0 for threshold in iou_thresholds}
    all_fps = {threshold: 0 for threshold in iou_thresholds}
    all_fns = {threshold: 0 for threshold in iou_thresholds}
    
    # Collect all predictions and ground truths by class
    class_predictions = {c: [] for c in range(1, num_classes)}  # Skip background class
    class_gt_counts = {c: 0 for c in range(1, num_classes)}
    
    # Process each image
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        
        # Count ground truths for each class
        for label in gt_labels:
            if label > 0:  # Skip background
                class_gt_counts[label] += 1
        
        # Store predictions by class
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            if label > 0:  # Skip background
                class_predictions[label].append({'box': box, 'score': score, 'matched': {t: False for t in iou_thresholds}})
        
        # Match predictions to ground truths
        for threshold in iou_thresholds:
            # Create array to track which ground truths have been matched
            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            
            # Sort predictions by confidence
            sorted_indices = np.argsort(-pred_scores)
            sorted_boxes = pred_boxes[sorted_indices]
            sorted_labels = pred_labels[sorted_indices]
            
            # Count TP, FP
            for pred_idx, (box, label) in enumerate(zip(sorted_boxes, sorted_labels)):
                if label == 0:  # Skip background
                    continue
                    
                # Find matching ground truth (same class, best IoU)
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_label != label or gt_matched[gt_idx]:
                        continue
                        
                    # Calculate IoU
                    iou = box_iou(
                        torch.tensor([box]), 
                        torch.tensor([gt_box])
                    )[0, 0].item()
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Check if we found a match
                if best_gt_idx != -1 and best_iou >= threshold:
                    gt_matched[best_gt_idx] = True
                    all_tps[threshold] += 1
                    
                    # Mark this prediction as matched for this threshold
                    pred_idx_in_class = class_predictions[label].index({'box': box, 'score': pred_scores[sorted_indices[pred_idx]], 
                                                                       'matched': {t: False for t in iou_thresholds}})
                    class_predictions[label][pred_idx_in_class]['matched'][threshold] = True
                else:
                    all_fps[threshold] += 1
            
            # Count FN (unmatched ground truths)
            all_fns[threshold] += np.sum(~gt_matched)
    
    # Calculate AP for each class at each threshold
    for threshold in iou_thresholds:
        for class_id in range(1, num_classes):
            # Get predictions for this class
            predictions = class_predictions[class_id]
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['score'], reverse=True)
            
            # Calculate precision and recall points
            tp = np.zeros(len(predictions))
            fp = np.zeros(len(predictions))
            
            for i, pred in enumerate(predictions):
                if pred['matched'][threshold]:
                    tp[i] = 1
                else:
                    fp[i] = 1
            
            # Compute cumulative values
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            # Calculate precision and recall
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            recall = tp_cumsum / (class_gt_counts[class_id] + 1e-10)
            
            # Append sentinel values
            precision = np.concatenate(([1], precision, [0]))
            recall = np.concatenate(([0], recall, [1]))
            
            # Ensure precision is decreasing
            for i in range(precision.size - 1, 0, -1):
                precision[i - 1] = max(precision[i - 1], precision[i])
            
            # Compute AP as the area under the PR curve
            indices = np.where(recall[1:] != recall[:-1])[0]
            ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
            
            metrics['AP_per_class'][threshold][class_id] = ap
    
    # Calculate mAP (average AP across classes)
    for threshold in iou_thresholds:
        valid_classes = metrics['AP_per_class'][threshold][1:] >= 0
        if np.sum(valid_classes) > 0:
            metrics['mAP'][threshold] = np.mean(metrics['AP_per_class'][threshold][1:][valid_classes])
        else:
            metrics['mAP'][threshold] = 0
    
    # Calculate global precision, recall, F1
    for threshold in iou_thresholds:
        if all_tps[threshold] + all_fps[threshold] > 0:
            metrics['precision'][threshold] = all_tps[threshold] / (all_tps[threshold] + all_fps[threshold])
        else:
            metrics['precision'][threshold] = 0
            
        if all_tps[threshold] + all_fns[threshold] > 0:
            metrics['recall'][threshold] = all_tps[threshold] / (all_tps[threshold] + all_fns[threshold])
        else:
            metrics['recall'][threshold] = 0
            
        if metrics['precision'][threshold] + metrics['recall'][threshold] > 0:
            metrics['f1'][threshold] = (
                2 * metrics['precision'][threshold] * metrics['recall'][threshold] / 
                (metrics['precision'][threshold] + metrics['recall'][threshold])
            )
        else:
            metrics['f1'][threshold] = 0
    
    return metrics

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot confusion matrix for most confused classes
    """
    # Get top confused classes (off-diagonal elements)
    n_classes = len(class_names)
    confusion_scores = []
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusion_scores.append((i, j, cm[i, j]))
    
    # Sort by confusion count
    confusion_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Take top 20 confusions
    top_confusions = confusion_scores[:20]
    
    if len(top_confusions) == 0:
        print("No confusion found between classes")
        return
    
    # Create labels for confusion pairs
    labels = [f"{class_names.get(true, f'Class {true}')} → {class_names.get(pred, f'Class {pred}')}" 
              for true, pred, _ in top_confusions]
    
    values = [count for _, _, count in top_confusions]
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(labels, values)
    plt.xlabel('Count')
    plt.title('Top Class Confusions')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_class_ap(class_ap, class_names, output_path, top_n=20):
    """
    Plot AP for top and bottom classes
    """
    # Create dictionary of class name to AP
    class_ap_dict = {class_names.get(i, f"Class {i}"): ap for i, ap in enumerate(class_ap) if i > 0}
    
    # Sort by AP
    sorted_classes = sorted(class_ap_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N and bottom N classes
    top_classes = sorted_classes[:top_n]
    bottom_classes = sorted_classes[-top_n:]
    
    # Plot top classes
    plt.figure(figsize=(12, 8))
    classes, aps = zip(*top_classes)
    plt.barh(classes, aps)
    plt.xlabel('Average Precision')
    plt.title(f'Top {top_n} Classes by AP')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_top.png'))
    plt.close()
    
    # Plot bottom classes
    plt.figure(figsize=(12, 8))
    classes, aps = zip(*bottom_classes)
    plt.barh(classes, aps)
    plt.xlabel('Average Precision')
    plt.title(f'Bottom {top_n} Classes by AP')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_bottom.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Validate Faster R-CNN checkpoints')
    parser.add_argument('--checkpoints_dir', type=str, required=True,
                        help='Directory containing checkpoint files')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                        help='Directory to save validation results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load validation dataset
    val_dir = os.path.join(args.data_dir, 'val')
    val_dataset = OMRDataset(
        root_dir=val_dir,
        transforms=get_transform(train=False),
        is_train=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Load class names
    mapping_file = os.path.join(args.data_dir, 'mapping.txt')
    class_names = load_class_names(mapping_file)
    num_classes = len(class_names) + 1  # Add background class
    
    # Find all checkpoint files
    checkpoint_files = [os.path.join(args.checkpoints_dir, f) for f in os.listdir(args.checkpoints_dir)
                       if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Add 'best.pth' and 'latest.pth' if they exist
    best_path = os.path.join(args.checkpoints_dir, 'best.pth')
    if os.path.exists(best_path):
        checkpoint_files.append(best_path)
    
    latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
    if os.path.exists(latest_path):
        checkpoint_files.append(latest_path)
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Validate each checkpoint
    results = {}
    for checkpoint_path in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"Validating {checkpoint_name}...")
        
        metrics = validate_checkpoint(checkpoint_path, val_loader, device, num_classes, class_names)
        results[checkpoint_name] = metrics
        
        print(f"  Loss: {metrics['loss']:.4f}, mAP@0.5: {metrics['mAP'][0.5]:.4f}")
        
        # Generate detailed reports for the latest epoch
        if checkpoint_name.startswith('checkpoint_epoch_') and checkpoint_name == checkpoint_files[-1]:
            # Plot confusion matrix
            print("Generating confusion matrix...")
            plot_confusion_matrix(metrics['confusion_matrix'], class_names, 
                                 os.path.join(args.output_dir, 'confusion_matrix.png'))
            
            # Plot class AP
            print("Generating class AP plots...")
            plot_class_ap(metrics['class_ap'], class_names,
                         os.path.join(args.output_dir, 'class_ap.png'))
            
            # Print size accuracy
            print("\nAccuracy by Object Size:")
            for size, acc in metrics['size_accuracy'].items():
                print(f"  {size.capitalize()}: {acc:.4f}")
    
    # Save results
    with open(os.path.join(args.output_dir, 'validation_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for checkpoint, metrics in results.items():
            serializable_results[checkpoint] = {}
            for k, v in metrics.items():
                if k == 'confusion_matrix':
                    serializable_results[checkpoint][k] = v.tolist()
                elif k == 'class_ap':
                    serializable_results[checkpoint][k] = v.tolist()
                else:
                    serializable_results[checkpoint][k] = v
        
        json.dump(serializable_results, f, indent=4)
    
    # Extract metrics for plotting
    epochs = []
    losses = []
    component_losses = {'classifier': [], 'box_reg': [], 'objectness': [], 'rpn_box_reg': []}
    maps = []
    
    # Extract metrics from results
    for checkpoint_name, metrics in results.items():
        if checkpoint_name.startswith('checkpoint_epoch_'):
            epoch = int(checkpoint_name.split('_')[-1].split('.')[0])
            epochs.append(epoch)
            losses.append(metrics['loss'])
            
            # Component losses
            component_losses['classifier'].append(metrics['loss_classifier'])
            component_losses['box_reg'].append(metrics['loss_box_reg'])
            component_losses['objectness'].append(metrics['loss_objectness'])
            component_losses['rpn_box_reg'].append(metrics['loss_rpn_box_reg'])
            
            maps.append({
                'mAP': metrics['mAP'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            })
    
    # Sort by epoch
    sorted_indices = np.argsort(epochs)
    epochs = [epochs[i] for i in sorted_indices]
    losses = [losses[i] for i in sorted_indices]
    
    for component in component_losses:
        component_losses[component] = [component_losses[component][i] for i in sorted_indices]
    
    maps = [maps[i] for i in sorted_indices]
    
    # Create plots
    # Main metrics
    plt.figure(figsize=(15, 12))
    
    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses, 'b-o', label='Validation Loss')
    plt.title('Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot component losses
    plt.subplot(2, 2, 2)
    for component, values in component_losses.items():
        plt.plot(epochs, values, 'o-', label=f'Loss {component}')
    plt.title('Component Losses per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot mAP
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['mAP'][0.5] for m in maps], 'r-o', label='mAP@0.5')
    if 0.75 in maps[0]['mAP']:
        plt.plot(epochs, [m['mAP'][0.75] for m in maps], 'g-o', label='mAP@0.75')
    plt.title('Mean Average Precision per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.legend()
    
    # Plot Precision/Recall/F1
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [m['precision'][0.5] for m in maps], 'b-o', label='Precision@0.5')
    plt.plot(epochs, [m['recall'][0.5] for m in maps], 'r-o', label='Recall@0.5')
    plt.plot(epochs, [m['f1'][0.5] for m in maps], 'g-o', label='F1@0.5')
    plt.title('Precision, Recall, F1 per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'validation_metrics.png'))
    
    # Also save as CSV for easier analysis
    with open(os.path.join(args.output_dir, 'validation_metrics.csv'), 'w') as f:
        # Write header
        header = ['epoch', 'loss', 'loss_classifier', 'loss_box_reg', 
                  'loss_objectness', 'loss_rpn_box_reg', 'mAP@0.5']
        if 0.75 in maps[0]['mAP']:
            header.append('mAP@0.75')
        header.extend(['precision@0.5', 'recall@0.5', 'f1@0.5'])
        f.write(','.join(header) + '\n')
        
        # Write data
        for i, epoch in enumerate(epochs):
            row = [
                str(epoch),
                str(losses[i]),
                str(component_losses['classifier'][i]),
                str(component_losses['box_reg'][i]),
                str(component_losses['objectness'][i]),
                str(component_losses['rpn_box_reg'][i]),
                str(maps[i]['mAP'][0.5])
            ]
            if 0.75 in maps[0]['mAP']:
                row.append(str(maps[i]['mAP'][0.75]))
            row.extend([
                str(maps[i]['precision'][0.5]),
                str(maps[i]['recall'][0.5]),
                str(maps[i]['f1'][0.5])
            ])
            f.write(','.join(row) + '\n')
    
    print(f"Validation results saved to {args.output_dir}")
    
    # Find best checkpoint by mAP
    if maps:
        best_index = np.argmax([m['mAP'][0.5] for m in maps])
        best_epoch = epochs[best_index]
        best_map = maps[best_index]['mAP'][0.5]
        print(f"Best checkpoint by mAP: epoch {best_epoch} with mAP {best_map:.4f}")

if __name__ == "__main__":
    main()