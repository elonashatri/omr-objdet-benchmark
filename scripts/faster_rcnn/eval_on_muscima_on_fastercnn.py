import os
import sys
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ONNX model on MUSCIMA dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--image_dir', type=str, 
                        default='/import/c4dm-05/elona/MusicObjectDetector-TF/MusicObjectDetector/images',
                        help='Directory containing test images')
    parser.add_argument('--annotation_dir', type=str,
                        default='/import/c4dm-05/elona/muscima-doremi-annotation',
                        help='Directory containing XML annotations')
    parser.add_argument('--mapping_file', type=str,
                        default='/import/c4dm-05/elona/muscima-doremi-annotation/eval_mapping.json',
                        help='Path to class mapping file')
    parser.add_argument('--num_images', type=int, default=50,
                        help='Number of images to evaluate (0 for all)')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for mAP calculation')
    
    return parser.parse_args()

def load_onnx_model(model_path):
    """Load ONNX model using ONNX Runtime"""
    try:
        session = ort.InferenceSession(model_path)
        print(f"ONNX model loaded from {model_path}")
        
        # Get model input and output names
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        return session, input_name, output_names
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None, None, None

def preprocess_image(image, input_size=(640, 640)):
    """Preprocess image for TensorFlow model in ONNX format (NHWC format)"""
    # Resize image
    resized = cv2.resize(image, input_size)
    
    # Convert to RGB if needed (TensorFlow typically uses RGB)
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    elif resized.shape[2] == 4:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
    
    # Keep as uint8 (0-255) as the model expects
    # Add batch dimension but keep in NHWC format (do NOT transpose)
    # [height, width, channels] -> [1, height, width, channels]
    batched = np.expand_dims(resized, axis=0)
    
    return batched


def run_inference(session, input_name, output_names, image, conf_threshold=0.3):
    """Run inference with ONNX model (TensorFlow Object Detection API format)"""
    # Get original image dimensions
    orig_height, orig_width = image.shape[:2]
    
    # Preprocess image
    input_data = preprocess_image(image)
    
    # Run inference
    outputs = session.run(output_names, {input_name: input_data})
    
    # Process outputs based on TensorFlow Object Detection API format
    # Typical outputs: detection_boxes, detection_scores, detection_classes
    
    # Find the correct output tensors
    boxes = None
    scores = None
    class_ids = None
    
    # Try to identify outputs by name
    for i, name in enumerate(output_names):
        if 'box' in name.lower():
            boxes = outputs[i]
        elif 'score' in name.lower():
            scores = outputs[i]
        elif 'class' in name.lower():
            class_ids = outputs[i]
    
    # If naming doesn't help, try by output shape
    if boxes is None or scores is None or class_ids is None:
        # Assuming first dimension is batch size
        for output in outputs:
            # Boxes typically have shape [batch, num_detections, 4]
            if len(output.shape) == 3 and output.shape[2] == 4:
                boxes = output
            # Scores typically have shape [batch, num_detections]
            elif len(output.shape) == 2:
                scores = output
            # Classes also typically have shape [batch, num_detections]
            elif len(output.shape) == 2 and np.issubdtype(output.dtype, np.number):
                class_ids = output
    
    # If still can't find outputs, use position-based fallback
    if boxes is None or scores is None or class_ids is None:
        print("Warning: Could not identify outputs by name or shape. Using fallback.")
        # Common order in TF Object Detection API: boxes, scores, classes, num_detections
        if len(outputs) >= 3:
            boxes = outputs[0]
            scores = outputs[1]
            class_ids = outputs[2]
    
    # Extract first batch
    if boxes is not None and boxes.shape[0] > 0:
        boxes = boxes[0]
    if scores is not None and scores.shape[0] > 0:
        scores = scores[0]
    if class_ids is not None and class_ids.shape[0] > 0:
        class_ids = class_ids[0]
    
    # Validate outputs
    if boxes is None or scores is None or class_ids is None:
        print("Error: Could not extract detection outputs")
        return [], [], []
    
    # Apply confidence threshold
    mask = scores > conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]
    
    # Convert to int for class IDs
    filtered_class_ids = filtered_class_ids.astype(int)
    
    # TF Object Detection API uses normalized coordinates [y1, x1, y2, x2]
    # Convert to pixel coordinates and reorder to [x1, y1, x2, y2]
    scaled_boxes = []
    for box in filtered_boxes:
        # Check if format is [y1, x1, y2, x2] (TF ODAPI) or [x1, y1, x2, y2]
        if len(box) == 4:
            if len(box.shape) == 2 and box.shape[1] == 4:  # If box has shape [N, 4]
                y1, x1, y2, x2 = box[0]  # Use first set
            else:
                # Assume TF ODAPI format: [y1, x1, y2, x2]
                y1, x1, y2, x2 = box
            
            # Convert normalized coordinates to pixel values
            scaled_boxes.append([
                int(x1 * orig_width),
                int(y1 * orig_height),
                int(x2 * orig_width),
                int(y2 * orig_height)
            ])
    
    print(f"Found {len(scaled_boxes)} detections above threshold {conf_threshold}")
    return scaled_boxes, filtered_scores, filtered_class_ids

def load_xml_annotation(annotation_path, class_mapping=None):
    """Load XML annotation in MUSCIMA++ format"""
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Get image dimensions
        size_elem = root.find('./size')
        if size_elem is not None:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
        else:
            # Default dimensions if not specified
            width, height = 1000, 1000
        
        # Extract objects
        objects = []
        for obj in root.findall('./object'):
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            
            class_name = name_elem.text
            
            # Get class ID from mapping
            class_id = None
            if class_mapping and class_name in class_mapping:
                class_id = class_mapping[class_name]
            
            # Skip if no class ID is available
            if class_id is None:
                continue
            
            # Get bounding box
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            objects.append({
                'class_id': class_id,
                'class_name': class_name,
                'bbox': [xmin, ymin, xmax, ymax]
            })
        
        return objects, width, height
    except Exception as e:
        print(f"Error parsing annotation: {e}")
        return [], 0, 0

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # No intersection
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union
    union = box1_area + box2_area - intersection
    
    return intersection / union

def calculate_map(all_detections, all_ground_truth, iou_threshold=0.5):
    """Calculate mAP over all classes"""
    # Initialize per-class APs
    aps = {}
    
    # Process each class
    for class_name in all_ground_truth.keys():
        if class_name not in all_detections:
            aps[class_name] = 0.0
            continue
        
        # Extract detections and ground truth for this class
        detections = all_detections[class_name]
        ground_truth = all_ground_truth[class_name]
        
        # Sort detections by confidence (descending)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Create list of ground truth objects with 'used' flag
        gt_with_used = []
        for gt in ground_truth:
            gt_with_used.append({
                'bbox': gt['bbox'],
                'used': False
            })
        
        # Initialize precision/recall arrays
        num_gt = len(ground_truth)
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        
        # Process each detection
        for i, detection in enumerate(detections):
            # Find the ground truth with highest IoU
            max_iou = -1
            max_idx = -1
            
            for j, gt in enumerate(gt_with_used):
                if gt['used']:
                    continue
                
                iou = calculate_iou(detection['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            # Check if detection matches a ground truth
            if max_iou >= iou_threshold:
                tp[i] = 1
                gt_with_used[max_idx]['used'] = True
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        if num_gt > 0:
            recall = cumsum_tp / num_gt
        else:
            recall = np.zeros_like(cumsum_tp)
        
        precision = np.zeros_like(cumsum_tp)
        for i in range(len(precision)):
            if cumsum_tp[i] + cumsum_fp[i] > 0:
                precision[i] = cumsum_tp[i] / (cumsum_tp[i] + cumsum_fp[i])
        
        # Calculate AP (area under precision-recall curve)
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        
        aps[class_name] = ap
    
    # Calculate mAP
    if len(aps) > 0:
        map_score = sum(aps.values()) / len(aps)
    else:
        map_score = 0.0
    
    return map_score, aps

def main():
    args = parse_args()
    
    # Load model
    session, input_name, output_names = load_onnx_model(args.model_path)
    if session is None:
        print("Failed to load ONNX model")
        return
    
    # Load class mapping
    class_mapping = {}
    if args.mapping_file and os.path.exists(args.mapping_file):
        import json
        with open(args.mapping_file, 'r') as f:
            class_mapping = json.load(f)
        print(f"Loaded {len(class_mapping)} classes from mapping file")
    
    # Get image paths
    image_paths = []
    for file in os.listdir(args.image_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(args.image_dir, file))
    
    # Limit number of images if specified
    if args.num_images > 0 and args.num_images < len(image_paths):
        image_paths = image_paths[:args.num_images]
    
    print(f"Evaluating {len(image_paths)} images")
    
    # Initialize results storage
    all_detections = defaultdict(list)
    all_ground_truth = defaultdict(list)
    
    # In your main() function, add debug information
    print(f"Looking for annotations in: {args.annotation_dir}")
    print(f"First few files in annotation directory:")
    try:
        for file in list(os.listdir(args.annotation_dir))[:5]:
            print(f"  - {file}")
    except Exception as e:
        print(f"Error listing annotation directory: {e}")

    # Then when processing each image:
    for image_path in tqdm(image_paths):
        # Get base filename for matching annotations
        base_name = os.path.basename(image_path)
        image_name = os.path.splitext(base_name)[0]
        
        print(f"Processing image: {image_name}")
        
        # Find matching annotation file with more detailed debugging
        annotation_path = None
        for file in os.listdir(args.annotation_dir):
            if file.endswith('.xml'):
                # Try exact match first
                if file == f"{image_name}.xml":
                    annotation_path = os.path.join(args.annotation_dir, file)
                    print(f"Found exact annotation match: {file}")
                    break
                # Then try substring matches
                elif image_name in file:
                    annotation_path = os.path.join(args.annotation_dir, file)
                    print(f"Found substring match: {file} for image {image_name}")
                    break
        
        if annotation_path is None:
            print(f"WARNING: No annotation found for {base_name} - skipping this image")
            continue
        
        # Continue with the rest of your processing...
        # Load ground truth annotations
        gt_objects, img_width, img_height = load_xml_annotation(annotation_path, class_mapping)
        
        # Run inference
        boxes, scores, class_ids = run_inference(
            session, input_name, output_names, image, args.conf_threshold
        )
        
        # Store ground truth
        for gt in gt_objects:
            class_name = gt['class_name']
            all_ground_truth[class_name].append({
                'image': base_name,
                'bbox': gt['bbox']
            })
        
        # Store detections
        for i in range(len(boxes)):
            class_id = int(class_ids[i])
            class_name = f"class_{class_id}"  # Use class mapping if available
            
            # Find class name from mapping
            for name, id in class_mapping.items():
                if id == class_id:
                    class_name = name
                    break
            
            all_detections[class_name].append({
                'image': base_name,
                'bbox': boxes[i],
                'confidence': float(scores[i])
            })
    
    # Calculate mAP
    map_score, class_aps = calculate_map(
        all_detections, all_ground_truth, args.iou_threshold
    )
    
    # Print results
    print(f"\nResults for {os.path.basename(args.model_path)}:")
    print(f"mAP@{args.iou_threshold}: {map_score:.4f}")
    print("\nPer-class AP:")
    for class_name, ap in sorted(class_aps.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {ap:.4f}")

if __name__ == "__main__":
    main()
    
    
    
    """
    python /homes/es314/omr-objdet-benchmark/scripts/faster_rcnn/eval_on_muscima_on_fastercnn.py \
        --model_path "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex003/may_2023_ex003.onnx" \
        --image_dir "/import/c4dm-05/elona/MusicObjectDetector-TF/MusicObjectDetector/images" \
        --annotation_dir "/import/c4dm-05/elona/muscima-doremi-annotation" \
        --mapping_file "/import/c4dm-05/elona/muscima-doremi-annotation/eval_mapping.json" \
        --num_images 50 \
        --conf_threshold 0.3 \
        --iou_threshold 0.5
        
        
        need to convert the mapping ----- since it has a different set of classes from doremi v1 and doremi v2
    """