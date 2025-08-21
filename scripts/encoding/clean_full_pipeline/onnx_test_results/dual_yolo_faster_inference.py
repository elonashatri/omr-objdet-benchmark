import os
import cv2
import json
import torch
import numpy as np
import onnxruntime
from PIL import Image
import matplotlib.pyplot as plt


def load_class_mapping(mapping_path):
    """
    Load class mappings from JSON file.
    Can handle both id->name and name->id formats.
    Returns both id_to_name and name_to_id dictionaries.
    """
    import json
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Check which format the mapping is in
    first_key = next(iter(mapping))
    try:
        # If we can convert the first key to int, it's likely id->name format
        int(first_key)
        id_to_name = {int(k): v for k, v in mapping.items()}
        name_to_id = {v: int(k) for k, v in mapping.items()}
        print(f"Loaded mapping in id->name format with {len(id_to_name)} entries")
    except ValueError:
        # If we can't, it's likely name->id format
        name_to_id = {k: int(v) if isinstance(v, str) else v for k, v in mapping.items()}
        id_to_name = {v: k for k, v in name_to_id.items()}
        print(f"Loaded mapping in name->id format with {len(name_to_id)} entries")
    
    return name_to_id, id_to_name



class FasterRCNNOnnxDetector:
    def __init__(self, model_path, id_to_name, conf_threshold=0.2, max_detections=1000, iou_threshold=0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.iou_threshold = iou_threshold
        self.id_to_name = id_to_name

        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Faster R-CNN input name: {self.input_name}")
        print(f"Faster R-CNN output names: {self.output_names}")

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        return np.expand_dims(image_np, axis=0), image.size

    def nms(self, boxes, scores):
        # Use OpenCV NMS instead of PyTorch for compatibility
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold)
        
        if len(indices) > 0:
            if isinstance(indices, tuple):  # For older OpenCV versions
                indices = indices[0]
            return indices
        return []

    def detect(self, image_path):
        image_np, (width, height) = self.preprocess_image(image_path)
        print(f"Original image size: {width}x{height}")
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: image_np})
        
        print(f"Faster R-CNN outputs length: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Output {i} shape: {output.shape}")
        
        # Parse outputs (adjust based on actual model output format)
        boxes = outputs[0][0]
        scores = outputs[1][0]
        class_ids = outputs[2][0]
        
        # Print first few detections
        print("First 5 Faster R-CNN detections:")
        for i in range(min(5, len(scores))):
            print(f"  Detection {i}: box={boxes[i]}, score={scores[i]}, class={int(class_ids[i])}")
        
        # Filter by confidence
        valid_idx = np.where(scores >= self.conf_threshold)[0]
        boxes = boxes[valid_idx]
        scores = scores[valid_idx]
        class_ids = class_ids[valid_idx]
        
        # Convert normalized coordinates to pixel coordinates AND fix the order
        # Faster R-CNN outputs [y1, x1, y2, x2] format normalized (0-1)
        absolute_boxes = []
        for box in boxes:
            y1, x1, y2, x2 = box
            # Convert to absolute coordinates
            x1_abs = int(x1 * width)
            y1_abs = int(y1 * height)
            x2_abs = int(x2 * width)
            y2_abs = int(y2 * height)
            # Store in [x1, y1, x2, y2] format
            absolute_boxes.append([x1_abs, y1_abs, x2_abs, y2_abs])
        
        absolute_boxes = np.array(absolute_boxes)
        
        # Apply NMS
        if len(absolute_boxes) > 0:
            keep = self.nms(absolute_boxes, scores)
            return absolute_boxes[keep], scores[keep], class_ids[keep]
        
        return np.array([]), np.array([]), np.array([])


class YOLOOnnxDetector:
    def __init__(self, model_path, id_to_name, conf_threshold=0.01, iou_threshold=0.45, img_size=1280):
        self.model_path = model_path
        self.conf_threshold = conf_threshold  # Even lower threshold for debugging
        self.iou_threshold = iou_threshold
        self.id_to_name = id_to_name
        
        # Create ONNX Runtime session
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"YOLOv8 input name: {self.input_name}")
        print(f"YOLOv8 output names: {self.output_names}")
        
        # Get input shape info
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"YOLOv8 model input shape: {self.input_shape}")
        
        # For dynamic dimensions, use the provided img_size parameter
        self.input_height = img_size
        self.input_width = img_size
        
        print(f"Using input dimensions: {self.input_width}x{self.input_height}")

    def preprocess_image(self, image_path):
        original_image = Image.open(image_path).convert("RGB")
        original_width, original_height = original_image.size
        print(f"Original image size for YOLOv8: {original_width}x{original_height}")
        
        # Resize while maintaining aspect ratio
        input_height, input_width = self.input_height, self.input_width
        scale = min(input_width / original_width, input_height / original_height)
        resized_height = int(original_height * scale)
        resized_width = int(original_width * scale)
        
        print(f"Resized to: {resized_width}x{resized_height} (scale: {scale})")
        
        # Create image with padding
        resized_image = original_image.resize((resized_width, resized_height), Image.LANCZOS)
        new_image = Image.new("RGB", (input_width, input_height), (114, 114, 114))
        new_image.paste(resized_image, ((input_width - resized_width) // 2, 
                                        (input_height - resized_height) // 2))
        
        # Calculate padding
        pad_w = (input_width - resized_width) // 2
        pad_h = (input_height - resized_height) // 2
        print(f"Padding: left/right={pad_w}, top/bottom={pad_h}")
        
        # Convert to numpy and normalize
        input_image = np.array(new_image).astype(np.float32) / 255.0
        
        # Change HWC to CHW format
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, 0)
        
        return input_image, original_width, original_height, scale, pad_w, pad_h

    def detect(self, image_path):
        # Preprocess image
        image_tensor, orig_w, orig_h, scale, pad_w, pad_h = self.preprocess_image(image_path)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: image_tensor})
        
        print(f"YOLOv8 outputs length: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Output {i} shape: {output.shape}")
        
        output = outputs[0]
        print(f"YOLOv8 output shape: {output.shape}")
        
        # Print some statistics about the output
        print(f"YOLOv8 output min value: {np.min(output)}, max value: {np.max(output)}")
        
        # YOLO output format: [batch, boxes, features]
        # features: [x, y, w, h, confidence, class_prob1, class_prob2, ...]
        
        # Get boxes from model output
        boxes = []
        scores = []
        class_ids = []
        
        # For YOLOv8, output is typically (1, 84+5, num_boxes)
        # Transpose to (num_boxes, 84+5)
        predictions = output[0].transpose(1, 0)
        
        # Get number of classes from output shape
        num_classes = predictions.shape[1] - 5
        print(f"Number of classes in YOLOv8 output: {num_classes}")
        
        # For verification and understanding the model's output, examine the first few predictions
        max_conf_index = np.argmax(predictions[:, 4])  # Index of box with highest objectness
        print(f"Box with highest objectness score (index {max_conf_index}):")
        print(f"  Coordinates: {predictions[max_conf_index, :4]}")
        print(f"  Objectness: {predictions[max_conf_index, 4]}")
        print(f"  Class probabilities (top 3): {np.argsort(predictions[max_conf_index, 5:])[-3:]}")
        
        # Process each box
        for i in range(len(predictions)):
            # Get objectness score
            obj_conf = predictions[i, 4]
            
            # Skip if objectness score is too low
            if obj_conf < self.conf_threshold:
                continue
            
            # Get class probabilities and best class
            class_probs = predictions[i, 5:5+num_classes]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]
            
            # Final confidence score
            confidence = obj_conf * class_score
            
            if confidence < self.conf_threshold:
                continue
            
            # Get box coordinates (normalized [cx, cy, w, h])
            cx, cy, w, h = predictions[i, :4]
            
            # Convert to corner format (normalized [x1, y1, x2, y2])
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            
            # Convert normalized coords to pixels in the input space
            x1 = x1 * self.input_width
            y1 = y1 * self.input_height
            x2 = x2 * self.input_width
            y2 = y2 * self.input_height
            
            # Remove padding
            x1 -= pad_w
            y1 -= pad_h
            x2 -= pad_w
            y2 -= pad_h
            
            # Scale coordinates to original image
            x1 /= scale
            y1 /= scale
            x2 /= scale
            y2 /= scale
            
            # Clip to image boundaries
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            # Skip invalid boxes
            if x1 >= x2 or y1 >= y2:
                continue
                
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(confidence))
            class_ids.append(int(class_id))
            
        print(f"Found {len(boxes)} valid detections after filtering")
        
        # Apply NMS if we have any valid boxes
        if len(boxes) > 0:
            boxes_np = np.array(boxes)
            scores_np = np.array(scores)
            class_ids_np = np.array(class_ids)
            
            # Apply NMS for each class
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.conf_threshold, self.iou_threshold)
            
            if len(indices) > 0:
                if isinstance(indices, tuple):  # For older OpenCV versions
                    indices = indices[0]
                
                print(f"After NMS: {len(indices)} detections")
                return boxes_np[indices], scores_np[indices], class_ids_np[indices]
        
        return np.array([]), np.array([]), np.array([])


def visualize(image_path, boxes, scores, class_ids, id_to_name, output_path, thickness=2, debug_mode=True):
    print(f"Creating visualization for {len(boxes)} detections in {output_path}")
    
    # Load image
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Get image dimensions for debugging
        h, w, c = image.shape
        print(f"Image dimensions: {w}x{h}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create a color map for different classes
    np.random.seed(42)  # For reproducible colors
    max_class_id = max(list(id_to_name.keys())) if id_to_name else 100
    colors = np.random.randint(0, 255, size=(max_class_id+1, 3), dtype=np.uint8).tolist()
    
    # Debug box values
    if debug_mode and len(boxes) > 0:
        print("Sample of boxes to be drawn:")
        for i in range(min(5, len(boxes))):
            try:
                box = boxes[i]
                score = scores[i]
                cls_id = int(class_ids[i])
                class_name = id_to_name.get(cls_id, f"Unknown ({cls_id})")
                print(f"Box {i}: {box}, score: {score:.4f}, class: {class_name}")
            except Exception as e:
                print(f"Error processing box {i}: {e}")
    
    # Draw boxes
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        try:
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, [max(0, b) for b in box])
            
            # Skip boxes with invalid dimensions
            if x1 >= x2 or y1 >= y2 or x1 >= w or y1 >= h:
                print(f"Skipping invalid box {i}: {(x1, y1, x2, y2)}")
                continue
            
            # Cap box dimensions to image size
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            # Get class name from ID
            cls_id = int(cls_id)
            class_name = id_to_name.get(cls_id, f"Unknown ({cls_id})")
            label = f"{class_name}: {score:.2f}"
            
            # Get color for this class
            color = tuple(map(int, colors[cls_id % len(colors)]))
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw filled background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
            cv2.rectangle(image, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
            
            # Draw text
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness//2)
        except Exception as e:
            print(f"Error drawing box {i}: {e}")
    
    # Add detection count
    count_text = f"Detected objects: {len(boxes)}"
    cv2.putText(image, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Save image
    try:
        result = cv2.imwrite(output_path, image)
        if result:
            print(f"Visualization saved to {output_path}")
        else:
            print(f"Error: Failed to save visualization to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Faster R-CNN and YOLOv8 ONNX models")
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--fasterrcnn_model', required=True, help='Path to Faster R-CNN ONNX model')
    parser.add_argument('--yolo_model', required=True, help='Path to YOLOv8 ONNX model')
    parser.add_argument('--frcnn_mapping', required=True, help='Path to Faster R-CNN class mapping JSON file')
    parser.add_argument('--yolo_mapping', required=True, help='Path to YOLO class mapping JSON file') 
    parser.add_argument('--output_dir', default="results", help='Directory to save output visualizations')
    parser.add_argument('--conf_threshold', type=float, default=0.01, help='Confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--img_size', type=int, default=1280, help='Input image size for YOLO model')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load class mappings separately for each model
    frcnn_name_to_id, frcnn_id_to_name = load_class_mapping(args.frcnn_mapping)
    yolo_name_to_id, yolo_id_to_name = load_class_mapping(args.yolo_mapping)
    
    # Print sample of class mappings to verify
    print("Sample of Faster R-CNN class mappings:")
    sample_keys = list(frcnn_id_to_name.keys())[:5]  # First 5 keys
    for key in sample_keys:
        print(f"  ID {key}: {frcnn_id_to_name[key]}")
        
    print("Sample of YOLO class mappings:")
    sample_keys = list(yolo_id_to_name.keys())[:5]  # First 5 keys
    for key in sample_keys:
        print(f"  ID {key}: {yolo_id_to_name[key]}")
    
    # Initialize detectors
    f_detector = FasterRCNNOnnxDetector(
        args.fasterrcnn_model, 
        frcnn_id_to_name, 
        conf_threshold=args.conf_threshold, 
        iou_threshold=args.iou_threshold
    )
    
    y_detector = YOLOOnnxDetector(
        args.yolo_model, 
        yolo_id_to_name, 
        conf_threshold=args.conf_threshold, 
        iou_threshold=args.iou_threshold,
        img_size=args.img_size
    )

    # Run detection
    print(f"Running Faster R-CNN detection on {args.image}")
    boxes_f, scores_f, class_ids_f = f_detector.detect(args.image)
    print(f"Faster R-CNN found {len(boxes_f)} objects")
    
    print(f"Running YOLOv8 detection on {args.image}")
    boxes_y, scores_y, class_ids_y = y_detector.detect(args.image)
    print(f"YOLOv8 found {len(boxes_y)} objects")

    # Generate output filenames
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    fasterrcnn_output = os.path.join(args.output_dir, f"{img_name}_fasterrcnn.jpg")
    yolo_output = os.path.join(args.output_dir, f"{img_name}_yolo.jpg")

    # Create visualizations - use the correct mapping for each model
    visualize(args.image, boxes_f, scores_f, class_ids_f, frcnn_id_to_name, fasterrcnn_output)
    visualize(args.image, boxes_y, scores_y, class_ids_y, yolo_id_to_name, yolo_output)

    print("Processing complete!")


if __name__ == "__main__":
    main()
    
    
# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_test_results/dual_yolo_faster_inference.py \
#   --fasterrcnn_model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex003/may_2023_ex003.onnx \
#   --yolo_model /import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.onnx \
#   --frcnn_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/frcnn_id_to_name.json \
#   --yolo_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/yolo_id_to_name.json \
#   --image /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/12-Etudes-001.png \
#   --output_dir /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/onnx_test_results
