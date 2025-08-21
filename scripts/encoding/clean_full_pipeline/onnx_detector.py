import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image
import json
import matplotlib.pyplot as plt
import re

class FasterRCNNOnnxDetector:
    def __init__(self, model_path, class_mapping_path, conf_threshold=0.2, max_detections=600):
        """
        Initialize the ONNX-based Faster R-CNN detector
        
        Args:
            model_path: Path to the ONNX model
            class_mapping_path: Path to the class mapping file
            conf_threshold: Confidence threshold for detections
            max_detections: Maximum number of detections to return
        """
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        
        # Load the ONNX model
        print(f"Loading ONNX model from {model_path}")
        self.session = onnxruntime.InferenceSession(model_path)
        
        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"Model loaded with input name: {self.input_name}")
        print(f"Output names: {self.output_names}")
        
        # Load class mapping
        self.class_names = self.load_class_mapping(class_mapping_path)
        print(f"Loaded {len(self.class_names)} class mappings")
    
    def load_class_mapping(self, mapping_file):
        """Load class ID to name mapping from file"""
        class_map = {}
        try:
            with open(mapping_file, 'r') as f:
                content = f.read()
            
            # Parse the mapping file
            items = re.findall(r'item\{\s*id:\s*(\d+)\s*name:\s*\'([^\']+)\'\s*\}', content)
            for item_id, item_name in items:
                class_map[int(item_id)] = item_name
            
            print(f"Loaded {len(class_map)} class mappings from {mapping_file}")
            return class_map
        
        except Exception as e:
            print(f"Error loading class names: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def preprocess_image(self, image_path):
        """Load and preprocess image for inference"""
        # Load image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            print(f"Converting image from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        print(f"Image shape: {image_np.shape}")
        
        # Add batch dimension
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        return image_np_expanded, image.size
    
    def detect(self, image_path):
        """Run detection on the image"""
        # Preprocess image
        image, (original_width, original_height) = self.preprocess_image(image_path)
        
        # Run inference
        print(f"Running inference on {image_path}")
        start_time = cv2.getTickCount()
        
        outputs = self.session.run(None, {self.input_name: image})
        
        end_time = cv2.getTickCount()
        inference_time = (end_time - start_time) / cv2.getTickFrequency()
        print(f"Inference completed in {inference_time:.4f} seconds")
        
        # Process outputs
        # The output order is typically: boxes, scores, classes, num_detections
        boxes = outputs[0]
        scores = outputs[1]
        class_ids = outputs[2]
        num_detections = int(outputs[3][0])
        
        print(f"Found {num_detections} raw detections")
        
        # Filter by confidence threshold and limit to max_detections
        valid_indices = np.where(scores[0] >= self.conf_threshold)[0]
        if len(valid_indices) > self.max_detections:
            # Sort by score and take top detections
            score_order = np.argsort(scores[0][valid_indices])[::-1][:self.max_detections]
            valid_indices = valid_indices[score_order]
        
        filtered_boxes = boxes[0][valid_indices]
        filtered_scores = scores[0][valid_indices]
        filtered_class_ids = class_ids[0][valid_indices]
        
        print(f"Filtered to {len(filtered_boxes)} detections with confidence >= {self.conf_threshold}")
        
        return {
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'classes': filtered_class_ids,
            'num_detections': len(filtered_boxes),
            'image_size': (original_height, original_width)
        }
    
    def visualize_detections(self, image_path, results, output_path):
        """Create visualization of detections"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image from {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        boxes, scores, class_ids = results['boxes'], results['scores'], results['classes']
        
        # Generate colors for each class
        unique_classes = np.unique(class_ids)
        colors = {}
        for i, cls_id in enumerate(unique_classes):
            hue = (i * 0.15) % 1.0
            rgb = plt.cm.hsv(hue)[:3]  # tuple in [0,1]
            colors[cls_id] = tuple((np.array(rgb) * 255).astype(int).tolist())
        
        # Draw bounding boxes
        detection_count = 0
        for box, score, cls_id in zip(boxes, scores, class_ids):
            detection_count += 1
            # Boxes are in [y1, x1, y2, x2] format from the model
            ymin, xmin, ymax, xmax = box
            
            # Convert normalized coordinates to pixel values
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            
            # Draw rectangle
            color = colors.get(cls_id, (0, 255, 0))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Add label
            class_id = int(cls_id)
            class_name = self.class_names.get(class_id, f"Unknown-{class_id}")
            label = f"{class_name}: {score:.2f}"
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Draw label background
            cv2.rectangle(image, (xmin, ymin - text_size[1] - 5), (xmin + text_size[0], ymin), color, -1)
            
            # Draw text
            cv2.putText(image, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (0, 0, 0), thickness)
        
        # Add detection count
        cv2.putText(image, f"Detections: {detection_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Save image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Detection visualization saved to {output_path}")
    
    def save_detection_data(self, results, image_name, output_dir):
        """Save detection data to JSON and CSV files"""
        boxes, scores, class_ids = results['boxes'], results['scores'], results['classes']
        height, width = results['image_size']
        
        # Create a list of detections
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            # Boxes are in [y1, x1, y2, x2] format
            ymin, xmin, ymax, xmax = box
            
            # Convert normalized coordinates to pixel values
            xmin_px = xmin * width
            xmax_px = xmax * width
            ymin_px = ymin * height
            ymax_px = ymax * height
            
            width_px = xmax_px - xmin_px
            height_px = ymax_px - ymin_px
            center_x = xmin_px + width_px / 2
            center_y = ymin_px + height_px / 2
            
            class_id = int(class_id)
            class_name = self.class_names.get(class_id, f"cls_{class_id}")
            
            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(score),
                "bbox": {
                    "x1": float(xmin_px),
                    "y1": float(ymin_px),
                    "x2": float(xmax_px),
                    "y2": float(ymax_px),
                    "width": float(width_px),
                    "height": float(height_px),
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
        import csv
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

# Standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Faster R-CNN ONNX model for object detection")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping file")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--max_detections", type=int, default=600, help="Maximum number of detections")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    detector = FasterRCNNOnnxDetector(
        args.model,
        args.class_mapping,
        conf_threshold=args.conf,
        max_detections=args.max_detections
    )
    
    # Run detection
    results = detector.detect(args.image)
    
    # Save results
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    json_path, csv_path = detector.save_detection_data(results, img_name, args.output_dir)
    
    # Visualize detections
    output_path = os.path.join(args.output_dir, f"{img_name}_onnx_detection.jpg")
    detector.visualize_detections(args.image, results, output_path)
    
    print(f"Complete! Results saved to {args.output_dir}")