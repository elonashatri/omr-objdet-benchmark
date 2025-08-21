import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image
import json
import matplotlib.pyplot as plt
import re
from collections import defaultdict

class FasterRCNNOnnxDetector:
    def __init__(self, model_path, class_mapping_path, conf_threshold=0.25, max_detections=600):
        """
        Initialize the ONNX-based Faster R-CNN detector
        
        Args:
            model_path: Path to the ONNX model
            class_mapping_path: Path to class mapping file
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
        
        # Typical notehead dimensions (will be estimated from detections if noteheads are found)
        self.avg_notehead_width = None
        
        # Barline distance threshold (will be calculated based on noteheads or barlines)
        self.barline_distance_threshold = None
    
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
    
    def get_image_width(self, image_path):
        """Get the width of an image in pixels"""
        try:
            with Image.open(image_path) as img:
                return img.width
        except Exception as e:
            print(f"Error getting image width: {e}")
            return 1000  # Reasonable default
    
    def estimate_notehead_dimensions(self, boxes, class_ids):
        """
        Estimate the average dimensions of noteheads
        Returns normalized width (as a fraction of image width)
        """
        notehead_widths = []
        
        for i, class_id in enumerate(class_ids):
            class_name = self.class_names.get(int(class_id), "").lower()
            
            # Check if the class is any type of notehead
            if "notehead" in class_name or "note" in class_name:
                # Boxes are in [y1, x1, y2, x2] format
                _, x1, _, x2 = boxes[i]
                width = x2 - x1
                notehead_widths.append(width)
        
        if notehead_widths:
            # Use median instead of mean to be more robust to outliers
            avg_width = np.median(notehead_widths)
            print(f"Estimated median notehead width: {avg_width:.4f} (normalized)")
            return avg_width
        else:
            # If no noteheads found, check for other common elements like quarter notes
            note_widths = []
            for i, class_id in enumerate(class_ids):
                class_name = self.class_names.get(int(class_id), "").lower()
                if "quarter" in class_name or "eighth" in class_name or "half" in class_name:
                    _, x1, _, x2 = boxes[i]
                    width = x2 - x1
                    note_widths.append(width)
            
            if note_widths:
                avg_width = np.median(note_widths)
                print(f"No noteheads found, using median note width: {avg_width:.4f} (normalized)")
                return avg_width
            
            # Default value if no noteheads or notes are detected
            print("No noteheads or notes detected, using default width estimate")
            return 0.005  # 0.5% of the image width as a default
    
    def filter_barlines(self, boxes, scores, class_ids, image_width):
        """
        Improved filter for barlines that are too close to each other.
        This version respects vertical position (staff lines).
        """
        # Convert to numpy arrays for easier manipulation
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        
        print("\n=== BARLINE FILTERING DIAGNOSTICS ===")
        print(f"Total objects before filtering: {len(boxes)}")
        
        # DEBUG: Print all class names in the detection
        unique_classes = np.unique(class_ids)
        print(f"Unique class IDs detected: {unique_classes}")
        for cls_id in unique_classes:
            class_name = self.class_names.get(int(cls_id), f"Unknown-{cls_id}")
            count = np.sum(class_ids == cls_id)
            print(f"  Class {cls_id} ({class_name}): {count} instances")
        
        # Identify barline classes
        barline_indices = []
        
        print("\nSearching for barline classes...")
        for i, class_id in enumerate(class_ids):
            class_name = self.class_names.get(int(class_id), "")
            lower_class_name = class_name.lower()
            if "barline" in lower_class_name:
                barline_indices.append(i)
                print(f"  Found barline at index {i}: {class_name} (ID: {class_id}), position: {boxes[i]}, score: {scores[i]:.3f}")
        
        if not barline_indices:
            print("No barlines detected, no filtering needed")
            return boxes, scores, class_ids
        
        print(f"\nFound {len(barline_indices)} barlines before filtering")
        
        # Use a fixed distance threshold for horizontal proximity
        horizontal_threshold = 0.05 * image_width  # 1% of image width
        
        # Use a fixed distance threshold for vertical proximity (to determine the same staff)
        vertical_threshold = 0.05  # 5% of normalized height
        
        print(f"Using horizontal threshold: {horizontal_threshold:.2f} pixels (1% of image width)")
        print(f"Using vertical threshold: {vertical_threshold:.4f} (5% of normalized height)")
        
        # Group barlines by vertical position (staff line) first
        staff_groups = []
        processed_barlines = set()
        
        # Sort barlines by y-coordinate to help with staff line identification
        y_centers = []
        for idx in barline_indices:
            y1, x1, y2, x2 = boxes[idx]
            y_center = (y1 + y2) / 2
            y_centers.append((idx, y_center))
        
        y_centers.sort(key=lambda x: x[1])  # Sort by y center position
        
        print("\nGrouping barlines by staff line (vertical position)...")
        for idx, y_center in y_centers:
            if idx in processed_barlines:
                continue
            
            current_staff = [idx]
            processed_barlines.add(idx)
            
            for j, y_center_j in y_centers:
                if j != idx and j not in processed_barlines:
                    # If they are close enough vertically, consider them on the same staff
                    if abs(y_center_j - y_center) < vertical_threshold:
                        current_staff.append(j)
                        processed_barlines.add(j)
            
            if current_staff:
                print(f"  Found staff group with {len(current_staff)} barlines at y≈{y_center:.3f}")
                staff_groups.append(current_staff)
        
        print(f"\nFound {len(staff_groups)} staff groups")
        
        # Now for each staff group, filter barlines that are too close horizontally
        keep_indices = []
        
        for staff_idx, staff_group in enumerate(staff_groups):
            print(f"\nProcessing staff group {staff_idx+1} with {len(staff_group)} barlines")
            
            # Group by horizontal proximity
            x_groups = []
            processed_in_staff = set()
            
            # Sort barlines in this staff by x-coordinate
            x_positions = []
            for idx in staff_group:
                y1, x1, y2, x2 = boxes[idx]
                x_center = (x1 + x2) / 2 * image_width  # Convert to pixel space
                x_positions.append((idx, x_center))
            
            x_positions.sort(key=lambda x: x[1])  # Sort by x center position
            
            for idx, x_center in x_positions:
                if idx in processed_in_staff:
                    continue
                
                current_x_group = [idx]
                processed_in_staff.add(idx)
                
                # Look for close barlines within this staff group
                for j, x_center_j in x_positions:
                    if j != idx and j not in processed_in_staff:
                        if abs(x_center_j - x_center) < horizontal_threshold:
                            current_x_group.append(j)
                            processed_in_staff.add(j)
                            print(f"  Grouping barlines {j} and {idx} - distance: {abs(x_center_j - x_center):.2f} px (threshold: {horizontal_threshold:.2f} px)")
                
                if current_x_group:
                    x_groups.append(current_x_group)
            
            # Now keep the highest confidence barline from each x-group
            for group in x_groups:
                if len(group) == 1:
                    # Only one barline in this group
                    keep_indices.append(group[0])
                else:
                    # Multiple barlines - keep the one with highest confidence
                    group_scores = [scores[i] for i in group]
                    best_idx = group[np.argmax(group_scores)]
                    keep_indices.append(best_idx)
                    
                    # Log what we're doing
                    class_name = self.class_names.get(int(class_ids[best_idx]), "unknown")
                    filtered_out = [i for i in group if i != best_idx]
                    filtered_names = []
                    for i in filtered_out:
                        c_name = self.class_names.get(int(class_ids[i]), "unknown")
                        filtered_names.append(f"{c_name} ({scores[i]:.3f})")
                    
                    print(f"  Keeping {class_name} with confidence {scores[best_idx]:.3f}, filtered out: {', '.join(filtered_names)}")
        
        # Keep all non-barline detections plus the filtered barlines
        all_keep_indices = list(set([i for i in range(len(boxes)) if i not in barline_indices] + keep_indices))
        all_keep_indices.sort()  # Sort to maintain original order
        
        total_barlines_kept = len([i for i in all_keep_indices if i in barline_indices])
        print(f"\nAfter filtering, keeping {len(all_keep_indices)} detections ({total_barlines_kept} barlines out of original {len(barline_indices)})")
        print("=================================\n")
        
        return boxes[all_keep_indices], scores[all_keep_indices], class_ids[all_keep_indices]
    
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
        
        # Apply class-specific thresholds
        custom_thresholds = {
            "barline": 0.009,
            "systemicBarline": 0.10,
            "stem": 0.09,
        }

        filtered_indices = []
        for i in range(int(outputs[3][0])):  # num_detections
            class_id = int(class_ids[0][i])
            class_name = self.class_names.get(class_id, "").lower()
            score = scores[0][i]
            
            # Check for custom threshold, else use default
            threshold = next((v for k, v in custom_thresholds.items() if k.lower() in class_name), self.conf_threshold)
            
            if score >= threshold:
                filtered_indices.append(i)

        # Optional: sort and truncate top max_detections
        if len(filtered_indices) > self.max_detections:
            top_scores = np.array([scores[0][i] for i in filtered_indices])
            top_indices = np.argsort(top_scores)[::-1][:self.max_detections]
            filtered_indices = [filtered_indices[i] for i in top_indices]

        filtered_boxes = boxes[0][filtered_indices]
        filtered_scores = scores[0][filtered_indices]
        filtered_class_ids = class_ids[0][filtered_indices]

        print(f"[INFO] Kept {len(filtered_boxes)} detections after class-specific thresholding")
        
        # Filter barlines that are too close to each other
        filtered_boxes, filtered_scores, filtered_class_ids = self.filter_barlines(
            filtered_boxes, 
            filtered_scores, 
            filtered_class_ids,
            original_width
        )

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
        barline_count = 0
        
        for box, score, cls_id in zip(boxes, scores, class_ids):
            detection_count += 1
            # Boxes are in [y1, x1, y2, x2] format from the model
            ymin, xmin, ymax, xmax = box
            
            # Convert normalized coordinates to pixel values
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            
            # Check if this is a barline
            class_id = int(cls_id)
            class_name = self.class_names.get(class_id, f"Unknown-{class_id}")
            is_barline = "barline" in class_name.lower()
            
            if is_barline:
                barline_count += 1
            
            # Draw rectangle
            color = colors.get(cls_id, (0, 255, 0))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Add label
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
        stats_text = f"Detections: {detection_count} (Barlines: {barline_count})"
        cv2.putText(image, stats_text, (10, 30),
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
        barline_count = 0
        
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
            
            # Count barlines
            if "barline" in class_name.lower():
                barline_count += 1
            
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
        
        # Add summary information
        metadata = {
            "total_detections": len(detections),
            "barline_count": barline_count,
            "image_size": {
                "width": width,
                "height": height
            },
            "filtering_applied": True,
            "notes": "Barlines too close together have been filtered to keep only the highest confidence one"
        }
        
        # Save detections to JSON file
        json_output_path = os.path.join(output_dir, f"{image_name}_detections.json")
        with open(json_output_path, 'w') as f:
            json.dump({
                "metadata": metadata,
                "detections": detections
            }, f, indent=2)
        
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
        print(f"Summary: {len(detections)} total detections, {barline_count} barlines after filtering")
        return json_output_path, csv_output_path


# Standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Faster R-CNN ONNX model for object detection")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping file")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max_detections", type=int, default=600, help="Maximum number of detections")
    parser.add_argument("--barline_distance_threshold", type=float, default=None, 
                      help="Override distance threshold for barline filtering (in pixels)")
    parser.add_argument("--disable_barline_filtering", action="store_true",
                      help="Disable barline filtering")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Faster R-CNN ONNX Detector with Barline Filtering")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Class mapping: {args.class_mapping}")
    print(f"Conf threshold: {args.conf}")
    print(f"Barline filtering: {'Disabled' if args.disable_barline_filtering else 'Enabled'}")
    if args.barline_distance_threshold:
        print(f"Using custom barline distance threshold: {args.barline_distance_threshold} pixels")
    print("-" * 50)
    
    # Initialize detector
    detector = FasterRCNNOnnxDetector(
        args.model,
        args.class_mapping,
        conf_threshold=args.conf,
        max_detections=args.max_detections
    )
    
    # Override barline distance threshold if provided
    if args.barline_distance_threshold is not None:
        detector.barline_distance_threshold = args.barline_distance_threshold / max(1, 
                                                             detector.get_image_width(args.image))
    
    # Disable barline filtering if requested
    if args.disable_barline_filtering:
        # Save the original method
        original_filter_barlines = detector.filter_barlines
        
        # Replace with a pass-through method that doesn't filter
        def no_filter(boxes, scores, class_ids, image_width):
            print("Barline filtering disabled by command-line argument")
            return boxes, scores, class_ids
        
        detector.filter_barlines = no_filter
    
    # Run detection
    results = detector.detect(args.image)
    
    # Save results
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    json_path, csv_path = detector.save_detection_data(results, img_name, args.output_dir)
    
    # Visualize detections
    output_path = os.path.join(args.output_dir, f"{img_name}_onnx_detection.jpg")
    detector.visualize_detections(args.image, results, output_path)
    
    # If we disabled filtering, restore the original method
    if args.disable_barline_filtering:
        detector.filter_barlines = original_filter_barlines
    
    print(f"Complete! Results saved to {args.output_dir}")
    print(f"Visualization: {output_path}")
    print(f"JSON data: {json_path}")
    print(f"CSV data: {csv_path}")
    print("=" * 50)