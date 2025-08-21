from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import json

# Load the PyTorch model
model_path = '/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.pt'
model = YOLO(model_path)

# Load class mapping to verify class names
class_mapping_path = '/homes/es314/omr-objdet-benchmark/data/class_mapping.json'
with open(class_mapping_path, 'r') as f:
    class_mapping = json.load(f)

print(f"Loaded model: {model_path}")
print(f"Model task: {model.task}")
print(f"Class mapping contains {len(class_mapping)} classes")

# Print the first few class mappings to understand the format
print("\nSample of class mapping:")
sample_items = list(class_mapping.items())[:5]
for k, v in sample_items:
    print(f"  {k}: {v}")

# Test image
image_path = '/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/12-Etudes-001.png'
output_path = '/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/yolov8_pt_detections.jpg'

# Run inference with PyTorch model
start_time = time.time()
results = model(image_path, conf=0.25)  # Confidence threshold
elapsed = time.time() - start_time

# Get detection results
result = results[0]  # First image result
boxes = result.boxes.xyxy.cpu().numpy()  # Boxes (xyxy format)
confidences = result.boxes.conf.cpu().numpy()  # Confidence
class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

print(f"Inference completed in {elapsed:.3f} seconds")
print(f"Found {len(boxes)} objects")

# Get all class IDs in the results
unique_cls_ids = np.unique(class_ids)
print(f"Unique class IDs in detections: {unique_cls_ids}")

# Create id_to_name mapping based on the format of class_mapping
if isinstance(next(iter(class_mapping.keys())), str) and next(iter(class_mapping.keys())).isdigit():
    # Format is {"0": "class0", "1": "class1", ...}
    id_to_name = {int(k): v for k, v in class_mapping.items()}
else:
    # Format is {"class0": 0, "class1": 1, ...}
    id_to_name = {v: k for k, v in class_mapping.items()}

# Count detections by class
class_counts = {}
for cls_id in class_ids:
    cls_name = id_to_name.get(cls_id, f"Unknown-{cls_id}")
    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

print("\nDetections by class:")
for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cls_name}: {count}")

# Print first few detections
if len(boxes) > 0:
    print("\nFirst 5 detections:")
    for i in range(min(5, len(boxes))):
        box = boxes[i]
        conf = confidences[i]
        cls_id = class_ids[i]
        cls_name = id_to_name.get(cls_id, f"Unknown-{cls_id}")
        print(f"  Box {i}: {box} - Class: {cls_name} ({cls_id}) - Confidence: {conf:.4f}")

# Save the output visualization
result.save(filename=output_path)
print(f"Visualization saved to {output_path}")

# Also save a custom visualization with class names
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read image from {image_path}")
else:
    # Create a colormap based on class IDs
    np.random.seed(42)  # For reproducible colors
    colors = np.random.randint(0, 255, size=(max(id_to_name.keys()) + 1, 3)).tolist()

    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        
        # Get class name and color
        cls_name = id_to_name.get(cls_id, f"Unknown-{cls_id}")
        color = colors[cls_id]
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{cls_name}: {conf:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1-label_height-5), (x1+label_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Add detection count
    cv2.putText(image, f"Detections: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save custom visualization
    custom_output_path = output_path.replace('.jpg', '_custom.jpg')
    cv2.imwrite(custom_output_path, image)
    print(f"Custom visualization saved to {custom_output_path}")