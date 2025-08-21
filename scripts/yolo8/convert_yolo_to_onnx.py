from ultralytics import YOLO
import json

# Path to your model
model_path = '/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.pt'

# Load the class mapping
with open('/homes/es314/omr-objdet-benchmark/data/class_mapping.json', 'r') as f:
    class_mapping = json.load(f)

# Load the model
model = YOLO(model_path)

# Export to ONNX
# The default output path will be in the same directory as the .pt file with .onnx extension
model.export(format='onnx', 
             dynamic=True,  # Dynamic axes for variable batch size
             simplify=True)  # Simplify the model where possible