import torch
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision

# Paths and settings
checkpoint_path = "./faster_rcnn_omr/latest.pth"
test_dir = "/homes/es314/omr-objdet-benchmark/scripts/faster_rcnn/2-solo-Christmas_Greeting-003.png"
output_dir = "./inference_results"
device = torch.device("cuda:1")  # Use a different GPU
os.makedirs(output_dir, exist_ok=True)

# Load the model
checkpoint = torch.load(checkpoint_path, map_location=device)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 217  # Adjust to your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Load test images
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))]

# Run inference
with torch.no_grad():
    for img_path in tqdm(test_images):
        # Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torchvision.transforms.functional.to_tensor(image).to(device)
        
        # Run inference
        outputs = model([img_tensor])
        
        # Draw predictions
        output = outputs[0]
        boxes = output['boxes'].cpu().numpy().astype(np.int32)
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        
        # Draw predictions with good scores
        threshold = 0.5
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= threshold:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)
                draw.text((box[0], box[1]), f"{label}: {score:.2f}", fill=(255, 0, 0))
        
        # Save output
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        img_pil.save(output_path)

print(f"Inference complete. Results saved to {output_dir}")