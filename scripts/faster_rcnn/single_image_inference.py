import torch
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import torchvision
from torchvision.transforms import functional as F

def load_class_names(mapping_file):
    """Load class names from mapping file"""
    class_names = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) >= 2:
                class_id = int(parts[0])
                class_name = ':'.join(parts[1:])  # Join with colon in case class name contains colons
                class_names[class_id] = class_name
    return class_names

def run_inference_on_image(checkpoint_path, image_path, mapping_file, output_path=None, threshold=0.5, device=None):
    """
    Run inference on a single image using a trained Faster R-CNN model.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        image_path: Path to the input image
        mapping_file: Path to class ID to name mapping file
        output_path: Path to save the output image (optional)
        threshold: Detection confidence threshold
        device: Device to run inference on (default: cuda if available, else cpu)
    
    Returns:
        Tuple of (original image, processed image with detections, detection results)
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load class names
    class_names = load_class_names(mapping_file)
    print(f"Loaded {len(class_names)} class names")
    
    # Load model
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model architecture
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Get number of classes from checkpoint if available
    num_classes = len(class_names) + 1  # +1 for background
    if 'args' in checkpoint and hasattr(checkpoint['args'], 'num_classes'):
        num_classes = checkpoint['args'].num_classes
    
    # Replace the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)
    
    # Load weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Load and preprocess image
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img_tensor = F.to_tensor(image).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model([img_tensor])
    
    # Process predictions
    output = outputs[0]
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    
    # Filter predictions by threshold
    keep = scores >= threshold
    boxes = boxes[keep].astype(np.int32)
    scores = scores[keep]
    labels = labels[keep]
    
    print(f"Found {len(boxes)} objects with confidence >= {threshold}")
    
    # Draw predictions on image
    result_image = image.copy()
    img_pil = Image.fromarray(result_image)
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Generate colors for classes
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    
    # Draw boxes and labels
    for box, score, label in zip(boxes, scores, labels):
        # Get color for this class
        color = tuple(map(int, colors[label % len(colors)]))
        
        # Get class name
        class_name = class_names.get(label, f"Class {label}")
        
        # Draw box
        draw.rectangle(box.tolist(), outline=color, width=3)
        
        # Draw label and score
        text = f"{class_name}: {score:.2f}"
        text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
        draw.rectangle([box[0], box[1], box[0] + text_w, box[1] + text_h], fill=color)
        draw.text((box[0], box[1]), text, fill="white", font=font)
    
    result_image = np.array(img_pil)
    
    # Save output if requested
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        print(f"Saved result to {output_path}")
    
    return original_image, result_image, {'boxes': boxes, 'scores': scores, 'labels': labels, 'class_names': class_names}

if __name__ == "__main__":
    # Configuration
    checkpoint_path = "./older_config_faster_rcnn_omr_output/best.pth"  # Path to your model checkpoint
    image_path = "/homes/es314/omr-objdet-benchmark/scripts/faster_rcnn/demisemiquavers_simple-085.png"  # Path to your image
    mapping_file = "/homes/es314/omr-objdet-benchmark/data/faster_rcnn_prepared_dataset/mapping.txt"  # Path to class mapping file
    output_path = "./inference_result.png"  # Where to save the output
    threshold = 0.4  # Detection threshold
    
    # Run inference
    _, _, detections = run_inference_on_image(
        checkpoint_path=checkpoint_path,
        image_path=image_path,
        mapping_file=mapping_file,
        output_path=output_path,
        threshold=threshold
    )
    
    # Print detection summary
    print("\nDetection Summary:")
    class_names = detections['class_names']
    for i, (box, score, label) in enumerate(zip(
            detections['boxes'], detections['scores'], detections['labels'])):
        class_name = class_names.get(label, f"Class {label}")
        print(f"Detection {i+1}: {class_name}, Score: {score:.4f}, Box: {box.tolist()}")



