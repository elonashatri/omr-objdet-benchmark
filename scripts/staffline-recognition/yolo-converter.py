from ultralytics import YOLO
import argparse
import os
import torch
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import select_device
from ultralytics.data.augment import LetterBox

class StafflineFocusLoss(v8DetectionLoss):
    def __init__(self, *args, staffline_class_id=0, staffline_weight=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.staffline_class_id = staffline_class_id
        self.staffline_weight = staffline_weight
    
    def __call__(self, preds, targets, *args, **kwargs):
        loss, loss_items = super().__call__(preds, targets, *args, **kwargs)
        
        # Apply additional weight to staffline class loss
        if len(targets) and self.staffline_class_id is not None:
            # Find staffline targets
            staffline_targets = targets[targets[:, 1] == self.staffline_class_id]
            
            if len(staffline_targets):
                # Increase loss weight for stafflines
                loss = loss * self.staffline_weight
                
        return loss, loss_items

class StafflineTrainer(DetectionTrainer):
    def __init__(self, staffline_class_id=0, staffline_weight=1.5, *args, **kwargs):
        self.staffline_class_id = staffline_class_id
        self.staffline_weight = staffline_weight
        super().__init__(*args, **kwargs)
    
    def get_loss_and_metrics(self, batch, step_num=0):
        # Replace the default loss with our staffline-focused loss
        if not hasattr(self, 'staffline_loss'):
            self.staffline_loss = StafflineFocusLoss(
                self.model.model.nl,
                self.model.model.na,
                self.model.model.nc, 
                self.model.model.stride,
                staffline_class_id=self.staffline_class_id,
                staffline_weight=self.staffline_weight
            )
        
        preds = self.model(batch["img"])
        loss, loss_items = self.staffline_loss(preds, batch["cls"], batch["bboxes"], batch["batch_idx"])
        
        metrics = {
            "loss": loss.item(),
            "box_loss": loss_items[0].item(),
            "cls_loss": loss_items[1].item(),
            "dfl_loss": loss_items[2].item()
        }
        
        return loss, metrics

def train_with_staffline_focus(data_yaml, model_path, epochs, batch_size, device, staffline_class_id):
    """Two-phase training with special focus on stafflines"""
    print(f"Starting phase 1: General training ({epochs//2} epochs)")
    
    # First phase - general training
    model = YOLO(model_path)
    results1 = model.train(
        data=data_yaml,
        epochs=epochs // 2,  # First half of training
        batch=batch_size,
        imgsz=1280,
        device=device,
        patience=50,
        hsv_h=0.01,  # Minimal hue shift
        hsv_s=0.2,   # Moderate saturation
        hsv_v=0.1,   # Minimal brightness
        degrees=5,   # Small rotation
        scale=0.2,   # Scale jitter
        mosaic=0.5,  # Reduced mosaic
        box=7.5,     # Box loss weight
        cls=0.5,     # Class loss weight
        project="runs/train",
        name="phase1"
    )
    
    # Get best model from phase 1
    phase1_model_path = os.path.join("runs/train/phase1", "weights/best.pt")
    if not os.path.exists(phase1_model_path):
        phase1_model_path = os.path.join("runs/train/phase1", "weights/last.pt")
    
    print(f"Completed phase 1. Best model saved at: {phase1_model_path}")
    print(f"Starting phase 2: Staffline-focused training ({epochs//2} epochs)")
    
    # Second phase - focused on stafflines using custom trainer
    # Initialize the trainer with staffline focus
    model2 = YOLO(phase1_model_path)
    
    # Create custom trainer arguments
    custom_args = {
        "data": data_yaml,
        "epochs": epochs // 2,
        "batch": batch_size,
        "imgsz": 1280,
        "device": device,
        "patience": 50,
        "hsv_h": 0.01,
        "hsv_s": 0.2,
        "hsv_v": 0.1,
        "degrees": 5,
        "scale": 0.2,
        "mosaic": 0.5,
        "box": 10.0,      # Increased box loss weight for better localization
        "cls": 0.8,       # Increased class loss for better staffline classification
        "project": "runs/train",
        "name": "phase2",
        "exist_ok": True
    }
    
    # Start custom training
    trainer = StafflineTrainer(
        staffline_class_id=staffline_class_id,
        staffline_weight=2.0,  # Weight staffline loss higher
        cfg=custom_args
    )
    
    # Train and save model
    trainer.train()
    
    # Get the path to the best model from phase 2
    final_model_path = os.path.join("runs/train/phase2", "weights/best.pt")
    if not os.path.exists(final_model_path):
        final_model_path = os.path.join("runs/train/phase2", "weights/last.pt")
    
    print(f"Training complete! Final model saved at: {final_model_path}")
    
    # Return the final trained model
    return YOLO(final_model_path)

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 with staffline focus")
    parser.add_argument("--yaml", type=str, required=True, help="Path to dataset.yaml file")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="YOLO model to use")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--staffline_class_id", type=int, required=True, 
                        help="Class ID for stafflines in your dataset (0-indexed)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--skip_phase1", action="store_true", help="Skip phase 1 training")
    
    args = parser.parse_args()
    
    print(f"Training {args.model} on {args.yaml} for {args.epochs} epochs")
    print(f"Using GPU device {args.device}")
    print(f"Staffline class ID: {args.staffline_class_id}")
    
    if args.resume:
        print("Resuming from last checkpoint")
        model = YOLO(args.model)
        model.train(resume=True)
    else:
        # Train with staffline focus
        model = train_with_staffline_focus(
            args.yaml,
            args.model,
            args.epochs,
            args.batch,
            args.device,
            args.staffline_class_id
        )
    
    # Evaluate final model
    print("Evaluating final model...")
    val_results = model.val(data=args.yaml)
    
    print(f"Validation results: {val_results}")
    print("Training complete!")

def post_process_stafflines(model, image_path, conf=0.25, staffline_class_id=0):
    """Post-process to enhance staffline detection"""
    import cv2
    
    # Get initial predictions
    results = model.predict(image_path, conf=conf)[0]
    
    # Extract image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Extract staffline predictions
    staffline_boxes = []
    for i, box in enumerate(results.boxes):
        cls = int(box.cls.item())
        if cls == staffline_class_id:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            staffline_boxes.append((x1, y1, x2, y2))
    
    # If we have few stafflines, try to find more with Hough transform
    if len(staffline_boxes) < 5:  # Typical staff has 5 lines
        # Use Hough transform to find horizontal lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=w//3, maxLineGap=20)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if horizontal (small y difference)
                if abs(y2 - y1) < 5:
                    # Check if this line is already detected
                    new_line = True
                    for box in staffline_boxes:
                        box_y_center = (box[1] + box[3]) / 2
                        line_y_center = (y1 + y2) / 2
                        if abs(box_y_center - line_y_center) < 10:  # Close enough
                            new_line = False
                            break
                    
                    if new_line:
                        # Add as new detection
                        print(f"Adding new staffline at y={line_y_center}")
                        # You would add this to your results
                        # This requires modifying the Results object
    
    return results

if __name__ == "__main__":
    main()