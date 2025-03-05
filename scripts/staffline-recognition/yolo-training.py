from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8x for staffline detection")
    # Your existing arguments
    parser.add_argument("--staffline_class_id", type=int, required=True, 
                        help="Class ID for stafflines in your dataset")
    
    args = parser.parse_args()
    
    # 1. Load model
    model = YOLO(args.model)
    
    # 2. Train with specialized parameters for thin objects
    results = model.train(
        data=args.yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=1280,
        device=args.device,
        patience=50,
        hsv_h=0.01,  # Minimal hue shift
        hsv_s=0.2,   # Moderate saturation
        hsv_v=0.1,   # Minimal brightness
        degrees=5,   # Small rotation
        scale=0.2,   # Scale jitter
        mosaic=0.5,  # Reduced mosaic
        overlap_mask=True,
        box=10.0,    # Higher box loss weight for better localization
        cls=0.5      # Class loss weight
    )
    
    # 3. Evaluate with specific focus on stafflines
    val_results = model.val()
    print(f"Validation results: {val_results}")
    
    # 4. Save model
    model_path = os.path.join(model.save_dir, 'best.pt')
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()