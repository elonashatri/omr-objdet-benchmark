from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8x on dataset")
    parser.add_argument("--yaml", type=str, required=True, help="Path to dataset.yaml file")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="YOLO model to use")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--name", type=str, default="", help="Experiment name")
    parser.add_argument("--project", type=str, default="/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs", help="Output")
    
    args = parser.parse_args()
    
    print(f"Training {args.model} on {args.yaml} for {args.epochs} epochs")
    print(f"Using GPU device {args.device}")
    
    # Load model
    model = YOLO(args.model)
    
    # Train model
    model.train(
        data=args.yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=1280,
        device=args.device
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()