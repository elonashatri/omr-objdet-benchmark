from ultralytics import YOLO

def get_map_per_class(model_path, data_yaml, img_size=1280, device=0):
    # Load the model
    model = YOLO(model_path)
    
    # Run validation with verbose option to get per-class metrics
    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        device=device,
        verbose=True  # This ensures detailed metrics are printed
    )
    
    # The metrics are available in the results object
    print("\n--- mAP Per Class ---")
    for i, class_name in enumerate(results.names):
        # Check if class exists in validation set
        if i in results.metrics.class_ids:
            idx = results.metrics.class_ids.index(i)
            ap50 = results.metrics.ap50_per_class[idx]
            ap = results.metrics.ap_per_class[idx]
            print(f"Class {i} ({class_name}): mAP50 = {ap50:.4f}, mAP50-95 = {ap:.4f}")
    
    return results

# Example usage
get_map_per_class(
    "/homes/es314/omr-objdet-benchmark/runs/train/phase12/weights/best.pt",
    "results/yolo_fixed/dataset.yaml",
    img_size=1280,
    device=0
)