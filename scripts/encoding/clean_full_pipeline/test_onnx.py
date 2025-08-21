#!/usr/bin/env python3
"""
Test script for running detection with ONNX-converted Faster R-CNN model
"""
import os
import argparse
from onnx_detector import FasterRCNNOnnxDetector

def main():
    parser = argparse.ArgumentParser(description="Test ONNX Faster R-CNN detection")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--class_mapping", type=str, required=True, help="Path to class mapping file")
    parser.add_argument("--output_dir", type=str, default="onnx_results", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("ONNX Faster R-CNN Detection Test")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found at {args.model}")
        return
    
    if not os.path.exists(args.image):
        print(f"ERROR: Image file not found at {args.image}")
        return
    
    if not os.path.exists(args.class_mapping):
        print(f"ERROR: Class mapping file not found at {args.class_mapping}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using model: {args.model}")
    print(f"Processing image: {args.image}")
    print(f"Using class mapping: {args.class_mapping}")
    print(f"Saving results to: {args.output_dir}")
    print(f"Confidence threshold: {args.conf}")
    
    try:
        # Initialize detector
        detector = FasterRCNNOnnxDetector(
            args.model,
            args.class_mapping,
            conf_threshold=args.conf,
        )
        
        # Run detection
        print("\nRunning detection...")
        results = detector.detect(args.image)
        
        # Get image name without extension
        img_name = os.path.splitext(os.path.basename(args.image))[0]
        
        # Save results
        print("\nSaving detection results...")
        json_path, csv_path = detector.save_detection_data(results, img_name, args.output_dir)
        
        # Visualize detections
        print("\nCreating visualization...")
        output_path = os.path.join(args.output_dir, f"{img_name}_onnx_detection.jpg")
        detector.visualize_detections(args.image, results, output_path)
        
        print("\nTest completed successfully!")
        print(f"Detection results saved to: {json_path}")
        print(f"Visualization saved to: {output_path}")
        
        # Print detection summary
        boxes, scores, classes = results['boxes'], results['scores'], results['classes']
        print(f"\nDetection summary:")
        print(f"  Total detections: {len(boxes)}")
        
        # Show top 5 detections
        if len(boxes) > 0:
            print("\nTop 5 detections:")
            # Sort by score
            sorted_indices = scores.argsort()[::-1][:5]
            for i, idx in enumerate(sorted_indices):
                class_id = int(classes[idx])
                class_name = detector.class_names.get(class_id, f"Unknown-{class_id}")
                print(f"  {i+1}. {class_name} (ID: {class_id}) - Score: {scores[idx]:.4f}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    


# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/test_onnx.py \
#   --model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-lr-0.003-classes-88-steps-80000-2475x3504-09-10-2020-008-train/Faster_R-CNN_inception-lr-0.003-classes-88-steps-80000-2475x3504-09-10-2020-008-train.onnx \
#   --image /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/12-Etudes-001.png \
#   --class_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/class_mappings/class_mapping.txt \
#   --output_dir onnx_test_results