import os
import shutil
import argparse

def create_model_directories(base_dir, model_paths, copy_configs=True):
    """
    Create organized directories for ONNX models and copy files
    
    Args:
        base_dir: Base directory where model folders will be created
        model_paths: List of ONNX model paths
        copy_configs: Whether to copy pipeline.config files
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    print(f"Using base directory: {base_dir}")
    
    # Process each model
    for model_path in model_paths:
        try:
            # Extract model name (directory name)
            model_name = os.path.basename(os.path.dirname(model_path))
            
            # Create model directory
            model_dir = os.path.join(base_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Copy ONNX model
            onnx_file = os.path.basename(model_path)
            target_path = os.path.join(model_dir, onnx_file)
            print(f"Copying {model_path} to {target_path}")
            shutil.copy(model_path, target_path)
            
            # Copy pipeline.config if requested
            if copy_configs:
                # Get original model directory
                original_dir = os.path.dirname(model_path)
                config_path = os.path.join(original_dir, "pipeline.config")
                
                if os.path.exists(config_path):
                    target_config = os.path.join(model_dir, "pipeline.config")
                    print(f"Copying pipeline.config to {target_config}")
                    shutil.copy(config_path, target_config)
                else:
                    print(f"Warning: pipeline.config not found at {config_path}")
                    
            print(f"Successfully created directory for {model_name}")
            
        except Exception as e:
            print(f"Error processing {model_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Organize ONNX models into directories")
    parser.add_argument("--base_dir", type=str, default="/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models",
                        help="Base directory for model folders")
    parser.add_argument("--no_configs", action="store_true", help="Don't copy pipeline.config files")
    
    args = parser.parse_args()
    
    # List of ONNX model paths
    model_paths = [
        "/import/c4dm-05/elona/checkpoint_runs/Faster_R-CNN_inception-lr-0.003-classes-88-steps-80000-2475x3504-09-10-2020-008-train/Faster_R-CNN_inception-lr-0.003-classes-88-steps-80000-2475x3504-09-10-2020-008-train.onnx",
        # "/import/c4dm-05/elona/checkpoint_runs/Faster-R-CNN-resnet50-88-classes-100000-2475x3504-06-10-2020-006-train/Faster-R-CNN-resnet50-88-classes-100000-2475x3504-06-10-2020-006-train.onnx",
        # "/import/c4dm-05/elona/checkpoint_runs/Faster_R-CNN_resnet50-lr-0.003-classes-72-steps-100000-2475x3504-03-10-2020-004-train/Faster_R-CNN_resnet50-lr-0.003-classes-72-steps-100000-2475x3504-03-10-2020-004-train.onnx",
        # "/import/c4dm-05/elona/checkpoint_runs/Faster_R-CNN_inception-lr-0.003-classes-72-steps-80000-2475x3504-01-10-2020-003-train/Faster_R-CNN_inception-lr-0.003-classes-72-steps-80000-2475x3504-01-10-2020-003-train.onnx"
    ]
    
    # Create model directories
    create_model_directories(args.base_dir, model_paths, not args.no_configs)
    
    # Create a README with model information
    readme_path = os.path.join(args.base_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# ONNX Model Directory\n\n")
        f.write("This directory contains ONNX models converted from TensorFlow Faster R-CNN models.\n\n")
        f.write("## Models\n\n")
        
        for model_path in model_paths:
            model_name = os.path.basename(os.path.dirname(model_path))
            f.write(f"### {model_name}\n\n")
            f.write(f"- Original path: {os.path.dirname(model_path)}\n")
            f.write(f"- Class mapping: /homes/es314/DOREMI/data/train_validation_test_records/mapping.txt\n\n")
    
    print(f"\nCreated README at {readme_path}")
    print(f"\nSuccessfully organized {len(model_paths)} models in {args.base_dir}")

if __name__ == "__main__":
    main()