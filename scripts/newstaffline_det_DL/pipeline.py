"""
Pipeline Module for Staff Line Detection

This module provides an end-to-end pipeline for staff line detection,
from preprocessing to inference.
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Import components
from model import StaffLineDetectionNet
from preprocessing import preprocess_xmls_to_masks
from data_loader import get_dataloader
from training_evaluation import train_staff_line_model, evaluate_staff_line_model
from inference_ import run_inference


def setup_directories(base_dir):
    """
    Set up the necessary directories for the pipeline.
    
    Args:
        base_dir (str): Base directory for all outputs
        
    Returns:
        dict: Dictionary of directory paths
    """
    dirs = {
        "base": base_dir,
        "preprocessed": os.path.join(base_dir, "preprocessed"),
        "masks": os.path.join(base_dir, "preprocessed", "masks"),
        "models": os.path.join(base_dir, "models"),
        "logs": os.path.join(base_dir, "logs"),
        "results": os.path.join(base_dir, "results"),
        "visualizations": os.path.join(base_dir, "visualizations")
    }
    
    # Create directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def run_pipeline(args):
    """
    Run the complete staff line detection pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    
    # Welcome message
    print("\n" + "="*80)
    print("Staff Line Detection Pipeline".center(80))
    print("="*80 + "\n")
    
    # Setup directories
    dirs = setup_directories(args.output_dir)
    
    # Step 1: Preprocess XML annotations to masks
    if not args.skip_preprocess:
        print("\n--- Step 1: Preprocessing Data ---\n")
        preprocess_success = preprocess_xmls_to_masks(
            xml_dir=args.xml_dir,
            output_mask_dir=dirs["masks"],
            image_dir=args.image_dir,
            num_workers=args.num_workers,
            verify=args.verify,
            visualize=args.visualize,
            num_vis_samples=args.num_vis_samples,
            subset_fraction=args.subset_fraction
        )
        
        if not preprocess_success:
            print("Preprocessing failed. Exiting pipeline.")
            return False
    else:
        print("\n--- Skipping Preprocessing ---\n")
    
    # Step 2: Train the model
    if not args.skip_train:
        print("\n--- Step 2: Training Model ---\n")
        
        # Train model
        history = train_staff_line_model(
            img_dir=args.image_dir,
            mask_dir=dirs["masks"],
            model_dir=dirs["models"],
            log_dir=dirs["logs"],
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            device_id=args.gpu_id,
            num_workers=args.num_workers,
            augment=not args.no_augment,
            weights=args.weights,
            target_size=tuple(args.target_size),
            subset_fraction=args.subset_fraction,
            continue_from=args.continue_from  # Add the continue_from parameter
        )
    else:
        print("\n--- Skipping Training ---\n")
    
    # Step 3: Evaluate the model
    if not args.skip_eval:
        print("\n--- Step 3: Evaluating Model ---\n")
        
        # Determine model path
        if args.weights and args.skip_train:
            model_path = args.weights
        else:
            # Try to find best model in model_dir
            model_path = os.path.join(dirs["models"], "best_model.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(dirs["models"], "final_model.pth")
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Skipping evaluation.")
        else:
            # Evaluate model
            # Use test directory if specified, otherwise use training data
            test_img_dir = args.test_img_dir if args.test_img_dir else args.image_dir
            
            metrics = evaluate_staff_line_model(
                model_path=model_path,
                img_dir=test_img_dir,
                mask_dir=dirs["masks"],
                batch_size=args.batch_size,
                device_id=args.gpu_id,
                num_workers=args.num_workers
            )
    else:
        print("\n--- Skipping Evaluation ---\n")
    
    # Step 4: Run inference
    if not args.skip_inference:
        print("\n--- Step 4: Running Inference ---\n")
        
        # Determine model path
        if args.weights and args.skip_train:
            model_path = args.weights
        else:
            # Try to find best model in model_dir
            model_path = os.path.join(dirs["models"], "best_model.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(dirs["models"], "final_model.pth")
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Skipping inference.")
        else:
            # Use test directory if specified, otherwise use training data
            test_img_dir = args.test_img_dir if args.test_img_dir else args.image_dir
            
            # Run inference
            run_inference(
                model_path=model_path,
                input_path=test_img_dir,
                output_dir=dirs["results"],
                batch_mode=True,
                gpu_id=args.gpu_id,
                post_process=not args.no_post_process
            )
    else:
        print("\n--- Skipping Inference ---\n")
    
    # Calculate total time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Final message
    print("\n" + "="*80)
    print(f"Pipeline completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s".center(80))
    print("="*80 + "\n")
    
    print(f"Results are available in: {os.path.abspath(args.output_dir)}")
    print(f"- Masks: {dirs['masks']}")
    print(f"- Models: {dirs['models']}")
    print(f"- Results: {dirs['results']}")
    print(f"- Logs: {dirs['logs']}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Staff Line Detection Pipeline")
    
    # Data paths
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing images")
    parser.add_argument("--xml_dir", type=str, required=True, 
                        help="Directory containing XML annotations")
    parser.add_argument("--output_dir", type=str, default="./staff_line_output",
                        help="Directory to save all outputs")
    parser.add_argument("--test_img_dir", type=str, default="",
                        help="Directory containing test images (optional)")
    
    # Preprocessing options
    parser.add_argument("--verify", action="store_true",
                        help="Verify the dataset after preprocessing")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize sample images and masks")
    parser.add_argument("--num_vis_samples", type=int, default=5,
                        help="Number of samples to visualize")
    
    # Training options
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--weights", type=str, default="",
                        help="Path to pre-trained weights (optional)")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable data augmentation")
    # Add continue_from argument
    parser.add_argument("--continue_from", type=str, default="",
                        help="Path to checkpoint to continue training from")
    
    # Dataset options
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 512],
                        help="Target size for resizing images (height width)")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Fraction of dataset to use (0.0-1.0)")
    
    # Inference options
    parser.add_argument("--no_post_process", action="store_true",
                        help="Disable post-processing in inference")
    
    # Hardware options
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Pipeline control
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing step")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training step")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation step")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference step")
    
    args = parser.parse_args()
    
    # Check if image and XML directories exist
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist.")
        return 1
    
    if not os.path.exists(args.xml_dir):
        print(f"Error: XML directory '{args.xml_dir}' does not exist.")
        return 1
    
    # Run the pipeline
    success = run_pipeline(args)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())