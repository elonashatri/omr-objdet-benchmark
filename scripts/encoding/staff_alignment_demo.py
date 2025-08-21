#!/usr/bin/env python3
"""
Staff Line Alignment Demo
-------------------------
This script demonstrates how to use the improved staff line detection
and alignment methods to fix misaligned staff lines in music score images.
"""

import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from precise_staff_alignment import ImprovedStaffDetector
from improved_staff_alignment import PreciseStaffAlignment, fix_staff_alignment

def main():
    """Main function to demonstrate staff line detection and alignment"""
    # Configure paths
    image_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/testing_images/Accidentals-004.png"  # Replace with your image path
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output paths
    detection_json = os.path.join(output_dir, "staff_lines_detected.json")
    detection_viz = os.path.join(output_dir, "staff_lines_detected.png")
    aligned_json = os.path.join(output_dir, "staff_lines_aligned.json")
    aligned_viz = os.path.join(output_dir, "staff_lines_aligned.png")
    comparison_viz = os.path.join(output_dir, "staff_lines_comparison.png")
    
    print(f"Processing image: {image_path}")
    
    # STEP 1: Initial staff line detection using the improved method
    print("Step 1: Detecting staff lines with improved method...")
    detector = ImprovedStaffDetector(line_merging_threshold=3)
    staff_data = detector.detect(image_path)
    
    # Save detection results
    with open(detection_json, 'w') as f:
        json.dump(staff_data, f, indent=2)
    
    # Visualize initial detection
    detector.visualize(image_path, staff_data, detection_viz)
    print(f"Initial detection saved to {detection_json}")
    print(f"Initial visualization saved to {detection_viz}")
    
    # STEP 2: Precise alignment to fix misaligned staff lines
    print("\nStep 2: Precisely aligning staff lines with actual image content...")
    aligner = PreciseStaffAlignment(debug=True)
    aligned_data = aligner.align_staff_lines(image_path, detection_json, aligned_json)
    
    # Visualize aligned staff lines
    aligner.visualize_alignment(image_path, detection_json, aligned_json, comparison_viz)
    print(f"Aligned staff data saved to {aligned_json}")
    print(f"Comparison visualization saved to {comparison_viz}")
    
    # Create a clean visualization of just the aligned result
    detector.visualize(image_path, aligned_data, aligned_viz)
    print(f"Aligned visualization saved to {aligned_viz}")
    
    print("\nComplete! Check the output files to see the improvement.")
    
    # Optional: display visualizations
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(mpimg.imread(detection_viz))
        plt.title("Initial Detection")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mpimg.imread(aligned_viz))
        plt.title("Aligned Result")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not display visualizations: {e}")

def process_batch(directory, output_dir=None):
    """
    Process a batch of images to demonstrate the method's robustness
    
    Args:
        directory: Directory containing music score images
        output_dir: Directory to save results (defaults to 'results' inside directory)
    """
    if output_dir is None:
        output_dir = os.path.join(directory, "results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        image_files.extend([f for f in os.listdir(directory) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        # Output paths for this image
        detection_json = os.path.join(output_dir, f"{base_name}_detected.json")
        aligned_json = os.path.join(output_dir, f"{base_name}_aligned.json")
        comparison_viz = os.path.join(output_dir, f"{base_name}_comparison.png")
        
        print(f"\nProcessing: {image_file}")
        
        try:
            # Initial detection
            detector = ImprovedStaffDetector(line_merging_threshold=3)
            staff_data = detector.detect(image_path)
            
            with open(detection_json, 'w') as f:
                json.dump(staff_data, f, indent=2)
            
            # Alignment
            aligner = PreciseStaffAlignment()
            aligned_data = aligner.align_staff_lines(image_path, detection_json, aligned_json)
            
            # Visualization
            aligner.visualize_alignment(image_path, detection_json, aligned_json, comparison_viz)
            
            print(f"  Success: Results saved to {output_dir}")
            
        except Exception as e:
            print(f"  Error processing {image_file}: {e}")
    
    print("\nBatch processing complete!")

def analyze_alignment_quality(original_json, aligned_json, image_path=None):
    """
    Analyze the quality of staff line alignment
    
    Args:
        original_json: Path to original staff line JSON
        aligned_json: Path to aligned staff line JSON
        image_path: Path to original image (optional)
    """
    # Load staff data
    with open(original_json, 'r') as f:
        original_staff = json.load(f)
        
    with open(aligned_json, 'r') as f:
        aligned_staff = json.load(f)
    
    # Calculate metrics
    original_positions = []
    aligned_positions = []
    
    for system_idx, system in enumerate(original_staff.get("staff_systems", [])):
        # Find corresponding system in aligned data
        aligned_system = None
        for s in aligned_staff.get("staff_systems", []):
            if s["id"] == system["id"]:
                aligned_system = s
                break
        
        if aligned_system is None:
            continue
            
        for i, line_idx in enumerate(system.get("lines", [])):
            if i >= len(aligned_system.get("lines", [])):
                continue
                
            aligned_idx = aligned_system["lines"][i]
            
            if line_idx < len(original_staff.get("detections", [])) and \
               aligned_idx < len(aligned_staff.get("detections", [])):
                orig_y = original_staff["detections"][line_idx]["bbox"]["center_y"]
                aligned_y = aligned_staff["detections"][aligned_idx]["bbox"]["center_y"]
                
                original_positions.append(orig_y)
                aligned_positions.append(aligned_y)
    
    if not original_positions or not aligned_positions:
        print("No corresponding lines found for comparison")
        return
    
    # Calculate average displacement
    displacements = [abs(o - a) for o, a in zip(original_positions, aligned_positions)]
    avg_displacement = sum(displacements) / len(displacements)
    max_displacement = max(displacements)
    
    print(f"Alignment Quality Analysis:")
    print(f"  Number of staff lines compared: {len(displacements)}")
    print(f"  Average displacement: {avg_displacement:.2f} pixels")
    print(f"  Maximum displacement: {max_displacement:.2f} pixels")
    
    # If image is provided, we can measure actual content alignment
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Process image to enhance staff lines
            aligner = PreciseStaffAlignment()
            enhanced_img = aligner._preprocess_image(img)
            
            # Measure line evidence for original and aligned positions
            original_scores = []
            aligned_scores = []
            
            for orig_y, aligned_y in zip(original_positions, aligned_positions):
                orig_score = aligner._measure_line_evidence(enhanced_img, int(orig_y), 0, enhanced_img.shape[1])
                aligned_score = aligner._measure_line_evidence(enhanced_img, int(aligned_y), 0, enhanced_img.shape[1])
                
                original_scores.append(orig_score)
                aligned_scores.append(aligned_score)
            
            avg_original_score = sum(original_scores) / len(original_scores)
            avg_aligned_score = sum(aligned_scores) / len(aligned_scores)
            
            improvement = (avg_aligned_score - avg_original_score) / avg_original_score * 100
            
            print(f"  Original line evidence score: {avg_original_score:.4f}")
            print(f"  Aligned line evidence score: {avg_aligned_score:.4f}")
            print(f"  Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    main()
    
    # Uncomment to process a batch of images
    process_batch("/homes/es314/omr-objdet-benchmark/scripts/encoding/testing_images")
    
    # Uncomment to analyze alignment quality
    analyze_alignment_quality("results/staff_lines_detected.json", 
                             "results/staff_lines_aligned.json",
                             "/homes/es314/omr-objdet-benchmark/scripts/encoding/testing_images/v3-accidentals-004.png")