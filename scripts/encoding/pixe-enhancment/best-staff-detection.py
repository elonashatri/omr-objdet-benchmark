#!/usr/bin/env python3
"""
Integration of Pixel-Perfect Alignment with Element-Driven Staff Detector
------------------------------------------------------------------------
This script shows how to integrate pixel-perfect alignment directly into
the element-driven staff detection pipeline.
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from precise_pixel_alignment import align_staff_lines_pixel_perfect
from element_driven_staff_detection import ElementDrivenStaffDetector

class EnhancedStaffDetector(ElementDrivenStaffDetector):
    """
    Enhanced staff detector that combines element-driven approach with
    pixel-perfect alignment for maximum accuracy
    """
    
    def __init__(self, debug=True, pixel_perfect=False):
        """
        Initialize the enhanced staff detector
        
        Args:
            debug: Enable debug output
            pixel_perfect: Enable pixel-perfect alignment
        """
        super().__init__(debug=debug)
        self.pixel_perfect = pixel_perfect
    
    def detect(self, image_path, detected_elements_path):
        """
        Detect staff lines with enhanced pixel-perfect accuracy
        
        Args:
            image_path: Path to the music score image
            detected_elements_path: Path to the JSON file with YOLO detected elements
                
        Returns:
            Dictionary containing staff line information
        """
        # First, use element-driven approach for initial staff line detection
        if self.debug:
            print("Step 1: Performing element-driven staff detection...")
            
        initial_staff_data = super().detect(image_path, detected_elements_path)
        
        # If pixel-perfect alignment is enabled, refine the staff lines
        if self.pixel_perfect:
            if self.debug:
                print("Step 2: Applying pixel-perfect alignment...")
                
            # Load image for alignment
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image from {image_path}")
                
            # Apply pixel-perfect alignment
            aligned_staff_data = align_staff_lines_pixel_perfect(image_path, initial_staff_data)
            
            return aligned_staff_data
        else:
            # Return initial staff data without alignment
            return initial_staff_data
    
    def visualize_comparison(self, image_path, initial_staff_data, aligned_staff_data, output_path=None):
        """
        Create visualization showing both initial and pixel-perfect aligned staff lines
        
        Args:
            image_path: Path to the music score image
            initial_staff_data: Initial staff line data
            aligned_staff_data: Pixel-perfect aligned staff line data
            output_path: Path to save visualization
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        
        # Colors for visualization
        initial_color = (1, 0, 0, 0.7)    # Red with alpha
        aligned_color = (0, 0.7, 0, 0.8)  # Green with alpha
        
        # Draw initial staff lines
        for system in initial_staff_data.get("staff_systems", []):
            for line_idx in system.get("lines", []):
                if line_idx < len(initial_staff_data.get("detections", [])):
                    line = initial_staff_data["detections"][line_idx]
                    x1 = line["bbox"]["x1"]
                    y1 = line["bbox"]["center_y"]
                    x2 = line["bbox"]["x2"]
                    y2 = line["bbox"]["center_y"]
                    
                    # Plot initial staff line
                    plt.plot([x1, x2], [y1, y2], color=initial_color, linewidth=2, linestyle='--',
                             label='Initial' if line_idx == system["lines"][0] else "")
        
        # Draw aligned staff lines
        for system in aligned_staff_data.get("staff_systems", []):
            for line_idx in system.get("lines", []):
                if line_idx < len(aligned_staff_data.get("detections", [])):
                    line = aligned_staff_data["detections"][line_idx]
                    x1 = line["bbox"]["x1"]
                    y1 = line["bbox"]["center_y"]
                    x2 = line["bbox"]["x2"]
                    y2 = line["bbox"]["center_y"]
                    
                    # Plot aligned staff line
                    plt.plot([x1, x2], [y1, y2], color=aligned_color, linewidth=2,
                             label='Pixel-perfect' if line_idx == system["lines"][0] else "")
        
        # Add legend (only once for each color)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        
        plt.title("Staff Line Detection Comparison")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def detect_with_comparison(self, image_path, detected_elements_path, output_dir=None):
        """
        Detect staff lines and create comparison visualization
        
        Args:
            image_path: Path to the music score image
            detected_elements_path: Path to the JSON file with YOLO detected elements
            output_dir: Directory to save results (optional)
            
        Returns:
            Dictionary with pixel-perfect aligned staff line data
        """
        # Store original pixel_perfect setting and set to False for initial detection
        original_setting = self.pixel_perfect
        self.pixel_perfect = False
        
        # Get initial staff data without pixel-perfect alignment
        initial_staff_data = super().detect(image_path, detected_elements_path)
        
        # Restore original setting and get aligned data if needed
        self.pixel_perfect = original_setting
        
        if self.pixel_perfect:
            # Get aligned staff data
            aligned_staff_data = align_staff_lines_pixel_perfect(image_path, initial_staff_data)
            
            # Create comparison visualization
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                comparison_viz = os.path.join(output_dir, f"{base_name}_comparison.png")
                self.visualize_comparison(image_path, initial_staff_data, aligned_staff_data, comparison_viz)
                
                # Save both versions
                initial_json = os.path.join(output_dir, f"{base_name}_initial.json")
                aligned_json = os.path.join(output_dir, f"{base_name}_aligned.json")
                
                with open(initial_json, 'w') as f:
                    json.dump(initial_staff_data, f, indent=2)
                
                with open(aligned_json, 'w') as f:
                    json.dump(aligned_staff_data, f, indent=2)
                
                print(f"Comparison visualization saved to: {comparison_viz}")
                print(f"Initial staff data saved to: {initial_json}")
                print(f"Aligned staff data saved to: {aligned_json}")
            
            return aligned_staff_data
        else:
            # If pixel_perfect is disabled, just return initial data
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                initial_json = os.path.join(output_dir, f"{base_name}_staff_lines.json")
                
                with open(initial_json, 'w') as f:
                    json.dump(initial_staff_data, f, indent=2)
                
                print(f"Staff data saved to: {initial_json}")
            
            return initial_staff_data

def detect_staff_lines(image_path, detected_elements_path, output_dir=None, pixel_perfect=False):
    """
    Convenience function to detect staff lines with pixel-perfect accuracy
    
    Args:
        image_path: Path to the music score image
        detected_elements_path: Path to the JSON file with YOLO detected elements
        output_dir: Directory to save results (optional)
        pixel_perfect: Whether to apply pixel-perfect alignment
        
    Returns:
        Dictionary with staff line data
    """
    # Create detector with pixel-perfect alignment
    detector = EnhancedStaffDetector(debug=True, pixel_perfect=pixel_perfect)
    
    # Detect staff lines
    print(f"Detecting staff lines for: {os.path.basename(image_path)}")
    print(f"Using detected elements from: {os.path.basename(detected_elements_path)}")
    print(f"Pixel-perfect alignment: {'Enabled' if pixel_perfect else 'Disabled'}")
    
    staff_data = detector.detect(image_path, detected_elements_path)
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output paths
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        alignment_type = "pixel_perfect" if pixel_perfect else "standard"
        staff_json = os.path.join(output_dir, f"{base_name}_{alignment_type}.json")
        staff_viz = os.path.join(output_dir, f"{base_name}_{alignment_type}.png")
        
        # Save staff data
        with open(staff_json, 'w') as f:
            json.dump(staff_data, f, indent=2)
        
        # Create visualization
        detector.visualize(image_path, staff_data, staff_viz)
        
        print(f"Results saved to:")
        print(f"  - JSON: {staff_json}")
        print(f"  - Visualization: {staff_viz}")
    
    return staff_data

def replace_in_pipeline(image_path, detected_elements_path, output_dir="results"):
    """
    Example function for replacing staff detection in an existing pipeline
    
    Args:
        image_path: Path to the music score image
        detected_elements_path: Path to the JSON file with YOLO detected elements
        output_dir: Directory to save results
        
    Returns:
        Dictionary with staff line data
    """
    # Create output directories
    staff_dir = os.path.join(output_dir, "staff_lines")
    os.makedirs(staff_dir, exist_ok=True)
    
    # Generate output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    staff_json = os.path.join(staff_dir, f"{base_name}_staff_lines.json")
    
    # Create enhanced detector
    detector = EnhancedStaffDetector(debug=True, pixel_perfect=False)
    
    # Detect staff lines with comparison
    print("Detecting staff lines with enhanced pixel-perfect accuracy...")
    staff_data = detector.detect_with_comparison(
        image_path, 
        detected_elements_path,
        os.path.join(output_dir, "comparison")
    )
    
    # Save the final staff data in format compatible with rest of pipeline
    with open(staff_json, 'w') as f:
        json.dump(staff_data, f, indent=2)
    
    print(f"Staff line data saved to: {staff_json}")
    print("Ready for next pipeline step.")
    
    return staff_data, staff_json

if __name__ == "__main__":
    # Example usage
    # /homes/es314/omr-objdet-benchmark/scripts/encoding/demisemiquavers_simple-085.png
    
    image_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/testing_images/Abismo_de_Rosas__Canhoto_Amrico_Jacomino-002.png"  # Replace with your image path
    detected_elements_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/results/faster_rcnn_results/object_detections/Abismo_de_Rosas__Canhoto_Amrico_Jacomino-002_detections.json"  # Replace with your YOLO detections
    
    # Detect staff lines with pixel-perfect alignment
    staff_data = detect_staff_lines(
        image_path, 
        detected_elements_path,
        output_dir="/homes/es314/omr-objdet-benchmark/scripts/encoding/pixe-enhancment/results/enhanced"
    )
    
    # Alternative: Replace in existing pipeline
    # staff_data, staff_json = replace_in_pipeline(image_path, detected_elements_path)