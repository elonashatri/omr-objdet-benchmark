import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json

def align_staff_lines_pixel_perfect(image_path, staff_data_path, output_path=None):
    """
    Perform pixel-perfect alignment of staff lines with the actual image content
    
    Args:
        image_path: Path to the music score image
        staff_data_path: Path to the JSON file with staff line data
        output_path: Path to save aligned staff line data (optional)
        
    Returns:
        Dictionary with pixel-perfect aligned staff line data
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Load staff data
    if isinstance(staff_data_path, str):
        with open(staff_data_path, 'r') as f:
            staff_data = json.load(f)
    else:
        staff_data = staff_data_path
    
    # Ensure image is properly oriented (staff lines should be dark)
    # Calculate mean brightness across the entire image
    mean_brightness = np.mean(img)
    if mean_brightness > 127:  # If image is predominantly bright
        img = 255 - img  # Invert the image
    
    # Enhanced preprocessing to make staff lines more prominent
    enhanced_img = enhance_staff_lines(img)
    
    # Apply pixel-perfect alignment
    aligned_data = perfectly_align_staff_lines(staff_data, enhanced_img)
    
    # Save aligned data if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(aligned_data, f, indent=2)
    
    return aligned_data

def enhance_staff_lines(img):
    """
    Enhance staff lines in the image for better detection
    
    Args:
        img: Grayscale image
        
    Returns:
        Enhanced image with more prominent staff lines
    """
    # Apply bilateral filtering to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 4
    )
    
    # Create horizontal kernel for enhancing staff lines
    kernel_length = max(1, img.shape[1] // 80)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    
    # Apply morphological operations to enhance horizontal lines
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Dilate slightly to make staff lines more prominent
    horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=1)
    
    return horizontal

def perfectly_align_staff_lines(staff_data, enhanced_img):
    """
    Precisely align staff lines to actual dark pixels in the image
    
    Args:
        staff_data: Dictionary with staff line data
        enhanced_img: Enhanced binary image
        
    Returns:
        Staff data with pixel-perfect alignment
    """
    height, width = enhanced_img.shape
    aligned_data = staff_data.copy()
    aligned_detections = []
    
    # Process each staff system
    for system in staff_data.get("staff_systems", []):
        system_id = system["id"]
        new_line_indices = []
        
        # Process each line in this system
        for line_idx in system.get("lines", []):
            original_line = staff_data["detections"][line_idx]
            x1 = int(original_line["bbox"]["x1"])
            x2 = int(original_line["bbox"]["x2"])
            center_y = int(original_line["bbox"]["center_y"])
            thickness = original_line["bbox"]["height"]
            
            # Define a precise search window - just a few pixels around the current position
            # We only want small refinements, not major shifts
            window_size = max(3, min(9, int(thickness * 2)))  # Adaptive window size
            y_start = max(0, center_y - window_size)
            y_end = min(height - 1, center_y + window_size)
            
            # Skip if window is invalid
            if y_start >= y_end:
                aligned_detections.append(original_line)
                new_line_indices.append(len(aligned_detections) - 1)
                continue
            
            # Extract window from enhanced image
            window = enhanced_img[y_start:y_end+1, x1:x2]
            
            # If window is empty or too narrow, keep original position
            if window.size == 0 or window.shape[1] < 10:
                aligned_detections.append(original_line)
                new_line_indices.append(len(aligned_detections) - 1)
                continue
            
            # Calculate horizontal projection profile within window
            projection = np.sum(window, axis=1)
            
            # Find row with maximum darkness
            if np.max(projection) > 0:
                # If there are multiple peaks, use more sophisticated approach
                if np.count_nonzero(projection == np.max(projection)) > 1:
                    # Use find_peaks to identify the most prominent peak
                    peaks, _ = find_peaks(projection, height=np.max(projection)*0.8, distance=2)
                    
                    if len(peaks) > 0:
                        # Calculate peak prominence
                        prominences = []
                        for peak in peaks:
                            # Find local minimum on both sides
                            left_min = np.min(projection[:peak+1]) if peak > 0 else projection[0]
                            right_min = np.min(projection[peak:]) if peak < len(projection)-1 else projection[-1]
                            prominence = projection[peak] - max(left_min, right_min)
                            prominences.append(prominence)
                        
                        # Select peak with highest prominence
                        best_peak = peaks[np.argmax(prominences)]
                        best_y = y_start + best_peak
                    else:
                        # Fallback to maximum if no peaks found
                        best_y = y_start + np.argmax(projection)
                else:
                    # Simple case - just use the maximum
                    best_y = y_start + np.argmax(projection)
                
                # Only adjust if the shift is reasonably small
                if abs(best_y - center_y) <= window_size:
                    # Create new line with adjusted position
                    new_line = original_line.copy()
                    new_line["bbox"] = original_line["bbox"].copy()
                    
                    # Update y-coordinates
                    new_line["bbox"]["y1"] = float(best_y - thickness/2)
                    new_line["bbox"]["y2"] = float(best_y + thickness/2)
                    new_line["bbox"]["center_y"] = float(best_y)
                    
                    # Increase confidence slightly
                    new_line["confidence"] = min(1.0, original_line.get("confidence", 0.9) + 0.05)
                    
                    aligned_detections.append(new_line)
                else:
                    # If shift would be too large, keep original position
                    aligned_detections.append(original_line)
            else:
                # If no clear peak, keep original position
                aligned_detections.append(original_line)
            
            new_line_indices.append(len(aligned_detections) - 1)
        
        # Update system with new line indices
        system["lines"] = new_line_indices
    
    # Create final aligned staff data
    aligned_data["detections"] = aligned_detections
    
    return aligned_data

def visualize_alignment_improvement(image_path, original_staff_path, aligned_staff_path, output_path=None):
    """
    Create visualization showing the alignment improvement
    
    Args:
        image_path: Path to the music score image
        original_staff_path: Path to the original staff data
        aligned_staff_path: Path to the aligned staff data
        output_path: Path to save visualization
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load staff data
    if isinstance(original_staff_path, str):
        with open(original_staff_path, 'r') as f:
            original_staff = json.load(f)
    else:
        original_staff = original_staff_path
        
    if isinstance(aligned_staff_path, str):
        with open(aligned_staff_path, 'r') as f:
            aligned_staff = json.load(f)
    else:
        aligned_staff = aligned_staff_path
    
    # Create figure
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    
    # Colors for visualization
    original_color = (1, 0, 0, 0.7)    # Red with alpha
    aligned_color = (0, 0.7, 0, 0.8)   # Green with alpha
    
    # Draw original staff lines
    for system in original_staff.get("staff_systems", []):
        for line_idx in system.get("lines", []):
            if line_idx < len(original_staff.get("detections", [])):
                line = original_staff["detections"][line_idx]
                x1 = line["bbox"]["x1"]
                y1 = line["bbox"]["center_y"]
                x2 = line["bbox"]["x2"]
                y2 = line["bbox"]["center_y"]
                
                # Plot original staff line
                plt.plot([x1, x2], [y1, y2], color=original_color, linewidth=2,
                         label='Original' if line_idx == system["lines"][0] else "")
    
    # Draw aligned staff lines
    for system in aligned_staff.get("staff_systems", []):
        for line_idx in system.get("lines", []):
            if line_idx < len(aligned_staff.get("detections", [])):
                line = aligned_staff["detections"][line_idx]
                x1 = line["bbox"]["x1"]
                y1 = line["bbox"]["center_y"]
                x2 = line["bbox"]["x2"]
                y2 = line["bbox"]["center_y"]
                
                # Plot aligned staff line
                plt.plot([x1, x2], [y1, y2], color=aligned_color, linewidth=1.5,
                         label='Pixel-perfect' if line_idx == system["lines"][0] else "")
    
    # Add legend (only once for each color)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    plt.title("Staff Line Alignment Comparison")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

# Example usage function
def fix_staff_alignment(image_path, staff_json_path, output_path=None):
    """
    Convenience function to fix staff line alignment with pixel-perfect accuracy
    
    Args:
        image_path: Path to the music score image
        staff_json_path: Path to the staff line JSON file
        output_path: Path to save aligned data (optional)
        
    Returns:
        Dictionary with aligned staff data
    """
    # Default output path if not provided
    if output_path is None:
        import os
        base_dir = os.path.dirname(staff_json_path)
        base_name = os.path.splitext(os.path.basename(staff_json_path))[0]
        output_path = os.path.join(base_dir, f"{base_name}_pixel_perfect.json")
    
    # Visualiation path
    viz_path = output_path.replace('.json', '_comparison.png')
    
    # Align staff lines
    print(f"Performing pixel-perfect alignment of staff lines...")
    aligned_data = align_staff_lines_pixel_perfect(image_path, staff_json_path, output_path)
    
    # Create visualization
    print(f"Creating visualization of alignment improvement...")
    visualize_alignment_improvement(image_path, staff_json_path, aligned_data, viz_path)
    
    print(f"Pixel-perfect alignment complete!")
    print(f"- Aligned staff data saved to: {output_path}")
    print(f"- Visualization saved to: {viz_path}")
    
    return aligned_data