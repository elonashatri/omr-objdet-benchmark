import cv2
import numpy as np
import json
from collections import defaultdict

def load_detections(json_file_path):
    """
    Load detection results from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file with detection results
        
    Returns:
        Dictionary containing detection results
    """
    with open(json_file_path, 'r') as f:
        detections = json.load(f)
    return detections

def check_stem_region(binary, x1, y1, x2, y2, direction, notehead_width):
    """
    Check a region for a stem using multiple detection methods.
    """
    if x1 >= x2 or y1 >= y2:
        return None
        
    # Extract region of interest
    roi = binary[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    # Calculate expected stem width
    expected_stem_width = max(1, notehead_width // 5)
    
    # METHOD 1: Morphological operations
    # Create a more flexible vertical kernel
    kernel_width = max(1, expected_stem_width)
    vertical_kernel = np.ones((max(5, roi.shape[0] // 3), kernel_width), np.uint8)
    vertical_structures = cv2.morphologyEx(roi, cv2.MORPH_OPEN, vertical_kernel)
    
    # Use dilation to connect potentially broken stems
    dilate_kernel = np.ones((5, kernel_width * 2), np.uint8)
    vertical_structures = cv2.dilate(vertical_structures, dilate_kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(vertical_structures, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Try contour-based detection first
    if contours:
        # Filter contours based on shape and position
        stem_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = h / w if w > 0 else 0
            
            # Relaxed criteria
            if aspect_ratio < 1.5 or h < roi.shape[0] // 6:
                continue
            
            # Calculate alignment with center of ROI
            center_x = roi.shape[1] // 2
            distance_from_center = abs((x + w//2) - center_x)
            alignment_score = 1 - (distance_from_center / (roi.shape[1] // 2)) if roi.shape[1] > 0 else 0
            
            # Score based on height, aspect ratio, and alignment
            score = h * aspect_ratio * (alignment_score + 0.5)  # Boost alignment
            stem_candidates.append((x, y, w, h, score))
        
        if stem_candidates:
            # Choose the best candidate
            best_stem = max(stem_candidates, key=lambda s: s[4])
            x, y, w, h = best_stem[:4]
            
            # Convert back to original image coordinates
            return (x1 + x, y1 + y, x1 + x + w, y1 + y + h)
    
    # METHOD 2: Try Hough Line Transform
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, 
                            minLineLength=roi.shape[0]//3, maxLineGap=roi.shape[0]//4)
    
    if lines is not None:
        best_line = None
        best_score = 0
        
        for line in lines:
            x_start, y_start, x_end, y_end = line[0]
            
            # Check if the line is vertical enough
            dx = abs(x_end - x_start)
            dy = abs(y_end - y_start)
            
            if dx > dy / 3:  # Allow some slant, but must be mostly vertical
                continue
                
            # Calculate distance from center
            line_center_x = (x_start + x_end) // 2
            roi_center_x = roi.shape[1] // 2
            distance_from_center = abs(line_center_x - roi_center_x)
            
            # Score based on length and alignment
            line_length = np.sqrt(dx**2 + dy**2)
            alignment_score = 1 - (distance_from_center / (roi.shape[1] // 2)) if roi.shape[1] > 0 else 0
            
            score = line_length * (alignment_score + 0.5)
            
            if score > best_score:
                best_score = score
                best_line = line[0]
        
        # Fix for "ValueError: The truth value of an array with more than one element is ambiguous"
        if best_line is not None:  # Changed from "if best_line:"
            x_start, y_start, x_end, y_end = best_line
            
            # Get stem direction from line points
            if y_start > y_end:
                x_start, y_start, x_end, y_end = x_end, y_end, x_start, y_start
                
            # Create a rectangle around the line with proper width
            mid_x = (x_start + x_end) // 2
            stem_width = expected_stem_width * 2
            
            stem_x = mid_x - stem_width // 2
            stem_y = min(y_start, y_end)
            stem_w = stem_width
            stem_h = abs(y_end - y_start)
            
            # Convert back to original image coordinates
            return (x1 + stem_x, y1 + stem_y, x1 + stem_x + stem_w, y1 + stem_y + stem_h)
    
    # METHOD 3: Use vertical projection profile
    if roi.shape[1] > 0 and roi.shape[0] > 0:
        try:
            # Import scipy here to avoid import errors
            from scipy.signal import find_peaks
            
            # Sum each column to find vertical lines
            vertical_projection = np.sum(roi, axis=0)
            
            # Find peaks in the projection
            peaks, _ = find_peaks(vertical_projection, height=roi.shape[0]//3)
            
            if len(peaks) > 0:
                # Find peak closest to center
                center = roi.shape[1] // 2
                closest_peak = peaks[np.argmin(np.abs(peaks - center))]
                
                stem_width = expected_stem_width * 2
                stem_x = max(0, closest_peak - stem_width // 2)
                stem_y = 0
                stem_w = min(stem_width, roi.shape[1] - stem_x)
                stem_h = roi.shape[0]
                
                return (x1 + stem_x, y1 + stem_y, x1 + stem_x + stem_w, y1 + stem_y + stem_h)
        except ImportError:
            # Skip this method if scipy is not available
            pass
    
    return None


def detect_stems(image_path, detection_json_path):
    """
    Enhanced stem detection with multiple methods and fallbacks.
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Load detection results
    detection_results = load_detections(detection_json_path)
    
    # Extract notehead detections
    notehead_detections = []
    for detection in detection_results["detections"]:
        if detection["class_name"] in ["noteheadBlack", "noteheadHalf"]:
            notehead_detections.append(detection)
    
    # Enhanced preprocessing to make stems more visible
    # 1. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_enhanced = clahe.apply(image)
    
    # 2. Sharpen the image to enhance edges
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    image_sharpened = cv2.filter2D(image_enhanced, -1, kernel)
    
    # 3. Multiple binary versions to capture stems in different lighting conditions
    _, binary_standard = cv2.threshold(image_sharpened, 200, 255, cv2.THRESH_BINARY_INV)
    binary_adaptive = cv2.adaptiveThreshold(image_sharpened, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 2)
    
    # 4. Create a combined binary image
    binary_combined = cv2.bitwise_or(binary_standard, binary_adaptive)
    
    # Detect staff lines to inform stem direction
    staff_lines = detect_staff_lines(image_path)
    staff_line_y = sorted(staff_lines) if staff_lines else []
    
    stems = []
    failed_noteheads = []
    
    # First pass: Attempt normal stem detection for each notehead
    for detection in notehead_detections:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        width = x2 - x1
        height = y2 - y1
        
        # Determine expected stem direction based on staff lines if available
        expected_direction = None
        if staff_line_y:
            # Find the middle line position
            if len(staff_line_y) >= 3:
                middle_line = staff_line_y[len(staff_line_y) // 2]
                expected_direction = "down" if center_y < middle_line else "up"
        
        # Create search regions with larger areas
        padding_x = width * 2  # Much wider search area
        search_height = height * 6  # Much longer search height
        
        # Upward stem search area
        up_x1 = max(0, center_x - padding_x)
        up_x2 = min(image.shape[1], center_x + padding_x)
        up_y1 = max(0, y1 - search_height)
        up_y2 = y1 + height
        
        # Downward stem search area
        down_x1 = max(0, center_x - padding_x)
        down_x2 = min(image.shape[1], center_x + padding_x)
        down_y1 = y1
        down_y2 = min(image.shape[0], y2 + search_height)
        
        # Try both methods and binarizations
        stem_found = False
        
        # If we have expected direction, try that first
        if expected_direction:
            search_order = [expected_direction, "up" if expected_direction == "down" else "down"]
        else:
            search_order = ["up", "down"]
        
        for direction in search_order:
            for binary in [binary_combined, binary_standard, binary_adaptive]:
                # Skip iterations after finding a stem
                if stem_found:
                    break
                    
                if direction == "up":
                    stem = check_stem_region(binary, up_x1, up_y1, up_x2, up_y2, "up", width)
                else:
                    stem = check_stem_region(binary, down_x1, down_y1, down_x2, down_y2, "down", width)
                
                if stem:
                    stems.append({
                        "notehead_id": detection.get("id", len(stems)),
                        "direction": direction,
                        "x1": stem[0],
                        "y1": stem[1],
                        "x2": stem[2],
                        "y2": stem[3]
                    })
                    stem_found = True
        
        # If still not found, save for second pass
        if not stem_found:
            failed_noteheads.append(detection)
    
    # Second pass: Try more aggressive methods for failed noteheads
    for detection in failed_noteheads:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Try direct line detection on a larger area
        padding_x = width * 3
        search_height = height * 8
        
        roi_up_x1 = max(0, center_x - padding_x)
        roi_up_x2 = min(image.shape[1], center_x + padding_x)
        roi_up_y1 = max(0, y1 - search_height)
        roi_up_y2 = y1 + height
        
        roi_down_x1 = max(0, center_x - padding_x)
        roi_down_x2 = min(image.shape[1], center_x + padding_x)
        roi_down_y1 = y1
        roi_down_y2 = min(image.shape[0], y2 + search_height)
        
        # Try Hough Line Transform directly
        stem_found = False
        
        # Try both directions
        for direction, (roi_x1, roi_y1, roi_x2, roi_y2) in [
            ("up", (roi_up_x1, roi_up_y1, roi_up_x2, roi_up_y2)),
            ("down", (roi_down_x1, roi_down_y1, roi_down_x2, roi_down_y2))
        ]:
            if stem_found:
                break
                
            roi = binary_combined[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0:
                continue
                
            # Use Canny edge detection and Hough Lines
            edges = cv2.Canny(roi, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                   minLineLength=height, maxLineGap=height//2)
            
            if lines is not None:
                best_line = None
                max_length = 0
                
                for line in lines:
                    x_start, y_start, x_end, y_end = line[0]
                    
                    # Check if the line is vertical
                    if abs(x_start - x_end) > width // 2:
                        continue
                        
                    # Check if the line passes near the notehead center
                    x_center = (x_start + x_end) // 2
                    notehead_center_in_roi = center_x - roi_x1
                    if abs(x_center - notehead_center_in_roi) > width:
                        continue
                        
                    # Calculate line length
                    length = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
                    
                    if length > max_length:
                        max_length = length
                        best_line = line[0]
                
                if best_line and max_length > height * 1.5:
                    x_start, y_start, x_end, y_end = best_line
                    # Convert to original image coordinates
                    stem_x1 = roi_x1 + min(x_start, x_end) - width//4
                    stem_y1 = roi_y1 + min(y_start, y_end)
                    stem_x2 = roi_x1 + max(x_start, x_end) + width//4
                    stem_y2 = roi_y1 + max(y_start, y_end)
                    
                    stems.append({
                        "notehead_id": detection.get("id", len(stems)),
                        "direction": direction,
                        "x1": stem_x1,
                        "y1": stem_y1,
                        "x2": stem_x2,
                        "y2": stem_y2
                    })
                    stem_found = True
        
        # Last resort: If still not found, use a reasonable guess based on notehead position
        if not stem_found and staff_line_y:
            # Find the middle line position
            if len(staff_line_y) >= 3:
                middle_line = staff_line_y[len(staff_line_y) // 2]
                direction = "down" if center_y < middle_line else "up"
                
                # Create a synthetic stem
                stem_width = max(2, width // 10)
                stem_length = height * 3
                
                if direction == "up":
                    stem_x1 = center_x - stem_width // 2
                    stem_y1 = max(0, y1 - stem_length)
                    stem_x2 = center_x + stem_width // 2
                    stem_y2 = y1
                else:
                    stem_x1 = center_x - stem_width // 2
                    stem_y1 = y2
                    stem_x2 = center_x + stem_width // 2
                    stem_y2 = min(image.shape[0], y2 + stem_length)
                
                stems.append({
                    "notehead_id": detection.get("id", len(stems)),
                    "direction": direction,
                    "x1": stem_x1,
                    "y1": stem_y1,
                    "x2": stem_x2,
                    "y2": stem_y2,
                    "is_synthetic": True  # Mark as synthetic
                })
    
    return stems, notehead_detections

def visualize_results(image_path, noteheads, stems, output_path="detected_stems.jpg"):
    """
    Visualize detection results with improved clarity.
    """
    image = cv2.imread(image_path)
    
    # Create a copy for overlay
    overlay = image.copy()
    
    # Draw noteheads
    for notehead in noteheads:
        bbox = notehead["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw stems
    for stem in stems:
        x1, y1, x2, y2 = stem["x1"], stem["y1"], stem["x2"], stem["y2"]
        
        # Check if stem is synthetic
        is_synthetic = stem.get("is_synthetic", False)
        
        if is_synthetic:
            # Draw synthetic stems in purple
            color = (255, 0, 255)  # Purple
        else:
            # Draw detected stems in blue
            color = (255, 0, 0)  # Blue
            
        # Draw stem bounding box with lower opacity
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        
        # Draw a thicker vertical line down the center for emphasis
        center_x = (x1 + x2) // 2
        cv2.line(image, (center_x, y1), (center_x, y2), color, 2)
    
    # Blend overlay with lower opacity
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Add detection statistics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"Noteheads: {len(noteheads)}, Stems: {len(stems)}, Rate: {len(stems)/len(noteheads)*100:.1f}%", 
                (10, 30), font, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, image)
    print(f"Results saved to {output_path}")

def detect_staff_lines(image_path):
    """
    Detect staff lines using Hough Line Transform.
    
    Args:
        image_path: Path to the score image
        
    Returns:
        List of staff line y-coordinates
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Binarize the image
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Use horizontal morphology to enhance staff lines
    horizontal_kernel = np.ones((1, 50), np.uint8)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    staff_line_ys = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter only horizontal lines
            if abs(y2 - y1) < 5:  # Allow slight slant
                staff_line_ys.append((y1 + y2) // 2)
    
    # Sort and cluster close y-coordinates (staff lines are usually 5 lines grouped together)
    staff_line_ys.sort()
    
    return staff_line_ys

def main():
    """Main function to run the stem detection pipeline."""
    image_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/results/Accidentals-004.png"
    detection_json_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/results/object_detections/Accidentals-004_detections.json"  # This is your YOLOv8 detection JSON file
    
    # Detect stems
    stems, noteheads = detect_stems(image_path, detection_json_path)
    
    # Visualize results
    visualize_results(image_path, noteheads, stems)
    
    # Print statistics
    print(f"Found {len(noteheads)} noteheads and {len(stems)} stems")
    print(f"Stem detection rate: {len(stems)/len(noteheads)*100:.2f}%")

    # Optional: Detect staff lines
    staff_lines = detect_staff_lines(image_path)
    print(f"Detected {len(staff_lines)} potential staff lines")

if __name__ == "__main__":
    main()