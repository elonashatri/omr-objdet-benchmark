"""
Inference Module for Staff Line Detection

This module handles inference and visualization using a trained staff line detection model.
With enhanced staff line detection using musical knowledge.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
import json

# Import model
from model import StaffLineDetectionNet, post_process_staff_lines


def load_image(image_path, max_size=512):
    """
    Load and preprocess an image for inference, with size limits to prevent memory issues.
    
    Args:
        image_path (str): Path to the image
        max_size (int): Maximum dimension (width or height) to resize to
        
    Returns:
        tuple: (torch.Tensor, numpy.ndarray, float) - Preprocessed image tensor, original image, and scale factor
    """
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Store original image and its dimensions
    original_image = image.copy()
    original_height, original_width = image.shape
    
    # Calculate scale factor to resize image if it's too large
    scale = 1.0
    if max(original_height, original_width) > max_size:
        scale = max_size / max(original_height, original_width)
        new_height = int(original_height * scale)
        new_width = int(original_width * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
    
    # Normalize and convert to tensor
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    return image_tensor, original_image, scale


def predict_staff_lines(model, image_tensor, device, post_process=True, min_line_length=50, max_line_gap=10):
    """
    Predict staff lines in an image, with memory-efficient processing.
    
    Args:
        model: The staff line detection model
        image_tensor (torch.Tensor): Input image tensor
        device (torch.device): Device to run inference on
        post_process (bool): Whether to apply post-processing
        min_line_length (int): Minimum length for line detection in post-processing
        max_line_gap (int): Maximum gap to connect in post-processing
        
    Returns:
        numpy.ndarray: Predicted staff line mask
    """
    # Check tensor size
    _, _, h, w = image_tensor.shape
    print(f"Processing image of size {w}x{h}")
    
    # Move image to device
    image_tensor = image_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    try:
        # Try to process the whole image
        with torch.no_grad():
            prediction = model(image_tensor)
            prediction = torch.sigmoid(prediction)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("WARNING: Out of memory error. Falling back to CPU processing.")
            # Move model and data to CPU
            model = model.cpu()
            image_tensor = image_tensor.cpu()
            
            # Process on CPU
            with torch.no_grad():
                prediction = model(image_tensor)
                prediction = torch.sigmoid(prediction)
                
            # Move model back to GPU for future processing
            model = model.to(device)
        else:
            # Re-raise if it's not a memory error
            raise e
    
    # Convert to numpy array
    prediction_np = prediction.squeeze().cpu().numpy()
    
    # Apply threshold
    binary_prediction = (prediction_np > 0.5).astype(np.float32)
    
    # Apply post-processing if enabled
    if post_process:
        binary_prediction = post_process_staff_lines(binary_prediction, min_line_length, max_line_gap)
    
    return binary_prediction


def visualize_prediction(image, prediction, title=None, save_path=None, show=True):
    """
    Visualize the prediction overlaid on the input image.
    
    Args:
        image (numpy.ndarray): Input image
        prediction (numpy.ndarray): Predicted staff line mask
        title (str, optional): Title for the visualization
        save_path (str, optional): Path to save the visualization
        show (bool): Whether to display the visualization
    """
    # Ensure prediction is in range [0, 1]
    if prediction.max() > 1.0:
        prediction = prediction / 255.0
    
    # Create a figure with subplots
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gray')
    plt.title('Staff Line Prediction')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    # Create RGB image for overlay
    if len(image.shape) == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = image
    
    # Create overlay (red for staff lines)
    overlay = np.zeros_like(rgb_image)
    if len(prediction.shape) == 2:
        mask = (prediction > 0.5)
        overlay[mask] = [255, 0, 0]  # Red
    else:
        # Handle case where prediction is already RGB
        overlay = prediction
    
    # Blend
    alpha = 0.5
    blended = cv2.addWeighted(rgb_image, 1, overlay, alpha, 0)
    
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title('Overlay')
    plt.axis('off')
    
    # Add title if provided
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save the visualization if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # Show the visualization if enabled
    if show:
        plt.show()
    else:
        plt.close()


def extract_staff_lines(prediction, min_length_ratio=0.2):
    """
    Extract staff line information from the prediction with improved detection.
    
    Args:
        prediction (numpy.ndarray): Predicted staff line mask
        min_length_ratio (float): Minimum length ratio for a valid staff line
        
    Returns:
        list: List of staff line coordinates [(y1, x1, x2), ...]
    """
    h, w = prediction.shape
    min_length = int(w * min_length_ratio)
    
    # Convert to 8-bit image if needed
    if prediction.dtype != np.uint8:
        pred_8bit = (prediction * 255).astype(np.uint8)
    else:
        pred_8bit = prediction
    
    # Apply morphological operations to enhance horizontal lines
    kernel_length = max(5, w // 100)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    
    # Apply morphology operations
    enhanced = cv2.morphologyEx(pred_8bit, cv2.MORPH_OPEN, horizontal_kernel)
    enhanced = cv2.dilate(enhanced, horizontal_kernel, iterations=1)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(enhanced, connectivity=8)
    
    # Process each component
    staff_lines = []
    for i in range(1, num_labels):  # Skip background (label 0)
        # Get component information
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Check if it's a potential staff line (long and thin)
        if width > min_length and height < 10:
            # Find the average y-coordinate (row) of the line
            mask = (labels == i)
            y_coords = np.where(mask)[0]
            y = int(np.mean(y_coords))
            
            # Find the leftmost and rightmost x-coordinates
            x_coords = np.where(mask)[1]
            x1 = np.min(x_coords)
            x2 = np.max(x_coords)
            
            staff_lines.append((y, x1, x2))
    
    # If we didn't find enough lines, try Hough transform
    if len(staff_lines) < 5:  # A standard staff should have 5 lines
        lines = cv2.HoughLinesP(
            enhanced, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=min_length,
            maxLineGap=20
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Filter for horizontal lines (small y difference)
                if abs(y2 - y1) < 5:
                    # Use average y-coordinate
                    y = (y1 + y2) // 2
                    
                    # Ensure x1 < x2
                    if x1 > x2:
                        x1, x2 = x2, x1
                    
                    staff_lines.append((y, x1, x2))
    
    # Sort staff lines by y-coordinate
    staff_lines.sort()
    
    # Remove duplicate lines (lines that are very close to each other)
    filtered_lines = []
    if staff_lines:
        filtered_lines.append(staff_lines[0])
        for i in range(1, len(staff_lines)):
            prev_y = filtered_lines[-1][0]
            curr_y = staff_lines[i][0]
            
            # If lines are more than 3 pixels apart, consider them different
            if curr_y - prev_y > 3:
                filtered_lines.append(staff_lines[i])
    
    return filtered_lines


def analyze_staff_systems(staff_lines, max_gap_ratio=0.3):
    """
    Group staff lines into staff systems with improved handling of incomplete systems.
    
    Args:
        staff_lines (list): List of staff line coordinates [(y1, x1, x2), ...]
        max_gap_ratio (float): Maximum gap ratio between lines in the same system
        
    Returns:
        list: List of staff systems, each containing 5 staff lines
    """
    if not staff_lines:
        return []
    
    # Calculate the median line spacing
    spacings = []
    for i in range(1, len(staff_lines)):
        spacing = staff_lines[i][0] - staff_lines[i-1][0]
        spacings.append(spacing)
    
    median_spacing = np.median(spacings) if spacings else 0
    max_gap = median_spacing * (1 + max_gap_ratio)
    
    # Group lines into systems
    staff_systems = []
    current_system = [staff_lines[0]]
    
    for i in range(1, len(staff_lines)):
        current_line = staff_lines[i]
        prev_line = staff_lines[i-1]
        
        # Check if this line belongs to the current system
        if current_line[0] - prev_line[0] <= max_gap:
            current_system.append(current_line)
        else:
            # Start a new system
            staff_systems.append(current_system)
            current_system = [current_line]
    
    # Add the last system
    if current_system:
        staff_systems.append(current_system)
    
    # Process each system to handle incomplete systems
    valid_systems = []
    for system in staff_systems:
        # If we already have 5 lines, it's a standard staff
        if len(system) == 5:
            valid_systems.append(system)
        elif 3 <= len(system) < 5:
            # Try to complete the system
            completed_system = complete_staff_system(system, staff_lines)
            valid_systems.append(completed_system)
        # Ignore systems with fewer than 3 lines
    
    return valid_systems


def save_json_results(staff_lines, staff_systems, output_path, scale=1.0):
    """
    Save staff line detection results in JSON format with proper scaling.
    """
    json_data = {"staff_systems": [], "detections": []}
    line_to_system_mapping = {}
    line_index = 0

    for system_idx, system in enumerate(staff_systems):
        line_indices = []
        for i, line in enumerate(system):
            line_indices.append(line_index)
            key = f"{line[0]}_{line[1]}_{line[2]}"
            line_to_system_mapping[key] = (system_idx, i)
            line_index += 1

        json_data["staff_systems"].append({
            "id": system_idx,
            "lines": line_indices
        })

    for line in staff_lines:
        y, x1, x2 = line
        key = f"{y}_{x1}_{x2}"
        system_idx, line_idx = line_to_system_mapping.get(key, (-1, -1))

        height = 4  # Default height
        y1 = y - height // 2
        y2 = y + height // 2
        width = x2 - x1
        center_x = x1 + width // 2
        center_y = y

        detection_entry = {
            "class_id": 0,
            "class_name": "staff_line",
            "confidence": 1.0,
            "bbox": {
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "width": float(width),
                "height": float(height),
                "center_x": float(center_x),
                "center_y": float(center_y)
            },
            "staff_system": system_idx,
            "line_number": line_idx
        }
        json_data["detections"].append(detection_entry)

    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"Saved JSON with {len(json_data['detections'])} detections.")
    return json_data



def save_inferred_json_results(output_path, scale=1.0, prediction=None, image=None):
    """
    Create a JSON result using musical knowledge when standard detection fails.
    This function will attempt to infer staff lines from the prediction or image.
    
    Args:
        output_path (str): Path to save the JSON file
        scale (float): Scale factor used during inference
        prediction (numpy.ndarray, optional): Predicted staff line mask
        image (numpy.ndarray, optional): Original image
    
    Returns:
        dict: The JSON data structure that was saved
    """
    print("Using musical knowledge to infer staff lines...")
    
    # Initialize the JSON structure
    json_data = {
        "staff_systems": [],
        "detections": []
    }
    
    # If we have a prediction, try to enhance it
    if prediction is not None:
        # Apply further processing to find even faint horizontal lines
        enhanced_prediction = enhance_staff_line_detection(prediction)
        
        # Prediction already resized to original dimensions
        staff_lines = find_horizontal_lines(prediction, min_length_ratio=0.7)
        merged_lines = merge_staff_lines(staff_lines, vertical_merge_thresh=8)
        staff_systems = enforce_five_lines_per_staff(merged_lines)

        # Continue saving JSON and visualization
        save_json_results(merged_lines, staff_systems, json_path, scale=1.0)

        # # If we found lines, group them into staff systems
        # if len(staff_lines) > 0:
        #     # Group lines into systems (standard music has 5 lines per staff)
        #     staff_systems = group_lines_into_systems(staff_lines)
            
            # Populate JSON data
            # populate_json_with_lines(json_data, staff_lines, staff_systems, scale)
    
    # If we still don't have any results but have the original image
    if len(json_data["detections"]) == 0 and image is not None:
        # Apply direct line detection on the original image
        staff_lines = detect_lines_from_image(image)
        
        if len(staff_lines) > 0:
            staff_systems = group_lines_into_systems(staff_lines)
            populate_json_with_lines(json_data, staff_lines, staff_systems, scale)
    
    # If we still couldn't find any staff lines, create an empty result
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Inferred JSON with {len(json_data['detections'])} detections and {len(json_data['staff_systems'])} systems")
    return json_data


def enhance_staff_line_detection(prediction):
    """
    Enhance the prediction to better detect staff lines.
    
    Args:
        prediction (numpy.ndarray): Predicted staff line mask
        
    Returns:
        numpy.ndarray: Enhanced prediction
    """
    # Ensure prediction is in the right format (8-bit single channel)
    if prediction.dtype != np.uint8:
        prediction_8bit = (prediction * 255).astype(np.uint8)
    else:
        prediction_8bit = prediction.copy()
    
    # Apply morphological operations to enhance horizontal lines
    # Create a horizontal kernel
    kernel_length = max(5, prediction.shape[1] // 100)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    
    # Apply morphology operations
    enhanced = cv2.morphologyEx(prediction_8bit, cv2.MORPH_OPEN, horizontal_kernel)
    enhanced = cv2.dilate(enhanced, horizontal_kernel, iterations=1)
    
    # Apply adaptive thresholding to handle varying line intensities
    enhanced = cv2.adaptiveThreshold(
        cv2.GaussianBlur(enhanced, (3, 3), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    return enhanced


def find_horizontal_lines(image, min_length_ratio=0.7, max_line_gap=30):
    """
    Improved parameters for robust horizontal line detection.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    width = image.shape[1]
    lines = cv2.HoughLinesP(
        image, 
        rho=1,
        theta=np.pi / 180,
        threshold=100,  # increased threshold to reduce noise
        minLineLength=int(width * min_length_ratio),  # long enough to represent real staff lines
        maxLineGap=max_line_gap  # more generous gap allowance to reduce fragmentation
    )
    
    staff_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 5:  # strict horizontal tolerance
                y_avg = (y1 + y2) // 2
                staff_lines.append((y_avg, min(x1, x2), max(x1, x2)))

    staff_lines.sort(key=lambda line: line[0])
    return staff_lines

def merge_staff_lines(lines, vertical_merge_thresh=8):
    """
    Merge lines that are vertically close, assuming they represent the same staff line.
    """
    if not lines:
        return []

    lines.sort(key=lambda l: l[0])
    merged_lines = []
    current_y, x1, x2 = lines[0]

    for line in lines[1:]:
        y, line_x1, line_x2 = line
        if abs(y - current_y) <= vertical_merge_thresh:
            current_y = (current_y + y) // 2
            x1 = min(x1, line_x1)
            x2 = max(x2, line_x2)
        else:
            merged_lines.append((current_y, x1, x2))
            current_y, x1, x2 = line
    merged_lines.append((current_y, x1, x2))

    return merged_lines



def merge_interrupted_lines(lines, vertical_tolerance=5, horizontal_gap=20):
    """
    Merge horizontally aligned line segments into single lines.
    
    Args:
        lines (list): List of detected lines [(y, x1, x2), ...]
        vertical_tolerance (int): Vertical distance threshold to consider lines the same
        horizontal_gap (int): Maximum horizontal gap to merge lines
        
    Returns:
        list: Merged lines [(y, x1, x2), ...]
    """
    if not lines:
        return []

    # Sort lines vertically first, then horizontally
    lines.sort(key=lambda l: (l[0], l[1]))
    
    merged_lines = []
    current_line = lines[0]
    
    for line in lines[1:]:
        y, x1, x2 = line
        curr_y, curr_x1, curr_x2 = current_line
        
        # Check if lines are vertically close enough
        if abs(y - curr_y) <= vertical_tolerance:
            # Check if horizontally close to merge
            if x1 - curr_x2 <= horizontal_gap:
                # Extend current line horizontally
                current_line = (int((y + curr_y) / 2), min(curr_x1, x1), max(curr_x2, x2))
            else:
                merged_lines.append(current_line)
                current_line = line
        else:
            merged_lines.append(current_line)
            current_line = line
    
    merged_lines.append(current_line)
    return merged_lines

def group_lines_into_staff_systems(merged_lines, expected_lines_per_staff=5, vertical_spacing_tolerance=5):
    """
    Group lines strictly into staff systems containing exactly five lines each.
    
    Args:
        merged_lines (list): Merged staff lines [(y, x1, x2), ...]
        expected_lines_per_staff (int): Number of lines per staff system
        vertical_spacing_tolerance (int): Allowed vertical spacing variation
    
    Returns:
        list: Valid staff systems with exactly five lines
    """
    if len(merged_lines) < expected_lines_per_staff:
        return []

    merged_lines.sort(key=lambda l: l[0])
    
    staff_systems = []
    temp_group = []

    for line in merged_lines:
        if not temp_group:
            temp_group.append(line)
            continue
        
        last_line_y = temp_group[-1][0]
        spacing = line[0] - last_line_y
        
        if len(temp_group) < expected_lines_per_staff:
            if not temp_group or abs(spacing - (temp_group[1][0] - temp_group[0][0])) <= vertical_spacing_tolerance:
                temp_group.append(line)
            else:
                temp_group = [line]  # reset if spacing deviates too much
        if len(temp_group) == expected_lines_per_staff:
            staff_systems.append(temp_group)
            temp_group = []
    
    return staff_systems


def group_lines_into_systems(staff_lines):
    """
    Group staff lines into systems based on musical knowledge.
    
    Args:
        staff_lines (list): List of staff line coordinates [(y, x1, x2), ...]
        
    Returns:
        list: List of staff systems, each containing 5 staff lines
    """
    if not staff_lines:
        return []
    
    # Calculate distances between adjacent lines
    distances = []
    for i in range(1, len(staff_lines)):
        distances.append(staff_lines[i][0] - staff_lines[i-1][0])
    
    if not distances:
        return []
    
    # Find the most common distance (staff line spacing)
    hist, bins = np.histogram(distances, bins=10)
    most_common_spacing = bins[np.argmax(hist)]
    
    # Group lines into potential systems
    systems = []
    current_system = [staff_lines[0]]
    
    for i in range(1, len(staff_lines)):
        current_line = staff_lines[i]
        prev_line = staff_lines[i-1]
        
        # Calculate distance between lines
        distance = current_line[0] - prev_line[0]
        
        # If distance is close to the expected spacing, add to current system
        if 0.7 * most_common_spacing <= distance <= 1.3 * most_common_spacing:
            current_system.append(current_line)
        else:
            # If we have a complete or nearly complete system, add it
            if len(current_system) >= 3:  # At least 3 lines to consider a staff system
                systems.append(current_system)
            
            # Start a new system
            current_system = [current_line]
    
    # Add the last system if it has enough lines
    if len(current_system) >= 3:
        systems.append(current_system)
    
    # Music theory: Each staff system typically has 5 lines
    # Complete systems that have fewer than 5 lines
    completed_systems = []
    for system in systems:
        if len(system) == 5:
            # Perfect system, keep as is
            completed_systems.append(system)
        elif 3 <= len(system) < 5:
            # Incomplete system, try to infer missing lines
            completed_system = complete_staff_system(system, staff_lines)
            completed_systems.append(completed_system)
    
    return completed_systems

def enforce_five_lines_per_staff(merged_lines, spacing_variation=0.3):
    """
    Ensure exactly five lines per staff using spacing consistency.
    """
    if len(merged_lines) < 5:
        return []

    merged_lines.sort(key=lambda l: l[0])
    systems = []
    temp_group = []

    for line in merged_lines:
        if not temp_group:
            temp_group.append(line)
            continue
        
        expected_spacing = np.median([temp_group[i+1][0] - temp_group[i][0] for i in range(len(temp_group)-1)]) \
                           if len(temp_group) > 1 else None
        if expected_spacing is None:
            temp_group.append(line)
        else:
            current_spacing = line[0] - temp_group[-1][0]
            if abs(current_spacing - expected_spacing) / expected_spacing <= spacing_variation:
                temp_group.append(line)
            else:
                if len(temp_group) == 5:
                    systems.append(temp_group)
                temp_group = [line]

    if len(temp_group) == 5:
        systems.append(temp_group)

    return systems


def complete_staff_system(incomplete_system, all_lines):
    """
    Complete a staff system that has fewer than 5 lines.
    
    Args:
        incomplete_system (list): Incomplete staff system (3-4 lines)
        all_lines (list): All detected lines
        
    Returns:
        list: Completed staff system with 5 lines
    """
    if len(incomplete_system) == 5:
        return incomplete_system
    
    # Sort lines by y-coordinate
    system_lines = sorted(incomplete_system, key=lambda line: line[0])
    
    # Calculate the average line spacing in this system
    spacings = []
    for i in range(1, len(system_lines)):
        spacings.append(system_lines[i][0] - system_lines[i-1][0])
    
    avg_spacing = sum(spacings) / len(spacings) if spacings else 0
    
    # Create a complete 5-line system
    completed_system = list(system_lines)  # Copy existing lines
    
    # Standard staff has lines at positions 0, 1, 2, 3, 4
    # Check which positions are missing
    positions = [0, 1, 2, 3, 4]
    existing_positions = []
    
    # Try to determine which positions the existing lines occupy
    if len(system_lines) >= 3:
        # If we have 3+ lines, we can reasonably guess their positions
        # by analyzing their spacing
        
        # First, assume the lines we have are consecutive
        start_pos = 0
        if len(system_lines) == 3:
            # Could be positions 0,1,2 or 1,2,3 or 2,3,4
            # Check spacing pattern to decide
            if len(spacings) >= 2 and abs(spacings[0] - spacings[1]) < 0.2 * avg_spacing:
                # Equal spacing suggests consecutive lines
                # Look at surrounding lines to determine position
                prev_lines = [l for l in all_lines if l[0] < system_lines[0][0]]
                next_lines = [l for l in all_lines if l[0] > system_lines[-1][0]]
                
                if prev_lines and system_lines[0][0] - prev_lines[-1][0] > 2 * avg_spacing:
                    # Large gap before first line suggests these are 0,1,2
                    start_pos = 0
                elif next_lines and next_lines[0][0] - system_lines[-1][0] > 2 * avg_spacing:
                    # Large gap after last line suggests these are 2,3,4
                    start_pos = 2
                else:
                    # Otherwise, assume middle positions 1,2,3
                    start_pos = 1
        elif len(system_lines) == 4:
            # Could be positions 0,1,2,3 or 1,2,3,4
            # Check for large gap before or after
            prev_lines = [l for l in all_lines if l[0] < system_lines[0][0]]
            next_lines = [l for l in all_lines if l[0] > system_lines[-1][0]]
            
            if prev_lines and system_lines[0][0] - prev_lines[-1][0] > 2 * avg_spacing:
                # Large gap before first line suggests these are 0,1,2,3
                start_pos = 0
            else:
                # Otherwise, assume 1,2,3,4
                start_pos = 1
    
    # Assign positions
    for i in range(len(system_lines)):
        existing_positions.append(start_pos + i)
    
    # Find missing positions
    missing_positions = [pos for pos in positions if pos not in existing_positions]
    
    # Calculate expected coordinates for missing lines
    y_base = system_lines[0][0] - start_pos * avg_spacing
    x1 = min(line[1] for line in system_lines)
    x2 = max(line[2] for line in system_lines)
    
    # Create missing lines
    for pos in missing_positions:
        expected_y = int(y_base + pos * avg_spacing)
        new_line = (expected_y, x1, x2)
        completed_system.append(new_line)
    
    # Resort the completed system
    completed_system.sort(key=lambda line: line[0])
    
    # Ensure we have exactly 5 lines
    if len(completed_system) > 5:
        # Take the 5 lines closest to expected positions
        ideal_positions = [int(y_base + pos * avg_spacing) for pos in range(5)]
        scored_lines = [(line, min(abs(line[0] - ideal_y) for ideal_y in ideal_positions)) 
                        for line in completed_system]
        scored_lines.sort(key=lambda x: x[1])  # Sort by score (distance to ideal position)
        completed_system = [line for line, score in scored_lines[:5]]
        completed_system.sort(key=lambda line: line[0])
    
    return completed_system


def detect_lines_from_image(image):
    """
    Directly detect staff lines from the original image.
    
    Args:
        image (numpy.ndarray): Original grayscale image
        
    Returns:
        list: List of staff line coordinates [(y, x1, x2), ...]
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply adaptive thresholding to handle varying brightness
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        2
    )
    
    # Remove noise
    binary = cv2.medianBlur(binary, 3)
    
    # Create a horizontal kernel for morphological operations
    kernel_length = max(5, gray.shape[1] // 50)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    
    # Apply morphology operations to isolate horizontal lines
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    detected_lines = cv2.dilate(detected_lines, horizontal_kernel, iterations=1)
    
    # Use Hough transform to find horizontal lines
    return find_horizontal_lines(detected_lines)


def populate_json_with_lines(json_data, staff_lines, staff_systems, scale=1.0):
    """
    Populate JSON data structure with staff lines and systems.
    
    Args:
        json_data (dict): JSON data to populate
        staff_lines (list): List of staff line coordinates
        staff_systems (list): List of staff systems
        scale (float): Scale factor for coordinates
    """
    # Map each staff line to its system and line number
    line_to_system_mapping = {}
    line_index = 0
    
    # Process staff systems
    for system_idx, system in enumerate(staff_systems):
        # Create system entry with line indices
        line_indices = []
        for i in range(len(system)):
            line_indices.append(line_index)
            # Map this staff line to its system and line number
            line_to_system_mapping[f"{system[i][0]}_{system[i][1]}_{system[i][2]}"] = (system_idx, i)
            line_index += 1
            
        system_entry = {
            "id": system_idx,
            "lines": line_indices
        }
        json_data["staff_systems"].append(system_entry)
    
    # Process all staff lines
    for line in staff_lines:
        y, x1, x2 = line
        
        # Create line key for lookup
        line_key = f"{y}_{x1}_{x2}"
        
        # Find system and line number
        system_idx, line_idx = line_to_system_mapping.get(line_key, (-1, -1))
        
        # Calculate dimensions for the bounding box
        height = 4  # Default height for staff lines
        y1 = y - height // 2
        y2 = y + height // 2
        width = x2 - x1
        center_x = x1 + width // 2
        center_y = y
        
        # Apply scaling if needed (to convert back to original image coordinates)
        if scale != 1.0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)
            width = int(width / scale)
            height = int(height / scale)
            center_x = int(center_x / scale)
            center_y = int(center_y / scale)
        
        # Only include lines that are part of a system
        if system_idx >= 0 and line_idx >= 0:
            # Create detection entry
            detection_entry = {
                "class_id": 0,
                "class_name": "staff_line",
                "confidence": 1.0,
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "width": float(width),
                    "height": float(height),
                    "center_x": float(center_x),
                    "center_y": float(center_y)
                },
                "staff_system": system_idx,
                "line_number": line_idx
            }
            json_data["detections"].append(detection_entry)


def run_inference(model_path, input_path, output_dir, batch_mode=False, 
                 gpu_id=0, post_process=True, max_size=512, subset=None):
    """
    Run staff line detection inference with memory-efficient processing.
    
    Args:
        model_path (str): Path to model weights
        input_path (str): Path to input image or directory
        output_dir (str): Path to output directory
        batch_mode (bool): Whether to process a directory of images
        gpu_id (int): GPU ID
        post_process (bool): Whether to apply post-processing
        max_size (int): Maximum dimension for resizing to prevent memory issues
        subset (int): Number of images to process (for testing, None=all)
    """
    # Set device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize and load model
    model = StaffLineDetectionNet(n_channels=1, n_classes=1)
    try:
        # First try loading just the model state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Could not load state dict directly: {e}")
        print("Trying to load full model...")
        # If that fails, try loading the full model
        model = torch.load(model_path, map_location=device)
    
    model = model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if batch_mode:
        # Process all images in the directory
        process_directory(model, input_path, output_dir, device, post_process, max_size)
    else:
        # Process a single image
        image_tensor, original_image, scale = load_image(input_path, max_size=max_size)
        
        # Predict staff lines
        prediction = predict_staff_lines(model, image_tensor, device, post_process)
        
        # If we resized for prediction, resize the prediction back to original size
        if scale < 1.0:
            h, w = original_image.shape
            prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Extract staff line information using our enhanced methods
        staff_lines = extract_staff_lines(prediction, min_length_ratio=0.2)
        staff_systems = analyze_staff_systems(staff_lines, max_gap_ratio=0.3)
        
        # Create base name for output
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Save mask
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        mask_image = (prediction * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_image)
        
        # Visualize and save
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        visualize_prediction(original_image, prediction, save_path=vis_path)
        
        # Print results
        print(f"Total staff lines detected: {len(staff_lines)}")
        print(f"Number of valid staff systems (with 5 lines): {len(staff_systems)}")
        
        # Save analysis
        analysis_path = os.path.join(output_dir, f"{base_name}_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write(f"Staff Line Analysis for {os.path.basename(input_path)}\n")
            f.write(f"Total staff lines detected: {len(staff_lines)}\n")
            f.write(f"Number of valid staff systems (with 5 lines): {len(staff_systems)}\n\n")
            
            for i, system in enumerate(staff_systems):
                f.write(f"Staff System {i+1}:\n")
                for j, line in enumerate(system):
                    y, x1, x2 = line
                    f.write(f"  Line {j+1}: y={y}, x1={x1}, x2={x2}, length={x2-x1}\n")
                
                # Calculate average line spacing
                if len(system) > 1:
                    spacings = [system[j+1][0] - system[j][0] for j in range(len(system)-1)]
                    avg_spacing = sum(spacings) / len(spacings)
                    f.write(f"  Average line spacing: {avg_spacing:.2f} pixels\n\n")
        
        # Save JSON results with fallback to enhanced methods
        json_path = os.path.join(output_dir, f"{base_name}_results.json")
        if len(staff_lines) == 0 or len(staff_systems) == 0:
            # Use music-specific inferencing if normal detection failed
            save_inferred_json_results(json_path, 1.0/scale if scale < 1.0 else 1.0, prediction, original_image)
        else:
            # Use normal JSON export with our enhanced staff lines
            save_json_results(staff_lines, staff_systems, json_path, 1.0/scale if scale < 1.0 else 1.0)
        print(f"JSON results saved to: {json_path}")


def process_directory(model, input_dir, output_dir, device, post_process=True, max_size=512, batch_size=1):
    """
    Process all images in a directory with memory-efficient handling.
    
    Args:
        model: The staff line detection model
        input_dir (str): Input directory containing images
        output_dir (str): Output directory for visualizations and results
        device (torch.device): Device to run inference on
        post_process (bool): Whether to apply post-processing
        max_size (int): Maximum dimension for images to prevent memory issues
        batch_size (int): Number of images to process at once (use 1 for large images)
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    mask_dir = os.path.join(output_dir, "masks")
    analysis_dir = os.path.join(output_dir, "analysis")
    json_dir = os.path.join(output_dir, "json")
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    # Process each image individually to avoid memory issues
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image_path = os.path.join(input_dir, image_file)
            image_tensor, original_image, scale = load_image(image_path, max_size=max_size)
            
            # Predict staff lines
            prediction = predict_staff_lines(model, image_tensor, device, post_process)
            
            # If we resized for prediction, resize the prediction back to original size
            if scale < 1.0:
                h, w = original_image.shape
                prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Convert prediction to image format for saving
            pred_image = (prediction * 255).astype(np.uint8)
            
            # Extract staff line information using our enhanced methods
            staff_lines = extract_staff_lines(prediction, min_length_ratio=0.2)
            staff_systems = analyze_staff_systems(staff_lines, max_gap_ratio=0.3)
            
            # Create base name for output files
            base_name = os.path.splitext(image_file)[0]
            
            # Save mask
            mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, pred_image)
            
            # Visualize and save
            vis_path = os.path.join(vis_dir, f"{base_name}_visualization.png")
            visualize_prediction(original_image, prediction, save_path=vis_path, show=False)
            
            # Save staff line analysis
            analysis_path = os.path.join(analysis_dir, f"{base_name}_analysis.txt")
            with open(analysis_path, 'w') as f:
                f.write(f"Staff Line Analysis for {image_file}\n")
                f.write(f"Total staff lines detected: {len(staff_lines)}\n")
                f.write(f"Number of valid staff systems (with 5 lines): {len(staff_systems)}\n\n")
                
                for i, system in enumerate(staff_systems):
                    f.write(f"Staff System {i+1}:\n")
                    for j, line in enumerate(system):
                        y, x1, x2 = line
                        f.write(f"  Line {j+1}: y={y}, x1={x1}, x2={x2}, length={x2-x1}\n")
                    
                    # Calculate average line spacing
                    if len(system) > 1:
                        spacings = [system[j+1][0] - system[j][0] for j in range(len(system)-1)]
                        avg_spacing = sum(spacings) / len(spacings)
                        f.write(f"  Average line spacing: {avg_spacing:.2f} pixels\n\n")
            
            # Save JSON results with fallback to enhanced methods
            json_path = os.path.join(json_dir, f"{base_name}_results.json")
            if len(staff_lines) == 0 or len(staff_systems) == 0:
                # Use music-specific inferencing if normal detection failed
                save_inferred_json_results(json_path, 1.0/scale if scale < 1.0 else 1.0, prediction, original_image)
            else:
                # Use normal JSON export with our enhanced staff lines
                save_json_results(staff_lines, staff_systems, json_path, 1.0/scale if scale < 1.0 else 1.0)
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staff Line Detection Inference with JSON Export")
    
    parser.add_argument("--weights", type=str, required=True, 
                        help="Path to trained model weights")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image or directory")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to output directory")
    parser.add_argument("--batch", action="store_true",
                        help="Process a directory of images")
    parser.add_argument("--gpu_id", type=int, default=0, 
                        help="GPU ID")
    parser.add_argument("--no_post_process", action="store_true",
                        help="Disable post-processing")
    parser.add_argument("--max_size", type=int, default=512,
                        help="Maximum image dimension (will resize larger images)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Number of images to process (for testing, default=all)")
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.weights,
        input_path=args.input,
        output_dir=args.output,
        batch_mode=args.batch,
        gpu_id=args.gpu_id,
        post_process=not args.no_post_process,
        max_size=args.max_size,
        subset=args.subset
    )