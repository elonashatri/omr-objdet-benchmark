"""
Inference Module for Staff Line Detection

This module handles inference and visualization using a trained staff line detection model.
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


def load_image(image_path, max_size=512):  # Changed default max_size to 512
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


def extract_staff_lines(prediction, min_length_ratio=0.3):
    """
    Extract staff line information from the prediction.
    
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
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_8bit, connectivity=8)
    
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
    
    # Sort staff lines by y-coordinate
    staff_lines.sort()
    
    return staff_lines


def analyze_staff_systems(staff_lines, max_gap_ratio=0.2):
    """
    Group staff lines into staff systems.
    
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
    
    # Filter systems that have exactly 5 staff lines (standard staff)
    valid_systems = [sys for sys in staff_systems if len(sys) == 5]
    
    return valid_systems


def save_json_results(staff_lines, staff_systems, output_path, scale=1.0):
    """
    Save staff line detection results in JSON format.
    
    Args:
        staff_lines (list): List of staff line coordinates [(y, x1, x2), ...]
        staff_systems (list): List of staff systems, each containing staff lines
        output_path (str): Path to save the JSON file
        scale (float): Scale factor used during inference (1.0 means no scaling)
    
    Returns:
        dict: The JSON data structure that was saved
    """
    # Initialize the JSON structure
    json_data = {
        "staff_systems": [],
        "detections": []
    }
    
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
    
    # Process all staff lines (including those that might not be part of any system)
    detection_index = 0
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
        detection_index += 1
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return json_data


def process_directory(model, input_dir, output_dir, device, post_process=True, max_size=512, batch_size=1):  # Changed max_size to 512
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
            
            # Extract staff line information
            staff_lines = extract_staff_lines(prediction)
            staff_systems = analyze_staff_systems(staff_lines)
            
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
            
            # Save JSON results
            json_path = os.path.join(json_dir, f"{base_name}_results.json")
            save_json_results(staff_lines, staff_systems, json_path, 1.0/scale if scale < 1.0 else 1.0)
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue


def run_inference(model_path, input_path, output_dir, batch_mode=False, 
                 gpu_id=0, post_process=True, max_size=512, subset=None):  # Changed max_size to 512
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
        
        # Extract staff line information
        staff_lines = extract_staff_lines(prediction)
        staff_systems = analyze_staff_systems(staff_lines)
        
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
        
        # Save JSON results
        json_path = os.path.join(output_dir, f"{base_name}_results.json")
        save_json_results(staff_lines, staff_systems, json_path, 1.0/scale if scale < 1.0 else 1.0)
        print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staff Line Detection Inference")
    
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
    parser.add_argument("--max_size", type=int, default=512,  # Changed default to 512
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