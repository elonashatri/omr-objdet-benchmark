import cv2
import numpy as np
import os
import math

def apply_page_curvature(image_path, curvature_intensity=0.2, output_path=None):
    """
    Apply a page curvature effect to an image, similar to an open book.
    
    Parameters:
    image_path (str): Path to the input image
    curvature_intensity (float): Intensity of the curvature effect (0.0 to 1.0)
    output_path (str): Path to save the curved image (if None, creates a new filename)
    
    Returns:
    str: Path to the saved curved image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get dimensions
    height, width = img.shape[:2]
    
    # Create the mapping function for the transformation
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    
    # Calculate the curvature effect
    for y in range(height):
        for x in range(width):
            # Normalize x position (0 to 1)
            norm_x = x / width - 0.5
            
            # Apply subtle curvature - stronger in the middle
            offset_y = curvature_intensity * (norm_x ** 2) * height
            
            # Set the mapping
            map_x[y, x] = x
            map_y[y, x] = y - offset_y
    
    # Apply the remapping
    curved_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Create output path if not specified
    if output_path is None:
        base_name = os.path.basename(image_path)
        file_name, ext = os.path.splitext(base_name)
        output_path = os.path.join(os.path.dirname(image_path), f"{file_name}_curved{ext}")
    
    # Save the curved image
    cv2.imwrite(output_path, curved_img)
    
    return output_path

def apply_blur_and_curvature(image_path, blur_amount=5, curvature_intensity=0.2, output_path=None):
    """
    Apply both Gaussian blur and page curvature to an image.
    
    Parameters:
    image_path (str): Path to the input image
    blur_amount (int): Amount of blur to apply (must be odd)
    curvature_intensity (float): Intensity of the curvature effect (0.0 to 1.0)
    output_path (str): Path to save the processed image (if None, creates a new filename)
    
    Returns:
    str: Path to the saved processed image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Make sure blur_amount is odd (required by GaussianBlur)
    if blur_amount % 2 == 0:
        blur_amount += 1
    
    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (blur_amount, blur_amount), 0)
    
    # Get dimensions
    height, width = blurred_img.shape[:2]
    
    # Create the mapping function for the transformation
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    
    # Calculate the curvature effect
    for y in range(height):
        for x in range(width):
            # Normalize x position (0 to 1)
            norm_x = x / width - 0.5
            
            # Apply subtle curvature - stronger in the middle
            offset_y = curvature_intensity * (norm_x ** 2) * height
            
            # Set the mapping
            map_x[y, x] = x
            map_y[y, x] = y - offset_y
    
    # Apply the remapping
    result_img = cv2.remap(blurred_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Create output path if not specified
    if output_path is None:
        base_name = os.path.basename(image_path)
        file_name, ext = os.path.splitext(base_name)
        output_path = os.path.join(os.path.dirname(image_path), f"{file_name}_blurred_curved{ext}")
    
    # Save the processed image
    cv2.imwrite(output_path, result_img)
    
    return output_path

if __name__ == "__main__":
    # Path to your image
    image_path = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/v2-accidentals-004.png"
    
    # Apply only curvature
    curved_path = apply_page_curvature(image_path, curvature_intensity=0.1)
    print(f"Curved image saved to: {curved_path}")
    
    # Apply both blur and curvature
    # combined_path = apply_blur_and_curvature(image_path, blur_amount=5, curvature_intensity=0.1)
    # print(f"Blurred and curved image saved to: {combined_path}")