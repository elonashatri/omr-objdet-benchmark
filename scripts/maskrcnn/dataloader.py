import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import xml.etree.ElementTree as ET

# Import our dataset utilities
from omr_dataset_utils import decode_rle_mask

def test_with_sample():
    """
    Test the mask decoding with a real sample XML and PNG file
    """
    # Define paths to sample files
    xml_path = "/homes/es314/omr-objdet-benchmark/scripts/maskrcnn/data/example_1.xml"
    image_path = "/homes/es314/omr-objdet-benchmark/scripts/maskrcnn/data/example_1.png"
    
    # Check if files exist
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    print(f"Loading XML from: {xml_path}")
    print(f"Loading image from: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]
    print(f"Image dimensions: {image_width}x{image_height}")
    
    # Parse XML
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return
    
    # Extract nodes with masks
    nodes_with_masks = []
    
    # First, try with the structure in the example
    for page in root.findall('.//Page'):
        for node in page.findall('.//Node'):
            class_name = node.get('name')
            if not class_name and node.find('ClassName') is not None:
                class_name = node.find('ClassName').text
                
            mask_elem = node.find('Mask')
            if mask_elem is not None and mask_elem.text:
                # Get bounding box info
                top = int(node.find('Top').text) if node.find('Top') is not None else 0
                left = int(node.find('Left').text) if node.find('Left') is not None else 0
                width = int(node.find('Width').text) if node.find('Width') is not None else 0
                height = int(node.find('Height').text) if node.find('Height') is not None else 0
                
                # Add to our list
                nodes_with_masks.append({
                    'class_name': class_name,
                    'mask_rle': mask_elem.text,
                    'top': top,
                    'left': left,
                    'width': width,
                    'height': height
                })
    
    # If no nodes found with the expected structure, print info and return
    if not nodes_with_masks:
        print("No nodes with masks found in the XML file.")
        # Print the structure of the XML for debugging
        print("XML structure:")
        for child in root:
            print(f"  {child.tag}")
            for grandchild in child:
                print(f"    {grandchild.tag}")
        return
    
    print(f"Found {len(nodes_with_masks)} nodes with masks")
    
    # Create a figure to display results
    fig, axes = plt.subplots(len(nodes_with_masks), 3, figsize=(15, 5*len(nodes_with_masks)))
    
    # If there's only one mask, wrap axes in a list
    if len(nodes_with_masks) == 1:
        axes = [axes]
    
    # Process each mask
    for i, node_info in enumerate(nodes_with_masks):
        # Decode the mask
        mask_rle = node_info['mask_rle']
        mask_height = node_info['height']
        mask_width = node_info['width']
        
        print(f"Decoding mask for {node_info['class_name']}")
        print(f"  RLE string starts with: {mask_rle[:50]}...")
        print(f"  Mask dimensions: {mask_width}x{mask_height}")
        
        try:
            mask = decode_rle_mask(mask_rle, mask_height, mask_width)
            
            # Position on full image
            full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            
            # Calculate bounds to ensure we don't go outside the image
            top, left = node_info['top'], node_info['left']
            h = min(mask_height, image_height - top)
            w = min(mask_width, image_width - left)
            
            # Only copy if dimensions are positive
            if h > 0 and w > 0:
                full_mask[top:top+h, left:left+w] = mask[:h, :w]
            
            # Draw bounding box on image
            image_with_box = image.copy()
            box_color = (255, 0, 0)  # Red
            box_thickness = 2
            cv2.rectangle(
                image_with_box,
                (node_info['left'], node_info['top']),
                (node_info['left'] + node_info['width'], node_info['top'] + node_info['height']),
                box_color,
                box_thickness
            )
            
            # Display results
            axes[i][0].imshow(image)
            axes[i][0].set_title(f"Original Image")
            axes[i][0].axis('off')
            
            axes[i][1].imshow(image_with_box)
            axes[i][1].set_title(f"Image with Bbox: {node_info['class_name']}")
            axes[i][1].axis('off')
            
            axes[i][2].imshow(full_mask, cmap='gray')
            axes[i][2].set_title(f"Mask: {node_info['class_name']}")
            axes[i][2].axis('off')
            
            print(f"  Successfully decoded mask")
            
        except Exception as e:
            print(f"Error decoding mask: {e}")
            axes[i][2].text(0.5, 0.5, f"Error: {str(e)}", 
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=axes[i][2].transAxes)
    
    plt.tight_layout()
    plt.savefig("omr_masks_visualization.png")
    print("Visualization saved to 'omr_masks_visualization.png'")
    plt.close()

if __name__ == "__main__":
    test_with_sample()