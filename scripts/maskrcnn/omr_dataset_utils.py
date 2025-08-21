import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json

def decode_rle_mask(mask_string, height, width):
    """
    Decode Run-Length Encoded (RLE) mask.
    Format: "0: X1 1: Y1 0: X2 1: Y2 ..."
    Where X and Y are the run lengths of 0s and 1s.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not mask_string:
        return mask
    
    # Clean up the mask string to handle any formatting issues
    mask_string = mask_string.strip()
    
    # Split the string into parts
    parts = []
    for part in mask_string.split():
        if ':' in part:
            # This is a value indicator (e.g. "0:")
            value = int(part.rstrip(':'))
            parts.append(value)
        else:
            # This is a length
            length = int(part)
            parts.append(length)
    
    # If we have an odd number of parts, something is wrong
    if len(parts) % 2 != 0:
        print(f"Warning: RLE mask has odd number of parts ({len(parts)})")
        return mask
    
    # Fill the mask
    row, col = 0, 0
    
    for i in range(0, len(parts), 2):
        value = parts[i]
        length = parts[i+1]
        
        for _ in range(length):
            if row < height and col < width:
                mask[row, col] = value
                col += 1
                if col == width:
                    col = 0
                    row += 1
                    if row == height:
                        break
    
    return mask

class OMRDataset(Dataset):
    def __init__(self, xml_files, transform=None, img_ext=".png"):
        self.xml_files = xml_files
        self.transform = transform
        self.img_ext = img_ext
        
        # Load class mapping from JSON file
        class_map_path = "/homes/es314/omr-objdet-benchmark/data/class_mapping.json"
        if os.path.exists(class_map_path):
            with open(class_map_path, 'r') as f:
                self.class_map = json.load(f)
            print(f"Loaded {len(self.class_map)} classes from class mapping file")
        else:
            # Fallback to default class map
            self.class_map = {
                'kStaffLine': 1,
                'gClef': 2,
                # Add more classes as needed
            }
            print("Warning: Using default class mapping - class mapping file not found")
    
    def __len__(self):
        return len(self.xml_files)
    
    def __getitem__(self, idx):
        xml_path = self.xml_files[idx]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find the corresponding image - correctly handling the path
        # Construct proper image path (replace annotations dir with images dir)
        if '/annotations/' in xml_path:
            img_path = xml_path.replace('/annotations/', '/images/').replace('.xml', self.img_ext)
        else:
            # Fallback to simple extension replacement
            img_path = xml_path.replace('.xml', self.img_ext)
        
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            canvas_height, canvas_width = image.shape[:2]
        else:
            # Fallback to dummy image if real image is not found
            print(f"Warning: Image not found at {img_path}, using dummy image")
            canvas_height, canvas_width = 1000, 2500  # Default dimensions
            image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        masks = []
        boxes = []
        labels = []
        
        # Try to find nodes in the expected structure
        nodes = []
        
        # First try with Page/Nodes structure as seen in the sample
        for page in root.findall('.//Page'):
            for nodes_container in page.findall('.//Nodes'):
                for node in nodes_container.findall('Node'):
                    nodes.append(node)
        
        # If no nodes found, try a more general approach
        if not nodes:
            nodes = root.findall('.//Node')
        
        for node in nodes:
            # Get class name from either attribute or child element
            class_name = node.get('name')
            if not class_name and node.find('ClassName') is not None:
                class_name = node.find('ClassName').text
                
            # Skip if class name is not recognized or missing
            if not class_name or class_name not in self.class_map:
                continue
                
            # Get bounding box (with error handling)
            try:
                top = int(node.find('Top').text) if node.find('Top') is not None else 0
                left = int(node.find('Left').text) if node.find('Left') is not None else 0
                width = int(node.find('Width').text) if node.find('Width') is not None else 0
                height = int(node.find('Height').text) if node.find('Height') is not None else 0
                
                # Skip if box has zero dimensions
                if width <= 0 or height <= 0:
                    continue
                    
                # Get mask
                mask_elem = node.find('Mask')
                if mask_elem is not None and mask_elem.text:
                    mask = decode_rle_mask(mask_elem.text, height, width)
                    
                    # Position the mask on the full canvas
                    full_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
                    
                    # Ensure we don't go out of bounds
                    h, w = min(height, canvas_height - top), min(width, canvas_width - left)
                    if h > 0 and w > 0:  # Only proceed if dimensions are positive
                        full_mask[top:top+h, left:left+w] = mask[:h, :w]
                        
                        masks.append(full_mask)
                        boxes.append([left, top, left + width, top + height])
                        labels.append(self.class_map[class_name])
            except Exception as e:
                print(f"Error processing node: {e}")
        
        # Convert to tensor format for Mask R-CNN
        if len(masks) > 0:
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
            
            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((len(masks),), dtype=torch.int64)
            }
            
            if self.transform:
                image, target = self.transform(image, target)
                
            return image, target
        else:
            # Handle empty case
            return torch.zeros((3, canvas_height, canvas_width), dtype=torch.float32), {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, canvas_height, canvas_width), dtype=torch.uint8),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }

def visualize_sample(dataset, idx):
    """Visualize a sample from the dataset"""
    image, target = dataset[idx]
    
    # Convert tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    
    boxes = target['boxes'].numpy()
    masks = target['masks'].numpy()
    labels = target['labels'].numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display image with bounding boxes
    ax[0].imshow(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
        ax[0].add_patch(rect)
    ax[0].set_title('Image with bounding boxes')
    
    # Display masks
    if len(masks) > 0:
        combined_mask = np.zeros(masks[0].shape, dtype=np.uint8)
        for i, mask in enumerate(masks):
            color_val = (i + 1) * 50  # Different color for each mask
            combined_mask[mask > 0] = color_val
        
        ax[1].imshow(combined_mask, cmap='viridis')
        ax[1].set_title('Instance masks')
    else:
        ax[1].set_title('No masks found')
    
    plt.tight_layout()
    plt.savefig(f"sample_{idx}_visualization.png")
    plt.close()
    
    print(f"Visualization saved to sample_{idx}_visualization.png")

def load_xml_dataset(root_dir):
    """Load all XML files from the root directory"""
    xml_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    return xml_files