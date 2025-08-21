import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm


class DoremiDataset(Dataset):
    """Dataset for Doremi Optical Music Recognition dataset"""
    
    def __init__(self, data_dir, annotation_dir, class_mapping_file=None, transform=None, min_box_size=5, max_classes=None):
        """
        Args:
            data_dir (str): Directory with all the images
            annotation_dir (str): Directory with XML annotation files
            class_mapping_file (str, optional): Path to JSON file with class mapping
            transform (callable, optional): Optional transform to be applied on a sample
            min_box_size (int): Minimum size for bounding boxes
            max_classes (int): Maximum number of classes to use (most frequent ones)
        """
        self.data_dir = Path(data_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transform = transform
        self.min_box_size = min_box_size
        
        # Get all XML annotation files
        self.annotation_files = list(self.annotation_dir.glob("*.xml"))
        print(f"Found {len(self.annotation_files)} annotation files")
        
        # Load class mapping or create from annotations
        self.class_to_idx = self.load_class_mapping(class_mapping_file, max_classes)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(f"Using {len(self.class_to_idx)} classes")
        
        # Create samples list
        self.samples = self._prepare_samples()
        
    def load_class_mapping(self, class_mapping_file, max_classes=None):
        """Load class mapping from file or create from annotations"""
        if class_mapping_file and os.path.exists(class_mapping_file):
            with open(class_mapping_file, 'r') as f:
                class_freq = json.load(f)
                
            # Sort classes by frequency (descending)
            sorted_classes = sorted(class_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Limit to max_classes most frequent classes if specified
            if max_classes and max_classes < len(sorted_classes):
                print(f"Limiting to {max_classes} most frequent classes")
                sorted_classes = sorted_classes[:max_classes]
            
            # Create class mapping (starting from 1, 0 is background)
            return {cls_name: idx + 1 for idx, (cls_name, _) in enumerate(sorted_classes)}
        else:
            print("Class mapping file not found. Creating from annotations...")
            class_counts = {}
            
            # Count class occurrences in annotations
            for ann_file in tqdm(self.annotation_files[:100], desc="Scanning annotations"):  # Sample first 100 files
                try:
                    tree = ET.parse(ann_file)
                    root = tree.getroot()
                    
                    for node in root.findall(".//Node"):
                        cls_name = node.find("ClassName").text
                        if cls_name not in class_counts:
                            class_counts[cls_name] = 0
                        class_counts[cls_name] += 1
                except Exception as e:
                    print(f"Error reading {ann_file}: {e}")
            
            # Sort classes by frequency (descending)
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Limit to max_classes most frequent classes if specified
            if max_classes and max_classes < len(sorted_classes):
                sorted_classes = sorted_classes[:max_classes]
            
            # Create class mapping (starting from 1, 0 is background)
            return {cls_name: idx + 1 for idx, (cls_name, _) in enumerate(sorted_classes)}
    
    def _prepare_samples(self):
        """Prepare samples list with image paths and annotation files"""
        samples = []
        
        for ann_file in tqdm(self.annotation_files, desc="Preparing dataset"):
            # Get corresponding image file
            img_file = self.data_dir / (ann_file.stem + ".png")
            if not img_file.exists():
                img_file = self.data_dir / (ann_file.stem + ".jpg")
            
            if img_file.exists():
                samples.append({
                    'image_path': str(img_file),
                    'annotation_path': str(ann_file),
                    'image_id': len(samples)
                })
        
        print(f"Prepared {len(samples)} valid samples with both images and annotations")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def parse_annotation(self, ann_path):
        """Parse XML annotation file and extract bounding boxes and classes"""
        boxes = []
        labels = []
        
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            # Find all Node elements
            for node in root.findall(".//Node"):
                cls_name = node.find("ClassName").text
                
                # Skip if class is not in our mapping
                if cls_name not in self.class_to_idx:
                    continue
                
                # Get bounding box
                top = int(node.find("Top").text)
                left = int(node.find("Left").text)
                width = int(node.find("Width").text)
                height = int(node.find("Height").text)
                
                # Skip too small boxes
                if width < self.min_box_size or height < self.min_box_size:
                    continue
                
                # Convert to [x1, y1, x2, y2] format
                boxes.append([left, top, left + width, top + height])
                labels.append(self.class_to_idx[cls_name])
        
        except Exception as e:
            print(f"Error parsing annotation {ann_path}: {e}")
        
        return boxes, labels
    
    def __getitem__(self, idx):
        """Get dataset item"""
        sample = self.samples[idx]
        image_path = sample['image_path']
        ann_path = sample['annotation_path']
        image_id = sample['image_id']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image {image_path}")
            # Return a dummy sample if image can't be read
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            boxes = []
            labels = []
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Parse annotation
            boxes, labels = self.parse_annotation(ann_path)
        
        # If no boxes, add a dummy box to avoid errors
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]  # Background class
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        
        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def convert_doremi_to_coco(data_dir, annotation_dir, output_path, class_mapping_file=None, min_box_size=5, max_classes=None):
    """
    Convert Doremi dataset to COCO format
    
    Args:
        data_dir (str): Directory with images
        annotation_dir (str): Directory with XML annotations
        output_path (str): Path to save COCO JSON
        class_mapping_file (str, optional): Path to class mapping JSON
        min_box_size (int): Minimum box size to include
        max_classes (int): Maximum number of classes to use
    """
    # Load dataset
    dataset = DoremiDataset(data_dir, annotation_dir, class_mapping_file, 
                           min_box_size=min_box_size, max_classes=max_classes)
    
    # Create COCO format
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for class_name, class_id in dataset.class_to_idx.items():
        coco["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "music_notation"
        })
    
    # Process each sample
    annotation_id = 0
    
    for idx, sample in enumerate(tqdm(dataset.samples, desc="Converting to COCO")):
        image_path = sample['image_path']
        ann_path = sample['annotation_path']
        image_id = sample['image_id']
        
        # Read image to get dimensions
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image {image_path}, skipping")
            continue
        
        height, width, _ = img.shape
        
        # Add image to COCO
        coco["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        })
        
        # Parse annotation
        boxes, labels = dataset.parse_annotation(ann_path)
        
        # Add annotations
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            
            annotation_id += 1
    
    # Save COCO JSON
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=4)
    
    print(f"Conversion complete. Saved to {output_path}")
    print(f"Included {len(coco['categories'])} categories")
    print(f"Processed {len(coco['images'])} images")
    print(f"Created {len(coco['annotations'])} annotations")


def create_class_mapping(vocab_file, output_file, max_classes=None):
    """
    Create class mapping from frequency JSON file
    
    Args:
        vocab_file (str): Path to frequency JSON file
        output_file (str): Path to save class mapping
        max_classes (int): Maximum number of classes to include
    """
    with open(vocab_file, 'r') as f:
        class_freq = json.load(f)
    
    # Sort classes by frequency
    sorted_classes = sorted(class_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to max_classes if specified
    if max_classes and max_classes < len(sorted_classes):
        print(f"Limiting to {max_classes} most frequent classes")
        sorted_classes = sorted_classes[:max_classes]
    
    # Create class mapping (starting from 1, 0 is background)
    class_mapping = {cls_name: idx + 1 for idx, (cls_name, _) in enumerate(sorted_classes)}
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(class_mapping, f, indent=4)
    
    print(f"Created class mapping with {len(class_mapping)} classes, saved to {output_file}")
    return class_mapping


def split_dataset(data_dir, annotation_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        data_dir (str): Directory with images
        annotation_dir (str): Directory with annotations
        output_dir (str): Directory to save splits
        train_ratio (float): Ratio of training set
        val_ratio (float): Ratio of validation set
        test_ratio (float): Ratio of test set
        seed (int): Random seed
    """
    import random
    random.seed(seed)
    
    # Get all annotation files
    annotation_files = list(Path(annotation_dir).glob("*.xml"))
    
    # Shuffle files
    random.shuffle(annotation_files)
    
    # Calculate split sizes
    train_size = int(len(annotation_files) * train_ratio)
    val_size = int(len(annotation_files) * val_ratio)
    
    # Split files
    train_files = annotation_files[:train_size]
    val_files = annotation_files[train_size:train_size + val_size]
    test_files = annotation_files[train_size + val_size:]
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create splits
    for files, split_dir in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
        for file in files:
            # Create symlinks
            dst = os.path.join(split_dir, file.name)
            os.symlink(os.path.abspath(file), dst)
            
            # Find corresponding image
            img_file = Path(data_dir) / (file.stem + ".png")
            if not img_file.exists():
                img_file = Path(data_dir) / (file.stem + ".jpg")
            
            if img_file.exists():
                img_dst = os.path.join(split_dir, img_file.name)
                os.symlink(os.path.abspath(img_file), img_dst)
    
    print(f"Split dataset into:")
    print(f"  Training: {len(train_files)} files ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(val_files)} files ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(test_files)} files ({test_ratio*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Doremi dataset")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create class mapping command
    class_parser = subparsers.add_parser("create_mapping", help="Create class mapping")
    class_parser.add_argument("--vocab_file", type=str, required=True, help="Path to frequency JSON file")
    class_parser.add_argument("--output_file", type=str, required=True, help="Path to save class mapping")
    class_parser.add_argument("--max_classes", type=int, default=None, help="Maximum number of classes")
    
    # Convert to COCO command
    coco_parser = subparsers.add_parser("convert_coco", help="Convert to COCO format")
    coco_parser.add_argument("--data_dir", type=str, required=True, help="Directory with images")
    coco_parser.add_argument("--annotation_dir", type=str, required=True, help="Directory with XML annotations")
    coco_parser.add_argument("--output_path", type=str, required=True, help="Path to save COCO JSON")
    coco_parser.add_argument("--class_mapping", type=str, default=None, help="Path to class mapping file")
    coco_parser.add_argument("--min_box_size", type=int, default=5, help="Minimum box size")
    coco_parser.add_argument("--max_classes", type=int, default=None, help="Maximum number of classes")
    
    # Split dataset command
    split_parser = subparsers.add_parser("split", help="Split dataset")
    split_parser.add_argument("--data_dir", type=str, required=True, help="Directory with images")
    split_parser.add_argument("--annotation_dir", type=str, required=True, help="Directory with annotations")
    split_parser.add_argument("--output_dir", type=str, required=True, help="Directory to save splits")
    split_parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    split_parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    split_parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    split_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.command == "create_mapping":
        create_class_mapping(args.vocab_file, args.output_file, args.max_classes)
    elif args.command == "convert_coco":
        convert_doremi_to_coco(args.data_dir, args.annotation_dir, args.output_path, 
                              args.class_mapping, args.min_box_size, args.max_classes)
    elif args.command == "split":
        split_dataset(args.data_dir, args.annotation_dir, args.output_dir, 
                     args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    else:
        parser.print_help()