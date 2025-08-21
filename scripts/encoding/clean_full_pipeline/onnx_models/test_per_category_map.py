import torch
import onnx
import onnxruntime as ort
import numpy as np
import cv2
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import xml.etree.ElementTree as ET
import random


class OMRModelEvaluator:
    def __init__(self, model_paths, test_data_path):
        """
        Initialize evaluator for multiple ONNX models
        
        Args:
            model_paths: List of paths to ONNX models
            test_data_path: Path to test dataset (images and annotations)
        """
        self.model_paths = model_paths
        self.test_data_path = test_data_path
        
        # Define class to category mapping based on provided class map
        self.class_categories = {
            'stafflines': [53],  # kStaffLine
            'clefs': [26, 42, 51],  # cClef, fClef, gClef
            'accidentals': [2, 3, 4, 5, 6, 7, 8],  # All accidentals
            'noteheads': list(range(54, 64)),  # All notehead types (54-63)
            'rests': list(range(64, 72)),  # All rest types (64-71)
            'dynamics': list(range(27, 41)) + [52],  # All dynamics
            'time_signatures': list(range(76, 88)),  # All time signature components
            'flags': [43, 44, 45, 46, 47, 48, 49, 50],  # All flag types
            'articulations': list(range(9, 23)),  # All articulation marks
            'other': [1, 23, 24, 25, 72, 73, 74, 75, 88]  # Other symbols
        }
        
        # Reverse mapping from class ID to category
        self.class_to_category = {}
        for category, class_ids in self.class_categories.items():
            for class_id in class_ids:
                self.class_to_category[class_id] = category
        
        # Load models
        self.models = {}
        for path in model_paths:
            model_name = os.path.basename(os.path.dirname(path))
            session = ort.InferenceSession(path)
            
            self.models[model_name] = {
                'session': session,
                'input_name': session.get_inputs()[0].name,
                'input_shape': session.get_inputs()[0].shape
            }
            print(f"Loaded model: {model_name}")
        
        # Create a nice readable class name mapping
        self.class_names = {
            1: '6stringTabClef', 2: 'accidentalDoubleFlat', 3: 'accidentalDoubleSharp',
            4: 'accidentalFlat', 5: 'accidentalNatural', 6: 'accidentalQuarterToneFlatStein',
            7: 'accidentalQuarterToneSharpStein', 8: 'accidentalSharp', 9: 'articAccentAbove',
            10: 'articAccentBelow', 11: 'articMarcatoAbove', 12: 'articMarcatoBelow',
            13: 'articStaccatissimoAbove', 14: 'articStaccatissimoBelow', 15: 'articStaccatoAbove',
            16: 'articStaccatoBelow', 17: 'articStressAbove', 18: 'articStressBelow',
            19: 'articTenutoAbove', 20: 'articTenutoBelow', 21: 'articUnstressAbove',
            22: 'articUnstressBelow', 23: 'augmentationDot', 24: 'barline',
            25: 'beam', 26: 'cClef', 27: 'dynamicFF', 28: 'dynamicFFF',
            29: 'dynamicFFFF', 30: 'dynamicForte', 31: 'dynamicFortePiano', 32: 'dynamicForzando',
            33: 'dynamicMF', 34: 'dynamicMP', 35: 'dynamicPP', 36: 'dynamicPPP',
            37: 'dynamicPPPP', 38: 'dynamicPiano', 39: 'dynamicSforzato', 40: 'dynamicSforzatoFF',
            41: 'dynamicText', 42: 'fClef', 43: 'flag16thDown', 44: 'flag16thUp',
            45: 'flag32ndDown', 46: 'flag32ndUp', 47: 'flag64thDown', 48: 'flag64thUp',
            49: 'flag8thDown', 50: 'flag8thUp', 51: 'gClef', 52: 'gradualDynamic',
            53: 'kStaffLine', 54: 'mensuralNoteheadMinimaWhite', 55: 'noteheadBlack',
            56: 'noteheadDiamondBlack', 57: 'noteheadDiamondBlackOld', 58: 'noteheadDiamondHalfOld',
            59: 'noteheadDiamondWhole', 60: 'noteheadDoubleWholeSquare', 61: 'noteheadHalf',
            62: 'noteheadWhole', 63: 'noteheadXOrnateEllipse', 64: 'rest',
            65: 'rest16th', 66: 'rest32nd', 67: 'rest64th', 68: 'rest8th',
            69: 'restHalf', 70: 'restQuarter', 71: 'restWhole', 72: 'slur',
            73: 'stem', 74: 'systemicBarline', 75: 'tie', 76: 'timeSig1',
            77: 'timeSig2', 78: 'timeSig3', 79: 'timeSig4', 80: 'timeSig5',
            81: 'timeSig6', 82: 'timeSig7', 83: 'timeSig8', 84: 'timeSig9',
            85: 'timeSigCommon', 86: 'timeSigCutCommon', 87: 'timeSignatureComponent',
            88: 'unpitchedPercussionClef1'
        }





    def preprocess_image(self, image, model_name):
        """
        Preprocess image for model input - handle dynamic shapes
        """
        model_info = self.models[model_name]
        input_shape = model_info['input_shape']
        
        # Get input dimensions - handle dynamic shapes
        if len(input_shape) == 4:
            try:
                # IMPORTANT: For NCHW format, dimension order is [batch, channels, height, width]
                # Make sure to use correct indices for height and width
                if input_shape[2] == 3 or input_shape[3] == 3:
                    # This suggests the model expects [batch, height, width, channels] format (NHWC)
                    # or there might be confusion in the input shape
                    print(f"WARNING: Input shape looks suspicious: {input_shape}")
                    height, width = 1000, 1000  # Use default values
                else:
                    # Try to extract height and width properly
                    try:
                        height = int(input_shape[2]) if not isinstance(input_shape[2], str) or not input_shape[2].startswith('unk') else 1000
                        width = int(input_shape[3]) if not isinstance(input_shape[3], str) or not input_shape[3].startswith('unk') else 1000
                    except (ValueError, TypeError):
                        height, width = 1000, 1000
            except (ValueError, TypeError, IndexError):
                # If conversion fails or indices are out of range, use default values
                print(f"Using default dimensions due to problematic shape: {input_shape}")
                height, width = 1000, 1000
        else:
            # Default to dimensions suitable for sheet music
            height, width = 1000, 1000
        
        # Safety check - ensure reasonable dimensions
        if height < 100 or width < 100:
            print(f"WARNING: Dimensions too small ({height}x{width}), using defaults")
            height, width = 1000, 1000
        
        print(f"Processing image for {model_name} with dimensions: {width}x{height}")
        
        # Special handling for very large sheet music images
        if image.shape[0] > 3000 or image.shape[1] > 3000:
            # Resize to a more manageable size while maintaining aspect ratio
            scale_factor = min(height / image.shape[0], width / image.shape[1])
            new_height = int(image.shape[0] * scale_factor)
            new_width = int(image.shape[1] * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            
            # Create a blank canvas of the target size
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Place the resized image in the center
            y_offset = (height - new_height) // 2
            x_offset = (width - new_width) // 2
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
            
            # Use the canvas as our resized image
            resized_image = canvas
        else:
            # For smaller images, just resize directly
            resized_image = cv2.resize(image, (width, height))
        
        # Convert to RGB if grayscale
        if len(resized_image.shape) == 2:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        elif resized_image.shape[2] == 1:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        
        # Try both input formats since different models might expect different formats
        try:
            # First try without normalization (keeping as uint8)
            # Some ONNX models expect raw uint8 input (0-255)
            uint8_input = np.transpose(resized_image, (2, 0, 1))  # Convert to NCHW
            uint8_input = np.expand_dims(uint8_input, axis=0)     # Add batch dimension
            
            # Test run with a small subset
            test_input = uint8_input[:, :, :5, :5]
            try:
                session = self.models[model_name]['session']
                input_name = self.models[model_name]['input_name']
                session.run(None, {input_name: test_input})
                print(f"Model {model_name} accepts uint8 input")
                return uint8_input, (resized_image.shape[1], resized_image.shape[0])
            except Exception as e:
                if "INVALID_ARGUMENT" in str(e) and "tensor(float)" in str(e):
                    # Model expects float input, try normalized version
                    normalized = resized_image.astype(np.float32) / 255.0
                    float_input = np.transpose(normalized, (2, 0, 1))
                    float_input = np.expand_dims(float_input, axis=0)
                    
                    # Test with small subset
                    test_float = float_input[:, :, :5, :5]
                    try:
                        session.run(None, {input_name: test_float})
                        print(f"Model {model_name} accepts float input")
                        return float_input, (resized_image.shape[1], resized_image.shape[0])
                    except:
                        pass
                
                # If both fail or it's another error, default to float
                print(f"Could not determine input format for {model_name}, defaulting to float")
                normalized = resized_image.astype(np.float32) / 255.0
                float_input = np.transpose(normalized, (2, 0, 1))
                float_input = np.expand_dims(float_input, axis=0)
                return float_input, (resized_image.shape[1], resized_image.shape[0])
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Default to float input as fallback
            normalized = resized_image.astype(np.float32) / 255.0
            float_input = np.transpose(normalized, (2, 0, 1))
            float_input = np.expand_dims(float_input, axis=0)
            return float_input, (resized_image.shape[1], resized_image.shape[0])

    def run_inference(self, image, model_name):
        """
        Run inference with specified model
        """
        model_info = self.models[model_name]
        session = model_info['session']
        input_name = model_info['input_name']
        
        try:
            # Preprocess image
            input_data, input_dims = self.preprocess_image(image, model_name)
            
            # Run inference
            try:
                outputs = session.run(None, {input_name: input_data})
                
                # Check if outputs are valid
                if not outputs or len(outputs) < 3:
                    print(f"Warning: Model {model_name} did not return enough outputs. Got {len(outputs) if outputs else 0} outputs.")
                    return {
                        'boxes': np.array([]),
                        'scores': np.array([]),
                        'classes': np.array([]),
                        'input_dims': input_dims
                    }
                
                # Parse outputs (structure depends on the model - adjust as needed)
                # For Faster R-CNN models converted to ONNX format
                # Format might be: boxes, scores, classes, num_detections
                boxes = outputs[0] if outputs[0].size > 0 else np.array([])
                scores = outputs[1] if outputs[1].size > 0 else np.array([])
                classes = outputs[2] if outputs[2].size > 0 else np.array([])
                
                # Debug output shapes
                print(f"Model {model_name} output shapes: boxes={boxes.shape}, scores={scores.shape}, classes={classes.shape}")
                
                # Filter by confidence
                confidence_threshold = 0.3  # Lower threshold for evaluation
                if scores.size > 0:
                    mask = scores > confidence_threshold
                    filtered_boxes = boxes[mask]
                    filtered_scores = scores[mask]
                    filtered_classes = classes[mask].astype(np.int32)
                    print(f"After filtering: {len(filtered_boxes)} detections with score > {confidence_threshold}")
                else:
                    filtered_boxes = np.array([])
                    filtered_scores = np.array([])
                    filtered_classes = np.array([])
                    print(f"No detections from model {model_name}")
                
                return {
                    'boxes': filtered_boxes,
                    'scores': filtered_scores,
                    'classes': filtered_classes,
                    'input_dims': input_dims
                }
                
            except Exception as e:
                print(f"Error running inference with model {model_name}: {e}")
                if 'outputs' in locals() and outputs and len(outputs) > 0:
                    print(f"Output shapes: {[o.shape for o in outputs]}")
                return {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'classes': np.array([]),
                    'input_dims': input_dims
                }
        except Exception as e:
            print(f"Error preprocessing image for model {model_name}: {e}")
            # Return empty results
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([]),
                'input_dims': (1000, 1000)  # Default dimensions
            }
    def evaluate_model(self, model_name, test_images, ground_truth):
        """
        Evaluate a model's performance by category
        
        Args:
            model_name: Name of the model to evaluate
            test_images: List of image paths
            ground_truth: Dictionary mapping image paths to ground truth annotations
                          with format {'boxes': [...], 'classes': [...]}
        
        Returns:
            Dictionary with performance metrics by category
        """
        results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for image_path in tqdm(test_images, desc=f"Evaluating {model_name}"):
            # Skip images without ground truth
            if image_path not in ground_truth:
                continue
                
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                continue
            
            # Run inference
            model_results = self.run_inference(image, model_name)
            
            # Get ground truth for this image
            gt = ground_truth.get(image_path, {'boxes': [], 'classes': []})
            
            # Scale boxes to original image dimensions
            orig_height, orig_width = image.shape[:2]
            input_width, input_height = model_results['input_dims']
            
            scaled_boxes = []
            for box in model_results['boxes']:
                scaled_box = [
                    box[0] * orig_width / input_width,
                    box[1] * orig_height / input_height,
                    box[2] * orig_width / input_width,
                    box[3] * orig_height / input_height
                ]
                scaled_boxes.append(scaled_box)
            
            # Categorize ground truth boxes
            gt_by_category = defaultdict(list)
            for i, class_id in enumerate(gt['classes']):
                category = self.class_to_category.get(class_id, 'other')
                gt_by_category[category].append((gt['boxes'][i], class_id))
            
            # Categorize detection boxes
            det_by_category = defaultdict(list)
            for i, class_id in enumerate(model_results['classes']):
                category = self.class_to_category.get(class_id, 'other')
                det_by_category[category].append((scaled_boxes[i], class_id, model_results['scores'][i]))
            
            # Evaluate each category
            for category in set(list(gt_by_category.keys()) + list(det_by_category.keys())):
                gt_boxes = gt_by_category[category]
                det_boxes = det_by_category[category]
                
                # Calculate metrics
                true_positives = 0
                matched_gt = set()
                
                # For each detection, find the best matching ground truth
                for det_box, det_class, det_score in det_boxes:
                    best_iou = 0.5  # IoU threshold
                    best_gt_idx = -1
                    
                    for i, (gt_box, gt_class) in enumerate(gt_boxes):
                        if i in matched_gt:
                            continue  # Skip already matched ground truths
                        
                        # Calculate IoU
                        iou = calculate_iou(det_box, gt_box)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    
                    if best_gt_idx >= 0:
                        # Found a match
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        # False positive
                        results[category]['fp'] += 1
                
                # Add true positives
                results[category]['tp'] += true_positives
                
                # Add false negatives (ground truths without matches)
                results[category]['fn'] += len(gt_boxes) - len(matched_gt)
        
        # Calculate precision, recall, F1 for each category
        for category, metrics in results.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[category]['precision'] = precision
            results[category]['recall'] = recall
            results[category]['f1'] = f1
        
        return dict(results)
    
    # Add these debugging statements to the evaluate_all_models function
    def evaluate_all_models(self, test_images, ground_truth):
        """
        Evaluate all models and compare performance by category
        
        Args:
            test_images: List of image paths
            ground_truth: Dictionary with ground truth annotations
        
        Returns:
            DataFrame with performance metrics for all models by category
        """
        all_results = {}
        category_weights = {}
        
        # Debug: Print some basic info about the input data
        print(f"Number of test images: {len(test_images)}")
        print(f"Number of images with ground truth: {len(ground_truth)}")
        print(f"Sample of ground truth keys: {list(ground_truth.keys())[:3] if ground_truth else 'None'}")
        
        # Check overlap between test_images and ground_truth
        overlap = [img for img in test_images if img in ground_truth]
        print(f"Number of images with both test data and ground truth: {len(overlap)}")
        
        if len(overlap) == 0:
            print("WARNING: No overlap between test images and ground truth!")
            print("This will result in an empty evaluation DataFrame.")
            # Create a minimal DataFrame to avoid errors
            return pd.DataFrame(columns=['Model', 'Category', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN'])
        
        for model_name in self.models.keys():
            print(f"\nEvaluating model: {model_name}")
            results = self.evaluate_model(model_name, test_images, ground_truth)
            all_results[model_name] = results
            
            # Debug: Print summary of results for this model
            if results:
                print(f"  Categories found: {list(results.keys())}")
                for category, metrics in results.items():
                    print(f"  {category}: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
            else:
                print("  No categories found in results!")
        
        # Debug: Check if we have any results at all
        if not all_results or all(not results for results in all_results.values()):
            print("WARNING: No evaluation results were produced!")
            return pd.DataFrame(columns=['Model', 'Category', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN'])
        
        # Convert to DataFrame for easier comparison
        model_comparison = defaultdict(list)
        categories = set()
        
        for model_name, results in all_results.items():
            for category, metrics in results.items():
                categories.add(category)
        
        print(f"Categories found across all models: {categories}")
        
        if not categories:
            print("WARNING: No categories found in results!")
            return pd.DataFrame(columns=['Model', 'Category', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN'])
        
        for category in sorted(categories):
            # Create a dict for category weights
            category_weights[category] = {}
            
            for model_name, results in all_results.items():
                if category in results:
                    metrics = results[category]
                    model_comparison['Model'].append(model_name)
                    model_comparison['Category'].append(category)
                    model_comparison['Precision'].append(metrics['precision'])
                    model_comparison['Recall'].append(metrics['recall'])
                    model_comparison['F1'].append(metrics['f1'])
                    model_comparison['TP'].append(metrics['tp'])
                    model_comparison['FP'].append(metrics['fp'])
                    model_comparison['FN'].append(metrics['fn'])
                    
                    # Store F1 score for category weights
                    category_weights[category][model_name] = metrics['f1']
        
        if not model_comparison['Model']:
            print("WARNING: No data to add to the DataFrame!")
            return pd.DataFrame(columns=['Model', 'Category', 'Precision', 'Recall', 'F1', 'TP', 'FP', 'FN'])
        
        df = pd.DataFrame(model_comparison)
        
        # Debug: Print DataFrame shape and summary
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        if not df.empty:
            print(f"DataFrame first row: {df.iloc[0].to_dict()}")
        
        # Save category weights to file
        with open('category_weights.json', 'w') as f:
            json.dump(category_weights, f, indent=2)
        
        return df


    # And modify the plot_category_comparison function to handle empty DataFrames:
    def plot_category_comparison(self, df, metric='F1'):
        """
        Plot performance comparison between models by category
        
        Args:
            df: DataFrame with evaluation results
            metric: Which metric to plot ('Precision', 'Recall', or 'F1')
        """
        # Check if DataFrame is empty or missing required columns
        if df.empty or 'Category' not in df.columns or 'Model' not in df.columns:
            print(f"WARNING: Cannot plot comparison - DataFrame is empty or missing required columns")
            return
        
        plt.figure(figsize=(14, 8))
        
        categories = sorted(df['Category'].unique())
        models = sorted(df['Model'].unique())
        
        if len(categories) == 0 or len(models) == 0:
            print(f"WARNING: No categories or models to plot")
            return
        
        bar_width = 0.8 / len(models)
        index = np.arange(len(categories))
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            values = []
            
            for category in categories:
                cat_data = model_data[model_data['Category'] == category]
                if not cat_data.empty and metric in cat_data:
                    values.append(cat_data[metric].values[0])
                else:
                    values.append(0)
            
            plt.bar(index + i * bar_width, values, bar_width,
                    label=model, alpha=0.7)
        
        plt.xlabel('Category')
        plt.ylabel(metric)
        plt.title(f'{metric} Score by Category')
        plt.xticks(index + bar_width * (len(models) - 1) / 2, categories, rotation=45)
        plt.legend(loc='best')
        plt.tight_layout()
        
        plt.savefig(f'model_comparison_{metric.lower()}.png')
        print(f"Saved plot to model_comparison_{metric.lower()}.png")


    def generate_optimal_weights(self, df):
        """
        Generate optimal weights for each category based on F1 scores
        
        Args:
            df: DataFrame with evaluation results
        
        Returns:
            Dictionary with optimal weights
        """
        weights = {}
        
        for category in df['Category'].unique():
            category_df = df[df['Category'] == category]
            
            # Sum of F1 scores for this category
            total_f1 = category_df['F1'].sum()
            
            # If all F1 scores are 0, use equal weights
            if total_f1 == 0:
                weights[category] = {model: 1.0 / len(category_df) for model in category_df['Model']}
            else:
                # Normalize F1 scores to get weights
                weights[category] = {}
                for _, row in category_df.iterrows():
                    weights[category][row['Model']] = row['F1'] / total_f1
        
        return weights


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


#  Fix the select_test_subset function to handle the file naming pattern better
def select_test_subset(image_dir, annotation_dir, percentage=5):
    """
    Select a random subset of images and their annotations
    
    Args:
        image_dir: Directory containing image files
        annotation_dir: Directory containing annotation files
        percentage: Percentage of data to select (default: 10%)
        
    Returns:
        List of tuples (image_path, annotation_path)
    """
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} image files in {image_dir}")
    
    # Get all annotation files for reference
    anno_files = [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]
    print(f"Found {len(anno_files)} annotation files in {annotation_dir}")
    
    # Calculate number of files to select
    num_files = max(1, int(len(image_files) * percentage / 100))
    
    # Randomly select files
    selected_images = random.sample(image_files, num_files)
    
    # Match each image with its annotation
    selected_pairs = []
    for img_file in selected_images:
        # Extract base name without extension
        base_name = os.path.splitext(img_file)[0]
        print(f"Processing image: {img_file}, base name: {base_name}")
        
        matched = False
        
        # Try different patterns to match annotation files
        
        # Pattern 1: Direct conversion of hyphens to underscores
        pattern1 = base_name.replace('-', '_')
        matching_annos = [f for f in anno_files if pattern1 in f]
        if matching_annos:
            anno_file = matching_annos[0]
            selected_pairs.append((
                os.path.join(image_dir, img_file),
                os.path.join(annotation_dir, anno_file)
            ))
            print(f"  Matched to annotation: {anno_file} (pattern 1)")
            matched = True
            continue
        
        # Pattern 2: Extract page number and create mapping
        parts = base_name.split('-')
        if len(parts) >= 3:
            try:
                # Try to extract page number
                page_num = str(int(parts[-1]))
                
                # Try without the page number
                base_without_page = '-'.join(parts[:-1])
                
                # Try different annotation filename patterns
                patterns = [
                    f"{base_without_page.replace('-', '_')}_Page_{page_num}.xml",
                    f"{base_without_page}_Page_{page_num}.xml",
                    f"{base_name.replace('-', '_')}.xml",
                    f"{base_name}_Page_{page_num}.xml"
                ]
                
                for pattern in patterns:
                    if pattern in anno_files:
                        selected_pairs.append((
                            os.path.join(image_dir, img_file),
                            os.path.join(annotation_dir, pattern)
                        ))
                        print(f"  Matched to annotation: {pattern} (pattern 2)")
                        matched = True
                        break
                        
                if matched:
                    continue
                    
                # If still not matched, look for partial matches
                for anno_file in anno_files:
                    if base_without_page.replace('-', '_') in anno_file:
                        selected_pairs.append((
                            os.path.join(image_dir, img_file),
                            os.path.join(annotation_dir, anno_file)
                        ))
                        print(f"  Matched to annotation: {anno_file} (partial match)")
                        matched = True
                        break
            except ValueError:
                # Not a number - try other approaches
                pass
        
        if not matched:
            # Final fallback: look through all annotation files for any partial match
            for anno_file in anno_files:
                # Remove extensions for comparison
                anno_base = os.path.splitext(anno_file)[0]
                # Check if there's significant overlap in the filenames
                if base_name in anno_base or anno_base in base_name:
                    selected_pairs.append((
                        os.path.join(image_dir, img_file),
                        os.path.join(annotation_dir, anno_file)
                    ))
                    print(f"  Matched to annotation: {anno_file} (fallback match)")
                    matched = True
                    break
        
        if not matched:
            print(f"  WARNING: No matching annotation found for {img_file}")
    
    print(f"Selected {len(selected_pairs)} image-annotation pairs for testing")
    return selected_pairs



def parse_xml_annotation(xml_path):
    """
    Parse XML annotation file for DOREMI dataset
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        Tuple of (boxes, classes) lists
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        classes = []
        
        print(f"Parsing XML: {xml_path}, root tag: {root.tag}")
        
        # Find all nodes in the XML
        nodes = root.findall('.//Node')
        print(f"  Found {len(nodes)} nodes")
        
        mapped_classes = 0
        skipped_classes = 0
        invalid_boxes = 0
        
        for node in nodes:
            # Get class name from ClassName element
            class_elem = node.find('ClassName')
            if class_elem is None or class_elem.text is None:
                print("  WARNING: Node missing ClassName element")
                continue
                
            class_name = class_elem.text
            # print(f"  Node class: {class_name}")
            
            # Map class name to ID using your class mapping
            class_id = map_class_name_to_id(class_name)
            if class_id is None:
                skipped_classes += 1
                continue  # Skip classes that don't map
            
            mapped_classes += 1
            
            # Parse bounding box coordinates from Top, Left, Width, Height elements
            top_elem = node.find('Top')
            left_elem = node.find('Left')
            width_elem = node.find('Width')
            height_elem = node.find('Height')
            
            if None in (top_elem, left_elem, width_elem, height_elem):
                print(f"  WARNING: Incomplete bounding box for {class_name}")
                invalid_boxes += 1
                continue
                
            try:
                top = float(top_elem.text)
                left = float(left_elem.text)
                width = float(width_elem.text)
                height = float(height_elem.text)
                
                # Convert to [xmin, ymin, xmax, ymax] format
                xmin = left
                ymin = top
                xmax = left + width
                ymax = top + height
                
                # Validate box coordinates
                if width <= 0 or height <= 0:
                    print(f"  WARNING: Invalid box dimensions: width={width}, height={height}")
                    invalid_boxes += 1
                    continue
            except (ValueError, TypeError) as e:
                print(f"  WARNING: Error parsing box coordinates: {e}")
                invalid_boxes += 1
                continue
            
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(class_id)
        
        print(f"  Parsed {len(boxes)} valid boxes from {mapped_classes} mapped classes")
        print(f"  Skipped {skipped_classes} unmapped classes and {invalid_boxes} invalid boxes")
        
        return boxes, classes
    except Exception as e:
        print(f"ERROR parsing XML file {xml_path}: {e}")
        return [], []

def map_class_name_to_id(class_name):
    """
    Map class name from DOREMI annotations to class ID in your model
    
    Args:
        class_name: Class name from annotation
        
    Returns:
        Class ID or None if not mapped
    """
    if class_name is None:
        print("WARNING: Received None as class name")
        return None
        
    # Direct mappings from DOREMI class names to your model's class IDs
    # This mapping is based on the class names in your XML format
    class_map = {
        # Staff and barlines
        'kStaffLine': 53,            # kStaffLine
        'staffLine': 53,             # Alternative
        'staff': 53,                 # Alternative
        'barline': 24,               # barline
        'systemicBarline': 74,       # systemic barline
        
        # Clefs
        'gClef': 51,                 # gClef
        'fClef': 42,                 # fClef
        'cClef': 26,                 # cClef
        '6stringTabClef': 1,         # 6stringTabClef
        'unpitchedPercussionClef1': 88, # unpitchedPercussionClef
        
        # Accidentals
        'accidentalDoubleFlat': 2,   # accidentalDoubleFlat
        'accidentalDoubleSharp': 3,  # accidentalDoubleSharp
        'accidentalFlat': 4,         # accidentalFlat
        'accidentalNatural': 5,      # accidentalNatural
        'accidentalQuarterToneFlatStein': 6, # accidentalQuarterToneFlatStein
        'accidentalQuarterToneSharpStein': 7, # accidentalQuarterToneSharpStein
        'accidentalSharp': 8,        # accidentalSharp
        
        # Articulations
        'articAccentAbove': 9,       # articAccentAbove
        'articAccentBelow': 10,      # articAccentBelow
        'articMarcatoAbove': 11,     # articMarcatoAbove
        'articMarcatoBelow': 12,     # articMarcatoBelow
        'articStaccatissimoAbove': 13, # articStaccatissimoAbove
        'articStaccatissimoBelow': 14, # articStaccatissimoBelow
        'articStaccatoAbove': 15,    # articStaccatoAbove
        'articStaccatoBelow': 16,    # articStaccatoBelow
        'articStressAbove': 17,      # articStressAbove
        'articStressBelow': 18,      # articStressBelow
        'articTenutoAbove': 19,      # articTenutoAbove
        'articTenutoBelow': 20,      # articTenutoBelow
        'articUnstressAbove': 21,    # articUnstressAbove
        'articUnstressBelow': 22,    # articUnstressBelow
        
        # Notation elements
        'augmentationDot': 23,       # augmentationDot
        'beam': 25,                  # beam
        'slur': 72,                  # slur
        'stem': 73,                  # stem
        'tie': 75,                   # tie
        
        # Dynamics
        'dynamicFF': 27,             # dynamicFF
        'dynamicFFF': 28,            # dynamicFFF
        'dynamicFFFF': 29,           # dynamicFFFF
        'dynamicForte': 30,          # dynamicForte
        'dynamicFortePiano': 31,     # dynamicFortePiano
        'dynamicForzando': 32,       # dynamicForzando
        'dynamicMF': 33,             # dynamicMF
        'dynamicMP': 34,             # dynamicMP
        'dynamicPP': 35,             # dynamicPP
        'dynamicPPP': 36,            # dynamicPPP
        'dynamicPPPP': 37,           # dynamicPPPP
        'dynamicPiano': 38,          # dynamicPiano
        'dynamicSforzato': 39,       # dynamicSforzato
        'dynamicSforzatoFF': 40,     # dynamicSforzatoFF
        'dynamicText': 41,           # dynamicText
        'gradualDynamic': 52,        # gradualDynamic
        
        # Flags
        'flag16thDown': 43,          # flag16thDown
        'flag16thUp': 44,            # flag16thUp
        'flag32ndDown': 45,          # flag32ndDown
        'flag32ndUp': 46,            # flag32ndUp
        'flag64thDown': 47,          # flag64thDown
        'flag64thUp': 48,            # flag64thUp
        'flag8thDown': 49,           # flag8thDown
        'flag8thUp': 50,             # flag8thUp
        
        # Noteheads
        'mensuralNoteheadMinimaWhite': 54, # mensuralNoteheadMinimaWhite
        'noteheadBlack': 55,         # noteheadBlack
        'noteheadDiamondBlack': 56,  # noteheadDiamondBlack
        'noteheadDiamondBlackOld': 57, # noteheadDiamondBlackOld
        'noteheadDiamondHalfOld': 58, # noteheadDiamondHalfOld
        'noteheadDiamondWhole': 59,  # noteheadDiamondWhole
        'noteheadDoubleWholeSquare': 60, # noteheadDoubleWholeSquare
        'noteheadHalf': 61,          # noteheadHalf
        'noteheadWhole': 62,         # noteheadWhole
        'noteheadXOrnateEllipse': 63, # noteheadXOrnateEllipse
        
        # Rests
        'rest': 64,                  # rest
        'rest16th': 65,              # rest16th
        'rest32nd': 66,              # rest32nd
        'rest64th': 67,              # rest64th
        'rest8th': 68,               # rest8th
        'restHalf': 69,              # restHalf
        'restQuarter': 70,           # restQuarter
        'restWhole': 71,             # restWhole
        
        # Time signatures
        'timeSig1': 76,              # timeSig1
        'timeSig2': 77,              # timeSig2
        'timeSig3': 78,              # timeSig3
        'timeSig4': 79,              # timeSig4
        'timeSig5': 80,              # timeSig5
        'timeSig6': 81,              # timeSig6
        'timeSig7': 82,              # timeSig7
        'timeSig8': 83,              # timeSig8
        'timeSig9': 84,              # timeSig9
        'timeSigCommon': 85,         # timeSigCommon
        'timeSigCutCommon': 86,      # timeSigCutCommon
        'timeSignatureComponent': 87 # timeSignatureComponent
    }
    
    # Try exact match first
    if class_name in class_map:
        return class_map[class_name]
    
    # Try case-insensitive matching
    lower_name = class_name.lower()
    for key, value in class_map.items():
        if key.lower() == lower_name:
            # Print when we find a match this way
            print(f"Case-insensitive match for '{class_name}' with '{key}' -> class ID {value}")
            return value
    
    # Try partial matching only for certain cases
    # This can be risky, so we limit it to specific patterns
    if 'staff' in lower_name or 'line' in lower_name:
        # Staff-related classes
        return 53  # kStaffLine
    elif 'clef' in lower_name:
        if 'g' in lower_name:
            return 51  # gClef
        elif 'f' in lower_name:
            return 42  # fClef
        elif 'c' in lower_name:
            return 26  # cClef
    elif 'notehead' in lower_name:
        if 'black' in lower_name:
            return 55  # noteheadBlack
        elif 'half' in lower_name:
            return 61  # noteheadHalf
        elif 'whole' in lower_name:
            return 62  # noteheadWhole
    elif 'rest' in lower_name:
        return 64  # rest
    elif 'bar' in lower_name:
        return 24  # barline
    
    # Print unmatched classes to help refine the mapping
    print(f"No mapping found for class: '{class_name}'")
    return None  # No mapping found

def main():
    # Define paths
    model_paths = [
        '/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-lr-0.003-classes-88-steps-80000-2475x3504-09-10-2020-008-train/Faster_R-CNN_inception-lr-0.003-classes-88-steps-80000-2475x3504-09-10-2020-008-train.onnx',
        '/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_resnet50-lr-0.003-classes-72-steps-100000-2475x3504-03-10-2020-004-train/Faster_R-CNN_resnet50-lr-0.003-classes-72-steps-100000-2475x3504-03-10-2020-004-train.onnx',
        '/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster-R-CNN-resnet50-88-classes-100000-2475x3504-06-10-2020-006-train/Faster-R-CNN-resnet50-88-classes-100000-2475x3504-06-10-2020-006-train.onnx'
    ]
    
    # DOREMI dataset paths
    image_dir = '/homes/es314/DOREMI/data/images'
    annotation_dir = '/homes/es314/DOREMI/data/page_annotations'
    
    # Initialize evaluator
    evaluator = OMRModelEvaluator(model_paths, "")
    
    # Select 10% of data for testing
    test_pairs = select_test_subset(image_dir, annotation_dir, percentage=10)
    
    # Extract just the image paths for the evaluator
    test_images = [img_path for img_path, _ in test_pairs]
    
    # Load ground truth from the selected annotations
    ground_truth = {}
    for img_path, anno_path in test_pairs:
        # Parse the XML annotation file
        boxes, classes = parse_xml_annotation(anno_path)
        if boxes and classes:  # Only add if we have valid annotations
            ground_truth[img_path] = {
                'boxes': np.array(boxes),
                'classes': np.array(classes)
            }
    
    print(f"Loaded ground truth for {len(ground_truth)} images")
    
    # Evaluate all models
    comparison_df = evaluator.evaluate_all_models(test_images, ground_truth)
    
    # Save results to CSV
    comparison_df.to_csv('model_comparison_by_category.csv', index=False)
    
    # Plot results
    evaluator.plot_category_comparison(comparison_df, 'F1')
    evaluator.plot_category_comparison(comparison_df, 'Precision')
    evaluator.plot_category_comparison(comparison_df, 'Recall')
    
    # Generate optimal weights
    optimal_weights = evaluator.generate_optimal_weights(comparison_df)
    
    # Save optimal weights to file
    with open('optimal_category_weights.json', 'w') as f:
        json.dump(optimal_weights, f, indent=2)
    
    # Print best model for each category
    print("\nBest model for each category (by F1 score):")
    for category in comparison_df['Category'].unique():
        cat_df = comparison_df[comparison_df['Category'] == category]
        if not cat_df.empty:
            best_idx = cat_df['F1'].idxmax()
            if not pd.isna(best_idx):  # Check if there are any valid rows
                best_model = cat_df.loc[best_idx]
                print(f"{category}: {best_model['Model']} (F1: {best_model['F1']:.3f})")


if __name__ == "__main__":
    main()