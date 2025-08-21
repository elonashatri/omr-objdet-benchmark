#!/usr/bin/env python3
import os
import json
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple, Set, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import re
import glob

class OMREvaluator:
    """
    Evaluator for OMR detection results against ground truth.
    Computes mAP (mean Average Precision) per category and overall.
    """
    
    def __init__(self, iou_threshold=0.5):
        """
        Initialize the evaluator.
        
        Args:
            iou_threshold: Threshold for considering a detection as correct
        """
        self.iou_threshold = iou_threshold
        self.categories = set()  # Will be populated with all class names
        
    def calculate_iou(self, box1: Dict[str, float], box2: Dict[str, float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: First bounding box with x1, y1, x2, y2 coordinates
            box2: Second bounding box with x1, y1, x2, y2 coordinates
            
        Returns:
            IoU score (float between 0 and 1)
        """
        # Calculate intersection area
        x_left = max(box1['x1'], box2['x1'])
        y_top = max(box1['y1'], box2['y1'])
        x_right = min(box1['x2'], box2['x2'])
        y_bottom = min(box1['y2'], box2['y2'])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of each box
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        
        # Calculate union area
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou
    
    def parse_ground_truth_xml(self, xml_file: str) -> List[Dict[str, Any]]:
        """
        Parse ground truth XML file to extract bounding boxes.
        
        Args:
            xml_file: Path to the ground truth XML file
            
        Returns:
            List of ground truth objects with class_name and bbox
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        gt_objects = []
        
        # Extract all Node elements
        for node in root.findall(".//Node"):
            class_name = node.get('ClassName')
            if class_name:
                # Skip staff lines as they're usually not part of the evaluation
                if class_name == "kStaffLine":
                    continue
                
                left = float(node.get('Left', 0))
                top = float(node.get('Top', 0))
                width = float(node.get('Width', 0))
                height = float(node.get('Height', 0))
                
                # Extract bounding box coordinates
                bbox = {
                    'x1': left,
                    'y1': top,
                    'x2': left + width,
                    'y2': top + height,
                    'width': width,
                    'height': height,
                    'center_x': left + width/2,
                    'center_y': top + height/2
                }
                
                gt_objects.append({
                    'class_name': class_name,
                    'bbox': bbox
                })
                
                # Add to categories
                self.categories.add(class_name)
        
        return gt_objects
    
    def parse_detection_json(self, json_file: str) -> List[Dict[str, Any]]:
        """
        Parse detection JSON file to extract bounding boxes.
        
        Args:
            json_file: Path to the detection JSON file
            
        Returns:
            List of detection objects with class_name, confidence, and bbox
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        detections = []
        
        for item in data.get('detections', []):
            class_name = item.get('class_name')
            confidence = item.get('confidence', 0.0)
            bbox = item.get('bbox', {})
            
            detections.append({
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox
            })
            
            # Add to categories
            if class_name:
                self.categories.add(class_name)
        
        return detections
    
    def parse_staffline_json(self, json_file: str) -> List[Dict[str, Any]]:
        """
        Parse staffline JSON file to extract bounding boxes.
        
        Args:
            json_file: Path to the staffline JSON file
            
        Returns:
            List of staffline objects with class_name, confidence, and bbox
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        detections = []
        
        for item in data.get('detections', []):
            class_name = item.get('class_name')
            confidence = item.get('confidence', 0.0)
            bbox = item.get('bbox', {})
            
            # Add staff system and line number information if available
            staff_system = item.get('staff_system')
            line_number = item.get('line_number')
            
            detection = {
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox
            }
            
            # Add staff information if available
            if staff_system is not None:
                detection['staff_system'] = staff_system
            if line_number is not None:
                detection['line_number'] = line_number
                
            detections.append(detection)
            
            # Add to categories
            if class_name:
                self.categories.add(class_name)
        
        return detections
    
    def compute_average_precision(self, gt_objects: List[Dict], detections: List[Dict], 
                                class_name: str = None) -> float:
        """
        Compute Average Precision for a single class or overall.
        
        Args:
            gt_objects: List of ground truth objects
            detections: List of detection objects
            class_name: Optional class name to filter for
            
        Returns:
            Average Precision score
        """
        # Filter for class if specified
        if class_name:
            gt_class = [obj for obj in gt_objects if obj['class_name'] == class_name]
            det_class = [obj for obj in detections if obj['class_name'] == class_name]
        else:
            gt_class = gt_objects
            det_class = detections
        
        # Sort detections by confidence (descending)
        det_class = sorted(det_class, key=lambda x: x['confidence'], reverse=True)
        
        num_gt = len(gt_class)
        if num_gt == 0:
            return 0.0  # No ground truth objects for this class
            
        # Set for tracking matched GT objects
        matched_gt = set()
        
        # Lists for precision and recall calculation
        tp = np.zeros(len(det_class))
        fp = np.zeros(len(det_class))
        
        # Match detections to ground truth
        for i, det in enumerate(det_class):
            # Find best matching GT object
            best_iou = -1
            best_gt_idx = -1
            
            for j, gt in enumerate(gt_class):
                if j in matched_gt:
                    continue  # Skip already matched GT objects
                
                iou = self.calculate_iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Check if detection is matched to a GT object
            if best_iou >= self.iou_threshold:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1
        
        # Compute cumulative precision and recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / num_gt
        precision = cum_tp / (cum_tp + cum_fp + 1e-10)  # Avoid division by zero
        
        # Compute AP using 11-point interpolation (VOC2007 method)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
                
        return ap
    
    def evaluate(self, gt_file: str, detection_files: Dict[str, str], 
                staffline_file: str = None, output_dir: str = None, 
                skip_plots: bool = False) -> Dict[str, Any]:
        """
        Evaluate detection results against ground truth.
        
        Args:
            gt_file: Path to ground truth XML file
            detection_files: Dictionary mapping detection type to file path
            staffline_file: Optional path to staffline JSON file to include with combined detections
            output_dir: Optional directory to save results
            skip_plots: Whether to skip generating plots (for faster processing)
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Reading ground truth file: {gt_file}")
        gt_objects = self.parse_ground_truth_xml(gt_file)
        print(f"Found {len(gt_objects)} ground truth objects across {len(self.categories)} categories")
        
        # Parse each detection file
        detection_results = {}
        for det_type, file_path in detection_files.items():
            print(f"Reading {det_type} detection file: {file_path}")
            detections = self.parse_detection_json(file_path)
            
            # If this is the combined type and staffline file is provided, add stafflines
            if det_type == 'combined' and staffline_file:
                print(f"Reading staffline file: {staffline_file}")
                staffline_detections = self.parse_staffline_json(staffline_file)
                detections.extend(staffline_detections)
                print(f"Added {len(staffline_detections)} staffline detections to combined detections")
            
            detection_results[det_type] = detections
            print(f"Found {len(detections)} detections in {det_type}")
        
        # Compute AP per category and per detection type
        results = {
            'per_category': defaultdict(dict),
            'overall': {},
            'categories': sorted(list(self.categories))
        }
        
        for det_type, detections in detection_results.items():
            # Compute per-category AP
            category_aps = {}
            for category in self.categories:
                ap = self.compute_average_precision(gt_objects, detections, category)
                category_aps[category] = ap
                results['per_category'][category][det_type] = ap
            
            # Compute overall mAP
            overall_ap = self.compute_average_precision(gt_objects, detections)
            results['overall'][det_type] = overall_ap
            
            print(f"{det_type} - Overall mAP: {overall_ap:.4f}")
            for category, ap in sorted(category_aps.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {category}: {ap:.4f}")
        
        # Save results if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results as JSON
            with open(os.path.join(output_dir, 'map_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate bar charts only if not skipping plots
            if not skip_plots:
                self._generate_charts(results, output_dir)
        
        return results
    
    def _generate_charts(self, results: Dict[str, Any], output_dir: str):
        """
        Generate visualization charts for the results.
        
        Args:
            results: Dictionary with evaluation results
            output_dir: Directory to save charts
        """
        # Overall mAP comparison
        plt.figure(figsize=(10, 6))
        det_types = list(results['overall'].keys())
        overall_map = [results['overall'][dt] for dt in det_types]
        
        plt.bar(det_types, overall_map)
        plt.ylabel('mAP')
        plt.title('Overall mAP by Detection Type')
        plt.savefig(os.path.join(output_dir, 'overall_map.png'))
        plt.close()
        
        # Top categories per detection type
        for det_type in det_types:
            plt.figure(figsize=(12, 8))
            
            category_aps = {cat: results['per_category'][cat].get(det_type, 0) 
                           for cat in results['categories']}
            
            # Sort and get top 15 categories
            top_categories = sorted(category_aps.items(), key=lambda x: x[1], reverse=True)[:15]
            cats, aps = zip(*top_categories)
            
            plt.bar(cats, aps)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('AP')
            plt.title(f'Top 15 Categories by AP - {det_type}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{det_type}_top_categories.png'))
            plt.close()
            
        # Comparative bar plot across all detection types and categories
        plt.figure(figsize=(16, 10))
        
        # Get all categories and detection types
        categories = results['categories']
        det_types = list(results['overall'].keys())
        
        # Width of each bar group
        bar_width = 0.8 / len(det_types)
        
        # Colors for each detection type
        colors = ['#3274A1', '#E1812C', '#3A923A']  # Blue, Orange, Green
        
        # Set position of bars on x-axis
        indices = np.arange(len(categories))
        
        # Create grouped bars
        for i, det_type in enumerate(det_types):
            # Get AP values for this detection type across all categories
            ap_values = [results['per_category'][cat].get(det_type, 0) for cat in categories]
            
            # Calculate position for this group of bars
            positions = indices + (i - len(det_types)/2 + 0.5) * bar_width
            
            # Create bars
            plt.bar(positions, ap_values, bar_width, 
                   label=det_type, color=colors[i % len(colors)], alpha=0.8)
        
        # Add labels and legend
        plt.xlabel('Categories')
        plt.ylabel('AP')
        plt.title('Average Precision by Category and Detection Type')
        plt.xticks(indices, categories, rotation=90, fontsize=8)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'comparative_ap_all_categories.png'), dpi=300)
        plt.close()
        
        # Additional comparative bar plot for top performing categories
        plt.figure(figsize=(14, 8))
        
        # Get average AP across all detection types for each category
        avg_category_aps = {
            cat: np.mean([results['per_category'][cat].get(dt, 0) for dt in det_types]) 
            for cat in categories
        }
        
        # Select top 20 categories by average AP
        top_categories = sorted(avg_category_aps.items(), key=lambda x: x[1], reverse=True)[:20]
        top_cats = [item[0] for item in top_categories]
        
        # Width of each bar group
        bar_width = 0.8 / len(det_types)
        indices = np.arange(len(top_cats))
        
        # Create grouped bars for top categories
        for i, det_type in enumerate(det_types):
            # Get AP values for this detection type across top categories
            ap_values = [results['per_category'][cat].get(det_type, 0) for cat in top_cats]
            
            # Calculate position for this group of bars
            positions = indices + (i - len(det_types)/2 + 0.5) * bar_width
            
            # Create bars
            plt.bar(positions, ap_values, bar_width, 
                   label=det_type, color=colors[i % len(colors)], alpha=0.8)
        
        # Add labels and legend
        plt.xlabel('Categories')
        plt.ylabel('AP')
        plt.title('Average Precision for Top 20 Categories by Detection Type')
        plt.xticks(indices, top_cats, rotation=45, ha='right', fontsize=9)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'comparative_ap_top_categories.png'), dpi=300)
        plt.close()
        
        # Create a "performance gain" visualization showing how combined improves over individual methods
        plt.figure(figsize=(14, 10))
        
        # Get relevant data
        categories = results['categories']
        
        # Calculate the maximum AP between symbol and structure for each category
        max_individual_ap = []
        combined_ap = []
        improvement_percentage = []
        category_labels = []
        
        for cat in categories:
            symbol_ap = results['per_category'][cat].get('symbol', 0)
            structure_ap = results['per_category'][cat].get('structure', 0)
            combined_value = results['per_category'][cat].get('combined', 0)
            
            max_individual = max(symbol_ap, structure_ap)
            
            # Only include categories where we have meaningful data
            if max_individual > 0.01 or combined_value > 0.01:
                max_individual_ap.append(max_individual)
                combined_ap.append(combined_value)
                
                # Calculate improvement percentage
                if max_individual > 0:
                    improvement = ((combined_value - max_individual) / max_individual) * 100
                else:
                    improvement = 0 if combined_value == 0 else 100
                
                improvement_percentage.append(improvement)
                category_labels.append(cat)
        
        # Sort by improvement percentage
        sorted_indices = np.argsort(improvement_percentage)
        
        # Get data in sorted order
        sorted_categories = [category_labels[i] for i in sorted_indices]
        sorted_max_individual = [max_individual_ap[i] for i in sorted_indices]
        sorted_combined = [combined_ap[i] for i in sorted_indices]
        sorted_improvement = [improvement_percentage[i] for i in sorted_indices]
        
        # Plot settings
        indices = np.arange(len(sorted_categories))
        bar_width = 0.35
        opacity = 0.8
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(16, 10))
        
        # First y-axis for AP values
        ax1.set_ylabel('Average Precision (AP)', fontsize=12)
        
        # Plot the max individual AP (symbol or structure, whichever is better)
        ax1.bar(indices - bar_width/2, sorted_max_individual, bar_width,
                alpha=opacity, color='#3A923A', label='Best Individual (Symbol or Structure)')
        
        # Plot the combined AP
        ax1.bar(indices + bar_width/2, sorted_combined, bar_width,
                alpha=opacity, color='#3274A1', label='Combined')
        
        # Create second y-axis for improvement percentage
        ax2 = ax1.twinx()
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        
        # Plot improvement as a line
        ax2.plot(indices, sorted_improvement, 'r-', marker='o', linewidth=2, label='Improvement %')
        
        # Set ticks and labels
        ax1.set_xticks(indices)
        ax1.set_xticklabels(sorted_categories, rotation=90, fontsize=9)
        
        # Add grid, title and legend
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax1.set_title('Combined vs. Best Individual Method (Symbol or Structure)', fontsize=14)
        
        # Add legends for both axes
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_improvement_analysis.png'), dpi=300)
        plt.close()
        
        # Create a scatter plot showing the relationship between max individual performance and combined performance
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot
        plt.scatter(max_individual_ap, combined_ap, alpha=0.7, s=100)
        
        # Add a diagonal line (y=x) to represent "no improvement"
        max_value = max(max(max_individual_ap), max(combined_ap))
        plt.plot([0, max_value], [0, max_value], 'k--', alpha=0.5, label='No Improvement Line')
        
        # Add category labels to points
        for i, cat in enumerate(category_labels):
            plt.annotate(cat, (max_individual_ap[i], combined_ap[i]), 
                        fontsize=7, alpha=0.8, 
                        xytext=(5, 5), textcoords='offset points')
        
        # Add labels and title
        plt.xlabel('Best Individual Method AP (Symbol or Structure)', fontsize=12)
        plt.ylabel('Combined Method AP', fontsize=12)
        plt.title('Combined vs. Best Individual Method Performance', fontsize=14)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Points above the line show improvement
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_vs_individual_scatter.png'), dpi=300)
        plt.close()

def extract_base_filename(path):
    """
    Extract the base filename without suffixes like '_detections' or '_pixel_perfect'
    
    Args:
        path: Path to the file
        
    Returns:
        Base filename that can be used to match with ground truth
    """
    filename = os.path.basename(path)
    # Remove common suffixes
    filename = re.sub(r'_(detections|combined_detections|pixel_perfect)\.json$', '', filename)
    return filename

def find_matching_gt_file(base_filename, gt_dir):
    """
    Find the matching ground truth XML file for a given base filename
    
    Args:
        base_filename: Base filename extracted from detection file
        gt_dir: Directory with ground truth XML files
        
    Returns:
        Path to matching ground truth file or None if not found
    """
    # Look for exact match
    gt_path = os.path.join(gt_dir, f"{base_filename}.xml")
    if os.path.exists(gt_path):
        return gt_path
    
    # Try fuzzy matching if exact match not found
    candidates = glob.glob(os.path.join(gt_dir, "*.xml"))
    for candidate in candidates:
        cand_base = os.path.splitext(os.path.basename(candidate))[0]
        if base_filename in cand_base or cand_base in base_filename:
            return candidate
    
    return None

def evaluate_all_files(combined_dir, symbol_dir, structure_dir, staffline_dir, gt_dir, output_base_dir, iou_threshold=0.5, skip_individual_plots=False):
    """
    Evaluate all matching files in the directories
    
    Args:
        combined_dir: Directory with combined detection JSON files
        symbol_dir: Directory with symbol detection JSON files
        structure_dir: Directory with structure detection JSON files
        staffline_dir: Directory with staffline JSON files
        gt_dir: Directory with ground truth XML files
        output_base_dir: Base directory for output results
        iou_threshold: IoU threshold for matching
        skip_individual_plots: Whether to skip generating plots for individual files
        
    Returns:
        Dictionary with overall results across all files
    """
    print(f"Debug: combined_dir = {combined_dir}")
    print(f"Debug: Looking for JSON files in {os.path.join(combined_dir, '*.json')}")
    
    # Get all combined detection files
    combined_files = glob.glob(os.path.join(combined_dir, "*.json"))
    print(f"Debug: Found {len(combined_files)} combined detection files")
    
    if len(combined_files) == 0:
        print(f"Debug: Directory exists? {os.path.exists(combined_dir)}")
        if os.path.exists(combined_dir):
            print(f"Debug: Directory contents: {os.listdir(combined_dir)}")
    
    all_results = {
        'per_file': {},
        'overall_avg_map': {},
        'per_category_avg': defaultdict(dict)  # To store average AP per category across all files
    }
    
    # Counters for categories across all files
    category_counts = defaultdict(int)
    category_sum_ap = defaultdict(lambda: defaultdict(float))
    
    for i, combined_file in enumerate(combined_files):
        base_filename = extract_base_filename(combined_file)
        print(f"\nProcessing file {i+1}/{len(combined_files)}: {base_filename}...")
        
        # Find corresponding files
        symbol_file = glob.glob(os.path.join(symbol_dir, f"{base_filename}*_detections.json"))
        structure_file = glob.glob(os.path.join(structure_dir, f"{base_filename}*_detections.json"))
        staffline_file = glob.glob(os.path.join(staffline_dir, f"{base_filename}*_pixel_perfect.json"))
        
        gt_file = find_matching_gt_file(base_filename, gt_dir)
        
        if not gt_file:
            print(f"No matching ground truth found for {base_filename}, skipping.")
            continue
        
        if not symbol_file or not structure_file:
            print(f"Missing detection files for {base_filename}, skipping.")
            continue
        
        # Create an output directory for this file
        file_output_dir = os.path.join(output_base_dir, base_filename) if not skip_individual_plots else None
        if file_output_dir:
            os.makedirs(file_output_dir, exist_ok=True)
        
        detection_files = {
            'combined': combined_file,
            'symbol': symbol_file[0],
            'structure': structure_file[0]
        }
        
        staffline_path = staffline_file[0] if staffline_file else None
        
        # Initialize evaluator and run evaluation
        evaluator = OMREvaluator(iou_threshold=iou_threshold)
        results = evaluator.evaluate(gt_file, detection_files, staffline_path, file_output_dir, skip_plots=skip_individual_plots)
        
        # Store results for this file
        all_results['per_file'][base_filename] = results
        
        # Accumulate category APs for averaging
        for category in results['categories']:
            for det_type in results['overall'].keys():
                if category in results['per_category'] and det_type in results['per_category'][category]:
                    ap_value = results['per_category'][category][det_type]
                    category_sum_ap[category][det_type] += ap_value
                    category_counts[category] += 1
    
    # Calculate average mAP across all files
    if all_results['per_file']:
        combined_maps = []
        symbol_maps = []
        structure_maps = []
        
        for file_results in all_results['per_file'].values():
            if 'overall' in file_results:
                if 'combined' in file_results['overall']:
                    combined_maps.append(file_results['overall']['combined'])
                if 'symbol' in file_results['overall']:
                    symbol_maps.append(file_results['overall']['symbol'])
                if 'structure' in file_results['overall']:
                    structure_maps.append(file_results['overall']['structure'])
        
        # Calculate averages
        if combined_maps:
            all_results['overall_avg_map']['combined'] = sum(combined_maps) / len(combined_maps)
        if symbol_maps:
            all_results['overall_avg_map']['symbol'] = sum(symbol_maps) / len(symbol_maps)
        if structure_maps:
            all_results['overall_avg_map']['structure'] = sum(structure_maps) / len(structure_maps)
        
        # Calculate per-category averages
        for category, count in category_counts.items():
            for det_type in ['combined', 'symbol', 'structure']:
                if det_type in category_sum_ap[category]:
                    all_results['per_category_avg'][category][det_type] = category_sum_ap[category][det_type] / count
        
        # Generate overall plots
        generate_overall_results(all_results, output_base_dir)
    
    return all_results

def generate_overall_results(all_results, output_dir):
    """
    Generate summary visualizations for results across all files
    
    Args:
        all_results: Dictionary with results for all files
        output_dir: Output directory for results
    """
    # Create overall mAP comparison chart
    plt.figure(figsize=(10, 6))
    
    # Extract overall average mAP values
    det_types = list(all_results['overall_avg_map'].keys())
    overall_maps = [all_results['overall_avg_map'][dt] for dt in det_types]
    
    # Bar chart for overall average mAP
    plt.bar(det_types, overall_maps)
    plt.ylabel('Average mAP across all files')
    plt.title('Overall Average mAP by Detection Type')
    plt.savefig(os.path.join(output_dir, 'overall_average_map.png'))
    plt.close()
    
    # Create a table of mAP values per file
    fig, ax = plt.subplots(figsize=(12, len(all_results['per_file']) * 0.4 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    for filename, results in all_results['per_file'].items():
        row = [filename]
        for det_type in det_types:
            if det_type in results['overall']:
                row.append(f"{results['overall'][det_type]:.4f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Create table
    header = ['Filename'] + det_types
    table = ax.table(cellText=table_data, colLabels=header, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Add title
    plt.title('mAP Results by File', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'map_results_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # If we have per-category average data, create a visualization
    if 'per_category_avg' in all_results and all_results['per_category_avg']:
        # Get top 30 categories by combined performance
        categories = list(all_results['per_category_avg'].keys())
        top_categories = sorted(
            categories,
            key=lambda cat: all_results['per_category_avg'][cat].get('combined', 0),
            reverse=True
        )[:30]
        
        # Create bar chart for top categories
        plt.figure(figsize=(14, 8))
        
        # Width of each bar group
        bar_width = 0.8 / len(det_types)
        indices = np.arange(len(top_categories))
        
        # Colors for each detection type
        colors = ['#3274A1', '#E1812C', '#3A923A']  # Blue, Orange, Green
        
        # Create grouped bars for top categories
        for i, det_type in enumerate(det_types):
            # Get AP values for this detection type across top categories
            ap_values = [all_results['per_category_avg'][cat].get(det_type, 0) for cat in top_categories]
            
            # Calculate position for this group of bars
            positions = indices + (i - len(det_types)/2 + 0.5) * bar_width
            
            # Create bars
            plt.bar(positions, ap_values, bar_width, 
                   label=det_type, color=colors[i % len(colors)], alpha=0.8)
        
        # Add labels and legend
        plt.xlabel('Categories')
        plt.ylabel('Average AP')
        plt.title('Average AP for Top Categories by Detection Type')
        plt.xticks(indices, top_categories, rotation=45, ha='right', fontsize=9)
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'top_categories_average_ap.png'), dpi=300)
        plt.close()
        
        # Performance gain visualization for combined vs. best individual
        plt.figure(figsize=(14, 10))
        
        # Calculate improvement for each category
        improvement_data = []
        
        for cat in top_categories:
            symbol_ap = all_results['per_category_avg'][cat].get('symbol', 0)
            structure_ap = all_results['per_category_avg'][cat].get('structure', 0)
            combined_value = all_results['per_category_avg'][cat].get('combined', 0)
            
            max_individual = max(symbol_ap, structure_ap)
            
            # Calculate improvement percentage
            if max_individual > 0:
                improvement = ((combined_value - max_individual) / max_individual) * 100
            else:
                improvement = 0 if combined_value == 0 else 100
                
            improvement_data.append((cat, max_individual, combined_value, improvement))
        
        # Sort by improvement percentage
        sorted_improvement = sorted(improvement_data, key=lambda x: x[3])
        
        # Get data in sorted order
        sorted_categories = [item[0] for item in sorted_improvement]
        sorted_max_individual = [item[1] for item in sorted_improvement]
        sorted_combined = [item[2] for item in sorted_improvement]
        sorted_improvement_pct = [item[3] for item in sorted_improvement]
        
        # Plot settings
        indices = np.arange(len(sorted_categories))
        bar_width = 0.35
        opacity = 0.8
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(16, 10))
        
        # First y-axis for AP values
        ax1.set_ylabel('Average Precision (AP)', fontsize=12)
        
        # Plot the max individual AP (symbol or structure, whichever is better)
        ax1.bar(indices - bar_width/2, sorted_max_individual, bar_width,
                alpha=opacity, color='#3A923A', label='Best Individual (Symbol or Structure)')
        
        # Plot the combined AP
        ax1.bar(indices + bar_width/2, sorted_combined, bar_width,
                alpha=opacity, color='#3274A1', label='Combined')
        
        # Create second y-axis for improvement percentage
        ax2 = ax1.twinx()
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        
        # Plot improvement as a line
        ax2.plot(indices, sorted_improvement_pct, 'r-', marker='o', linewidth=2, label='Improvement %')
        
        # Set ticks and labels
        ax1.set_xticks(indices)
        ax1.set_xticklabels(sorted_categories, rotation=90, fontsize=9)
        
        # Add grid, title and legend
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax1.set_title('Combined vs. Best Individual Method (Average across all files)', fontsize=14)
        
        # Add legends for both axes
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_improvement_analysis_avg.png'), dpi=300)
        plt.close()
    
    # Save overall results as JSON
    with open(os.path.join(output_dir, 'overall_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate OMR detection results against ground truth')
    parser.add_argument('--combined_dir', help='Directory with combined detection JSON files')
    parser.add_argument('--symbol_dir', help='Directory with symbol detection JSON files')
    parser.add_argument('--structure_dir', help='Directory with structure detection JSON files')
    parser.add_argument('--staffline_dir', help='Directory with staffline JSON files')
    parser.add_argument('--gt_dir', help='Directory with ground truth XML files')
    parser.add_argument('--output', default='evaluation_results', help='Output directory for results')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--skip_plots', action='store_true', help='Skip generating plots for individual files')
    
    # Also keep the individual file options for backward compatibility
    parser.add_argument('--gt', help='Path to ground truth XML file (for evaluating a single file)')
    parser.add_argument('--combined', help='Path to combined detections JSON file')
    parser.add_argument('--symbol', help='Path to symbol detections JSON file')
    parser.add_argument('--structure', help='Path to structure detections JSON file')
    parser.add_argument('--staffline', help='Path to staffline JSON file')
    
    args = parser.parse_args()
    
    # Check if we're evaluating directories or individual files
    if args.combined_dir and args.symbol_dir and args.structure_dir and args.gt_dir:
        # Directory-based evaluation
        print("Running directory-based evaluation...")
        all_results = evaluate_all_files(
            args.combined_dir,
            args.symbol_dir,
            args.structure_dir,
            args.staffline_dir or "",  # Empty string if staffline_dir is None
            args.gt_dir,
            args.output,
            args.iou,
            args.skip_plots
        )
        print(f"\nEvaluation complete. Overall results saved to: {args.output}")
        print("\nAverage mAP across all files:")
        for det_type, map_value in all_results['overall_avg_map'].items():
            print(f"  {det_type}: {map_value:.4f}")
        
    elif args.gt and args.combined and args.symbol and args.structure:
        # Single file evaluation (backward compatibility)
        print("Running single file evaluation...")
        evaluator = OMREvaluator(iou_threshold=args.iou)
        
        detection_files = {
            'combined': args.combined,
            'symbol': args.symbol,
            'structure': args.structure
        }
        
        results = evaluator.evaluate(args.gt, detection_files, args.staffline, args.output, args.skip_plots)
        print(f"\nEvaluation complete. Results saved to: {args.output}")
    
    else:
        parser.print_help()
        print("\nError: You must either provide directory paths or individual file paths.")
        return 1

if __name__ == "__main__":
    main()

# python omr_evaluator.py \
#   --gt "/import/c4dm-05/elona/EVAL/1-OMR_EVAL/GT/restructured_with_key_sig/1-2-Kirschner_-_Chissà_che_cosa_pensa-001.xml" \
#   --combined "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/results/1-2-Kirschner_-_Chissà_che_cosa_pensa-001/combined_detections/1-2-Kirschner_-_Chissà_che_cosa_pensa-001_combined_detections.json" \
#   --symbol "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/results/1-2-Kirschner_-_Chissà_che_cosa_pensa-001/symbol_detections/1-2-Kirschner_-_Chissà_che_cosa_pensa-001_detections.json" \
#   --structure "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/complete_pipeline_1/results/1-2-Kirschner_-_Chissà_che_cosa_pensa-001/structure_detections/1-2-Kirschner_-_Chissà_che_cosa_pensa-001_detections.json" \
#   --output "omr_evaluation_results"

# python omr_evaluator.py --output "omr_evaluation_results"