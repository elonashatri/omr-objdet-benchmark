#!/usr/bin/env python3
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def count_by_class(detections):
    """Count detections by class."""
    class_counts = {}
    for det in detections:
        class_name = det.get('class_name', str(det.get('class_id', 'unknown')))
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts

def count_by_inference(detections):
    """Count original vs inferred detections."""
    original = 0
    inferred = 0
    for det in detections:
        if det.get('inferred', False):
            inferred += 1
        else:
            original += 1
    return {'original': original, 'inferred': inferred}

def analyze_enhancement_results(original_file, enhanced_file):
    """Analyze differences between original and enhanced detection files."""
    original_data = load_json(original_file)
    enhanced_data = load_json(enhanced_file)
    
    original_detections = original_data.get('detections', [])
    enhanced_detections = enhanced_data.get('detections', [])
    
    print(f"Original detections: {len(original_detections)}")
    print(f"Enhanced detections: {len(enhanced_detections)}")
    
    # Count by class
    original_class_counts = count_by_class(original_detections)
    enhanced_class_counts = count_by_class(enhanced_detections)
    
    print("\nDetections by class:")
    for class_name in sorted(set(list(original_class_counts.keys()) + list(enhanced_class_counts.keys()))):
        orig_count = original_class_counts.get(class_name, 0)
        enhanced_count = enhanced_class_counts.get(class_name, 0)
        diff = enhanced_count - orig_count
        diff_str = f"(+{diff})" if diff > 0 else f"({diff})" if diff < 0 else ""
        print(f"  {class_name}: {orig_count} → {enhanced_count} {diff_str}")
    
    # Count by inference
    inference_counts = count_by_inference(enhanced_detections)
    print(f"\nOriginal elements: {inference_counts['original']}")
    print(f"Inferred elements: {inference_counts['inferred']}")
    
    # Plot comparison
    plot_comparison(original_class_counts, enhanced_class_counts)
    
    return {
        'original_count': len(original_detections),
        'enhanced_count': len(enhanced_detections),
        'original_class_counts': original_class_counts,
        'enhanced_class_counts': enhanced_class_counts,
        'inference_counts': inference_counts
    }

def plot_comparison(original_counts, enhanced_counts):
    """Create a bar chart comparing original and enhanced detection counts by class."""
    all_classes = sorted(set(list(original_counts.keys()) + list(enhanced_counts.keys())))
    
    # Filter out classes that might have too many instances to visualize clearly
    # You can adjust or remove this if you want to see all classes
    # classes_to_plot = [c for c in all_classes if original_counts.get(c, 0) < 100 and enhanced_counts.get(c, 0) < 100]
    classes_to_plot = all_classes
    
    x = np.arange(len(classes_to_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    original_values = [original_counts.get(c, 0) for c in classes_to_plot]
    enhanced_values = [enhanced_counts.get(c, 0) for c in classes_to_plot]
    
    ax.bar(x - width/2, original_values, width, label='Original')
    ax.bar(x + width/2, enhanced_values, width, label='Enhanced')
    
    ax.set_xlabel('Element Class')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Original vs Enhanced Detection Counts')
    ax.set_xticks(x)
    ax.set_xticklabels(classes_to_plot, rotation=45, ha='right')
    ax.legend()
    
    fig.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Analyze OMR detection enhancement results')
    parser.add_argument('--original', required=True, help='Path to original detections JSON file')
    parser.add_argument('--enhanced', required=True, help='Path to enhanced detections JSON file')
    parser.add_argument('--output', help='Path to save analysis visualization')
    
    args = parser.parse_args()
    
    print(f"Analyzing enhancement results:")
    print(f"  Original: {args.original}")
    print(f"  Enhanced: {args.enhanced}")
    
    results = analyze_enhancement_results(args.original, args.enhanced)
    
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Analysis visualization saved to: {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()