#!/usr/bin/env python3
"""
Script to create a mapping between DOREMI and MUSCIMA++ class names and
modify MUSCIMA++ XML annotations to use DOREMI class names.
"""

import os
import json
import re
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# Define mapping between MUSCIMA++ and DOREMI class names
# Format: "muscima_class_name": "doremi_class_name"
CLASS_NAME_MAPPING = {
    # Core notation elements
    "notehead-full": "noteheadBlack",
    "notehead-empty": "noteheadHalf",
    "stem": "stem",
    "beam": "beam",
    "staff_line": "kStaffLine",
    "measure_separator": "barline",
    "thin_barline": "barline",
    "thick_barline": "systemicBarline",
    "flat": "accidentalFlat",
    "sharp": "accidentalSharp",
    "natural": "accidentalNatural",
    "whole_rest": "restWhole",
    "half_rest": "restHalf",
    "quarter_rest": "restQuarter",
    "8th_rest": "rest8th",
    "16th_rest": "rest16th",
    "g-clef": "gClef",
    "f-clef": "fClef",
    "c-clef": "cClef",
    "slur": "slur", 
    "tie": "tie",
    "duration-dot": "augmentationDot",
    "curved-line_(tie-or-slur)": "slur",  # Default to slur, could be tie
    "ledger_line": "kStaffLine",  # Not exact but closest
    
    # Additional mappings from previously unmapped classes
    "accent": "articAccentAbove",  # Position would need checking
    "double_sharp": "accidentalDoubleSharp",
    "fermata": "fermata",
    "hairpin-cresc": "gradualDynamic",  # Both hairpins map to gradualDynamic
    "hairpin-decr": "gradualDynamic",
    "dynamics_text": "dynamicForte",  # Default, would need content checking
    "trill": "ornamentTrill",
    "trill_wobble": "wiggleTrill",
    "tenuto": "articTenutoAbove",  # Position would need checking
    "tuple": "tupletText",
    "tuple_bracketline": "tupletBracket",
    "time_signature": "timeSignatureComponent",
    "multi-measure_rest": "rest",  # General rest category
    "other-clef": "unpitchedPercussionClef1",  # Approximate mapping
    "other-dot": "augmentationDot",  # Default mapping, might not be accurate
    "grace-notehead-full": "noteheadBlack",  # Could be specialized in DOREMI
    "numeral_0": "timeSig0",  # Approximate mapping to time signature numbers
    "numeral_1": "timeSig1",
    "numeral_2": "timeSig2",
    "numeral_3": "timeSig3",
    "numeral_4": "timeSig4",
    "numeral_5": "timeSig5",
    "numeral_6": "timeSig6",
    "numeral_7": "timeSig7",
    "numeral_8": "timeSig8",
    "repeat-dot": "repeatDot",
    "whole-time_mark": "timeSigCommon",
    "staccato-dot": "articStaccatoAbove",  # Will need position checking
    
    # Special handling for flags that need direction
    "8th_flag": "flag8thUp",  # Default, will be determined by context
    "16th_flag": "flag16thUp",  # Default, will be determined by context
}

def get_bbox(obj_elem):
    """Extract bounding box coordinates from object element"""
    bbox_elem = obj_elem.find('bndbox')
    if bbox_elem is None:
        return None
    
    xmin = int(bbox_elem.find('xmin').text)
    ymin = int(bbox_elem.find('ymin').text)
    xmax = int(bbox_elem.find('xmax').text)
    ymax = int(bbox_elem.find('ymax').text)
    
    return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}

def determine_context(all_objects):
    """
    Analyze object relationships to determine contextual information
    like stem direction, position relative to staff, etc.
    """
    # Find staff lines first to establish reference positions
    staff_lines = [obj for obj in all_objects if obj['name'] in ('staff_line', 'kStaffLine')]
    
    # Group staff lines by their y-position to identify staff systems
    if staff_lines:
        avg_y = {}
        for obj in staff_lines:
            y_center = (obj['bbox']['ymin'] + obj['bbox']['ymax']) / 2
            for id in avg_y:
                # If this y is close to an existing staff line, group them
                if abs(y_center - avg_y[id]) < 50:  # Adjust threshold as needed
                    avg_y[id] = (avg_y[id] + y_center) / 2
                    obj['staff_id'] = id
                    break
            else:
                # New staff system
                new_id = len(avg_y) + 1
                avg_y[new_id] = y_center
                obj['staff_id'] = new_id
    
    # For each notehead, find the nearest stem and staff line
    for obj in all_objects:
        if obj['name'] in ('notehead-full', 'notehead-empty', 'noteheadBlack', 'noteheadHalf'):
            # Find connected stem
            stems = [s for s in all_objects if s['name'] == 'stem']
            if stems:
                # Estimate stem connection by proximity
                obj_center_x = (obj['bbox']['xmin'] + obj['bbox']['xmax']) / 2
                obj_center_y = (obj['bbox']['ymin'] + obj['bbox']['ymax']) / 2
                
                for stem in stems:
                    stem_x = (stem['bbox']['xmin'] + stem['bbox']['xmax']) / 2
                    stem_y = (stem['bbox']['ymin'] + stem['bbox']['ymax']) / 2
                    
                    # Check if stem overlaps or is very close to notehead
                    x_overlap = (stem['bbox']['xmin'] <= obj['bbox']['xmax'] and 
                                 stem['bbox']['xmax'] >= obj['bbox']['xmin'])
                    close_x = abs(stem_x - obj_center_x) < 20  # Threshold for x-distance
                    
                    if x_overlap or close_x:
                        # Determine stem direction
                        stem_height = stem['bbox']['ymax'] - stem['bbox']['ymin']
                        note_center_y = (obj['bbox']['ymin'] + obj['bbox']['ymax']) / 2
                        stem_center_y = (stem['bbox']['ymin'] + stem['bbox']['ymax']) / 2
                        
                        # If stem extends more above the notehead, it's stem-up
                        # Otherwise, it's stem-down
                        upper_part = note_center_y - stem['bbox']['ymin']
                        lower_part = stem['bbox']['ymax'] - note_center_y
                        
                        if upper_part > lower_part:
                            stem['direction'] = 'up'
                        else:
                            stem['direction'] = 'down'
                        
                        obj['connected_stem'] = stem
                        break
    
    # For each flag, find its stem and inherit direction
    for obj in all_objects:
        if obj['name'] in ('8th_flag', '16th_flag'):
            # Find nearest stem
            stems = [s for s in all_objects if s['name'] == 'stem' and 'direction' in s]
            if stems:
                flag_center_x = (obj['bbox']['xmin'] + obj['bbox']['xmax']) / 2
                flag_center_y = (obj['bbox']['ymin'] + obj['bbox']['ymax']) / 2
                
                nearest_stem = None
                min_distance = float('inf')
                
                for stem in stems:
                    stem_x = (stem['bbox']['xmin'] + stem['bbox']['xmax']) / 2
                    stem_y = (stem['bbox']['ymin'] + stem['bbox']['ymax']) / 2
                    
                    # Check if flag is near a stem endpoint
                    distance = ((flag_center_x - stem_x)**2 + (flag_center_y - stem_y)**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_stem = stem
                
                if nearest_stem and min_distance < 50:  # Threshold for proximity
                    obj['direction'] = nearest_stem['direction']
    
    # For articulations, determine if they're above or below based on nearest notehead
    for obj in all_objects:
        if obj['name'] in ('staccato-dot', 'accent', 'tenuto'):
            # Find nearest notehead
            noteheads = [n for n in all_objects if n['name'] in 
                         ('notehead-full', 'notehead-empty', 'noteheadBlack', 'noteheadHalf')]
            
            if noteheads:
                art_center_x = (obj['bbox']['xmin'] + obj['bbox']['xmax']) / 2
                art_center_y = (obj['bbox']['ymin'] + obj['bbox']['ymax']) / 2
                
                nearest_note = None
                min_distance = float('inf')
                
                for note in noteheads:
                    note_x = (note['bbox']['xmin'] + note['bbox']['xmax']) / 2
                    note_y = (note['bbox']['ymin'] + note['bbox']['ymax']) / 2
                    
                    distance = ((art_center_x - note_x)**2 + (art_center_y - note_y)**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_note = note
                
                if nearest_note and min_distance < 100:  # Threshold for proximity
                    note_center_y = (nearest_note['bbox']['ymin'] + nearest_note['bbox']['ymax']) / 2
                    
                    # If articulation is above notehead
                    if art_center_y < note_center_y:
                        obj['position'] = 'above'
                    else:
                        obj['position'] = 'below'
    
    return all_objects

def map_muscima_to_doremi(muscima_class, context):
    """
    Map MUSCIMA++ class to DOREMI class with context awareness
    context: dict with additional info like position, direction, etc.
    """
    if muscima_class in CLASS_NAME_MAPPING:
        doremi_class = CLASS_NAME_MAPPING[muscima_class]
        
        # Special handling for flags based on direction
        if muscima_class == "8th_flag" and 'direction' in context:
            if context['direction'] == 'up':
                return "flag8thUp"
            elif context['direction'] == 'down':
                return "flag8thDown"
            
        # Special handling for 16th flags based on direction
        if muscima_class == "16th_flag" and 'direction' in context:
            if context['direction'] == 'up':
                return "flag16thUp"
            elif context['direction'] == 'down':
                return "flag16thDown"
        
        # Special handling for articulations based on position
        if muscima_class == "staccato-dot" and 'position' in context:
            if context['position'] == 'above':
                return "articStaccatoAbove"
            elif context['position'] == 'below':
                return "articStaccatoBelow"
        
        if muscima_class == "accent" and 'position' in context:
            if context['position'] == 'above':
                return "articAccentAbove"
            elif context['position'] == 'below':
                return "articAccentBelow"
            
        if muscima_class == "tenuto" and 'position' in context:
            if context['position'] == 'above':
                return "articTenutoAbove"
            elif context['position'] == 'below':
                return "articTenutoBelow"
        
        return doremi_class
    
    return None  # No mapping found

def load_doremi_mapping(json_file):
    """Load DOREMI class mapping {name: id}"""
    with open(json_file, 'r') as f:
        return json.load(f)

def process_xml_annotations(xml_dir, output_dir, doremi_mapping, use_context=False, dry_run=False):
    """
    Process XML annotations to convert MUSCIMA++ class names to DOREMI class names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all XML files
    xml_files = list(Path(xml_dir).glob("**/*.xml"))
    print(f"Found {len(xml_files)} XML files to process")
    
    # Track statistics
    stats = {
        "processed_files": 0,
        "modified_objects": 0,
        "skipped_objects": 0,
        "unmapped_classes": set()
    }
    
    for xml_path in tqdm(xml_files):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            file_modified = False
            
            if use_context:
                # First pass: extract all objects with their bounding boxes
                all_objects = []
                for obj_idx, obj in enumerate(root.findall('./object')):
                    name_elem = obj.find('name')
                    if name_elem is None:
                        continue
                    
                    bbox = get_bbox(obj)
                    if bbox is None:
                        continue
                    
                    all_objects.append({
                        'idx': obj_idx,
                        'name': name_elem.text,
                        'bbox': bbox
                    })
                
                # Determine context for all objects
                all_objects = determine_context(all_objects)
                
                # Create lookup by index
                obj_by_idx = {obj['idx']: obj for obj in all_objects}
                
                # Second pass: update class names with context
                for obj_idx, obj in enumerate(root.findall('./object')):
                    name_elem = obj.find('name')
                    if name_elem is None:
                        continue
                    
                    muscima_class = name_elem.text
                    
                    # Get context if available
                    context = obj_by_idx.get(obj_idx, {})
                    
                    # Map to DOREMI class with context
                    doremi_class = map_muscima_to_doremi(muscima_class, context)
                    
                    if doremi_class:
                        name_elem.text = doremi_class
                        file_modified = True
                        stats["modified_objects"] += 1
                    else:
                        stats["skipped_objects"] += 1
                        stats["unmapped_classes"].add(muscima_class)
            else:
                # Simple mapping without context
                for obj in root.findall('./object'):
                    name_elem = obj.find('name')
                    if name_elem is None:
                        continue
                    
                    muscima_class = name_elem.text
                    
                    # Apply mapping if exists
                    if muscima_class in CLASS_NAME_MAPPING:
                        doremi_class = CLASS_NAME_MAPPING[muscima_class]
                        
                        # Update the class name to DOREMI format
                        if doremi_class:
                            name_elem.text = doremi_class
                            file_modified = True
                            stats["modified_objects"] += 1
                        else:
                            stats["skipped_objects"] += 1
                    else:
                        stats["unmapped_classes"].add(muscima_class)
                        stats["skipped_objects"] += 1
            
            # Save the modified XML file if changes were made
            if file_modified and not dry_run:
                output_path = os.path.join(output_dir, os.path.basename(xml_path))
                tree.write(output_path)
                stats["processed_files"] += 1
        
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Processed files: {stats['processed_files']}")
    print(f"  Modified objects: {stats['modified_objects']}")
    print(f"  Skipped objects: {stats['skipped_objects']}")
    print(f"\nUnmapped classes ({len(stats['unmapped_classes'])}):")
    for cls in sorted(stats['unmapped_classes']):
        print(f"  - {cls}")
    
    return stats

def create_evaluation_mapping(doremi_mapping, output_file):
    """
    Create a mapping file for evaluation that maps DOREMI class IDs
    to the corresponding MUSCIMA++ class names
    """
    # Create inverse of doremi_mapping {id: name}
    doremi_id_to_name = {v: k for k, v in doremi_mapping.items()}
    
    # Create inverse of CLASS_NAME_MAPPING {doremi_name: muscima_name}
    inverse_mapping = {v: k for k, v in CLASS_NAME_MAPPING.items() if v}
    
    # Create the evaluation mapping {id: muscima_name}
    eval_mapping = {}
    for doremi_id, doremi_name in doremi_id_to_name.items():
        if doremi_name in inverse_mapping:
            muscima_name = inverse_mapping[doremi_name]
            eval_mapping[doremi_id] = muscima_name
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(eval_mapping, f, indent=2)
    
    print(f"Created evaluation mapping with {len(eval_mapping)} classes at {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert MUSCIMA++ annotations to DOREMI class names')
    parser.add_argument('--doremi-mapping', required=True, help='Path to DOREMI class mapping JSON file')
    parser.add_argument('--xml-dir', required=True, help='Directory containing MUSCIMA++ XML annotations')
    parser.add_argument('--output-dir', required=True, help='Directory to save modified XML files')
    parser.add_argument('--create-eval-mapping', help='Create evaluation mapping file')
    parser.add_argument('--use-context', action='store_true', help='Use context-aware class mapping')
    parser.add_argument('--dry-run', action='store_true', help='Only print statistics, do not save files')
    
    args = parser.parse_args()
    
    # Load DOREMI mapping
    doremi_mapping = load_doremi_mapping(args.doremi_mapping)
    print(f"Loaded DOREMI mapping with {len(doremi_mapping)} classes")
    
    # Process XML annotations
    stats = process_xml_annotations(
        args.xml_dir, 
        args.output_dir, 
        doremi_mapping, 
        use_context=args.use_context,
        dry_run=args.dry_run
    )
    
    # Optionally create evaluation mapping
    if args.create_eval_mapping:
        create_evaluation_mapping(doremi_mapping, args.create_eval_mapping)

if __name__ == "__main__":
    main()
    
    
# Advanced usage with context-aware mapping (recommended)
# python /homes/es314/omr-objdet-benchmark/scripts/yolo8/convert_mapping_muscmia.py \
#   --doremi-mapping /homes/es314/omr-objdet-benchmark/data/class_mapping.json \
#   --xml-dir /import/c4dm-05/elona/MusicObjectDetector-TF/MusicObjectDetector/data/Stavewise_Annotations \
#   --output-dir /import/c4dm-05/elona/muscima-doremi-annotation \
#   --use-context \
#   --create-eval-mapping /import/c4dm-05/elona/muscima-doremi-annotation/eval_mapping.json

