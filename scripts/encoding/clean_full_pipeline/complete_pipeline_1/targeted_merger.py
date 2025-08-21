import numpy as np
import json
import logging

logger = logging.getLogger('omr_pipeline.merge')

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Box format: {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    
    Returns:
        float: IoU value between 0 and 1
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
    x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area

def elements_are_same_type(class_name1, class_name2):
    """
    Determine if two elements are of the same type for merging purposes
    
    Args:
        class_name1: First class name
        class_name2: Second class name
        
    Returns:
        bool: True if elements are of the same type
    """
    class_name1 = class_name1.lower()
    class_name2 = class_name2.lower()
    
    # Exact match
    if class_name1 == class_name2:
        return True
    
    # Check for barlines
    if 'barline' in class_name1 and 'barline' in class_name2:
        return True
    
    # Check for beams
    if class_name1 == 'beam' and class_name2 == 'beam':
        return True
    
    # Check for stems 
    if class_name1 == 'stem' and class_name2 == 'stem':
        return True
    
    # Check for whole rests
    if ('restwhole' in class_name1 or 'whole rest' in class_name1) and \
       ('restwhole' in class_name2 or 'whole rest' in class_name2):
        return True
    
    return False

def are_duplicates(box1, box2, class_name1, class_name2, image_width=1000, image_height=1000):
    """
    Check if two elements are duplicates using very strict criteria
    Specialized for beams, barlines, stems, and whole rests
    
    Returns:
        bool: True if elements are considered duplicates
    """
    # Calculate centers and distances
    center_x1 = (box1['x1'] + box1['x2']) / 2
    center_x2 = (box2['x1'] + box2['x2']) / 2
    center_y1 = (box1['y1'] + box1['y2']) / 2 
    center_y2 = (box2['y1'] + box2['y2']) / 2
    
    # Calculate horizontal and vertical distances (in normalized coordinates)
    x_distance = abs(center_x1 - center_x2)
    y_distance = abs(center_y1 - center_y2)
    
    # Calculate widths and heights
    width1 = box1['x2'] - box1['x1']
    width2 = box2['x2'] - box2['x1']
    height1 = box1['y2'] - box1['y1']
    height2 = box2['y2'] - box2['y1']
    
    # Calculate area ratio
    area1 = width1 * height1
    area2 = width2 * height2
    area_ratio = min(area1, area2) / max(area1, area2)
    
    # Get lowercase class names
    class_name1 = class_name1.lower()
    class_name2 = class_name2.lower()
    
    # Only check specific element types - extremely specific criteria for each type
    
    # Barlines
    if 'barline' in class_name1 and 'barline' in class_name2:
        # Barlines must be very close horizontally and overlap vertically
        x_threshold = 0.007  # 0.7% of image width
        y_threshold = 0.05   # 5% of image height (vertical position matters less)
        return x_distance < x_threshold and y_distance < y_threshold
    
    # Beams
    elif class_name1 == 'beam' and class_name2 == 'beam':
        # Beams must be very close in both position and size
        x_threshold = 0.01   # 1% of image width
        y_threshold = 0.005  # 0.5% of image height
        # Also check for similar size and some overlap
        iou = calculate_iou(box1, box2)
        return (x_distance < x_threshold and 
                y_distance < y_threshold and 
                area_ratio > 0.7 and  # Similar size 
                iou > 0.3)            # Some overlap
    
    # Stems
    elif class_name1 == 'stem' and class_name2 == 'stem':
        # Stems must be very close horizontally but can differ vertically
        x_threshold = 0.005  # 0.5% of image width
        y_threshold = 0.02   # 2% of image height
        return x_distance < x_threshold and y_distance < y_threshold
    
    # Whole rests
    elif ('restwhole' in class_name1 or 'whole rest' in class_name1) and \
         ('restwhole' in class_name2 or 'whole rest' in class_name2):
        # Whole rests must be very close in both dimensions
        x_threshold = 0.07  # 0.7% of image width
        y_threshold = 0.07  # 0.7% of image height
        
        # Don't require overlap, just proximity in position
        # Two whole rests anywhere in proximity should be considered duplicates
        return (x_distance < x_threshold and y_distance < y_threshold)
    
    # Not a type we're interested in merging
    return False

def merge_rests_properly(all_detections):
    """
    Apply music engraving rules to merge all types of rests properly
    Handles: rest, restHalf, restQuarter, restWhole, rest8th, rest16th, etc.
    """
    # Find all rests
    rests = []
    other_objects = []
    
    # First, separate rests from other objects
    for detection in all_detections:
        class_name = detection.get('class_name', '').lower()
        if 'rest' in class_name:  # This catches all rest types
            rests.append(detection)
        else:
            other_objects.append(detection)
    
    # Group rests by type first
    rest_types = {}
    for rest in rests:
        class_name = rest.get('class_name', '').lower()
        if class_name not in rest_types:
            rest_types[class_name] = []
        rest_types[class_name].append(rest)
    
    # Process each type of rest separately
    all_kept_rests = []
    
    for rest_type, type_rests in rest_types.items():
        # Group rests by staff line AND horizontal proximity
        rest_groups = []
        
        for rest in type_rests:
            y_pos = rest['bbox']['center_y']
            x_pos = rest['bbox']['center_x']
            
            # Check if this rest belongs to an existing group
            found_group = False
            for group in rest_groups:
                first_rest = group[0]
                first_y = first_rest['bbox']['center_y']
                first_x = first_rest['bbox']['center_x']
                
                # Only group if they're on same staff (close vertically) AND close horizontally
                # Vertical threshold: 5% of position
                # Horizontal threshold: 3% of image width
                y_threshold = 0.05 * max(y_pos, first_y)
                x_threshold = 0.03  # 3% of normalized width
                
                if abs(y_pos - first_y) < y_threshold and abs(x_pos - first_x) < x_threshold:
                    group.append(rest)
                    found_group = True
                    break
            
            # If no matching group, create a new one
            if not found_group:
                rest_groups.append([rest])
        
        # For each group, keep only the highest confidence rest
        type_kept_rests = []
        for group in rest_groups:
            if len(group) == 1:
                type_kept_rests.append(group[0])
            else:
                # Sort by confidence and keep the best one
                group.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                type_kept_rests.append(group[0])
                print(f"Merged {len(group)} {rest_type} rests at (x,y)≈({group[0]['bbox']['center_x']:.1f},{group[0]['bbox']['center_y']:.1f})")
        
        all_kept_rests.extend(type_kept_rests)
    
    # Combine with other objects
    return other_objects + all_kept_rests




def get_structure_elements(detections):
    """
    Extract structure elements (beams, barlines, stems, whole rests) from detections
    
    Args:
        detections: List of detection objects
        
    Returns:
        List of structural elements
    """
    structure_elements = []
    
    for detection in detections:
        class_name = detection.get('class_name', '').lower()
        
        # Keep only beams, barlines, stems, and whole rests
        if (class_name == 'beam' or 
            'barline' in class_name or 
            class_name == 'stem' or
            'restwhole' in class_name or 
            'whole rest' in class_name):
            structure_elements.append(detection)
            
    return structure_elements



def combine_detections_targeted(structure_detections, symbol_detections, high_conf_threshold=0.5, 
                               image_width=1000, image_height=1000):
    """
    Specialized function to combine detections with targeted merging for specific elements
    
    Args:
        structure_detections: List of structure model detections
        symbol_detections: List of symbol model detections
        high_conf_threshold: Confidence threshold for filtering
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Dictionary with combined detections
    """
    # Define our model specialization logic based on model performance
    rest_specialization = {
        "restWhole": "structure",    # Use structure model (FR-CNN) - 46.06% better
        "restHalf": "symbol",        # Use symbol model (YOLO) - 32.52% better
        "rest16th": "structure",     # Use structure model (FR-CNN) - 1.11% better
        "rest8th": "structure",      # Use structure model (FR-CNN) - 24.24% better
        "restQuarter":"symbol",  # Use structure model (FR-CNN) - 12.99% better
        "rest32nd": "either"         # Both models equally good
    }
    
    logger.info("Using specialized model preferences for different rest types")

    # Mark each detection with its source model
    for det in structure_detections:
        det["source_model"] = "structure"
    for det in symbol_detections:
        det["source_model"] = "symbol"
    
    # DEBUGGING: Log initial stem counts
    structure_stems = [d for d in structure_detections if d.get("class_name", "").lower() == "stem"]
    symbol_stems = [d for d in symbol_detections if d.get("class_name", "").lower() == "stem"]
    logger.info(f"Input stems: {len(structure_stems)} from structure model, {len(symbol_stems)} from symbol model")
    
    # 1. Extract all rests from both models 
    structure_rests = [d for d in structure_detections if "rest" in d.get("class_name", "").lower()]
    symbol_rests = [d for d in symbol_detections if "rest" in d.get("class_name", "").lower()]
    
    logger.info(f"Found {len(structure_rests)} rests in structure model and {len(symbol_rests)} rests in symbol model")
    
    # 2. Keep track of which rests we'll include in our final result
    kept_rests = []
    rest_positions = set()  # To avoid duplicates
    
    # 3. For each specialized rest type, prioritize the preferred model
    for rest_type, preferred_model in rest_specialization.items():
        if preferred_model == "structure":
            primary_rests = structure_rests
            secondary_rests = symbol_rests
        elif preferred_model == "symbol":
            primary_rests = symbol_rests
            secondary_rests = structure_rests
        else:  # "either" - prefer higher confidence
            # For "either", merge them all and sort by confidence
            combined_rests = structure_rests + symbol_rests
            combined_rests.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            primary_rests = combined_rests
            secondary_rests = []
        
        # Find rests of this type in the primary model
        type_rests_primary = [r for r in primary_rests 
                              if rest_type.lower() in r.get("class_name", "").lower()]
        
        # Keep high confidence rests from the primary model
        for rest in type_rests_primary:
            if rest.get("confidence", 0) >= high_conf_threshold:
                pos = (round(rest["bbox"]["center_x"], 1), round(rest["bbox"]["center_y"], 1))
                if pos not in rest_positions:
                    kept_rests.append(rest)
                    rest_positions.add(pos)
        
        # If the preferred model is "either", we've already merged and sorted them
        if preferred_model == "either":
            continue
        
        # For rests not yet handled, try the secondary model
        type_rests_secondary = [r for r in secondary_rests 
                               if rest_type.lower() in r.get("class_name", "").lower()]
        
        for rest in type_rests_secondary:
            if rest.get("confidence", 0) >= high_conf_threshold * 1.2:  # Higher threshold for secondary
                pos = (round(rest["bbox"]["center_x"], 1), round(rest["bbox"]["center_y"], 1))
                if pos not in rest_positions:
                    kept_rests.append(rest)
                    rest_positions.add(pos)
    
    # 4. Add any remaining high-confidence rests that haven't been handled yet
    all_remaining_rests = []
    for rest in structure_rests + symbol_rests:
        pos = (round(rest["bbox"]["center_x"], 1), round(rest["bbox"]["center_y"], 1))
        if pos not in rest_positions and rest.get("confidence", 0) >= high_conf_threshold:
            all_remaining_rests.append(rest)
            rest_positions.add(pos)
    
    # Sort remaining rests by confidence and add to our kept rests
    all_remaining_rests.sort(key=lambda r: r.get("confidence", 0), reverse=True)
    kept_rests.extend(all_remaining_rests)
    
    logger.info(f"After specialized processing, keeping {len(kept_rests)} rests")
    
    # 5. Now handle all non-rest elements using the standard targeted merger
    # Remove rests from both detection lists
    structure_non_rests = [d for d in structure_detections if "rest" not in d.get("class_name", "").lower()]
    symbol_non_rests = [d for d in symbol_detections if "rest" not in d.get("class_name", "").lower()]
    
    # CRITICAL CHANGE: Directly extract and keep all stems from both models without merging
    all_stems = []
    all_stems.extend(structure_stems)  # Keep all structure model stems
    
    # Only add symbol model stems if they don't overlap with structure stems
    for sym_stem in symbol_stems:
        # Check if this stem position is close to any structure stem
        sym_x = sym_stem["bbox"]["center_x"]
        sym_y = sym_stem["bbox"]["center_y"]
        
        has_overlap = False
        for struct_stem in structure_stems:
            struct_x = struct_stem["bbox"]["center_x"]
            struct_y = struct_stem["bbox"]["center_y"]
            
            # Use a very conservative overlap check (5 pixels horizontal, 10 pixels vertical)
            if abs(sym_x - struct_x) < 5 and abs(sym_y - struct_y) < 10:
                has_overlap = True
                break
        
        # If no overlap, add this stem
        if not has_overlap:
            all_stems.append(sym_stem)
    
    logger.info(f"Keeping all {len(all_stems)} stems without merging/filtering")
    
    # Remove stems from further processing
    structure_non_stem = [d for d in structure_non_rests if d.get("class_name", "").lower() != "stem"]
    symbol_non_stem = [d for d in symbol_non_rests if d.get("class_name", "").lower() != "stem"]
    
    # 6. Get remaining structural elements (beams, barlines) - follows your original logic
    structure_elements = [d for d in structure_non_stem if (
        d.get("class_name", "").lower() == "beam" or
        "barline" in d.get("class_name", "").lower()
    )]
    
    logger.info(f"Found {len(structure_elements)} other structural elements (beams, barlines)")
    
    # 7. Get high confidence structural elements from symbol model
    symbol_structure_elements = [d for d in symbol_non_stem if (
        d.get("class_name", "").lower() == "beam" or
        "barline" in d.get("class_name", "").lower()
    )]
    
    high_conf_symbol_structure = [d for d in symbol_structure_elements 
                                if d.get('confidence', 0) >= high_conf_threshold]
    
    logger.info(f"Found {len(high_conf_symbol_structure)} high-confidence structural elements from symbol model")
    
    # 8. Merge duplicate structural elements - follows your original logic
    merged_structure_elements = []
    to_skip = set()
    
    all_structure_elements = sorted(
        structure_elements + high_conf_symbol_structure,
        key=lambda x: x.get('confidence', 0),
        reverse=True
    )
    
    # Process in order of confidence
    for i, detection in enumerate(all_structure_elements):
        if i in to_skip:
            continue
            
        bbox_i = detection.get('bbox', {})
        class_name_i = detection.get('class_name', '')
        confidence_i = detection.get('confidence', 0)
        
        # Find duplicates
        duplicates = []
        for j, other in enumerate(all_structure_elements):
            if i == j or j in to_skip:
                continue
                
            bbox_j = other.get('bbox', {})
            class_name_j = other.get('class_name', '')
            confidence_j = other.get('confidence', 0)
            
            if are_duplicates(bbox_i, bbox_j, class_name_i, class_name_j, image_width, image_height):
                duplicates.append((j, other))
                to_skip.add(j)
                logger.info(f"Merging duplicates: {class_name_i} at ({bbox_i['center_x']:.1f}, {bbox_i['center_y']:.1f}) and {class_name_j}")
        
        merged_structure_elements.append(detection)
    
    logger.info(f"After merging, kept {len(merged_structure_elements)} structural elements")
    
    # 9. Get all remaining non-structural elements from the symbol model
    non_structure_elements = []
    for detection in symbol_non_stem:  # Use non_stem here instead of non_rests
        class_name = detection.get('class_name', '').lower()
        
        # Skip structural elements (already handled)
        if (class_name == 'beam' or 'barline' in class_name):
            continue
            
        non_structure_elements.append(detection)
    
    logger.info(f"Adding {len(non_structure_elements)} non-structural elements from symbol model")

    # 10. Combine all elements: kept rests + all_stems + merged structural elements + non-structural elements
    # CHANGE: Keep all stems, don't filter them
    combined_detections = {
        "detections": kept_rests + all_stems + merged_structure_elements + non_structure_elements
    }

    # 11. Final cleanup using existing merge_rests_properly for any duplicate rests we missed
    combined_detections["detections"] = merge_rests_properly(combined_detections["detections"])

    # Final stem count check
    final_stems = [d for d in combined_detections["detections"] if d.get("class_name", "").lower() == "stem"]
    logger.info(f"Final stem count: {len(final_stems)} (should match initial count of {len(structure_stems) + len(symbol_stems)})")
    logger.info(f"Final combined detections: {len(combined_detections['detections'])} elements")

    return combined_detections

# For direct testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Targeted merger for structural elements")
    parser.add_argument("--structure", required=True, help="Structure model detections file")
    parser.add_argument("--symbol", required=True, help="Symbol model detections file")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--width", type=int, default=1000, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=1000, help="Image height in pixels")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    # Load detections
    with open(args.structure, 'r') as f:
        structure_data = json.load(f)
    
    with open(args.symbol, 'r') as f:
        symbol_data = json.load(f)
    
    # Combine detections
    combined = combine_detections_targeted(
        structure_data.get('detections', []),
        symbol_data.get('detections', []),
        args.conf,
        args.width,
        args.height
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"Combined detections saved to {args.output}")