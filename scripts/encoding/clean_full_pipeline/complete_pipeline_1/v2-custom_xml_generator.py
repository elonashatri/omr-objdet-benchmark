#!/usr/bin/env python3
"""
Complete Custom XML Generator for OMR Pipeline

This module provides comprehensive functionality to generate a custom XML output format
that matches the structure in the provided ground truth format. It includes:
- All musical element types (staffs, notes, beams, stems, barlines, etc.)
- Proper element linking (note-stem, stem-beam, etc.)
- Rich semantic data (pitch, duration, onset, etc.)
- Accurate positioning and properties

Usage:
    from complete_custom_xml_generator import generate_custom_xml
    
    # Generate XML from OMRProcessor object
    generate_custom_xml(processor, 'output.xml')
"""

import xml.dom.minidom as minidom
from xml.etree.ElementTree import Element, SubElement, tostring
import os
import json
from pathlib import Path
import logging
from collections import defaultdict
import hashlib
import re

# Configure logging
logger = logging.getLogger('complete_custom_xml_generator')

# Comprehensive class mapping for all musical elements
CLASS_MAPPING = {
    # Staff elements
    "staffLine": "kStaffLine",
    
    # Clefs
    "gClef": "gClef",
    "fClef": "fClef",
    "cClef": "cClef",
    "gClef8va": "gClef8va",
    "gClef8vb": "gClef8vb",
    "fClef8vb": "fClef8vb",
    "cClefTenor": "cClefTenor",
    "cClefAlto": "cClefAlto",
    "cClefSoprano": "cClefSoprano",
    "percussionClef": "percussionClef",
    
    # Accidentals
    "accidentalSharp": "accidentalSharp",
    "accidentalFlat": "accidentalFlat",
    "accidentalNatural": "accidentalNatural",
    "accidentalDoubleSharp": "accidentalDoubleSharp",
    "accidentalDoubleFlat": "accidentalDoubleFlat",
    
    # Time signatures
    "timeSignature": "timeSig",
    "timeSignatureCommon": "timeSigCommon",
    "timeSignatureCutCommon": "timeSigCutCommon",
    
    # Note heads
    "noteheadBlack": "noteheadBlack",
    "noteheadHalf": "noteheadHalf",
    "noteheadWhole": "noteheadWhole",
    
    # Rests
    "restWhole": "restWhole",
    "restHalf": "restHalf",
    "restQuarter": "restQuarter",
    "rest8th": "rest8th",
    "rest16th": "rest16th",
    "rest32nd": "rest32nd",
    
    # Stems, beams, and flags
    "stem": "stem",
    "beam": "beam",
    "flag8thUp": "flag8thUp",
    "flag8thDown": "flag8thDown",
    "flag16thUp": "flag16thUp",
    "flag16thDown": "flag16thDown",
    
    # Other notations
    "barline": "barline",
    "systemicBarline": "systemicBarline",
    "dot": "augmentationDot", 
    "slur": "slur",
    "tie": "tie",
    
    # Articulations
    "staccato": "articStaccato",
    "accent": "articAccent",
    "tenuto": "articTenuto",
    "marcato": "articMarcato",
    
    # Dynamics
    "dynamicForte": "dynamicForte",
    "dynamicPiano": "dynamicPiano",
    "dynamicMF": "dynamicMF",
    "dynamicMP": "dynamicMP",
    "dynamicFF": "dynamicFF",
    "dynamicPP": "dynamicPP",
    "crescendo": "crescendo",
    "diminuendo": "diminuendo",
    
    # Ornaments
    "trill": "trill",
    "mordent": "mordent",
    "turn": "turn",
    
    # Tuplets
    "tuplet": "tuplet",
    
    # Fallback for unknown classes
    "unknown": "kUnknown"
}

# Clef hotspot mapping (which note the clef marks)
CLEF_HOTSPOT_MAP = {
    "G": "G4",
    "F": "F3",
    "C": "C4",
    "G8va": "G5",
    "G8vb": "G3",
    "F8vb": "F2",
    "cClefTenor": "C4",
    "cClefAlto": "C4",
    "cClefSoprano": "C4",
}

# Key signature descriptions
KEY_SIGNATURE_DESCRIPTIONS = {
    "sharp": {
        1: "G Major",
        2: "D Major",
        3: "A Major",
        4: "E Major",
        5: "B Major",
        6: "F# Major",
        7: "C# Major"
    },
    "flat": {
        1: "F Major",
        2: "Bb Major",
        3: "Eb Major",
        4: "Ab Major",
        5: "Db Major",
        6: "Gb Major",
        7: "Cb Major"
    }
}

def load_raw_detections(processor):
    """Load raw detections from processor or detection file"""
    raw_detections = []
    
    # First try to get from processor attributes
    if hasattr(processor, 'detections') and processor.detections:
        if isinstance(processor.detections, list):
            raw_detections = processor.detections
        elif isinstance(processor.detections, dict) and 'detections' in processor.detections:
            raw_detections = processor.detections['detections']
    
    # If no detections found or empty, try to load from file
    if not raw_detections and hasattr(processor, 'detection_path'):
        try:
            path = processor.detection_path
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    raw_detections = data
                elif isinstance(data, dict) and 'detections' in data:
                    raw_detections = data['detections']
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load detections from file: {e}")
    
    # If we have merged_path attribute, try that
    if not raw_detections and hasattr(processor, 'merged_path'):
        try:
            with open(processor.merged_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    raw_detections = data
                elif isinstance(data, dict) and 'detections' in data:
                    raw_detections = data['detections']
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load merged detections: {e}")
    
    # Last resort - if processor has a json_path attribute, try that
    if not raw_detections and hasattr(processor, 'json_path'):
        try:
            with open(processor.json_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    raw_detections = data
                elif isinstance(data, dict) and 'detections' in data:
                    raw_detections = data['detections']
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load detections from json_path: {e}")
    
    return raw_detections

def extract_id_mapping(raw_detections):
    """Extract ID mappings from raw detections"""
    # Maps to track relationships between elements
    element_ids = {}  # Maps element_id -> original detection
    note_stem_map = {}  # Maps notehead_id -> stem_id
    stem_beam_map = {}  # Maps stem_id -> beam_id
    beam_stems_map = {}  # Maps beam_id -> list of stem_ids
    
    # Process all detections to build ID mappings
    for detection in raw_detections:
        # Skip if no class_name
        if 'class_name' not in detection:
            continue
            
        # Get element ID (either from 'id' field or generate from class and position)
        element_id = detection.get('id')
        if not element_id:
            class_name = detection['class_name']
            x = detection['bbox']['center_x'] if 'bbox' in detection else 0
            y = detection['bbox']['center_y'] if 'bbox' in detection else 0
            element_id = f"{class_name}_{x:.1f}_{y:.1f}"
        
        # Store element by ID
        element_ids[element_id] = detection
        
        # Track note-stem relationships
        if 'notehead' in detection.get('class_name', '').lower():
            # Check if this note has a stem reference
            if 'stem_id' in detection:
                note_stem_map[element_id] = detection['stem_id']
        
        # Track stem-beam relationships
        if detection.get('class_name', '').lower() == 'stem':
            # Check if this stem has a beam reference
            if 'beam_id' in detection:
                stem_beam_map[element_id] = detection['beam_id']
                
                # Also track in the reverse map
                if detection['beam_id'] not in beam_stems_map:
                    beam_stems_map[detection['beam_id']] = []
                beam_stems_map[detection['beam_id']].append(element_id)
            
            # Check for note reference
            if 'notehead_id' in detection:
                # Add to note-stem map from stem perspective
                note_stem_map[detection['notehead_id']] = element_id
    
    # Return all mappings
    return {
        'element_ids': element_ids,
        'note_stem_map': note_stem_map,
        'stem_beam_map': stem_beam_map,
        'beam_stems_map': beam_stems_map
    }

def generate_unique_id(prefix, idx):
    """Generate a unique ID for an element"""
    return f"{prefix}{idx}"

def map_class_name(class_name):
    """Map internal class names to XML class names"""
    if not class_name:
        return "kUnknown"
    
    # Convert to lowercase for case-insensitive matching
    lower_name = class_name.lower()
    
    # Try exact match first
    if class_name in CLASS_MAPPING:
        return CLASS_MAPPING[class_name]
    
    # Try partial match
    for key, value in CLASS_MAPPING.items():
        if key.lower() in lower_name:
            return value
    
    # Return original if no match
    return class_name

def create_staff_line_nodes(root, processor, raw_detections):
    """Create staff line nodes in XML"""
    nodes = root.find("Page/Nodes")
    node_id = 100  # Start with high IDs for staff lines
    
    # Track spacing run IDs for each system
    spacing_run_ids = {}
    
    # Process each staff system
    for system_idx, system in enumerate(processor.staff_systems):
        staff_id = system_idx
        
        # Generate a unique spacing run ID for this system
        spacing_run_id = str(hash(f"system_{system_idx}") % 1000000000000000000)
        spacing_run_ids[system_idx] = spacing_run_id
        
        # Get staff line positions
        staff_lines = []
        if hasattr(system, 'lines') and system.lines:
            staff_lines = sorted(system.lines.values())  # Get y positions
        
        if not staff_lines and hasattr(system, 'staves') and system.staves:
            # Try to get from staves structure
            for staff in system.staves:
                if staff:  # staff might be a list of y positions
                    staff_lines.extend(staff)
            staff_lines = sorted(staff_lines)
        
        # If we still don't have staff lines, try from raw detections
        if not staff_lines:
            # Extract staff lines from raw detections
            detected_staff_lines = [d for d in raw_detections 
                                  if d.get('class_name', '').lower() == 'staffline' or 
                                     d.get('class_name', '').lower() == 'kstaffline']
            
            if detected_staff_lines:
                # Group by y position (with some tolerance)
                positions = {}
                for line in detected_staff_lines:
                    y = line['bbox']['center_y'] if 'bbox' in line else line.get('y', 0)
                    found = False
                    for key in positions:
                        if abs(key - y) < 5:  # 5 pixel tolerance
                            positions[key].append(line)
                            found = True
                            break
                    if not found:
                        positions[y] = [line]
                
                # Get median y for each group
                staff_lines = [sum(line['bbox']['center_y'] if 'bbox' in line else line.get('y', 0) 
                               for line in group) / len(group) 
                               for group in positions.values()]
                staff_lines = sorted(staff_lines)[:5]  # Take top 5 lines
        
        # If we still don't have staff lines, skip this system
        if not staff_lines:
            logger.warning(f"No staff lines found for system {system_idx}")
            continue
        
        # Take the first 5 lines (standard staff)
        staff_lines = staff_lines[:5]
        
        # Create staff line nodes
        for line_idx, line_y in enumerate(staff_lines):
            node = SubElement(nodes, "Node")
            
            # Add ID element (reverse order like in ground truth)
            id_elem = SubElement(node, "Id")
            id_elem.text = str(node_id - line_idx)  # Reverse order
            
            class_elem = SubElement(node, "ClassName")
            class_elem.text = "kStaffLine"
            
            top_elem = SubElement(node, "Top")
            top_elem.text = str(int(line_y))
            
            left_elem = SubElement(node, "Left")
            left_elem.text = "550"  # Standard from ground truth
            
            width_elem = SubElement(node, "Width")
            width_elem.text = "1750"  # Standard from ground truth
            
            height_elem = SubElement(node, "Height")
            height_elem.text = "3"  # Standard from ground truth
            
            # Add mask (simplified)
            mask_elem = SubElement(node, "Mask")
            mask_elem.text = "0: 0 1: 5250 "  # Standard mask from ground truth
            
            # Add data section with staff information
            data_elem = SubElement(node, "Data")
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "staff_id")
            data_item.set("type", "int")
            data_item.text = str(staff_id)
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "spacing_run_id")
            data_item.set("type", "int")
            data_item.text = spacing_run_id
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "ordered_staff_id")
            data_item.set("type", "int")
            data_item.text = str(staff_id)
            
            # Add order ID
            order_elem = SubElement(node, "order_id")
            order_elem.text = str(line_idx + 1)  # Order starts from 1
            
            node_id -= 1
    
    return spacing_run_ids

def add_clef_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add clef nodes to XML"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = id_mappings.get('element_ids', {}).values()
    
    # Get all clefs from raw detections
    clef_detections = [d for d in raw_detections 
                     if 'clef' in d.get('class_name', '').lower()]
    
    # If no clefs in raw detections, try from processor
    if not clef_detections:
        # Extract clefs from all systems
        for system in processor.staff_systems:
            clefs = [e for e in system.elements if hasattr(e, 'class_name') and 'clef' in e.class_name.lower()]
            clef_detections.extend(clefs)
    
    # Process each clef
    for clef_idx, clef in enumerate(clef_detections):
        # Determine which staff this clef belongs to
        staff_id = 0
        if hasattr(clef, 'staff_system') and clef.staff_system:
            staff_id = processor.staff_systems.index(clef.staff_system)
        elif 'staff_system' in clef:
            staff_id = clef['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Map clef type to class name
        class_elem = SubElement(node, "ClassName")
        
        # Determine clef type from class name and attributes
        clef_type = "gClef"  # Default to G clef
        class_name = clef.get('class_name', '') if isinstance(clef, dict) else getattr(clef, 'class_name', '')
        
        if isinstance(clef, dict) and 'type' in clef:
            clef_type = clef['type']
        elif hasattr(clef, 'type'):
            clef_type = clef.type
        
        # Check for special clefs (8va, 8vb)
        if "8vb" in class_name.lower():
            clef_type += "8vb"
        elif "8va" in class_name.lower():
            clef_type += "8va"
        
        # Map to class name
        mapped_class = map_class_name(clef_type)
        class_elem.text = mapped_class
        
        # Extract bbox information
        bbox = {}
        if isinstance(clef, dict) and 'bbox' in clef:
            bbox = clef['bbox']
        elif hasattr(clef, 'bbox'):
            bbox = clef.bbox
        else:
            # Create bbox from position and dimensions
            x = getattr(clef, 'x', 0)
            y = getattr(clef, 'y', 0)
            width = getattr(clef, 'width', 50)
            height = getattr(clef, 'height', 150)
            bbox = {
                'x1': x - width/2,
                'y1': y - height/2,
                'x2': x + width/2,
                'y2': y + height/2,
                'width': width,
                'height': height,
                'center_x': x,
                'center_y': y
            }
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 50)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 150)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add event ID
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "dorico_event_id")
        data_item.set("type", "int")
        data_item.text = str(11000 + current_id)  # Generate unique ID
        
        # Add clef type
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "clef_type")
        data_item.set("type", "str")
        data_item.text = f"k{clef_type.replace('8va', '').replace('8vb', '')}Clef"
        
        # Add clef hotspot
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "clef_hotspot")
        data_item.set("type", "str")
        data_item.text = CLEF_HOTSPOT_MAP.get(clef_type, "G4")  # Default to G4
        
        # Add required stave lines
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "clef_required_stave_lines")
        data_item.set("type", "int")
        data_item.text = "5"
        
        # Add clef stave position
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "clef_stave_position")
        data_item.set("type", "int")
        data_item.text = "2"
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def add_key_signature_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add key signature nodes to XML"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = id_mappings.get('element_ids', {}).values()
    
    # Get key signature accidentals from raw detections
    key_sig_detections = [d for d in raw_detections 
                        if (d.get('class_name', '').lower().startswith('accidental') and 
                            d.get('is_key_signature', False))]
    
    # If no key signatures in raw detections, try from processor
    if not key_sig_detections:
        # Extract key signatures from all systems
        for system in processor.staff_systems:
            key_accs = []
            if hasattr(system, 'key_signature') and system.key_signature:
                key_accs.extend(system.key_signature)
            else:
                # Try to find accidentals marked as key signatures
                key_accs.extend([e for e in system.elements 
                               if hasattr(e, 'is_key_signature') and e.is_key_signature])
            key_sig_detections.extend(key_accs)
    
    # Group key signatures by staff/system
    key_sigs_by_staff = defaultdict(list)
    
    for acc in key_sig_detections:
        # Determine which staff this accidental belongs to
        staff_id = 0
        if hasattr(acc, 'staff_system') and acc.staff_system:
            staff_id = processor.staff_systems.index(acc.staff_system)
        elif 'staff_system' in acc:
            staff_id = acc['staff_system']
        
        key_sigs_by_staff[staff_id].append(acc)
    
    # Process key signatures for each staff
    for staff_id, accidentals in key_sigs_by_staff.items():
        # Sort accidentals by x position
        if all(isinstance(acc, dict) and 'bbox' in acc for acc in accidentals):
            accidentals.sort(key=lambda a: a['bbox']['center_x'])
        elif all(hasattr(acc, 'x') for acc in accidentals):
            accidentals.sort(key=lambda a: a.x)
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        # Process each accidental
        for acc_idx, acc in enumerate(accidentals):
            node = SubElement(nodes, "Node")
            
            # Add ID
            id_elem = SubElement(node, "Id")
            id_elem.text = str(current_id)
            current_id += 1
            
            # Map accidental type to class name
            class_elem = SubElement(node, "ClassName")
            
            # Get accidental type
            acc_type = "sharp"  # Default
            if isinstance(acc, dict) and 'type' in acc:
                acc_type = acc['type']
            elif hasattr(acc, 'type'):
                acc_type = acc.type
            
            # Determine class name
            if acc_type.lower() == "sharp":
                class_elem.text = "accidentalSharp"
            elif acc_type.lower() == "flat":
                class_elem.text = "accidentalFlat"
            elif acc_type.lower() == "natural":
                class_elem.text = "accidentalNatural"
            else:
                class_elem.text = f"accidental{acc_type.capitalize()}"
            
            # Extract bbox information
            bbox = {}
            if isinstance(acc, dict) and 'bbox' in acc:
                bbox = acc['bbox']
            elif hasattr(acc, 'bbox'):
                bbox = acc.bbox
            else:
                # Create bbox from position and dimensions
                x = getattr(acc, 'x', 0)
                y = getattr(acc, 'y', 0)
                width = getattr(acc, 'width', 20)
                height = getattr(acc, 'height', 60)
                bbox = {
                    'x1': x - width/2,
                    'y1': y - height/2,
                    'x2': x + width/2,
                    'y2': y + height/2,
                    'width': width,
                    'height': height,
                    'center_x': x,
                    'center_y': y
                }
            
            # Position elements
            top_elem = SubElement(node, "Top")
            top_elem.text = str(int(bbox.get('y1', 0)))
            
            left_elem = SubElement(node, "Left")
            left_elem.text = str(int(bbox.get('x1', 0)))
            
            width_elem = SubElement(node, "Width")
            width_elem.text = str(int(bbox.get('width', 20)))
            
            height_elem = SubElement(node, "Height")
            height_elem.text = str(int(bbox.get('height', 60)))
            
            # Add data section
            data_elem = SubElement(node, "Data")
            
            # Add event ID
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "dorico_event_id")
            data_item.set("type", "int")
            data_item.text = str(11500 + current_id)  # Generate unique ID
            
            # Determine key signature description based on count
            key_count = len(accidentals)
            key_desc = "C Major"  # Default
            
            if acc_type.lower() == "sharp":
                key_desc = KEY_SIGNATURE_DESCRIPTIONS["sharp"].get(key_count, "C Major")
            elif acc_type.lower() == "flat":
                key_desc = KEY_SIGNATURE_DESCRIPTIONS["flat"].get(key_count, "C Major")
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "key_signature_description")
            data_item.set("type", "str")
            data_item.text = key_desc
            
            # Add staff data
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "staff_id")
            data_item.set("type", "int")
            data_item.text = str(staff_id)
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "spacing_run_id")
            data_item.set("type", "int")
            data_item.text = spacing_run_id
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "ordered_staff_id")
            data_item.set("type", "int")
            data_item.text = str(staff_id)
            
            # Add order ID
            order_elem = SubElement(node, "order_id")
            order_elem.text = str(order_id)
            order_id += 1
    
    return current_id, order_id

def add_time_signature_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add time signature nodes to XML"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = id_mappings.get('element_ids', {}).values()
    
    # Get time signatures from raw detections
    time_sig_detections = [d for d in raw_detections 
                        if 'timesig' in d.get('class_name', '').lower()]
    
    # If no time signatures in raw detections, try from processor
    if not time_sig_detections:
        # Extract time signatures from all systems
        for system_idx, system in enumerate(processor.staff_systems):
            if hasattr(system, 'time_signature') and system.time_signature:
                # Create a dictionary representation of the time signature
                ts = system.time_signature
                time_sig_detections.append({
                    'class_name': 'timeSignature',
                    'beats': ts.beats if hasattr(ts, 'beats') else 4,
                    'beat_type': ts.beat_type if hasattr(ts, 'beat_type') else 4,
                    'staff_system': system_idx,
                    'bbox': {
                        'x1': 700,  # Default position
                        'y1': 688 - 21,
                        'width': 35,
                        'height': 42,
                        'center_x': 718,
                        'center_y': 688
                    }
                })
    
    # Process each time signature
    for time_sig_idx, time_sig in enumerate(time_sig_detections):
        # Determine which staff this time signature belongs to
        staff_id = 0
        if hasattr(time_sig, 'staff_system') and time_sig.staff_system:
            staff_id = processor.staff_systems.index(time_sig.staff_system)
        elif 'staff_system' in time_sig:
            staff_id = time_sig['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Determine time signature class name
        class_elem = SubElement(node, "ClassName")
        
        # Get time signature properties
        beats = 4  # Default
        beat_type = 4  # Default
        
        if isinstance(time_sig, dict):
            beats = time_sig.get('beats', 4)
            beat_type = time_sig.get('beat_type', 4)
        elif hasattr(time_sig, 'beats') and hasattr(time_sig, 'beat_type'):
            beats = time_sig.beats
            beat_type = time_sig.beat_type
        
        # Determine class name
        if beats == 4 and beat_type == 4:
            class_elem.text = "timeSigCommon"
        elif beats == 2 and beat_type == 2:
            class_elem.text = "timeSigCutCommon"
        else:
            class_elem.text = "timeSig"
        
        # Extract bbox information
        bbox = {}
        if isinstance(time_sig, dict) and 'bbox' in time_sig:
            bbox = time_sig['bbox']
        elif hasattr(time_sig, 'bbox'):
            bbox = time_sig.bbox
        else:
            # Create bbox from position and dimensions
            x = getattr(time_sig, 'x', 700)
            y = getattr(time_sig, 'y', 688)
            bbox = {
                'x1': x - 17,
                'y1': y - 21,
                'x2': x + 18,
                'y2': y + 21,
                'width': 35,
                'height': 42,
                'center_x': x,
                'center_y': y
            }
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 35)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 42)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add event ID
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "dorico_event_id")
        data_item.set("type", "int")
        data_item.text = str(12000 + current_id)  # Generate unique ID
        
        # Build time signature description
        time_desc = f"{beats}/{beat_type} (q, "
        time_desc += "+".join(["1"] * beats) + ")"
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "time_signature_description")
        data_item.set("type", "str")
        data_item.text = time_desc
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def calculate_note_onset_map(processor):
    """Calculate onset times for notes based on position rather than object references"""
    position_onset_map = {}  # Maps (x, y) -> onset time
    
    # Process each staff system
    for system_idx, system in enumerate(processor.staff_systems):
        cumulative_onset = 0.0  # Initialize at the start of each system
        
        # Process measures if available
        if hasattr(system, 'measures') and system.measures:
            for measure_idx, measure in enumerate(system.measures):
                # Get notes and rests in this measure, sorted by x-position
                elements = [e for e in measure.elements if hasattr(e, 'duration')]
                elements.sort(key=lambda e: e.x)
                
                # Process elements
                for elem in elements:
                    # Get element position
                    if hasattr(elem, 'x') and hasattr(elem, 'y'):
                        x, y = elem.x, elem.y
                    elif isinstance(elem, dict) and 'bbox' in elem:
                        x = elem['bbox'].get('center_x', 0)
                        y = elem['bbox'].get('center_y', 0)
                    else:
                        continue  # Skip if no position
                    
                    # Store position with onset time - round to avoid floating point issues
                    position_onset_map[(round(x, 1), round(y, 1))] = cumulative_onset
                    
                    # Add duration to onset
                    if hasattr(elem, 'duration') and elem.duration is not None:
                        cumulative_onset += elem.duration
                    else:
                        # Default to quarter note duration
                        cumulative_onset += 1.0
        else:
            # If no measures, sort all notes by x-position
            elements = [e for e in system.elements if hasattr(e, 'duration')]
            elements.sort(key=lambda e: e.x)
            
            # Process elements
            for elem in elements:
                # Get element position
                if hasattr(elem, 'x') and hasattr(elem, 'y'):
                    x, y = elem.x, elem.y
                elif isinstance(elem, dict) and 'bbox' in elem:
                    x = elem['bbox'].get('center_x', 0)
                    y = elem['bbox'].get('center_y', 0)
                else:
                    continue  # Skip if no position
                
                # Store position with onset time - round to avoid floating point issues
                position_onset_map[(round(x, 1), round(y, 1))] = cumulative_onset
                
                # Add duration to onset
                if hasattr(elem, 'duration') and elem.duration is not None:
                    cumulative_onset += elem.duration
                else:
                    # Default to quarter note duration
                    cumulative_onset += 1.0
    
    return position_onset_map

def add_stem_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add stem nodes to XML first (before notes) so we can link to them"""
    current_id = next_id
    order_id = 100  # Start stems at order 100
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get all stems from raw detections
    stem_detections = [d for d in raw_detections 
                     if d.get('class_name', '').lower() == 'stem']
    
    # If no stems in raw detections, try from processor
    if not stem_detections:
        # Extract stems from all systems
        for system in processor.staff_systems:
            stems = [e for e in system.elements if hasattr(e, 'class_name') and e.class_name.lower() == 'stem']
            stem_detections.extend(stems)
    
    # Create XML nodes for all stems
    stem_id_map = {}  # Maps stem ID string -> xml ID
    stem_obj_to_id = {}  # Maps stem object to its ID string
    
    for stem_idx, stem in enumerate(stem_detections):
        # Generate stem ID
        xml_id = current_id
        current_id += 1
        
        # Create a string ID for this stem
        stem_string_id = None
        
        if isinstance(stem, dict) and 'id' in stem:
            # Use the existing ID if available
            stem_string_id = stem['id']
        else:
            # Create a string ID based on position
            x = 0
            y = 0
            if isinstance(stem, dict) and 'bbox' in stem:
                x = stem['bbox'].get('center_x', 0)
                y = stem['bbox'].get('center_y', 0)
            elif hasattr(stem, 'bbox'):
                x = getattr(stem.bbox, 'center_x', getattr(stem, 'x', 0))
                y = getattr(stem.bbox, 'center_y', getattr(stem, 'y', 0))
            else:
                x = getattr(stem, 'x', 0)
                y = getattr(stem, 'y', 0)
                
            stem_string_id = f"stem_{x:.1f}_{y:.1f}_{stem_idx}"
        
        # Store mapping from string ID to XML ID
        stem_id_map[stem_string_id] = xml_id
        
        # Also store a mapping from this stem object to its string ID
        # We'll use this to lookup during beam linking
        stem_obj_to_id[stem_idx] = stem_string_id
        
        # Determine which staff this stem belongs to
        staff_id = 0
        if hasattr(stem, 'staff_system') and stem.staff_system:
            staff_id = processor.staff_systems.index(stem.staff_system)
        elif isinstance(stem, dict) and 'staff_system' in stem:
            staff_id = stem['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(xml_id)
        
        # Class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "stem"
        
        # Extract bbox information
        bbox = {}
        if isinstance(stem, dict) and 'bbox' in stem:
            bbox = stem['bbox']
        elif hasattr(stem, 'bbox'):
            bbox = stem.bbox
        else:
            # Create bbox from position and dimensions
            x = getattr(stem, 'x', 0)
            y = getattr(stem, 'y', 0)
            width = getattr(stem, 'width', 3)
            height = getattr(stem, 'height', 80)
            bbox = {
                'x1': x - width/2,
                'y1': y - height/2,
                'x2': x + width/2,
                'y2': y + height/2,
                'width': width,
                'height': height,
                'center_x': x,
                'center_y': y
            }
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 3)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 80)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    # Also add mapping from notehead IDs to stem IDs based on notehead_id in stems
    for stem_idx, stem in enumerate(stem_detections):
        if isinstance(stem, dict) and 'notehead_id' in stem:
            # Get the stem's string ID
            stem_string_id = stem_obj_to_id.get(stem_idx)
            if stem_string_id and stem_string_id in stem_id_map:
                # Add the mapping from notehead ID to XML ID
                notehead_id = stem['notehead_id']
                stem_id_map[notehead_id] = stem_id_map[stem_string_id]
    
    # Return the updated ID and the stem mapping
    return current_id, order_id, stem_id_map

def add_beam_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings, stem_id_map):
    """Add beam nodes to XML"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    beam_stems_map = id_mappings.get('beam_stems_map', {})
    
    # Get all beams from raw detections
    beam_detections = [d for d in raw_detections 
                     if d.get('class_name', '').lower() == 'beam']
    
    # If no beams in raw detections, try from processor
    if not beam_detections:
        # Extract beams from all systems
        for system in processor.staff_systems:
            beams = [e for e in system.elements if hasattr(e, 'class_name') and e.class_name.lower() == 'beam']
            beam_detections.extend(beams)
    
    # Create beam ID mapping
    beam_id_map = {}  # Maps beam ID string -> XML ID
    beam_obj_to_id = {}  # Maps beam object index to its ID string
    
    # Process each beam
    for beam_idx, beam in enumerate(beam_detections):
        # Generate beam ID
        xml_id = current_id
        current_id += 1
        
        # Create a string ID for this beam
        beam_string_id = None
        
        if isinstance(beam, dict) and 'id' in beam:
            # Use the existing ID if available
            beam_string_id = beam['id']
        else:
            # Create a string ID based on position
            x = 0
            y = 0
            if isinstance(beam, dict) and 'bbox' in beam:
                x = beam['bbox'].get('center_x', 0)
                y = beam['bbox'].get('center_y', 0)
            elif hasattr(beam, 'bbox'):
                x = getattr(beam.bbox, 'center_x', getattr(beam, 'x', 0))
                y = getattr(beam.bbox, 'center_y', getattr(beam, 'y', 0))
            else:
                x = getattr(beam, 'x', 0)
                y = getattr(beam, 'y', 0)
                
            beam_string_id = f"beam_{x:.1f}_{y:.1f}_{beam_idx}"
        
        # Store mapping from string ID to XML ID
        beam_id_map[beam_string_id] = xml_id
        
        # Also store mapping from index to ID
        beam_obj_to_id[beam_idx] = beam_string_id
        
        # Determine which staff this beam belongs to
        staff_id = 0
        if hasattr(beam, 'staff_system') and beam.staff_system:
            staff_id = processor.staff_systems.index(beam.staff_system)
        elif isinstance(beam, dict) and 'staff_system' in beam:
            staff_id = beam['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(xml_id)
        
        # Class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "beam"
        
        # Extract bbox information
        bbox = {}
        if isinstance(beam, dict) and 'bbox' in beam:
            bbox = beam['bbox']
        elif hasattr(beam, 'bbox'):
            bbox = beam.bbox
        else:
            # Create bbox from position and dimensions
            x = getattr(beam, 'x', 0)
            y = getattr(beam, 'y', 0)
            width = getattr(beam, 'width', 100)
            height = getattr(beam, 'height', 10)
            bbox = {
                'x1': x - width/2,
                'y1': y - height/2,
                'x2': x + width/2,
                'y2': y + height/2,
                'width': width,
                'height': height,
                'center_x': x,
                'center_y': y
            }
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 100)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 10)))
        
        # Add outlinks to stems connected to this beam
        if beam_string_id in beam_stems_map and beam_stems_map[beam_string_id]:
            # Get stems connected to this beam
            stem_ids = []
            for stem_id in beam_stems_map[beam_string_id]:
                if stem_id in stem_id_map:
                    stem_ids.append(stem_id_map[stem_id])
            
            if stem_ids:
                # Add outlinks
                outlinks_elem = SubElement(node, "Outlinks")
                outlinks_elem.text = " ".join(map(str, stem_ids))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id, beam_id_map

def add_note_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings, stem_id_map):
    """Add note nodes to XML with proper linking to stems"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections and mappings
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    note_stem_map = id_mappings.get('note_stem_map', {})
    
    # Calculate note onset times
    note_onset_map = calculate_note_onset_map(processor)
    
    # Get all noteheads from raw detections
    note_detections = [d for d in raw_detections 
                     if 'notehead' in d.get('class_name', '').lower()]
    
    # Print comprehensive debug info
    print(f"\n========== NOTE PROCESSING DEBUG ==========")
    print(f"Total raw detections: {len(raw_detections)}")
    print(f"Total note detections: {len(note_detections)}")
    if note_detections:
        print(f"Note class names: {[d.get('class_name', 'unknown') for d in note_detections[:10]]}...")
    
    # If no notes in raw detections, try from processor
    if not note_detections:
        print("No note detections found in raw_detections, trying processor.notes...")
        # Extract notes from all systems
        for system in processor.staff_systems:
            notes = [e for e in system.elements if hasattr(e, 'step') and hasattr(e, 'octave')]
            note_detections.extend(notes)
        print(f"After processor search: {len(note_detections)} notes found")
    
    # Create XML nodes for all notes
    note_id_map = {}  # Maps note ID string -> xml ID
    
    # Process all notes with debug info at key points
    processed_count = 0
    for note_idx, note in enumerate(note_detections):
        processed_count += 1
        
        # Generate note ID
        xml_id = current_id
        current_id += 1
        
        # Create a string ID for this note
        note_string_id = None
        
        if isinstance(note, dict) and 'id' in note:
            # Use the existing ID if available
            note_string_id = note['id']
        else:
            # Create a string ID based on position
            x = 0
            y = 0
            if isinstance(note, dict) and 'bbox' in note:
                x = note['bbox'].get('center_x', 0)
                y = note['bbox'].get('center_y', 0)
            elif hasattr(note, 'bbox'):
                x = getattr(note.bbox, 'center_x', getattr(note, 'x', 0))
                y = getattr(note.bbox, 'center_y', getattr(note, 'y', 0))
            else:
                x = getattr(note, 'x', 0)
                y = getattr(note, 'y', 0)
                
            note_string_id = f"notehead_{x:.1f}_{y:.1f}_{note_idx}"
        
        # Store in note ID map
        note_id_map[note_string_id] = xml_id
        
        # Determine which staff this note belongs to
        staff_id = 0
        if hasattr(note, 'staff_system') and note.staff_system:
            staff_id = processor.staff_systems.index(note.staff_system)
        elif isinstance(note, dict) and 'staff_system' in note:
            staff_id = note['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(xml_id)
        
        # Map note type to class name
        class_elem = SubElement(node, "ClassName")
        
        # Determine note type
        note_type = "quarter"  # Default
        
        if isinstance(note, dict):
            if 'duration_type' in note:
                note_type = note['duration_type']
            elif 'class_name' in note:
                class_name = note['class_name'].lower()
                if 'whole' in class_name:
                    note_type = 'whole'
                elif 'half' in class_name:
                    note_type = 'half'
        elif hasattr(note, 'duration_type'):
            note_type = note.duration_type
        
        # Determine note head class based on duration
        if note_type in ['whole']:
            class_elem.text = "noteheadWhole"
        elif note_type in ['half']:
            class_elem.text = "noteheadHalf"
        else:
            class_elem.text = "noteheadBlack"
        
        # Extract bbox information
        bbox = {}
        if isinstance(note, dict) and 'bbox' in note:
            bbox = note['bbox']
        elif hasattr(note, 'bbox'):
            bbox = note.bbox
        else:
            # Create bbox from position and dimensions
            x = getattr(note, 'x', 0)
            y = getattr(note, 'y', 0)
            width = getattr(note, 'width', 25)
            height = getattr(note, 'height', 20)
            bbox = {
                'x1': x - width/2,
                'y1': y - height/2,
                'x2': x + width/2,
                'y2': y + height/2,
                'width': width,
                'height': height,
                'center_x': x,
                'center_y': y
            }
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 25)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 20)))
        
        # Find stem for this note and add outlinks
        stem_id = None
        
        # Check if note has a stem_id
        if isinstance(note, dict) and 'stem_id' in note:
            if note['stem_id'] in stem_id_map:
                stem_id = stem_id_map[note['stem_id']]
        
        # Check if note's ID is in the note-stem map
        if note_string_id in note_stem_map:
            stem_key = note_stem_map[note_string_id]
            if stem_key in stem_id_map:
                stem_id = stem_id_map[stem_key]
        
        # Check by matching note with stem that references its ID
        if note_string_id:
            for stem in [d for d in raw_detections if d.get('class_name', '').lower() == 'stem']:
                if isinstance(stem, dict) and 'notehead_id' in stem and stem['notehead_id'] == note_string_id:
                    stem_id_key = f"stem_{stem.get('bbox', {}).get('center_x', 0)}_{stem.get('bbox', {}).get('center_y', 0)}_0"
                    if stem_id_key in stem_id_map:
                        stem_id = stem_id_map[stem_id_key]
                        break
        
        # If we found a stem, add outlinks
        if stem_id is not None:
            outlinks_elem = SubElement(node, "Outlinks")
            outlinks_elem.text = str(stem_id)
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add duration data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "duration_beats")
        data_item.set("type", "float")
        
        # Determine note duration
        duration = 1.0  # Default to quarter note
        
        if isinstance(note, dict):
            if 'duration' in note and note['duration'] is not None:
                duration = note['duration']
            elif 'duration_type' in note:
                # Map duration_type to actual duration
                duration_map = {
                    'whole': 4.0, 
                    'half': 2.0, 
                    'quarter': 1.0, 
                    'eighth': 0.5, 
                    '8th': 0.5,
                    '16th': 0.25, 
                    '32nd': 0.125
                }
                duration = duration_map.get(note['duration_type'], 1.0)
        elif hasattr(note, 'duration') and note.duration is not None:
            duration = note.duration
        elif hasattr(note, 'duration_type'):
            # Map duration_type to actual duration
            duration_map = {
                'whole': 4.0, 
                'half': 2.0, 
                'quarter': 1.0, 
                'eighth': 0.5, 
                '8th': 0.5,
                '16th': 0.25, 
                '32nd': 0.125
            }
            duration = duration_map.get(note.duration_type, 1.0)
            
        data_item.text = f"{duration:.6f}"
        
        # Add onset data
        onset = 0.0

        # Get note position for position-based lookup
        note_x = 0
        note_y = 0
        if isinstance(note, dict) and 'bbox' in note:
            note_x = note['bbox'].get('center_x', 0)
            note_y = note['bbox'].get('center_y', 0)
        elif hasattr(note, 'x') and hasattr(note, 'y'):
            note_x = note.x
            note_y = note.y
            
        # Look up onset by position (rounded to handle float precision)
        position_key = (round(note_x, 1), round(note_y, 1))
        if position_key in note_onset_map:
            onset = note_onset_map[position_key]
            
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "onset_beats")
        data_item.set("type", "float")
        data_item.text = f"{onset:.6f}"
        
        # Add pitch data
        octave = 4  # Default
        step = "C"  # Default
        
        if isinstance(note, dict):
            if 'octave' in note:
                octave = note['octave']
            if 'step' in note:
                step = note['step']
        elif hasattr(note, 'octave') and hasattr(note, 'step'):
            octave = note.octave
            step = note.step
            
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "pitch_octave")
        data_item.set("type", "int")
        data_item.text = str(octave)
        
        # Calculate MIDI pitch with enhanced debugging
        midi_pitch = 0
        
        # Debug note information
        if note_idx < 10:  # Only print for first 10 notes to avoid log spam
            print(f"DEBUG: Processing note {note_idx} at position ({bbox.get('center_x', 0):.1f}, {bbox.get('center_y', 0):.1f})")
            print(f"DEBUG: Class name: {class_name if isinstance(note, dict) and 'class_name' in note else getattr(note, 'class_name', 'unknown')}")
            print(f"DEBUG: Extracted pitch step={step}, octave={octave}, type(step)={type(step)}, type(octave)={type(octave)}")
        
        # Check if we have valid step and octave values
        if step in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
            # Convert step to numeric value (C=0, D=2, E=4, etc.)
            step_values = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
            step_value = step_values[step]
            
            # Calculate MIDI pitch (C4 = 60)
            midi_pitch = (octave + 1) * 12 + step_value
            
            # Apply accidental if present
            alter = 0
            
            if isinstance(note, dict) and 'alter' in note:
                alter = note['alter']
                midi_pitch += alter
            elif hasattr(note, 'alter'):
                alter = note.alter
                midi_pitch += alter
            elif isinstance(note, dict) and 'accidental' in note and note['accidental']:
                # Get accidental type
                acc_alter_map = {'sharp': 1, 'flat': -1, 'natural': 0, 'double-sharp': 2, 'double-flat': -2}
                acc = note['accidental']
                acc_type = acc.get('type', 'natural') if isinstance(acc, dict) else 'natural'
                alter = acc_alter_map.get(acc_type, 0)
                midi_pitch += alter
            elif hasattr(note, 'accidental') and note.accidental:
                # Get accidental type
                acc_alter_map = {'sharp': 1, 'flat': -1, 'natural': 0, 'double-sharp': 2, 'double-flat': -2}
                acc_type = note.accidental.type if hasattr(note.accidental, 'type') else 'natural'
                alter = acc_alter_map.get(acc_type, 0)
                midi_pitch += alter
                
            if note_idx < 10:  # Debug for first 10 notes only
                print(f"DEBUG: MIDI pitch calculation: ({octave} + 1) * 12 + {step_value} + alter={alter} = {midi_pitch}")
        else:
            # Invalid pitch data - add debug info
            print(f"Invalid pitch data: step={step}, octave={octave} for note at ({bbox.get('center_x', 0):.1f}, {bbox.get('center_y', 0):.1f})")
            # Default to middle C (60) as fallback
            midi_pitch = 60
            
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "midi_pitch_code")
        data_item.set("type", "int")
        data_item.text = str(midi_pitch)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "normalized_pitch_step")
        data_item.set("type", "str")
        data_item.text = step
        
        # Add event ID
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "dorico_event_id")
        data_item.set("type", "int")
        data_item.text = str(18000 + note_idx)  # Generate unique ID
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    print(f"Processed {processed_count} out of {len(note_detections)} notes")
    print(f"============================================\n")
    
    return current_id, order_id, note_id_map

def add_rest_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add rest nodes to XML"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Calculate note onset times
    note_onset_map = calculate_note_onset_map(processor)
    
    # Get all rests from raw detections
    rest_detections = [d for d in raw_detections 
                     if 'rest' in d.get('class_name', '').lower()]
    
    # If no rests in raw detections, try from processor
    if not rest_detections:
        # Extract rests from all systems
        for system in processor.staff_systems:
            rests = [e for e in system.elements if hasattr(e, 'class_name') and 'rest' in e.class_name.lower()]
            rest_detections.extend(rests)
    
    # Process each rest
    for rest_idx, rest in enumerate(rest_detections):
        # Determine which staff this rest belongs to
        staff_id = 0
        if hasattr(rest, 'staff_system') and rest.staff_system:
            staff_id = processor.staff_systems.index(rest.staff_system)
        elif isinstance(rest, dict) and 'staff_system' in rest:
            staff_id = rest['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Map rest type to class name
        class_elem = SubElement(node, "ClassName")
        
        # Determine rest type
        rest_type = ""
        class_name = ""
        
        if isinstance(rest, dict):
            if 'duration_type' in rest:
                rest_type = rest['duration_type']
            elif 'class_name' in rest:
                class_name = rest['class_name'].lower()
        elif hasattr(rest, 'duration_type'):
            rest_type = rest.duration_type
        elif hasattr(rest, 'class_name'):
            class_name = rest.class_name.lower()
        
        # If we have a class name but no rest_type, extract from class name
        if not rest_type and class_name:
            if 'whole' in class_name:
                rest_type = 'whole'
            elif 'half' in class_name:
                rest_type = 'half'
            elif 'quarter' in class_name:
                rest_type = 'quarter'
            elif '8th' in class_name or 'eighth' in class_name:
                rest_type = '8th'
            elif '16th' in class_name:
                rest_type = '16th'
            elif '32nd' in class_name:
                rest_type = '32nd'
        
        # Set class name based on rest type
        if rest_type == 'whole':
            class_elem.text = "restWhole"
        elif rest_type == 'half':
            class_elem.text = "restHalf"
        elif rest_type == 'quarter':
            class_elem.text = "restQuarter"
        elif rest_type in ['8th', 'eighth']:
            class_elem.text = "rest8th"
        elif rest_type == '16th':
            class_elem.text = "rest16th"
        elif rest_type == '32nd':
            class_elem.text = "rest32nd"
        else:
            # If we have class_name, use that
            if class_name:
                # Map class name
                if 'whole' in class_name:
                    class_elem.text = "restWhole"
                elif 'half' in class_name:
                    class_elem.text = "restHalf"
                elif 'quarter' in class_name:
                    class_elem.text = "restQuarter"
                elif '8th' in class_name or 'eighth' in class_name:
                    class_elem.text = "rest8th"
                elif '16th' in class_name:
                    class_elem.text = "rest16th"
                elif '32nd' in class_name:
                    class_elem.text = "rest32nd"
                else:
                    # Default to quarter rest if unknown
                    class_elem.text = "restQuarter"
            else:
                # Default to quarter rest if unknown
                class_elem.text = "restQuarter"
        
        # Extract bbox information
        bbox = {}
        if isinstance(rest, dict) and 'bbox' in rest:
            bbox = rest['bbox']
        elif hasattr(rest, 'bbox'):
            bbox = rest.bbox
        else:
            # Create bbox from position and dimensions
            x = getattr(rest, 'x', 0)
            y = getattr(rest, 'y', 0)
            width = getattr(rest, 'width', 20)
            height = getattr(rest, 'height', 30)
            bbox = {
                'x1': x - width/2,
                'y1': y - height/2,
                'x2': x + width/2,
                'y2': y + height/2,
                'width': width,
                'height': height,
                'center_x': x,
                'center_y': y
            }
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 20)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 30)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add duration data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "duration_beats")
        data_item.set("type", "float")
        
        # Determine rest duration
        duration = 1.0  # Default to quarter note
        
        if isinstance(rest, dict):
            if 'duration' in rest and rest['duration'] is not None:
                duration = rest['duration']
            elif 'duration_type' in rest:
                # Map duration_type to actual duration
                duration_map = {
                    'whole': 4.0, 
                    'half': 2.0, 
                    'quarter': 1.0, 
                    'eighth': 0.5, 
                    '8th': 0.5,
                    '16th': 0.25, 
                    '32nd': 0.125
                }
                duration = duration_map.get(rest['duration_type'], 1.0)
        elif hasattr(rest, 'duration') and rest.duration is not None:
            duration = rest.duration
        elif hasattr(rest, 'duration_type'):
            # Map duration_type to actual duration
            duration_map = {
                'whole': 4.0, 
                'half': 2.0, 
                'quarter': 1.0, 
                'eighth': 0.5, 
                '8th': 0.5,
                '16th': 0.25, 
                '32nd': 0.125
            }
            duration = duration_map.get(rest.duration_type, 1.0)
        
        # If duration not found, infer from class_name
        if duration == 1.0 and class_name:
            duration_map = {
                'whole': 4.0, 
                'half': 2.0, 
                'quarter': 1.0, 
                'eighth': 0.5, 
                '8th': 0.5,
                '16th': 0.25, 
                '32nd': 0.125
            }
            
            for key, value in duration_map.items():
                if key in class_name:
                    duration = value
                    break
                    
        data_item.text = f"{duration:.6f}"
        
        # Add onset data
        # Add onset data with position-based lookup
        onset = 0.0
        # Get rest position for lookup
        rest_x = 0
        rest_y = 0
        if isinstance(rest, dict) and 'bbox' in rest:
            rest_x = rest['bbox'].get('center_x', 0)
            rest_y = rest['bbox'].get('center_y', 0)
        elif hasattr(rest, 'x') and hasattr(rest, 'y'):
            rest_x = rest.x
            rest_y = rest.y
            
        # Lookup position in onset map
        position_key = (round(rest_x, 1), round(rest_y, 1))
        if position_key in note_onset_map:
            onset = note_onset_map[position_key]
            
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "onset_beats")
        data_item.set("type", "float")
        data_item.text = f"{onset:.6f}"
        
        # Add event ID
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "dorico_event_id")
        data_item.set("type", "int")
        data_item.text = str(19000 + rest_idx)  # Generate unique ID
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def add_barline_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add barline nodes to XML"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get all barlines from raw detections
    barline_detections = [d for d in raw_detections 
                        if 'barline' in d.get('class_name', '').lower()]
    
    # If no barlines in raw detections, try from processor
    if not barline_detections:
        # Extract barlines from all systems
        for system in processor.staff_systems:
            barlines = [e for e in system.elements if hasattr(e, 'class_name') and 'barline' in e.class_name.lower()]
            barline_detections.extend(barlines)
    
    # Process each barline
    for barline_idx, barline in enumerate(barline_detections):
        # Determine which staff this barline belongs to
        staff_id = 0
        if hasattr(barline, 'staff_system') and barline.staff_system:
            staff_id = processor.staff_systems.index(barline.staff_system)
        elif isinstance(barline, dict) and 'staff_system' in barline:
            staff_id = barline['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Class name
        class_elem = SubElement(node, "ClassName")
        
        # Determine barline type
        is_systemic = False
        
        if isinstance(barline, dict):
            class_name = barline.get('class_name', '').lower()
            is_systemic = class_name == 'systemicbarline' or barline.get('is_systemic', False)
        elif hasattr(barline, 'class_name'):
            class_name = barline.class_name.lower()
            is_systemic = class_name == 'systemicbarline' or getattr(barline, 'is_systemic', False)
        
        if is_systemic:
            class_elem.text = "systemicBarline"
        else:
            class_elem.text = "barline"
        
        # Extract bbox information
        bbox = {}
        if isinstance(barline, dict) and 'bbox' in barline:
            bbox = barline['bbox']
        elif hasattr(barline, 'bbox'):
            bbox = barline.bbox
        else:
            # Create bbox from position and dimensions
            x = getattr(barline, 'x', 0)
            y = getattr(barline, 'y', 0)
            width = getattr(barline, 'width', 4)
            height = getattr(barline, 'height', 100)
            bbox = {
                'x1': x - width/2,
                'y1': y - height/2,
                'x2': x + width/2,
                'y2': y + height/2,
                'width': width,
                'height': height,
                'center_x': x,
                'center_y': y
            }
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 4)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 100)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add event ID for barline
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "dorico_event_id")
        data_item.set("type", "int")
        data_item.text = str(14000 + barline_idx)  # Generate unique ID
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id



def add_accidental_natural_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add accidental natural nodes to XML"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get all accidentals from raw detections
    accidental_detections = [d for d in raw_detections 
                           if 'accidental' in d.get('class_name', '').lower() and 
                           'natural' in d.get('class_name', '').lower()]
    
    # Process each accidental
    for acc_idx, acc in enumerate(accidental_detections):
        # Determine which staff this accidental belongs to
        staff_id = 0
        if hasattr(acc, 'staff_system') and acc.staff_system:
            staff_id = processor.staff_systems.index(acc.staff_system)
        elif isinstance(acc, dict) and 'staff_system' in acc:
            staff_id = acc['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Set class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "accidentalNatural"
        
        # Extract bbox information
        bbox = {}
        if isinstance(acc, dict) and 'bbox' in acc:
            bbox = acc['bbox']
        elif hasattr(acc, 'bbox'):
            bbox = acc.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 20)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 50)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def add_articulation_below_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add articulation nodes (accent below, staccato below)"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get articulations with "below" in the name
    articulation_detections = [d for d in raw_detections 
                             if any(art in d.get('class_name', '').lower() for art in 
                                   ['staccato', 'accent']) and 
                                'below' in d.get('class_name', '').lower()]
    
    # Process each articulation
    for art_idx, art in enumerate(articulation_detections):
        # Determine which staff this articulation belongs to
        staff_id = 0
        if hasattr(art, 'staff_system') and art.staff_system:
            staff_id = processor.staff_systems.index(art.staff_system)
        elif isinstance(art, dict) and 'staff_system' in art:
            staff_id = art['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Determine articulation type
        class_elem = SubElement(node, "ClassName")
        
        if 'accent' in art.get('class_name', '').lower():
            class_elem.text = "articAccentBelow"
        else:
            class_elem.text = "articStaccatoBelow"
        
        # Extract bbox information
        bbox = {}
        if isinstance(art, dict) and 'bbox' in art:
            bbox = art['bbox']
        elif hasattr(art, 'bbox'):
            bbox = art.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 15)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 15)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def add_dynamic_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add dynamic nodes (mf, mp, etc.)"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get all dynamics from raw detections
    dynamic_detections = [d for d in raw_detections 
                         if 'dynamic' in d.get('class_name', '').lower()]
    
    # Process each dynamic
    for dyn_idx, dyn in enumerate(dynamic_detections):
        # Determine which staff this dynamic belongs to
        staff_id = 0
        if hasattr(dyn, 'staff_system') and dyn.staff_system:
            staff_id = processor.staff_systems.index(dyn.staff_system)
        elif isinstance(dyn, dict) and 'staff_system' in dyn:
            staff_id = dyn['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Map dynamic type to class name
        class_elem = SubElement(node, "ClassName")
        
        # Default to MF but check for specific types
        class_elem.text = "dynamicMF"
        if isinstance(dyn, dict) and 'class_name' in dyn:
            class_name = dyn['class_name'].lower()
            if 'ff' in class_name:
                class_elem.text = "dynamicFF"
            elif 'f' in class_name and 'm' not in class_name:
                class_elem.text = "dynamicF"
            elif 'pp' in class_name:
                class_elem.text = "dynamicPP"
            elif 'p' in class_name and 'm' not in class_name:
                class_elem.text = "dynamicP"
            elif 'mp' in class_name:
                class_elem.text = "dynamicMP"
        
        # Extract bbox information
        bbox = {}
        if isinstance(dyn, dict) and 'bbox' in dyn:
            bbox = dyn['bbox']
        elif hasattr(dyn, 'bbox'):
            bbox = dyn.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 30)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 20)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def add_gradual_dynamic_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add gradual dynamic nodes (crescendo, diminuendo)"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get all gradual dynamics
    grad_dyn_detections = [d for d in raw_detections 
                         if ('gradual' in d.get('class_name', '').lower() or
                            'cresc' in d.get('class_name', '').lower() or
                            'dim' in d.get('class_name', '').lower())]
    
    # Process each gradual dynamic
    for dyn_idx, dyn in enumerate(grad_dyn_detections):
        # Determine which staff this gradual dynamic belongs to
        staff_id = 0
        if hasattr(dyn, 'staff_system') and dyn.staff_system:
            staff_id = processor.staff_systems.index(dyn.staff_system)
        elif isinstance(dyn, dict) and 'staff_system' in dyn:
            staff_id = dyn['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Set class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "gradualDynamic"
        
        # Extract bbox information
        bbox = {}
        if isinstance(dyn, dict) and 'bbox' in dyn:
            bbox = dyn['bbox']
        elif hasattr(dyn, 'bbox'):
            bbox = dyn.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 80)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 20)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def add_flag_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add flag nodes (8thUp, etc.)"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get all flags
    flag_detections = [d for d in raw_detections 
                      if 'flag' in d.get('class_name', '').lower()]
    
    # Process each flag
    for flag_idx, flag in enumerate(flag_detections):
        # Determine which staff this flag belongs to
        staff_id = 0
        if hasattr(flag, 'staff_system') and flag.staff_system:
            staff_id = processor.staff_systems.index(flag.staff_system)
        elif isinstance(flag, dict) and 'staff_system' in flag:
            staff_id = flag['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Set class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "flag8thUp"  # Default
        
        if isinstance(flag, dict) and 'class_name' in flag:
            class_name = flag['class_name'].lower()
            if 'down' in class_name:
                class_elem.text = "flag8thDown"
            if '16th' in class_name:
                class_elem.text = class_elem.text.replace('8th', '16th')
        
        # Extract bbox information
        bbox = {}
        if isinstance(flag, dict) and 'bbox' in flag:
            bbox = flag['bbox']
        elif hasattr(flag, 'bbox'):
            bbox = flag.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 20)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 30)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def add_rest_types(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add specific rest types (quarter, half)"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get specific rest types
    quarter_rest_detections = [d for d in raw_detections 
                            if 'restquarter' in d.get('class_name', '').lower()]
    half_rest_detections = [d for d in raw_detections 
                          if 'resthalf' in d.get('class_name', '').lower()]
    
    # Process quarter rests
    for rest_idx, rest in enumerate(quarter_rest_detections):
        # Set up the node
        staff_id = 0
        if hasattr(rest, 'staff_system') and rest.staff_system:
            staff_id = processor.staff_systems.index(rest.staff_system)
        elif isinstance(rest, dict) and 'staff_system' in rest:
            staff_id = rest['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Set class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "restQuarter"
        
        # Extract bbox information
        bbox = {}
        if isinstance(rest, dict) and 'bbox' in rest:
            bbox = rest['bbox']
        elif hasattr(rest, 'bbox'):
            bbox = rest.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 20)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 30)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    # Process half rests
    for rest_idx, rest in enumerate(half_rest_detections):
        # Set up the node
        staff_id = 0
        if hasattr(rest, 'staff_system') and rest.staff_system:
            staff_id = processor.staff_systems.index(rest.staff_system)
        elif isinstance(rest, dict) and 'staff_system' in rest:
            staff_id = rest['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Set class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "restHalf"
        
        # Extract bbox information
        bbox = {}
        if isinstance(rest, dict) and 'bbox' in rest:
            bbox = rest['bbox']
        elif hasattr(rest, 'bbox'):
            bbox = rest.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 20)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 20)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def add_slur_tie_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add slur and tie nodes"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get slurs and ties
    slur_detections = [d for d in raw_detections if 'slur' in d.get('class_name', '').lower()]
    tie_detections = [d for d in raw_detections if 'tie' in d.get('class_name', '').lower()]
    
    # Process slurs
    for slur_idx, slur in enumerate(slur_detections):
        # Set up the node
        staff_id = 0
        if hasattr(slur, 'staff_system') and slur.staff_system:
            staff_id = processor.staff_systems.index(slur.staff_system)
        elif isinstance(slur, dict) and 'staff_system' in slur:
            staff_id = slur['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Set class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "slur"
        
        # Extract bbox information
        bbox = {}
        if isinstance(slur, dict) and 'bbox' in slur:
            bbox = slur['bbox']
        elif hasattr(slur, 'bbox'):
            bbox = slur.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 50)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 30)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    # Process ties
    for tie_idx, tie in enumerate(tie_detections):
        # Set up the node
        staff_id = 0
        if hasattr(tie, 'staff_system') and tie.staff_system:
            staff_id = processor.staff_systems.index(tie.staff_system)
        elif isinstance(tie, dict) and 'staff_system' in tie:
            staff_id = tie['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Set class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "tie"
        
        # Extract bbox information
        bbox = {}
        if isinstance(tie, dict) and 'bbox' in tie:
            bbox = tie['bbox']
        elif hasattr(tie, 'bbox'):
            bbox = tie.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 40)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 20)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def add_timesig4_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings):
    """Add time signature 4 nodes"""
    current_id = next_id
    order_id = next_order_id
    
    # Get raw detections
    raw_detections = list(id_mappings.get('element_ids', {}).values())
    
    # Get all time signature digits
    timesig_detections = [d for d in raw_detections 
                        if 'timesig' in d.get('class_name', '').lower() and 
                        '4' in d.get('class_name', '')]
    
    # Process each time signature digit
    for ts_idx, ts in enumerate(timesig_detections):
        # Determine which staff this time sig belongs to
        staff_id = 0
        if hasattr(ts, 'staff_system') and ts.staff_system:
            staff_id = processor.staff_systems.index(ts.staff_system)
        elif isinstance(ts, dict) and 'staff_system' in ts:
            staff_id = ts['staff_system']
        
        # Get spacing run ID for this staff
        spacing_run_id = spacing_run_ids.get(staff_id, "0")
        
        node = SubElement(nodes, "Node")
        
        # Add ID
        id_elem = SubElement(node, "Id")
        id_elem.text = str(current_id)
        current_id += 1
        
        # Set class name
        class_elem = SubElement(node, "ClassName")
        class_elem.text = "timeSig4"
        
        # Extract bbox information
        bbox = {}
        if isinstance(ts, dict) and 'bbox' in ts:
            bbox = ts['bbox']
        elif hasattr(ts, 'bbox'):
            bbox = ts.bbox
        
        # Position elements
        top_elem = SubElement(node, "Top")
        top_elem.text = str(int(bbox.get('y1', 0)))
        
        left_elem = SubElement(node, "Left")
        left_elem.text = str(int(bbox.get('x1', 0)))
        
        width_elem = SubElement(node, "Width")
        width_elem.text = str(int(bbox.get('width', 20)))
        
        height_elem = SubElement(node, "Height")
        height_elem.text = str(int(bbox.get('height', 30)))
        
        # Add data section
        data_elem = SubElement(node, "Data")
        
        # Add staff data
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "spacing_run_id")
        data_item.set("type", "int")
        data_item.text = spacing_run_id
        
        data_item = SubElement(data_elem, "DataItem")
        data_item.set("key", "ordered_staff_id")
        data_item.set("type", "int")
        data_item.text = str(staff_id)
        
        # Add order ID
        order_elem = SubElement(node, "order_id")
        order_elem.text = str(order_id)
        order_id += 1
    
    return current_id, order_id

def generate_custom_xml(processor, output_path=None):
    """
    Generate comprehensive custom XML format similar to the provided ground truth structure.
    
    Args:
        processor: OMRProcessor object containing processed music elements
        output_path: Path to save the XML file (if None, returns XML string)
        
    Returns:
        XML string if output_path is None, otherwise saves to file and returns path
    """
    # Load raw detections from processor
    raw_detections = load_raw_detections(processor)
    if not raw_detections:
        logger.warning("No raw detections found in processor. XML may be incomplete.")
    
    # Extract ID mappings from raw detections
    id_mappings = extract_id_mapping(raw_detections)
    
    # Create root elements
    root = Element("masks")
    page = SubElement(root, "Page")
    page.set("pageIndex", "0")
    nodes = SubElement(page, "Nodes")
    
    # Create staff line nodes and get spacing run IDs
    spacing_run_ids = create_staff_line_nodes(root, processor, raw_detections)
    
    # Start ID counter after staff lines
    next_id = 106  # Start with higher ID
    next_order_id = 6  # Start order after staff lines
    
    # Add clef nodes
    next_id, next_order_id = add_clef_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)
    
    # Add key signature nodes
    next_id, next_order_id = add_key_signature_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)
    next_id, next_order_id = add_accidental_natural_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)

    
    # Add time signature nodes
    next_id, next_order_id = add_time_signature_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)
    next_id, next_order_id = add_timesig4_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)

    
    # Add stem nodes first
    next_id, next_order_id, stem_id_map = add_stem_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)
    
    # Add beam nodes
    next_id, next_order_id, beam_id_map = add_beam_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings, stem_id_map)
    
    # Add note nodes with stem links
    next_id, next_order_id, note_id_map = add_note_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings, stem_id_map)
    
    # Add rest nodes
    next_id, next_order_id = add_rest_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)
    next_id, next_order_id = add_rest_types(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)

    
    # Add barline nodes
    next_id, next_order_id = add_barline_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)
    
    next_id, next_order_id = add_articulation_below_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)

    # After dynamic nodes:
    next_id, next_order_id = add_dynamic_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings) 
    next_id, next_order_id = add_gradual_dynamic_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)
    
    next_id, next_order_id = add_slur_tie_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)



    next_id, next_order_id = add_flag_nodes(nodes, processor, spacing_run_ids, next_id, next_order_id, id_mappings)

    # Convert to string with pretty formatting
    xml_string = minidom.parseString(tostring(root, 'utf-8')).toprettyxml(indent="  ")
    
    # Save to file if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_string)
        logger.info(f"Complete custom XML saved to {output_path}")
        return output_path
    
    return xml_string


if __name__ == "__main__":
    # Example usage:
    # This code will run when the script is executed directly
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Generate complete custom XML format from OMR processor data")
    parser.add_argument("--input", required=True, help="Path to merged detection JSON file")
    parser.add_argument("--staff-lines", required=True, help="Path to staff lines JSON file")
    parser.add_argument("--output", required=True, help="Path to output XML file")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Import OMRProcessor dynamically to avoid circular imports
        from processor import OMRProcessor
        
        # Create processor and process data
        processor = OMRProcessor(args.input, args.staff_lines)
        processor.process()
        
        # Generate and save complete custom XML
        output_path = generate_custom_xml(processor, args.output)
        print(f"Complete custom XML saved to: {output_path}")
        
    except ImportError:
        logger.error("Failed to import OMRProcessor. Make sure it's available in the path.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating complete custom XML: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)