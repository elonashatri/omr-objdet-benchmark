#!/usr/bin/env python3
"""
Custom XML Generator for OMR Pipeline

This module provides functionality to generate a custom XML output format
that matches the structure in the provided ground truth format with
pitch and duration information included.

Usage:
    from custom_xml_generator import generate_custom_xml
    
    # Generate XML from OMRProcessor object
    generate_custom_xml(processor, 'output.xml')
"""

import xml.dom.minidom as minidom
from xml.etree.ElementTree import Element, SubElement, tostring
import os
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger('custom_xml_generator')

def generate_custom_xml(processor, output_path=None):
    """
    Generate custom XML format similar to the provided ground truth structure.
    
    Args:
        processor: OMRProcessor object containing processed music elements
        output_path: Path to save the XML file (if None, returns XML string)
        
    Returns:
        XML string if output_path is None, otherwise saves to file and returns path
    """
    # Create root elements
    root = Element("masks")
    page = SubElement(root, "Page")
    page.set("pageIndex", "0")
    nodes = SubElement(page, "Nodes")
    
    # Track node ID
    node_id = 1
    
    # Add staff lines first
    for system_idx, system in enumerate(processor.staff_systems):
        staff_id = system_idx
        
        # Generate a unique spacing run ID (can be any large number)
        spacing_run_id = str(hash(f"system_{system_idx}") % 10000000000000000000)
        
        # Get staff line positions
        staff_lines = sorted(system.lines.values()) if hasattr(system, 'lines') else []
        
        for line_idx, line_y in enumerate(staff_lines):
            if line_idx >= 5:  # Only process the first 5 lines (standard staff)
                continue
                
            node = SubElement(nodes, "Node")
            
            # Add staff line elements
            id_elem = SubElement(node, "Id")
            id_elem.text = str(100 - line_idx)  # Reverse order like in ground truth
            
            class_elem = SubElement(node, "ClassName")
            class_elem.text = "kStaffLine"
            
            # Determine position - these are estimates
            # In a real implementation, you'd extract this from the actual bbox
            top_elem = SubElement(node, "Top")
            top_elem.text = str(int(line_y))
            
            left_elem = SubElement(node, "Left")
            left_elem.text = "550"  # From ground truth example
            
            width_elem = SubElement(node, "Width")
            width_elem.text = "1750"  # From ground truth example
            
            height_elem = SubElement(node, "Height")
            height_elem.text = "3"  # From ground truth example
            
            # Add mask (simplified)
            mask_elem = SubElement(node, "Mask")
            mask_elem.text = "0: 0 1: 5250 "  # From ground truth example
            
            # Add data section
            data_elem = SubElement(node, "Data")
            
            # Add data items
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
            order_elem.text = str(line_idx + 1)
            
            node_id += 1
    
    # Process clefs
    for system_idx, system in enumerate(processor.staff_systems):
        staff_id = system_idx
        spacing_run_id = str(hash(f"system_{system_idx}") % 10000000000000000000)
        
        # Get clefs for this system
        clefs = [e for e in system.elements if hasattr(e, 'type') and 'clef' in e.class_name.lower()]
        
        for clef_idx, clef in enumerate(clefs):
            node = SubElement(nodes, "Node")
            
            # Add clef elements
            id_elem = SubElement(node, "Id")
            id_elem.text = str(node_id)
            
            class_elem = SubElement(node, "ClassName")
            
            # Map clef type to correct class name
            clef_type_map = {
                "G": "gClef",
                "G8va": "gClef8va",
                "G8vb": "gClef8vb",
                "F": "fClef",
                "F8vb": "fClef8vb",
                "C": "cClef",
            }
            
            clef_class = clef_type_map.get(clef.type, "gClef")  # Default to G clef
            if hasattr(clef, 'class_name'):
                if "8vb" in clef.class_name.lower():
                    clef_class += "8vb"
                elif "8va" in clef.class_name.lower():
                    clef_class += "8va"
                
            class_elem.text = clef_class
            
            # Position (from bbox)
            top_elem = SubElement(node, "Top")
            top_elem.text = str(int(clef.bbox['y1']))
            
            left_elem = SubElement(node, "Left")
            left_elem.text = str(int(clef.bbox['x1']))
            
            width_elem = SubElement(node, "Width")
            width_elem.text = str(int(clef.bbox['width']))
            
            height_elem = SubElement(node, "Height")
            height_elem.text = str(int(clef.bbox['height']))
            
            # Add data section
            data_elem = SubElement(node, "Data")
            
            # Add data items
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "dorico_event_id")
            data_item.set("type", "int")
            data_item.text = str(10000 + node_id)  # Generate unique ID
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "clef_type")
            data_item.set("type", "str")
            data_item.text = f"k{clef.type}Clef"
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "clef_hotspot")
            data_item.set("type", "str")
            if clef.type == "G":
                data_item.text = "G4"
            elif clef.type == "F":
                data_item.text = "F3"
            else:
                data_item.text = "C4"
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "clef_required_stave_lines")
            data_item.set("type", "int")
            data_item.text = "5"
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "clef_stave_position")
            data_item.set("type", "int")
            data_item.text = "2"
            
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
            order_id = 6 + clef_idx  # Start after staff lines
            order_elem = SubElement(node, "order_id")
            order_elem.text = str(order_id)
            
            node_id += 1
    
    # Process key signatures
    for system_idx, system in enumerate(processor.staff_systems):
        staff_id = system_idx
        spacing_run_id = str(hash(f"system_{system_idx}") % 10000000000000000000)
        
        # Get key signature accidentals for this system
        key_signature = getattr(system, 'key_signature', [])
        
        for acc_idx, acc in enumerate(key_signature):
            node = SubElement(nodes, "Node")
            
            # Add accidental elements
            id_elem = SubElement(node, "Id")
            id_elem.text = str(node_id)
            
            class_elem = SubElement(node, "ClassName")
            
            # Map accidental type to class name
            acc_type_map = {
                "flat": "accidentalFlat",
                "sharp": "accidentalSharp",
                "natural": "accidentalNatural",
                # Add other accidental types as needed
            }
            
            class_elem.text = acc_type_map.get(acc.type, "accidentalSharp")
            
            # Position (from bbox)
            top_elem = SubElement(node, "Top")
            top_elem.text = str(int(acc.bbox['y1']))
            
            left_elem = SubElement(node, "Left")
            left_elem.text = str(int(acc.bbox['x1']))
            
            width_elem = SubElement(node, "Width")
            width_elem.text = str(int(acc.bbox['width']))
            
            height_elem = SubElement(node, "Height")
            height_elem.text = str(int(acc.bbox['height']))
            
            # Add data section
            data_elem = SubElement(node, "Data")
            
            # Add data items
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "dorico_event_id")
            data_item.set("type", "int")
            data_item.text = str(11000 + node_id)  # Generate unique ID
            
            # Determine key signature description based on count
            key_count = len(key_signature)
            key_desc = "C Major"  # Default
            if acc.type == "sharp":
                sharp_counts = {1: "G Major", 2: "D Major", 3: "A Major", 4: "E Major", 
                                5: "B Major", 6: "F# Major", 7: "C# Major"}
                key_desc = sharp_counts.get(key_count, "Unknown Major")
            elif acc.type == "flat":
                flat_counts = {1: "F Major", 2: "Bb Major", 3: "Eb Major", 4: "Ab Major", 
                               5: "Db Major", 6: "Gb Major", 7: "Cb Major"}
                key_desc = flat_counts.get(key_count, "Unknown Major")
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "key_signature_description")
            data_item.set("type", "str")
            data_item.text = key_desc
            
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
            order_id = 7 + acc_idx  # Start after clefs
            order_elem = SubElement(node, "order_id")
            order_elem.text = str(order_id)
            
            node_id += 1
    
    # Process time signatures
    for system_idx, system in enumerate(processor.staff_systems):
        staff_id = system_idx
        spacing_run_id = str(hash(f"system_{system_idx}") % 10000000000000000000)
        
        # Get time signature for this system
        time_sig = getattr(system, 'time_signature', None)
        
        if time_sig:
            node = SubElement(nodes, "Node")
            
            # Add time signature elements
            id_elem = SubElement(node, "Id")
            id_elem.text = str(node_id)
            
            class_elem = SubElement(node, "ClassName")
            
            # Determine time signature class name
            if time_sig.beats == 4 and time_sig.beat_type == 4:
                class_elem.text = "timeSigCommon"
            elif time_sig.beats == 2 and time_sig.beat_type == 2:
                class_elem.text = "timeSigCutCommon"
            else:
                class_elem.text = "timeSig"
            
            # For position, find a reasonable spot after key signature
            # In a real implementation, use actual detected positions
            # Let's place it after the last key signature accidental
            x_pos = 700  # Default
            y_pos = 688  # Default (centered on staff)
            
            # Find key signature accidentals to position after them
            key_sigs = [e for e in system.elements if hasattr(e, 'is_key_signature') and e.is_key_signature]
            if key_sigs:
                x_pos = max(e.x for e in key_sigs) + 40
            
            top_elem = SubElement(node, "Top")
            top_elem.text = str(int(y_pos))
            
            left_elem = SubElement(node, "Left")
            left_elem.text = str(int(x_pos))
            
            width_elem = SubElement(node, "Width")
            width_elem.text = "35"  # Default from ground truth
            
            height_elem = SubElement(node, "Height")
            height_elem.text = "42"  # Default from ground truth
            
            # Add data section
            data_elem = SubElement(node, "Data")
            
            # Add data items
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "dorico_event_id")
            data_item.set("type", "int")
            data_item.text = str(11500 + node_id)  # Generate unique ID
            
            # Build time signature description
            time_desc = f"{time_sig.beats}/{time_sig.beat_type} (q, "
            time_desc += "+".join(["1"] * time_sig.beats) + ")"
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "time_signature_description")
            data_item.set("type", "str")
            data_item.text = time_desc
            
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
            order_id = 9  # After key signatures based on ground truth
            order_elem = SubElement(node, "order_id")
            order_elem.text = str(order_id)
            
            node_id += 1
    
    # Process notes and calculate their onset times
    note_onset_map = {}  # To track onset times for each note
    
    # First, calculate onset times for each note
    for system_idx, system in enumerate(processor.staff_systems):
        cumulative_onset = 0.0
        
        # Process each measure
        for measure_idx, measure in enumerate(system.measures):
            # Get notes and rests in this measure, sorted by x-position
            elements = [e for e in measure.elements if hasattr(e, 'duration')]
            elements.sort(key=lambda e: e.x)
            
            # Process elements
            for elem in elements:
                # Store onset time
                note_onset_map[elem] = cumulative_onset
                
                # Add duration to onset - with proper handling of None values
                if hasattr(elem, 'duration') and elem.duration is not None:
                    cumulative_onset += elem.duration
                else:
                    # Default to quarter note duration if no duration specified
                    cumulative_onset += 1.0
    
    # Now process notes with onset information
    for system_idx, system in enumerate(processor.staff_systems):
        staff_id = system_idx
        spacing_run_id = str(hash(f"system_{system_idx}") % 10000000000000000000)
        
        # Get notes for this system
        notes = [e for e in system.elements if hasattr(e, 'step') and hasattr(e, 'octave')]
        
        for note_idx, note in enumerate(notes):
            node = SubElement(nodes, "Node")
            
            # Add note elements
            id_elem = SubElement(node, "Id")
            id_elem.text = str(note_idx + 1)  # Start notes with ID 1
            
            class_elem = SubElement(node, "ClassName")
            
            # Map note type to class name
            note_type_map = {
                'quarter': 'noteheadBlack',
                'eighth': 'noteheadBlack',
                '16th': 'noteheadBlack',
                '32nd': 'noteheadBlack',
                'half': 'noteheadHalf',
                'whole': 'noteheadWhole'
            }
            
            class_elem.text = note_type_map.get(getattr(note, 'duration_type', 'quarter'), 'noteheadBlack')
            
            # Position (from bbox)
            top_elem = SubElement(node, "Top")
            top_elem.text = str(int(note.bbox['y1']))
            
            left_elem = SubElement(node, "Left")
            left_elem.text = str(int(note.bbox['x1']))
            
            width_elem = SubElement(node, "Width")
            width_elem.text = str(int(note.bbox['width']))
            
            height_elem = SubElement(node, "Height")
            height_elem.text = str(int(note.bbox['height']))
            
            # Add outlinks (if note is connected to a stem)
            if hasattr(note, 'stem') and note.stem:
                outlinks_elem = SubElement(node, "Outlinks")
                outlinks_elem.text = str(note.stem.id if hasattr(note.stem, 'id') else 54)  # Default if no ID
            
            # Add data section
            data_elem = SubElement(node, "Data")
            
            # Add duration data - with safe handling of None values
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "duration_beats")
            data_item.set("type", "float")
            duration = 1.0  # Default to quarter note
            if hasattr(note, 'duration') and note.duration is not None:
                duration = note.duration
            data_item.text = f"{duration:.6f}"
            
            # Add onset data (if available)
            onset = note_onset_map.get(note, 0.0)
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "onset_beats")
            data_item.set("type", "float")
            data_item.text = f"{onset:.6f}"
            
            # Add pitch data
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "pitch_octave")
            data_item.set("type", "int")
            data_item.text = str(note.octave)
            
            # Calculate MIDI pitch
            midi_pitch = 0
            if hasattr(note, 'step') and hasattr(note, 'octave'):
                # Convert step to numeric value (C=0, D=2, E=4, etc.)
                step_values = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
                step_value = step_values.get(note.step, 0)
                
                # Calculate MIDI pitch (C4 = 60)
                midi_pitch = (note.octave + 1) * 12 + step_value
                
                # Apply accidental if present
                if hasattr(note, 'alter'):
                    midi_pitch += note.alter
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "midi_pitch_code")
            data_item.set("type", "int")
            data_item.text = str(midi_pitch)
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "normalized_pitch_step")
            data_item.set("type", "str")
            data_item.text = note.step
            
            data_item = SubElement(data_elem, "DataItem")
            data_item.set("key", "dorico_event_id")
            data_item.set("type", "int")
            data_item.text = str(18000 + note_idx)  # Generate unique ID
            
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
            order_id = 10 + note_idx  # Start after all other elements
            order_elem = SubElement(node, "order_id")
            order_elem.text = str(order_id)
            
            node_id += 1
    
    # Convert to string with pretty formatting
    xml_string = minidom.parseString(tostring(root, 'utf-8')).toprettyxml(indent="  ")
    
    # Save to file if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_string)
        logger.info(f"Custom XML saved to {output_path}")
        return output_path
    
    return xml_string


if __name__ == "__main__":
    # Example usage:
    # This code will run when the script is executed directly
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Generate custom XML format from OMR processor data")
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
        
        # Generate and save custom XML
        output_path = generate_custom_xml(processor, args.output)
        print(f"Custom XML saved to: {output_path}")
        
    except ImportError:
        logger.error("Failed to import OMRProcessor. Make sure it's available in the path.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating custom XML: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)