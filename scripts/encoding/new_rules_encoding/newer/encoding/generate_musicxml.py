import xml.etree.ElementTree as ET
# Import music element classes
try:
    from music_elements import (
        Note, Rest, Accidental, Clef, Barline, Beam, Flag, 
        TimeSignatureElement, KeySignature, TimeSignature
    )
except ImportError:
    print("WARNING: Could not import music_elements module. XML generation may fail.")
    
# Import staff system classes if needed
try:
    from staff_system import StaffSystem, Measure
except ImportError:
    print("WARNING: Could not import staff_system module. XML generation may fail.")

def _is_single_instrument_score(staff_systems, clefs):
    """
    Determine if this is likely a single-instrument score by analyzing:
    1. Number of staves per system
    2. Clef consistency
    3. Presence of instrument-specific identifiers
    
    Returns: True if likely a single-instrument score, False otherwise
    """
    # Look for explicit instrument indicators
    instrument_types = []
    
    # Simple test: if all systems have a single staff, it's likely a single-instrument score
    total_systems = len(staff_systems)
    if total_systems == 0:
        return True  # Default to single instrument for empty score
    
    # Count systems with one staff vs. multiple staves
    single_staff_systems = 0
    
    for system in staff_systems:
        # Check number of staves in this system
        if hasattr(system, 'staves') and system.staves:
            num_staves = len(system.staves)
            if num_staves <= 1:
                single_staff_systems += 1
    
    # If most systems have one staff, it's likely a single-instrument score
    ratio = single_staff_systems / total_systems if total_systems > 0 else 1.0
    
    if ratio > 0.8:  # 80% or more have single staff
        return True
    
    # Check clef consistency across systems
    # For single-instrument scores, clef should be similar across systems
    clef_types = []
    for system in staff_systems:
        if hasattr(system, 'clef') and system.clef:
            clef_types.append(system.clef.type)
    
    # If most or all clefs are the same type, it's likely single instrument
    if clef_types:
        most_common = max(set(clef_types), key=clef_types.count)
        consistency = clef_types.count(most_common) / len(clef_types)
        if consistency > 0.8:
            return True
    
    # Default to single instrument if we can't determine clearly
    return True



def encode_clef_to_xml(attrs_element, clef_type, clef_line=None):
    """
    Encode a clef in MusicXML format with proper handling for all clef types.
    
    Args:
        attrs_element: The parent attributes XML element
        clef_type: The type of clef (G, F, C, G8va, etc.)
        clef_line: The staff line for the clef (if not specified, use default)
        
    Returns:
        The created clef element
    """
    clef_elem = ET.SubElement(attrs_element, 'clef')
    
    # Standard G clef (treble)
    if clef_type in ["G", "gClef"]:
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "G"
        line = ET.SubElement(clef_elem, 'line')
        line.text = "2" if clef_line is None else str(clef_line)
    
    # G clef octave up (treble 8va)
    elif clef_type in ["G8va", "gClef8va"]:
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "G"
        line = ET.SubElement(clef_elem, 'line')
        line.text = "2" if clef_line is None else str(clef_line)
        octave_change = ET.SubElement(clef_elem, 'clef-octave-change')
        octave_change.text = "1"  # Sounds one octave higher
    
    # G clef octave down (treble 8vb)
    elif clef_type in ["G8vb", "gClef8vb"]:
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "G"
        line = ET.SubElement(clef_elem, 'line')
        line.text = "2" if clef_line is None else str(clef_line)
        octave_change = ET.SubElement(clef_elem, 'clef-octave-change')
        octave_change.text = "-1"  # Sounds one octave lower
    
    # Standard F clef (bass)
    elif clef_type in ["F", "fClef"]:
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "F"
        line = ET.SubElement(clef_elem, 'line')
        line.text = "4" if clef_line is None else str(clef_line)
    
    # F clef octave down (bass 8vb)
    elif clef_type in ["F8vb", "fClef8vb"]:
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "F"
        line = ET.SubElement(clef_elem, 'line')
        line.text = "4" if clef_line is None else str(clef_line)
        octave_change = ET.SubElement(clef_elem, 'clef-octave-change')
        octave_change.text = "-1"  # Sounds one octave lower
    
    # C clef (alto, tenor, soprano)
    elif clef_type in ["C", "cClef"]:
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "C"
        line = ET.SubElement(clef_elem, 'line')
        # Default to alto clef if line not specified
        line.text = "3" if clef_line is None else str(clef_line)
    
    # C clef - tenor
    elif clef_type in ["cClefTenor"]:
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "C"
        line = ET.SubElement(clef_elem, 'line')
        line.text = "4" if clef_line is None else str(clef_line)
    
    # C clef - soprano
    elif clef_type in ["cClefSoprano"]:
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "C"
        line = ET.SubElement(clef_elem, 'line')
        line.text = "1" if clef_line is None else str(clef_line)
    
    # Percussion clef
    elif clef_type in ["percussion", "unpitchedPercussionClef1"]:
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "percussion"
        line = ET.SubElement(clef_elem, 'line')
        line.text = "2" if clef_line is None else str(clef_line)
    
    # Default to G clef if unknown
    else:
        print(f"Warning: Unknown clef type '{clef_type}'. Using G clef.")
        sign = ET.SubElement(clef_elem, 'sign')
        sign.text = "G"
        line = ET.SubElement(clef_elem, 'line')
        line.text = "2" if clef_line is None else str(clef_line)
    
    return clef_elem

def generate_musicxml(processor):
    """Generate MusicXML from the processed musical elements with improved handling for single-instrument scores."""

    # Extract data from processor
    staff_systems = processor.staff_systems
    notes = processor.notes
    rests = processor.rests
    accidentals = processor.accidentals
    clefs = processor.clefs
    barlines = processor.barlines
    beams = processor.beams
    flags = processor.flags
    time_signature_elements = processor.time_signature_elements
    key_signatures = processor.key_signatures
    measures = processor.measures
    
    # Create XML structure
    score_partwise = ET.Element('score-partwise', version='4.0')
    
    # Add part-list
    part_list = ET.SubElement(score_partwise, 'part-list')
    
    # Check if single instrument using the local helper function
    is_single_instrument = _is_single_instrument_score(staff_systems, clefs)
    

    if is_single_instrument:
        # For single-instrument score: create just one part
        print("Detected single-instrument score - generating with one part")
        score_part = ET.SubElement(part_list, 'score-part', id='P1')
        part_name = ET.SubElement(score_part, 'part-name')
        part_name.text = 'Classical Guitar'  # Can be customized or extracted from metadata
        
        # Create single part with all measures from all systems
        part = ET.SubElement(score_partwise, 'part', id='P1')
        
        # Track current state across all systems
        current_clef_type = None
        current_clef_line = None
        current_key_fifths = None
        current_time_sig = None
        
        # Measure number counter
        measure_num = 1
        
        # Process each system as a sequence of measures in the same part
        for system_idx, system in enumerate(staff_systems):
            # Add measures for this system
            for j, measure in enumerate(system.measures):
                measure_elem = ET.SubElement(part, 'measure', number=str(measure_num))
                measure_num += 1
                
                # Determine if we need to include attributes
                include_attributes = False
                
                # Calculate key signature fifths value
                key_fifths = 0
                if system.key_signature:
                    num_sharps = sum(1 for acc in system.key_signature if acc.type == 'sharp')
                    num_flats = sum(1 for acc in system.key_signature if acc.type == 'flat')
                    if num_sharps > 0:
                        key_fifths = num_sharps
                    elif num_flats > 0:
                        key_fifths = -num_flats
                
                # Get current clef type
                clef_type = None
                clef_line = None
                if system.clef:
                    clef_type = system.clef.type
                    clef_line = system.clef.line
                
                # Get time signature
                time_sig = system.time_signature
                
                # First measure or first measure of a new system gets attributes
                if measure_num == 2 or clef_type != current_clef_type or clef_line != current_clef_line or key_fifths != current_key_fifths or time_sig != current_time_sig:
                    include_attributes = True
                    current_clef_type = clef_type
                    current_clef_line = clef_line
                    current_key_fifths = key_fifths
                    current_time_sig = time_sig
                
                # Add attributes if needed
                if include_attributes:
                    attrs = ET.SubElement(measure_elem, 'attributes')
                    
                    # Add divisions (time base)
                    divisions = ET.SubElement(attrs, 'divisions')
                    divisions.text = '4'  # Quarter note = 4 divisions
                    
                    # Add key signature
                    key = ET.SubElement(attrs, 'key')
                    fifths = ET.SubElement(key, 'fifths')
                    fifths.text = str(key_fifths)
                    
                    # Add time signature if present
                    if time_sig:
                        time = ET.SubElement(attrs, 'time')
                        beats = ET.SubElement(time, 'beats')
                        beats.text = str(time_sig.beats)
                        beat_type = ET.SubElement(time, 'beat-type')
                        beat_type.text = str(time_sig.beat_type)
                    
                    # # Add clef if present
                    # if clef_type and clef_line:
                    #     clef_elem = ET.SubElement(attrs, 'clef')
                    #     sign = ET.SubElement(clef_elem, 'sign')
                    #     sign.text = clef_type
                    #     line = ET.SubElement(clef_elem, 'line')
                    #     line.text = str(clef_line)
                    if clef_type:
                        encode_clef_to_xml(attrs, clef_type, clef_line)
                # Add notes, rests, and other elements
                # Get all elements in this measure
                all_measure_elements = measure.elements
                
                # Sort elements by x-position
                all_measure_elements.sort(key=lambda e: e.x)
                
                for elem in all_measure_elements:
                    if isinstance(elem, Note):
                        # Skip notes that are part of a chord (except the first one)
                        if elem.is_chord_member and elem.chord[0] != elem:
                            continue
                            
                        # Create note element
                        note_elem = ET.SubElement(measure_elem, 'note')
                        
                        # If this is part of a chord, handle accordingly
                        if elem.is_chord_member and len(elem.chord) > 1:
                            # Add pitch for the first note
                            if elem.step and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))  # Convert to divisions
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add accidental if present and not part of key signature
                            if elem.accidental and not elem.accidental.is_key_signature:
                                acc_elem = ET.SubElement(note_elem, 'accidental')
                                acc_elem.text = elem.accidental.type
                            
                            # Add other chord notes
                            for chord_note in elem.chord[1:]:
                                # Create note element for chord note
                                chord_note_elem = ET.SubElement(measure_elem, 'note')
                                
                                # Add chord tag
                                ET.SubElement(chord_note_elem, 'chord')
                                
                                # Add pitch
                                if chord_note.step and chord_note.octave is not None:
                                    pitch_elem = chord_note.position_to_xml()
                                    chord_note_elem.append(pitch_elem)
                                
                                # Add duration
                                duration = ET.SubElement(chord_note_elem, 'duration')
                                duration.text = str(int(4 * chord_note.duration))
                                
                                # Add type
                                type_elem = ET.SubElement(chord_note_elem, 'type')
                                type_elem.text = chord_note.duration_type
                                
                                # Add accidental if present and not part of key signature
                                if chord_note.accidental and not chord_note.accidental.is_key_signature:
                                    acc_elem = ET.SubElement(chord_note_elem, 'accidental')
                                    acc_elem.text = chord_note.accidental.type
                        else:
                            # Regular note (not part of a chord)
                            # Add pitch
                            if elem.step and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add accidental if present and not part of key signature
                            if elem.accidental and not elem.accidental.is_key_signature:
                                acc_elem = ET.SubElement(note_elem, 'accidental')
                                acc_elem.text = elem.accidental.type
                    
                    elif isinstance(elem, Rest):
                        # Create rest element
                        rest_elem = ET.SubElement(measure_elem, 'note')
                        ET.SubElement(rest_elem, 'rest')
                        
                        # Add duration
                        duration = ET.SubElement(rest_elem, 'duration')
                        duration.text = str(int(4 * elem.duration))
                        
                        # Add type
                        type_elem = ET.SubElement(rest_elem, 'type')
                        type_elem.text = elem.duration_type
    else:
        # For multi-instrument scores: maintain current behavior with multiple parts
        print("Detected multi-instrument score - generating with multiple parts")
        # Create score-part elements (one per staff system)
        for i, system in enumerate(staff_systems):
            score_part = ET.SubElement(part_list, 'score-part', id=f'P{i+1}')
            part_name = ET.SubElement(score_part, 'part-name')
            part_name.text = f'Part {i+1}'
        
        # Create parts
        for i, system in enumerate(staff_systems):
            part = ET.SubElement(score_partwise, 'part', id=f'P{i+1}')
            
            # Track current state
            current_clef_type = None
            current_clef_line = None
            current_key_fifths = None
            current_time_sig = None
            
            # Add measures
            for j, measure in enumerate(system.measures):
                measure_elem = ET.SubElement(part, 'measure', number=str(j+1))
                
                # Determine if we need to include attributes
                include_attributes = False
                
                # Calculate key signature fifths value
                key_fifths = 0
                if system.key_signature:
                    num_sharps = sum(1 for acc in system.key_signature if acc.type == 'sharp')
                    num_flats = sum(1 for acc in system.key_signature if acc.type == 'flat')
                    if num_sharps > 0:
                        key_fifths = num_sharps
                    elif num_flats > 0:
                        key_fifths = -num_flats
                
                # Get current clef type
                clef_type = None
                clef_line = None
                if system.clef:
                    clef_type = system.clef.type
                    clef_line = system.clef.line
                
                # Get time signature
                time_sig = system.time_signature
                
                # First measure always gets attributes
                if j == 0:
                    include_attributes = True
                    current_clef_type = clef_type
                    current_clef_line = clef_line
                    current_key_fifths = key_fifths
                    current_time_sig = time_sig
                else:
                    # Check for clef changes
                    if clef_type != current_clef_type or clef_line != current_clef_line:
                        include_attributes = True
                        current_clef_type = clef_type
                        current_clef_line = clef_line
                    
                    # Check for key signature changes
                    if key_fifths != current_key_fifths:
                        include_attributes = True
                        current_key_fifths = key_fifths
                    
                    # Check for time signature changes
                    if time_sig != current_time_sig:
                        include_attributes = True
                        current_time_sig = time_sig
                
                # Add attributes if needed
                if include_attributes:
                    attrs = ET.SubElement(measure_elem, 'attributes')
                    
                    # Add divisions (time base)
                    divisions = ET.SubElement(attrs, 'divisions')
                    divisions.text = '4'  # Quarter note = 4 divisions
                    
                    # Add key signature
                    key = ET.SubElement(attrs, 'key')
                    fifths = ET.SubElement(key, 'fifths')
                    fifths.text = str(key_fifths)
                    
                    # Add time signature if present
                    if time_sig:
                        time = ET.SubElement(attrs, 'time')
                        beats = ET.SubElement(time, 'beats')
                        beats.text = str(time_sig.beats)
                        beat_type = ET.SubElement(time, 'beat-type')
                        beat_type.text = str(time_sig.beat_type)
                    
                    # # Add clef if present
                    # if clef_type and clef_line:
                    #     clef_elem = ET.SubElement(attrs, 'clef')
                    #     sign = ET.SubElement(clef_elem, 'sign')
                    #     sign.text = clef_type
                    #     line = ET.SubElement(clef_elem, 'line')
                    #     line.text = str(clef_line)
                    if clef_type:
                        encode_clef_to_xml(attrs, clef_type, clef_line)
                
                # Add notes, rests, and other elements
                # Get all elements in this measure
                all_measure_elements = measure.elements
                
                # Sort elements by x-position
                all_measure_elements.sort(key=lambda e: e.x)
                
                for elem in all_measure_elements:
                    if isinstance(elem, Note):
                        # Skip notes that are part of a chord (except the first one)
                        if elem.is_chord_member and elem.chord[0] != elem:
                            continue
                            
                        # Create note element
                        note_elem = ET.SubElement(measure_elem, 'note')
                        
                        # If this is part of a chord, handle accordingly
                        if elem.is_chord_member and len(elem.chord) > 1:
                            # Add pitch for the first note
                            if elem.step and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))  # Convert to divisions
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add accidental if present and not part of key signature
                            if elem.accidental and not elem.accidental.is_key_signature:
                                acc_elem = ET.SubElement(note_elem, 'accidental')
                                acc_elem.text = elem.accidental.type
                            
                            # Add other chord notes
                            for chord_note in elem.chord[1:]:
                                # Create note element for chord note
                                chord_note_elem = ET.SubElement(measure_elem, 'note')
                                
                                # Add chord tag
                                ET.SubElement(chord_note_elem, 'chord')
                                
                                # Add pitch
                                if chord_note.step and chord_note.octave is not None:
                                    pitch_elem = chord_note.position_to_xml()
                                    chord_note_elem.append(pitch_elem)
                                
                                # Add duration
                                duration = ET.SubElement(chord_note_elem, 'duration')
                                duration.text = str(int(4 * chord_note.duration))
                                
                                # Add type
                                type_elem = ET.SubElement(chord_note_elem, 'type')
                                type_elem.text = chord_note.duration_type
                                
                                # Add accidental if present and not part of key signature
                                if chord_note.accidental and not chord_note.accidental.is_key_signature:
                                    acc_elem = ET.SubElement(chord_note_elem, 'accidental')
                                    acc_elem.text = chord_note.accidental.type
                        else:
                            # Regular note (not part of a chord)
                            # Add pitch
                            if elem.step and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add accidental if present and not part of key signature
                            if elem.accidental and not elem.accidental.is_key_signature:
                                acc_elem = ET.SubElement(note_elem, 'accidental')
                                acc_elem.text = elem.accidental.type
                    
                    elif isinstance(elem, Rest):
                        # Create rest element
                        rest_elem = ET.SubElement(measure_elem, 'note')
                        ET.SubElement(rest_elem, 'rest')
                        
                        # Add duration
                        duration = ET.SubElement(rest_elem, 'duration')
                        duration.text = str(int(4 * elem.duration))
                        
                        # Add type
                        type_elem = ET.SubElement(rest_elem, 'type')
                        type_elem.text = elem.duration_type
    
    score_xml = ET.tostring(score_partwise, encoding='utf-8')
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    doc_type = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
    return xml_declaration + doc_type + score_xml.decode('utf-8')
