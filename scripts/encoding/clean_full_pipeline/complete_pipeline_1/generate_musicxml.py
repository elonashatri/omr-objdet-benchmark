import xml.etree.ElementTree as ET
# Import music element classes
try:
    from music_elements import (
        Note, Rest, Accidental, Clef, Barline, Beam, Flag, 
        TimeSignatureElement, KeySignature, TimeSignature,
        Tie, Slur, Dynamic, GradualDynamic, Tuplet, TupletBracket, TupletText,
        Articulation, Ornament, AugmentationDot, TextDirection
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
    # Count systems with different staff counts
    staff_counts = set()
    for system in staff_systems:
        if hasattr(system, 'staves') and system.staves:
            staff_counts.add(len(system.staves))
    
    # If we have mixed staff counts, this is not a single-instrument score
    if len(staff_counts) > 1:
        print(f"Detected mixed staff counts {staff_counts} - treating as multi-part score")
        return False
    
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
    
    # Check for systemic barlines which indicate multi-staff systems
    has_systemic_barlines = any(
        hasattr(system, 'elements') and 
        any(e.__class__.__name__ == 'Barline' and 
            hasattr(e, 'is_systemic') and e.is_systemic 
            for e in system.elements)
        for system in staff_systems
    )
    
    if has_systemic_barlines:
        print("Detected systemic barlines - treating as multi-part score")
        return False
    
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

def add_notations_to_note(note_elem, elem):
    """
    Add notation elements to a note (ties, slurs, articulations, ornaments, etc.)
    
    Args:
        note_elem: The XML note element
        elem: The Note object with notation properties
    """
    # Check if we need to add notations
    has_notations = False
    
    # Check for ties
    has_tie_start = hasattr(elem, 'tie_start') and elem.tie_start
    has_tie_end = hasattr(elem, 'tie_end') and elem.tie_end
    
    # Check for slurs
    has_slur_starts = hasattr(elem, 'slur_starts') and elem.slur_starts
    has_slur_ends = hasattr(elem, 'slur_ends') and elem.slur_ends
    
    # Check for articulations
    has_articulations = hasattr(elem, 'articulations') and elem.articulations
    
    # Check for tuplet notations
    is_tuplet = hasattr(elem, 'is_tuplet') and elem.is_tuplet
    tuplet_start = hasattr(elem, 'tuplet_start') and elem.tuplet_start
    tuplet_end = hasattr(elem, 'tuplet_end') and elem.tuplet_end
    
    # Check for ornaments
    has_ornaments = hasattr(elem, 'ornaments') and elem.ornaments
    
    # Check for fermata
    has_fermata = hasattr(elem, 'fermata') and elem.fermata
    
    # Check for beams
    has_beams = hasattr(elem, 'beam_types') and elem.beam_types
    
    # Create notations element if needed
    if (has_tie_start or has_tie_end or has_slur_starts or has_slur_ends or 
        has_articulations or (is_tuplet and (tuplet_start or tuplet_end)) or 
        has_ornaments or has_fermata):
        
        notations = ET.SubElement(note_elem, 'notations')
        
        # Add tie notations
        if has_tie_start:
            ET.SubElement(notations, 'tied', type='start')
        if has_tie_end:
            ET.SubElement(notations, 'tied', type='stop')
        
        # Add slur notations
        if has_slur_starts:
            for i, slur_id in enumerate(elem.slur_starts):
                ET.SubElement(notations, 'slur', type='start', number=str(slur_id))
        if has_slur_ends:
            for i, slur_id in enumerate(elem.slur_ends):
                ET.SubElement(notations, 'slur', type='stop', number=str(slur_id))
        
        # Add articulations
        if has_articulations:
            articulations = ET.SubElement(notations, 'articulations')
            for art in elem.articulations:
                art_type = art.type if hasattr(art, 'type') else art
                ET.SubElement(articulations, art_type)
                
                # If the articulation has a placement, set it
                if hasattr(art, 'placement'):
                    articulations[-1].set('placement', art.placement)
        
        # Add tuplet notation
        if is_tuplet:
            if tuplet_start:
                ET.SubElement(notations, 'tuplet', type='start', number='1')
            if tuplet_end:
                ET.SubElement(notations, 'tuplet', type='stop', number='1')
        
        # Add ornaments
        if has_ornaments:
            ornaments = ET.SubElement(notations, 'ornaments')
            for orn in elem.ornaments:
                orn_type = orn.type if hasattr(orn, 'type') else orn
                ET.SubElement(ornaments, orn_type)
        
        # Add fermata
        if has_fermata:
            ET.SubElement(notations, 'fermata')
    
    # Add beam elements outside notations
    if has_beams:
        for beam_type, level in elem.beam_types:
            beam = ET.SubElement(note_elem, 'beam', number=str(level))
            beam.text = beam_type

def add_tie_elements(note_elem, elem):
    """Add tie elements to a note"""
    if hasattr(elem, 'tie_start') and elem.tie_start:
        ET.SubElement(note_elem, 'tie', type='start')
    if hasattr(elem, 'tie_end') and elem.tie_end:
        ET.SubElement(note_elem, 'tie', type='stop')

def add_time_modification(note_elem, elem):
    """Add time modification for tuplets"""
    if hasattr(elem, 'is_tuplet') and elem.is_tuplet and hasattr(elem, 'tuplet_data'):
        actual, normal = elem.tuplet_data
        time_modification = ET.SubElement(note_elem, 'time-modification')
        actual_notes = ET.SubElement(time_modification, 'actual-notes')
        actual_notes.text = str(actual)
        normal_notes = ET.SubElement(time_modification, 'normal-notes')
        normal_notes.text = str(normal)

def add_dynamic_to_measure(measure_elem, dynamic, placement='below'):
    """Add a dynamic marking to a measure"""
    direction = ET.SubElement(measure_elem, 'direction', placement=placement)
    direction_type = ET.SubElement(direction, 'direction-type')
    dynamics = ET.SubElement(direction_type, 'dynamics')
    
    # Add the specific dynamic mark
    dynamic_type = dynamic.type if hasattr(dynamic, 'type') else dynamic
    ET.SubElement(dynamics, dynamic_type)
    
    # Add positioning if available
    if hasattr(dynamic, 'default_x'):
        direction.set('default-x', str(dynamic.default_x))
    if hasattr(dynamic, 'default_y'):
        direction.set('default-y', str(dynamic.default_y))
    
    # Add staff assignment if available
    if hasattr(dynamic, 'staff'):
        staff = ET.SubElement(direction, 'staff')
        staff.text = str(dynamic.staff)

def add_gradual_dynamic_to_measure(measure_elem, dynamic, placement='below'):
    """Add a gradual dynamic (crescendo/diminuendo) to a measure"""
    direction = ET.SubElement(measure_elem, 'direction', placement=placement)
    direction_type = ET.SubElement(direction, 'direction-type')
    
    # Create wedge element
    wedge_type = "crescendo" if dynamic.type == "crescendo" else "diminuendo"
    wedge = ET.SubElement(direction_type, 'wedge', type=wedge_type)
    
    # Add positioning if available
    if hasattr(dynamic, 'default_x'):
        direction.set('default-x', str(dynamic.default_x))
    if hasattr(dynamic, 'default_y'):
        direction.set('default-y', str(dynamic.default_y))
    
    # Add staff assignment if available
    if hasattr(dynamic, 'staff'):
        staff = ET.SubElement(direction, 'staff')
        staff.text = str(dynamic.staff)
        
    # If this is the end of a gradual dynamic, add a stop direction
    if wedge_type == "stop":
        stop_direction = ET.SubElement(measure_elem, 'direction', placement=placement)
        stop_direction_type = ET.SubElement(stop_direction, 'direction-type')
        ET.SubElement(stop_direction_type, 'wedge', type="stop")

def add_text_direction_to_measure(measure_elem, text_dir, placement='above'):
    """Add a text direction to a measure"""
    direction = ET.SubElement(measure_elem, 'direction', placement=placement)
    direction_type = ET.SubElement(direction, 'direction-type')
    text_element = ET.SubElement(direction_type, 'words')
    text_element.text = text_dir.text
    
    # Add positioning if available
    if hasattr(text_dir, 'default_x'):
        direction.set('default-x', str(text_dir.default_x))
    if hasattr(text_dir, 'default_y'):
        direction.set('default-y', str(text_dir.default_y))
    
    # Add staff assignment if available
    if hasattr(text_dir, 'staff'):
        staff = ET.SubElement(direction, 'staff')
        staff.text = str(text_dir.staff)

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
    
    # Extract new element types if available
    ties = getattr(processor, 'ties', [])
    slurs = getattr(processor, 'slurs', [])
    dynamics = getattr(processor, 'dynamics', [])
    gradual_dynamics = getattr(processor, 'gradual_dynamics', [])
    tuplets = getattr(processor, 'tuplets', [])
    tuplet_brackets = getattr(processor, 'tuplet_brackets', [])
    tuplet_texts = getattr(processor, 'tuplet_texts', [])
    articulations = getattr(processor, 'articulations', [])
    ornaments = getattr(processor, 'ornaments', [])
    augmentation_dots = getattr(processor, 'augmentation_dots', [])
    text_directions = getattr(processor, 'text_directions', [])

    print("\nDEBUG: Staff system structure for MusicXML generation:")
    for i, system in enumerate(staff_systems):
        is_multi = hasattr(system, 'is_multi_staff') and system.is_multi_staff
        num_staves = len(system.staves) if hasattr(system, 'staves') else 0
        print(f"  System {i}: multi-staff={is_multi}, staves={num_staves}")

    # Check if we have any multi-staff systems
    has_multi_staff_systems = any(hasattr(system, 'is_multi_staff') and system.is_multi_staff 
                                for system in staff_systems)

    # Create XML structure
    score_partwise = ET.Element('score-partwise', version='4.0')
    
    # Determine if this is a single-instrument score
    is_single_instrument = _is_single_instrument_score(staff_systems, clefs)
    
    # Add this debug statement here
    print(f"Final decision: is_single_instrument={is_single_instrument}")
    
    # Add part-list
    part_list = ET.SubElement(score_partwise, 'part-list')
    
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
                    
                    if clef_type:
                        encode_clef_to_xml(attrs, clef_type, clef_line)
                
                # Add dynamics for this measure (if any)
                measure_dynamics = [d for d in dynamics if hasattr(d, 'measure') and d.measure == measure]
                for dynamic in measure_dynamics:
                    add_dynamic_to_measure(measure_elem, dynamic)
                    print(f"Added {dynamic.type} dynamic to measure {j+1}")
                
                # Add gradual dynamics for this measure (if any)
                measure_gradual_dynamics = [d for d in gradual_dynamics if hasattr(d, 'measure') and d.measure == measure]
                for gradual in measure_gradual_dynamics:
                    add_gradual_dynamic_to_measure(measure_elem, gradual)
                    print(f"Added {gradual.type} gradual dynamic to measure {j+1}")
                
                # Add text directions for this measure (if any)
                measure_text_directions = [t for t in text_directions if hasattr(t, 'measure') and t.measure == measure]
                for text_dir in measure_text_directions:
                    add_text_direction_to_measure(measure_elem, text_dir, text_dir.placement if hasattr(text_dir, 'placement') else 'above')
                
                # Add notes, rests, and other elements
                # Get all elements in this measure
                all_measure_elements = measure.elements
                
                # Sort elements by x-position
                all_measure_elements.sort(key=lambda e: e.x)
                
                for elem in all_measure_elements:
                    if isinstance(elem, Note):
                        # Skip notes that are part of a chord (except the first one)
                        if hasattr(elem, 'is_chord_member') and elem.is_chord_member and hasattr(elem, 'chord') and elem.chord[0] != elem:
                            continue
                            
                        # Create note element
                        note_elem = ET.SubElement(measure_elem, 'note')
                        
                        # If this is part of a chord, handle accordingly
                        if hasattr(elem, 'is_chord_member') and elem.is_chord_member and hasattr(elem, 'chord') and len(elem.chord) > 1:
                            # Add pitch for the first note
                            if hasattr(elem, 'step') and elem.step and hasattr(elem, 'octave') and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))  # Convert to divisions
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add accidental if present and not part of key signature
                            if hasattr(elem, 'accidental') and elem.accidental and not elem.accidental.is_key_signature:
                                acc_elem = ET.SubElement(note_elem, 'accidental')
                                acc_elem.text = elem.accidental.type
                            
                            # Add ties, time modification, and notations for first note
                            add_tie_elements(note_elem, elem)
                            add_time_modification(note_elem, elem)
                            add_notations_to_note(note_elem, elem)
                            
                            # Add other chord notes
                            for chord_note in elem.chord[1:]:
                                # Create note element for chord note
                                chord_note_elem = ET.SubElement(measure_elem, 'note')
                                
                                # Add chord tag
                                ET.SubElement(chord_note_elem, 'chord')
                                
                                # Add pitch
                                if hasattr(chord_note, 'step') and chord_note.step and hasattr(chord_note, 'octave') and chord_note.octave is not None:
                                    pitch_elem = chord_note.position_to_xml()
                                    chord_note_elem.append(pitch_elem)
                                
                                # Add duration
                                duration = ET.SubElement(chord_note_elem, 'duration')
                                duration.text = str(int(4 * chord_note.duration))
                                
                                # Add type
                                type_elem = ET.SubElement(chord_note_elem, 'type')
                                type_elem.text = chord_note.duration_type
                                
                                # Add dots if present
                                if hasattr(chord_note, 'augmentation_dots') and chord_note.augmentation_dots > 0:
                                    for i in range(chord_note.augmentation_dots):
                                        ET.SubElement(chord_note_elem, 'dot')
                                
                                # Add accidental if present and not part of key signature
                                if hasattr(chord_note, 'accidental') and chord_note.accidental:
                                    if not hasattr(chord_note.accidental, 'is_key_signature') or not chord_note.accidental.is_key_signature:
                                        acc_elem = ET.SubElement(chord_note_elem, 'accidental')
                                        acc_elem.text = chord_note.accidental.type
                                                                
                                # Add ties, time modification, and notations for chord notes
                                add_tie_elements(chord_note_elem, chord_note)
                                add_time_modification(chord_note_elem, chord_note)
                                add_notations_to_note(chord_note_elem, chord_note)
                        else:
                            # Regular note (not part of a chord)
                            # Add pitch
                            if hasattr(elem, 'step') and elem.step and hasattr(elem, 'octave') and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add dots if present
                            if hasattr(elem, 'augmentation_dots') and elem.augmentation_dots > 0:
                                for i in range(elem.augmentation_dots):
                                    ET.SubElement(note_elem, 'dot')
                            
                            # Add accidental if present and not part of key signature
                            if hasattr(elem, 'accidental') and elem.accidental:
                                # First check if the accidental attribute exists and isn't None
                                if not hasattr(elem.accidental, 'is_key_signature') or not elem.accidental.is_key_signature:
                                    # Then check for is_key_signature attribute
                                    acc_elem = ET.SubElement(note_elem, 'accidental')
                                    acc_elem.text = elem.accidental.type
                            
                            # Add ties, time modification, and notations
                            add_tie_elements(note_elem, elem)
                            add_time_modification(note_elem, elem)
                            add_notations_to_note(note_elem, elem)
                    
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
                        
                        # Add dots if present
                        if hasattr(elem, 'augmentation_dots') and elem.augmentation_dots > 0:
                            for i in range(elem.augmentation_dots):
                                ET.SubElement(rest_elem, 'dot')
                        
                        # Add tuplet information if applicable
                        if hasattr(elem, 'is_tuplet') and elem.is_tuplet:
                            add_time_modification(rest_elem, elem)
                            if hasattr(elem, 'tuplet_start') or hasattr(elem, 'tuplet_end'):
                                add_notations_to_note(rest_elem, elem)

    else:
        # For multi-instrument scores: with proper staff elements
        print("Detected multi-instrument score - generating with separate parts")
        
        # Create score-part elements (one per SYSTEM, not per staff)
        for i, system in enumerate(staff_systems):
            score_part = ET.SubElement(part_list, 'score-part', id=f'P{i+1}')
            part_name = ET.SubElement(score_part, 'part-name')
            
            # Better naming for vocal parts
            if i == 0:
                part_name.text = 'Soprano Solo'  # First system (solo staff)
            else:
                part_name.text = f'Vocal {i}'  # Other systems
            
        # Create parts (one per SYSTEM)
        for i, system in enumerate(staff_systems):
            part = ET.SubElement(score_partwise, 'part', id=f'P{i+1}')
            
            # Extract and track clefs for this system
            system_clefs = []
            if hasattr(system, 'elements'):
                # Get all clefs in this system, sorted by vertical position
                all_clefs = [e for e in system.elements if isinstance(e, Clef)]
                if all_clefs:
                    all_clefs.sort(key=lambda c: c.y)
                    system_clefs = all_clefs
            
            # Track current state
            current_clef_types = {}  # Track clef types by staff number
            current_clef_lines = {}  # Track clef lines by staff number
            current_key_fifths = None
            current_time_sig = None
            
            # Determine if this is a multi-staff system
            is_multi_staff = hasattr(system, 'is_multi_staff') and system.is_multi_staff
            num_staves = len(system.staves) if hasattr(system, 'staves') and system.staves else 1
            
            # Add measures
            for j, measure in enumerate(system.measures):
                measure_elem = ET.SubElement(part, 'measure', number=str(j+1))
                
                # Determine if we need to include attributes
                include_attributes = False
                
                # Calculate key signature fifths value
                key_fifths = 0
                if hasattr(system, 'key_signature') and system.key_signature:
                    # Group key signature accidentals by staff
                    accidentals_by_staff = {}
                    for acc in system.key_signature:
                        staff_id = processor._get_staff_id(acc) if hasattr(processor, '_get_staff_id') else 0
                        if staff_id not in accidentals_by_staff:
                            accidentals_by_staff[staff_id] = []
                        accidentals_by_staff[staff_id].append(acc)
                    
                    # Calculate fifths for each staff
                    fifths_by_staff = {}
                    for staff_id, staff_accidentals in accidentals_by_staff.items():
                        num_sharps = sum(1 for acc in staff_accidentals if acc.type == 'sharp')
                        num_flats = sum(1 for acc in staff_accidentals if acc.type == 'flat')
                        
                        if num_sharps > 0:
                            fifths_by_staff[staff_id] = num_sharps
                        elif num_flats > 0:
                            fifths_by_staff[staff_id] = -num_flats
                        else:
                            fifths_by_staff[staff_id] = 0
                    
                    # Use the first staff's fifths by default (or merge somehow)
                    if fifths_by_staff:
                        key_fifths = next(iter(fifths_by_staff.values()))
                    else:
                        key_fifths = 0
                # Get time signature
                time_sig = system.time_signature if hasattr(system, 'time_signature') else None
                
                # First measure always gets attributes
                if j == 0:
                    include_attributes = True
                    current_key_fifths = key_fifths
                    current_time_sig = time_sig
                else:
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
                    
                    # For multi-staff systems, specify the number of staves
                    if is_multi_staff and num_staves > 1:
                        staves_elem = ET.SubElement(attrs, 'staves')
                        staves_elem.text = str(num_staves)
                        print(f"Added staves={num_staves} to measure {j+1} in part {i+1}")
                    
                    # Add clefs ONLY in the first measure or when they change
                    if j == 0:
                        if is_multi_staff and num_staves > 1:
                            # Map clefs to staves
                            staff_clef_map = {}
                            
                            # Try to assign detected clefs to staves based on vertical position
                            if system_clefs:
                                # Sort system_clefs by vertical position (top to bottom)
                                system_clefs.sort(key=lambda c: c.y)
                                
                                # Figure out how many clefs we have vs how many staves
                                for staff_idx in range(min(len(system_clefs), num_staves)):
                                    staff_num = staff_idx + 1  # Staff numbers are 1-based
                                    staff_clef_map[staff_num] = system_clefs[staff_idx]
                                    
                                    # Track current clef for this staff
                                    current_clef_types[staff_num] = system_clefs[staff_idx].type
                                    current_clef_lines[staff_num] = system_clefs[staff_idx].line if hasattr(system_clefs[staff_idx], 'line') else None
                            
                            # Add a clef for each staff
                            for staff_num in range(1, num_staves + 1):
                                clef_elem = ET.SubElement(attrs, 'clef')
                                clef_elem.set('number', str(staff_num))
                                
                                if staff_num in staff_clef_map:
                                    # Use detected clef for this staff
                                    staff_clef = staff_clef_map[staff_num]
                                    
                                    sign = ET.SubElement(clef_elem, 'sign')
                                    sign.text = staff_clef.type if hasattr(staff_clef, 'type') else "G"
                                    
                                    line = ET.SubElement(clef_elem, 'line')
                                    line.text = str(staff_clef.line) if hasattr(staff_clef, 'line') else "2"
                                else:
                                    # Use intelligent defaults
                                    if staff_num == 1:
                                        # First staff (upper): G clef (treble)
                                        sign = ET.SubElement(clef_elem, 'sign')
                                        sign.text = "G"
                                        line = ET.SubElement(clef_elem, 'line')
                                        line.text = "2"
                                        
                                        # Track this clef
                                        current_clef_types[staff_num] = "G"
                                        current_clef_lines[staff_num] = 2
                                    else:
                                        # Second or lower staff: F clef (bass) - typical for piano
                                        sign = ET.SubElement(clef_elem, 'sign')
                                        sign.text = "F"
                                        line = ET.SubElement(clef_elem, 'line')
                                        line.text = "4"
                                        
                                        # Track this clef
                                        current_clef_types[staff_num] = "F"
                                        current_clef_lines[staff_num] = 4
                        else:
                            # Single clef for single-staff system
                            system_clef = None
                            
                            # Try to find a suitable clef for this system
                            if system_clefs:
                                system_clef = system_clefs[0]  # Use the first detected clef
                            
                            if system_clef:
                                clef_type = system_clef.type if hasattr(system_clef, 'type') else "G"
                                clef_line = system_clef.line if hasattr(system_clef, 'line') else 2
                                
                                encode_clef_to_xml(attrs, clef_type, clef_line)
                                
                                # Track this clef
                                current_clef_types[1] = clef_type
                                current_clef_lines[1] = clef_line
                            else:
                                # Default to G clef if no detected clefs
                                encode_clef_to_xml(attrs, "G", 2)
                                
                                # Track this clef
                                current_clef_types[1] = "G"
                                current_clef_lines[1] = 2
                
                # Add dynamics for this measure (if any)
                measure_dynamics = [d for d in dynamics if hasattr(d, 'measure') and d.measure == measure]
                for dynamic in measure_dynamics:
                    add_dynamic_to_measure(measure_elem, dynamic)
                
                # Add gradual dynamics for this measure (if any)
                measure_gradual_dynamics = [d for d in gradual_dynamics if hasattr(d, 'measure') and d.measure == measure]
                for gradual in measure_gradual_dynamics:
                    add_gradual_dynamic_to_measure(measure_elem, gradual)
                
                # Add text directions for this measure (if any)
                measure_text_directions = [t for t in text_directions if hasattr(t, 'measure') and t.measure == measure]
                for text_dir in measure_text_directions:
                    add_text_direction_to_measure(measure_elem, text_dir, text_dir.placement if hasattr(text_dir, 'placement') else 'above')
                
                # Add notes, rests, and other elements
                # Get all elements in this measure
                all_measure_elements = measure.elements
                
                # Sort elements by x-position
                all_measure_elements.sort(key=lambda e: e.x)
                
                for elem in all_measure_elements:
                    if isinstance(elem, Note):
                        # Skip notes that are part of a chord (except the first one)
                        if hasattr(elem, 'is_chord_member') and elem.is_chord_member and hasattr(elem, 'chord') and elem.chord[0] != elem:
                            continue
                            
                        # Create note element
                        note_elem = ET.SubElement(measure_elem, 'note')
                        
                        # For multi-staff systems, determine which staff this note belongs to
                        if is_multi_staff and num_staves > 1:
                            # Check staff assignment
                            staff_id = processor._get_staff_id(elem) if hasattr(processor, '_get_staff_id') else 0
                            # Staff numbers in MusicXML are 1-based
                            staff_num = staff_id + 1
                            
                            # Add staff element
                            staff_elem = ET.SubElement(note_elem, 'staff')
                            staff_elem.text = str(staff_num)
                        
                        # If this is part of a chord, handle accordingly
                        if hasattr(elem, 'is_chord_member') and elem.is_chord_member and hasattr(elem, 'chord') and len(elem.chord) > 1:
                            # Add pitch for the first note
                            if hasattr(elem, 'step') and elem.step and hasattr(elem, 'octave') and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))  # Convert to divisions
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add accidental if present and not part of key signature
                            # FIX: Use elem instead of chord_note here
                            if hasattr(elem, 'accidental') and elem.accidental:
                                if not hasattr(elem.accidental, 'is_key_signature') or not elem.accidental.is_key_signature:
                                    acc_elem = ET.SubElement(note_elem, 'accidental')
                                    acc_elem.text = elem.accidental.type
                            
                            # Add ties, time modification, and notations for first note
                            add_tie_elements(note_elem, elem)
                            add_time_modification(note_elem, elem)
                            add_notations_to_note(note_elem, elem)
                            
                            # Add other chord notes
                            for chord_note in elem.chord[1:]:
                                # Create note element for chord note
                                chord_note_elem = ET.SubElement(measure_elem, 'note')
                                
                                # Add chord tag
                                ET.SubElement(chord_note_elem, 'chord')
                                
                                # For multi-staff systems, add staff assignment to chord notes too
                                if is_multi_staff and num_staves > 1:
                                    staff_id = processor._get_staff_id(chord_note) if hasattr(processor, '_get_staff_id') else 0
                                    staff_num = staff_id + 1
                                    
                                    staff_elem = ET.SubElement(chord_note_elem, 'staff')
                                    staff_elem.text = str(staff_num)
                                
                                # Add pitch
                                if hasattr(chord_note, 'step') and chord_note.step and hasattr(chord_note, 'octave') and chord_note.octave is not None:
                                    pitch_elem = chord_note.position_to_xml()
                                    chord_note_elem.append(pitch_elem)
                                
                                # Add duration
                                duration = ET.SubElement(chord_note_elem, 'duration')
                                duration.text = str(int(4 * chord_note.duration))
                                
                                # Add type
                                type_elem = ET.SubElement(chord_note_elem, 'type')
                                type_elem.text = chord_note.duration_type
                                
                                # Add accidental if present and not part of key signature
                                if hasattr(chord_note, 'accidental') and chord_note.accidental:
                                    if not hasattr(chord_note.accidental, 'is_key_signature') or not chord_note.accidental.is_key_signature:
                                        acc_elem = ET.SubElement(chord_note_elem, 'accidental')
                                        acc_elem.text = chord_note.accidental.type
                                
                                # Add ties, time modification, and notations for chord notes
                                add_tie_elements(chord_note_elem, chord_note)
                                add_time_modification(chord_note_elem, chord_note)
                                add_notations_to_note(chord_note_elem, chord_note)
                        else:
                            # Regular note (not part of a chord)
                            # Add pitch
                            if hasattr(elem, 'step') and elem.step and hasattr(elem, 'octave') and elem.octave is not None:
                                pitch_elem = elem.position_to_xml()
                                note_elem.append(pitch_elem)
                            
                            # Add duration
                            duration = ET.SubElement(note_elem, 'duration')
                            duration.text = str(int(4 * elem.duration))
                            
                            # Add type
                            type_elem = ET.SubElement(note_elem, 'type')
                            type_elem.text = elem.duration_type
                            
                            # Add accidental if present and not part of key signature
                            if hasattr(elem, 'accidental') and elem.accidental:
                                # Check if it's not a key signature accidental
                                if not hasattr(elem.accidental, 'is_key_signature') or not elem.accidental.is_key_signature:
                                    acc_elem = ET.SubElement(note_elem, 'accidental')
                                    acc_elem.text = elem.accidental.type
                            
                            # Add ties, time modification, and notations
                            add_tie_elements(note_elem, elem)
                            add_time_modification(note_elem, elem)
                            add_notations_to_note(note_elem, elem)
                    
                    elif isinstance(elem, Rest):
                        # Create rest element
                        rest_elem = ET.SubElement(measure_elem, 'note')
                        ET.SubElement(rest_elem, 'rest')
                        
                        # For multi-staff systems, determine which staff this rest belongs to
                        if is_multi_staff and num_staves > 1:
                            # Check staff assignment
                            staff_id = processor._get_staff_id(elem) if hasattr(processor, '_get_staff_id') else 0
                            # Staff numbers in MusicXML are 1-based
                            staff_num = staff_id + 1
                            
                            # Add staff element
                            staff_elem = ET.SubElement(rest_elem, 'staff')
                            staff_elem.text = str(staff_num)
                        
                        # Add duration
                        duration = ET.SubElement(rest_elem, 'duration')
                        duration.text = str(int(4 * elem.duration))
                        
                        # Add type
                        type_elem = ET.SubElement(rest_elem, 'type')
                        type_elem.text = elem.duration_type
                        
                        # Add tuplet information if applicable
                        if hasattr(elem, 'is_tuplet') and elem.is_tuplet:
                            add_time_modification(rest_elem, elem)
                            if hasattr(elem, 'tuplet_start') or hasattr(elem, 'tuplet_end'):
                                add_notations_to_note(rest_elem, elem)
    
    score_xml = ET.tostring(score_partwise, encoding='utf-8')
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>\n'
    doc_type = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
    return xml_declaration + doc_type + score_xml.decode('utf-8')