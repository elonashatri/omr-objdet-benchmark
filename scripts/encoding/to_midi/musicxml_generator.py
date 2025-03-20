"""
musicxml_generator.py - Generates MusicXML from processed OMR data
"""
import json
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generate_musicxml(note_data, pitch_data, rhythm_data, time_signature="4/4"):
    """Generate MusicXML from the processed data"""
    # Parse time signature
    time_sig_parts = time_signature.split('/')
    if len(time_sig_parts) != 2:
        time_sig_parts = ['4', '4']  # Default to 4/4
    beats_per_measure = int(time_sig_parts[0])
    beat_unit = int(time_sig_parts[1])
    
    # Extract data
    noteheads = [n for n in note_data if n.get('class_name') == 'noteheadBlack']
    clefs = [n for n in note_data if 'Clef' in n.get('class_name', '')]
    rests = [n for n in note_data if 'rest' in n.get('class_name', '').lower()]
    
    # Get key signature
    key_info = pitch_data.get('key_signature', {'fifths': 0, 'mode': 'major'})
    
    # Get note durations
    note_durations = rhythm_data.get('note_durations', {})
    
    # Get accidentals
    note_pitch_info = pitch_data.get('notes', {})
    
    # Sort all elements by x position
    all_elements = []
    
    # Process noteheads
    for i, note in enumerate(noteheads):
        x_pos = note['bbox']['center_x']
        note_id = i  # Using index as ID
        
        # Get duration info
        duration_info = note_durations.get(note_id, {
            'duration': 'quarter',
            'divisions': 16
        })
        
        # Get pitch info
        pitch_info = note_pitch_info.get(str(note_id), {})
        if not pitch_info:
            # Try with numeric key
            pitch_info = note_pitch_info.get(note_id, {
                'step': 'C',
                'octave': '4',
                'alter': 0,
                'accidental': None
            })
        
        all_elements.append({
            'type': 'note',
            'id': note_id,
            'x_pos': x_pos,
            'duration': duration_info.get('duration', 'quarter'),
            'divisions': duration_info.get('divisions', 16),
            'step': pitch_info.get('step', 'C'),
            'octave': pitch_info.get('octave', '4'),
            'alter': pitch_info.get('alter', 0),
            'accidental': pitch_info.get('accidental'),
            'beam_id': duration_info.get('beam_id'),
            'beam_pos': duration_info.get('beam_pos')
        })
    
    # Process rests
    for i, rest in enumerate(rests):
        x_pos = rest['bbox']['center_x']
        rest_class = rest['class_name']
        
        # Determine rest duration based on class name
        if 'rest32nd' in rest_class:
            duration = '32nd'
            divisions = 2
        elif 'rest16th' in rest_class:
            duration = '16th'
            divisions = 4
        elif 'rest8th' in rest_class:
            duration = 'eighth'
            divisions = 8
        elif 'restQuarter' in rest_class:
            duration = 'quarter'
            divisions = 16
        else:
            duration = 'quarter'
            divisions = 16
        
        all_elements.append({
            'type': 'rest',
            'x_pos': x_pos,
            'duration': duration,
            'divisions': divisions
        })
    
    # Sort elements by X position
    all_elements.sort(key=lambda e: e['x_pos'])
    
    # Create MusicXML structure
    score_partwise = Element('score-partwise', version="4.0")
    
    # Add work information
    work = SubElement(score_partwise, 'work')
    work_title = SubElement(work, 'work-title')
    work_title.text = 'OMR to MusicXML Conversion'
    
    # Add identification
    identification = SubElement(score_partwise, 'identification')
    creator = SubElement(identification, 'creator', type="composer")
    creator.text = 'OMR JSON to MusicXML Converter'
    encoding = SubElement(identification, 'encoding')
    software = SubElement(encoding, 'software')
    software.text = 'Custom OMR JSON to MusicXML Script'
    
    # Add part list
    part_list = SubElement(score_partwise, 'part-list')
    score_part = SubElement(part_list, 'score-part', id="P1")
    part_name = SubElement(score_part, 'part-name')
    part_name.text = 'Music Part'
    
    # Add part
    part = SubElement(score_partwise, 'part', id="P1")
    
    # Organize elements into measures based on durations
    measure_capacity = 16 * beats_per_measure  # Divisions per measure (16 = quarter note)
    current_measure = 1
    current_position = 0
    measure_elements = []
    current_measure_elements = []
    
    # Group elements into measures
    for elem in all_elements:
        # If adding this element would exceed measure capacity, start a new measure
        if current_position + elem.get('divisions', 16) > measure_capacity and current_measure_elements:
            measure_elements.append(current_measure_elements)
            current_measure += 1
            current_position = 0
            current_measure_elements = []
        
        # Add element to current measure
        current_measure_elements.append(elem)
        current_position += elem.get('divisions', 16)
    
    # Add the last measure if not empty
    if current_measure_elements:
        measure_elements.append(current_measure_elements)
    
    # Process each measure
    for measure_idx, measure_elems in enumerate(measure_elements):
        # Create measure element
        measure = SubElement(part, 'measure', number=str(measure_idx + 1))
        
        # Add attributes for the first measure
        if measure_idx == 0:
            attributes = SubElement(measure, 'attributes')
            
            # Add divisions (how many divisions per quarter note)
            divisions = SubElement(attributes, 'divisions')
            divisions.text = '16'  # 16 divisions per quarter note
            
            # Add key signature
            key = SubElement(attributes, 'key')
            fifths = SubElement(key, 'fifths')
            fifths.text = str(key_info['fifths'])
            mode = SubElement(key, 'mode')
            mode.text = key_info['mode']
            
            # Add time signature
            time = SubElement(attributes, 'time')
            beats = SubElement(time, 'beats')
            beats.text = str(beats_per_measure)
            beat_type = SubElement(time, 'beat-type')
            beat_type.text = str(beat_unit)
            
            # Add clef if available
            if clefs:
                clef = SubElement(attributes, 'clef')
                sign = SubElement(clef, 'sign')
                
                if 'gClef' in clefs[0]['class_name']:
                    sign.text = 'G'
                elif 'fClef' in clefs[0]['class_name']:
                    sign.text = 'F'
                else:
                    sign.text = 'G'  # Default to G clef
                
                line = SubElement(clef, 'line')
                if sign.text == 'G':
                    line.text = '2'
                elif sign.text == 'F':
                    line.text = '4'
                else:
                    line.text = '2'  # Default
        
        # Process elements in this measure
        for elem in measure_elems:
            if elem['type'] == 'note':
                note_elem = SubElement(measure, 'note')
                
                # Add pitch information
                pitch = SubElement(note_elem, 'pitch')
                step = SubElement(pitch, 'step')
                step.text = elem['step']
                
                # Add alter if needed
                if elem['alter'] != 0:
                    alter = SubElement(pitch, 'alter')
                    alter.text = str(elem['alter'])
                
                octave = SubElement(pitch, 'octave')
                octave.text = elem['octave']
                
                # Add duration
                duration = SubElement(note_elem, 'duration')
                duration.text = str(elem['divisions'])
                
                # Add type
                type_elem = SubElement(note_elem, 'type')
                type_elem.text = elem['duration']
                
                # Add accidental if needed (only if not in key signature)
                if elem['accidental'] and elem['accidental'] != 'natural':
                    accidental = SubElement(note_elem, 'accidental')
                    accidental.text = elem['accidental']
                
                # Add stem direction
                stem = SubElement(note_elem, 'stem')
                stem.text = 'up'  # Default to up
                
                # Handle beaming
                if elem.get('beam_id') and elem.get('beam_pos'):
                    # For 32nd notes, add 3 beams
                    if elem['duration'] == '32nd':
                        beam1 = SubElement(note_elem, 'beam', number="1")
                        beam1.text = elem['beam_pos']
                        
                        beam2 = SubElement(note_elem, 'beam', number="2")
                        beam2.text = elem['beam_pos']
                        
                        beam3 = SubElement(note_elem, 'beam', number="3")
                        beam3.text = elem['beam_pos']
                    
                    # For 16th notes, add 2 beams
                    elif elem['duration'] == '16th':
                        beam1 = SubElement(note_elem, 'beam', number="1")
                        beam1.text = elem['beam_pos']
                        
                        beam2 = SubElement(note_elem, 'beam', number="2")
                        beam2.text = elem['beam_pos']
                    
                    # For eighth notes, add 1 beam
                    elif elem['duration'] == 'eighth':
                        beam1 = SubElement(note_elem, 'beam', number="1")
                        beam1.text = elem['beam_pos']
            
            elif elem['type'] == 'rest':
                note_elem = SubElement(measure, 'note')
                rest = SubElement(note_elem, 'rest')
                
                # Add duration
                duration = SubElement(note_elem, 'duration')
                duration.text = str(elem['divisions'])
                
                # Add type
                type_elem = SubElement(note_elem, 'type')
                type_elem.text = elem['duration']
        
        # Add barline at the end of each measure
        barline = SubElement(measure, 'barline', location="right")
        bar_style = SubElement(barline, 'bar-style')
        
        # Use normal barline for all but the last measure
        if measure_idx < len(measure_elements) - 1:
            bar_style.text = 'light'
        else:
            bar_style.text = 'light-heavy'  # Final barline
    
    return prettify(score_partwise)

def process_data_for_musicxml(json_data, pitch_info, rhythm_info, time_signature="4/4"):
    """Process the combined data to generate MusicXML"""
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    symbols = data.get('detections', data.get('detections', []))
    
    
    # Generate the MusicXML
    musicxml = generate_musicxml(symbols, pitch_info, rhythm_info, time_signature)
    return musicxml

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 4:
        print("Usage: python musicxml_generator.py input.json pitch_info.json rhythm_info.json [time_signature]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    pitch_file = sys.argv[2]
    rhythm_file = sys.argv[3]
    time_signature = sys.argv[4] if len(sys.argv) > 4 else "4/4"
    
    # Load files
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    with open(pitch_file, 'r') as f:
        pitch_info = json.load(f)
    
    with open(rhythm_file, 'r') as f:
        rhythm_info = json.load(f)
    
    # Generate MusicXML
    musicxml = process_data_for_musicxml(data, pitch_info, rhythm_info, time_signature)
    
    # Write to output file
    output_file = os.path.splitext(input_file)[0] + ".musicxml"
    with open(output_file, 'w') as f:
        f.write(musicxml)
    
    print(f"MusicXML file created: {output_file}")