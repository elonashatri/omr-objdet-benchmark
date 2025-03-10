import json
import argparse
import math
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

parser = argparse.ArgumentParser(description='Convert OMR JSON to MusicXML with proper 32nd note beaming')
parser.add_argument('-i', '--input', default='paste.txt', help='Input JSON file (default: paste.txt)')
parser.add_argument('-o', '--output', default='output.musicxml', help='Output MusicXML file (default: output.musicxml)')
parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose information')
parser.add_argument('-t', '--time', default='4/4', help='Time signature (default: 4/4)')
parser.add_argument('-b', '--beamall', action='store_true', help='Treat all notes in beam groups as 32nd notes')
args = parser.parse_args()

# Parse time signature
time_sig_parts = args.time.split('/')
if len(time_sig_parts) != 2:
    time_sig_parts = ['4', '4']  # Default to 4/4
beats_per_measure = int(time_sig_parts[0])
beat_unit = int(time_sig_parts[1])

# Load the JSON data
with open(args.input, 'r') as file:
    data = json.load(file)

# Create a function to prettify XML output
def prettify(elem):
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

if args.verbose:
    print("Extracting musical symbols from JSON...")

# Extract all relevant musical symbols
symbols = data['symbols']
noteheads = [s for s in symbols if s['class_name'] == 'noteheadBlack']
stems = [s for s in symbols if s['class_name'] == 'stem']
rests = [s for s in symbols if 'rest' in s['class_name']]
flags = [s for s in symbols if 'flag' in s['class_name']]
beams = [s for s in symbols if s['class_name'] == 'beam']
clefs = [s for s in symbols if 'Clef' in s['class_name']]

if args.verbose:
    print(f"Found {len(noteheads)} noteheads, {len(stems)} stems, {len(rests)} rests, " +
          f"{len(flags)} flags, {len(beams)} beams, {len(clefs)} clefs")

# Check if there are any 32nd flags - important for determining default duration
has_32nd_flags = any('flag32nd' in flag['class_name'] for flag in flags)
if has_32nd_flags or args.beamall:
    if args.verbose:
        print("32nd flags detected or --beamall flag used - defaulting to 32nd notes for beamed groups")
    default_beam_duration = '32nd'
    default_beam_value = 2  # 2 divisions for 32nd note
else:
    default_beam_duration = '16th'
    default_beam_value = 4  # 4 divisions for 16th note

# Find all linked_symbols connections for beams
beam_connections = {}
for i, beam in enumerate(beams):
    if 'linked_symbols' in beam:
        beam_id = f"beam_{i}"
        note_ids = []
        
        for link in beam['linked_symbols']:
            if link['type'] == 'connects_notehead':
                note_id = link['id']
                note_ids.append(note_id)
        
        # Store the list of connected noteheads
        beam_connections[beam_id] = note_ids
        
        if args.verbose and note_ids:
            print(f"Beam {beam_id} connects note IDs: {note_ids}")

# Find which notes are in beam groups
note_to_beam_group = {}
for beam_id, note_ids in beam_connections.items():
    for note_id in note_ids:
        if 0 <= note_id < len(symbols):
            note = symbols[note_id]
            if note['class_name'] == 'noteheadBlack':
                note_to_beam_group[note_id] = beam_id

# Group notes by beam and sort by x position
beam_groups = {}
for beam_id, note_ids in beam_connections.items():
    beam_groups[beam_id] = []
    for note_id in note_ids:
        if 0 <= note_id < len(symbols):
            note = symbols[note_id]
            if note['class_name'] == 'noteheadBlack':
                beam_groups[beam_id].append(note)
    
    # Sort notes in the beam group by x position
    if beam_groups[beam_id]:
        beam_groups[beam_id].sort(key=lambda n: n['bbox']['center_x'])

if args.verbose:
    print("\nBeam groups after sorting:")
    for beam_id, notes in beam_groups.items():
        if notes:
            note_xs = [n['bbox']['center_x'] for n in notes]
            print(f"Beam {beam_id}: {len(notes)} notes at {[f'{x:.1f}' for x in note_xs]}")

# Identify 32nd notes by proximity to 32nd flags
flagged_notes = {}
for flag in flags:
    if 'flag32nd' in flag['class_name']:
        flag_x = flag['bbox']['center_x']
        flag_y = flag['bbox']['center_y']
        
        # Find the closest notehead
        closest_note = None
        min_distance = float('inf')
        
        for note in noteheads:
            note_x = note['bbox']['center_x']
            note_y = note['bbox']['center_y']
            distance = math.sqrt((note_x - flag_x)**2 + (note_y - flag_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_note = note
        
        if closest_note and min_distance < 100:  # Threshold for proximity
            note_id = symbols.index(closest_note)
            flagged_notes[note_id] = {
                'duration': '32nd',
                'value': 2  # 2 divisions for 32nd note
            }
            
            if args.verbose:
                note_x = closest_note['bbox']['center_x']
                print(f"32nd flag at x={flag_x:.1f} assigned to note ID {note_id} at x={note_x:.1f} (distance: {min_distance:.1f})")

# Build a list of all musical elements with their properties
all_elements = []

# Process noteheads
for i, note in enumerate(noteheads):
    note_id = symbols.index(note)
    x_pos = note['bbox']['center_x']
    
    # Determine duration
    if note_id in flagged_notes:
        duration = flagged_notes[note_id]['duration']
        duration_value = flagged_notes[note_id]['value']
    elif note_id in note_to_beam_group:
        # If note is in a beam group and we have 32nd flags or --beamall flag,
        # treat all beamed notes as 32nd notes
        duration = default_beam_duration
        duration_value = default_beam_value
    else:
        duration = '16th'  # Default for isolated notes
        duration_value = 4  # 4 divisions for 16th note
    
    # Get pitch information
    pitch_info = note.get('pitch', {'note_name': 'A4'})
    note_name = pitch_info.get('note_name', 'A4')
    
    # Extract step and octave
    if len(note_name) >= 2:
        step = note_name[0]
        octave = note_name[1] if note_name[1].isdigit() else '4'
    else:
        step = 'A'
        octave = '4'
    
    # Store beam group info
    beam_id = note_to_beam_group.get(note_id)
    
    all_elements.append({
        'type': 'note',
        'id': note_id,
        'x_pos': x_pos,
        'duration': duration,
        'duration_value': duration_value,
        'step': step,
        'octave': octave,
        'beam_id': beam_id
    })

# Process rests
for rest in rests:
    x_pos = rest['bbox']['center_x']
    
    # Determine rest type and duration value
    if 'rest32nd' in rest['class_name']:
        duration = '32nd'
        duration_value = 2  # 2 divisions for 32nd rest
    elif 'rest16th' in rest['class_name']:
        duration = '16th'
        duration_value = 4  # 4 divisions for 16th rest
    elif 'rest8th' in rest['class_name']:
        duration = 'eighth'
        duration_value = 8  # 8 divisions for 8th rest
    else:
        duration = '16th'  # Default
        duration_value = 4  # 4 divisions for 16th rest
    
    all_elements.append({
        'type': 'rest',
        'x_pos': x_pos,
        'duration': duration,
        'duration_value': duration_value
    })

# Sort all elements by x position
all_elements.sort(key=lambda e: e['x_pos'])

if args.verbose:
    print("\nSorted musical elements:")
    for i, elem in enumerate(all_elements):
        if elem['type'] == 'note':
            beam_info = f", beam_id={elem['beam_id']}" if elem.get('beam_id') else ""
            print(f"{i+1}. Note at x={elem['x_pos']:.1f}, {elem['duration']}, {elem['step']}{elem['octave']}{beam_info}")
        else:
            print(f"{i+1}. Rest at x={elem['x_pos']:.1f}, {elem['duration']}")

# Determine beam positions for notes in beam groups
for beam_id, notes in beam_groups.items():
    for i, note in enumerate(notes):
        note_id = symbols.index(note)
        
        # Find the corresponding element
        for elem in all_elements:
            if elem['type'] == 'note' and elem.get('id') == note_id:
                if i == 0:
                    elem['beam_pos'] = 'begin'
                elif i == len(notes) - 1:
                    elem['beam_pos'] = 'end'
                else:
                    elem['beam_pos'] = 'continue'
                
                if args.verbose:
                    print(f"Note ID {note_id} beam position: {elem['beam_pos']}")

# Create MusicXML structure
score_partwise = Element('score-partwise', version="4.0")

# Add work information
work = SubElement(score_partwise, 'work')
work_title = SubElement(work, 'work-title')
work_title.text = 'Converted from JSON'

# Add identification
identification = SubElement(score_partwise, 'identification')
creator = SubElement(identification, 'creator', type="composer")
creator.text = 'JSON to MusicXML Converter'

# Add part list
part_list = SubElement(score_partwise, 'part-list')
score_part = SubElement(part_list, 'score-part', id="P1")
part_name = SubElement(score_part, 'part-name')
part_name.text = 'Music Part'

# Add part
part = SubElement(score_partwise, 'part', id="P1")

# Organize elements into measures based on durations
measure_capacity = 16 * beats_per_measure  # Divisions per measure
current_measure = 1
current_position = 0
measure_elements = []
current_measure_elements = []

# Group elements into measures
for elem in all_elements:
    # If adding this element would exceed measure capacity, start a new measure
    if current_position + elem['duration_value'] > measure_capacity and current_measure_elements:
        measure_elements.append(current_measure_elements)
        current_measure += 1
        current_position = 0
        current_measure_elements = []
    
    # Add element to current measure
    current_measure_elements.append(elem)
    current_position += elem['duration_value']

# Add the last measure if not empty
if current_measure_elements:
    measure_elements.append(current_measure_elements)

if args.verbose:
    print(f"\nOrganized into {len(measure_elements)} measures")
    for i, measure in enumerate(measure_elements):
        print(f"Measure {i+1}: {len(measure)} elements")

# Process each measure
for measure_idx, measure_elems in enumerate(measure_elements):
    # Create measure element
    measure = SubElement(part, 'measure', number=str(measure_idx + 1))
    
    # Add attributes for the first measure
    if measure_idx == 0:
        attributes = SubElement(measure, 'attributes')
        
        # Add divisions (defines the meaning of duration values)
        divisions = SubElement(attributes, 'divisions')
        divisions.text = '16'  # 16 divisions per quarter note
        
        # Add key
        key = SubElement(attributes, 'key')
        fifths = SubElement(key, 'fifths')
        fifths.text = '0'  # C major / A minor
        
        # Add time signature
        time = SubElement(attributes, 'time')
        beats = SubElement(time, 'beats')
        beats.text = str(beats_per_measure)
        beat_type = SubElement(time, 'beat-type')
        beat_type.text = str(beat_unit)
        
        # Add clef
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
            octave = SubElement(pitch, 'octave')
            octave.text = elem['octave']
            
            # Add duration
            duration = SubElement(note_elem, 'duration')
            duration.text = str(elem['duration_value'])
            
            # Add type
            type_elem = SubElement(note_elem, 'type')
            type_elem.text = elem['duration']
            
            # Add stem direction (explicitly up for all notes)
            stem = SubElement(note_elem, 'stem')
            stem.text = 'up'
            
            # For 32nd notes, add proper flags/beams
            if elem['duration'] == '32nd':
                # If part of a beam group, add beam elements
                if elem.get('beam_pos'):
                    # Add primary beam
                    beam1 = SubElement(note_elem, 'beam', number="1")
                    beam1.text = elem['beam_pos']
                    
                    # Add secondary beam
                    beam2 = SubElement(note_elem, 'beam', number="2")
                    beam2.text = elem['beam_pos']
                    
                    # Add third beam for 32nd notes
                    beam3 = SubElement(note_elem, 'beam', number="3")
                    beam3.text = elem['beam_pos']
                else:
                    # For individual 32nd notes, add the flag
                    notations = SubElement(note_elem, 'notations')
            
            # For 16th notes in beam groups
            elif elem['duration'] == '16th' and elem.get('beam_pos'):
                # Add beams
                beam1 = SubElement(note_elem, 'beam', number="1")
                beam1.text = elem['beam_pos']
                
                beam2 = SubElement(note_elem, 'beam', number="2")
                beam2.text = elem['beam_pos']
        
        elif elem['type'] == 'rest':
            note_elem = SubElement(measure, 'note')
            rest = SubElement(note_elem, 'rest')
            
            # Add duration
            duration = SubElement(note_elem, 'duration')
            duration.text = str(elem['duration_value'])
            
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

# Write MusicXML to file
with open(args.output, 'w') as file:
    file.write(prettify(score_partwise))

print(f"MusicXML file created: {args.output} with {len(measure_elements)} measures")
print("Note: By default, beamed notes are treated as 16th notes unless 32nd flags are detected.")
print("If your score consists primarily of 32nd notes, use the --beamall flag to treat all beamed notes as 32nd notes.")