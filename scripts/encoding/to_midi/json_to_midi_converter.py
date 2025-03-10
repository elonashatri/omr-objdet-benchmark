import json
import mido
from mido import Message, MidiFile, MidiTrack
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Convert OMR JSON to MIDI with pattern recognition')
parser.add_argument('-i', '--input', default='paste.txt', help='Input JSON file')
parser.add_argument('-o', '--output', default='output.mid', help='Output MIDI file')
parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed analysis')
parser.add_argument('-p', '--plot', action='store_true', help='Generate visualization plot')
args = parser.parse_args()

# Load the JSON data
with open(args.input, 'r') as file:
    data = json.load(file)

# Create a new MIDI file with one track
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
mid.ticks_per_beat = 480  # High resolution for precise timing

# Set tempo (120 BPM)
track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))

# Extract all relevant musical symbols
symbols = []
for symbol in data['symbols']:
    if symbol['class_name'] in ['noteheadBlack', 'rest8th', 'rest32nd', 'flag32ndUp', 'gClef', 'beam']:
        symbols.append(symbol)

# Identify the beamed groups using beam elements
beam_groups = []
for symbol in symbols:
    if symbol['class_name'] == 'beam':
        group = []
        if 'linked_symbols' in symbol:
            for link in symbol['linked_symbols']:
                if link['type'] == 'connects_notehead':
                    notehead_id = link['id']
                    # Find notehead with this ID
                    for i, s in enumerate(data['symbols']):
                        if i == notehead_id and s['class_name'] == 'noteheadBlack':
                            group.append(s)
        if group:
            beam_groups.append(group)

# Create a visualization if requested
if args.plot:
    plt.figure(figsize=(10, 6))
    # Plot all symbols
    x_positions = [s['bbox']['center_x'] for s in symbols]
    y_positions = [s['bbox']['center_y'] for s in symbols]
    labels = [s['class_name'] for s in symbols]
    
    # Create color mapping for different symbol types
    colors = {'noteheadBlack': 'black', 'rest8th': 'red', 'rest32nd': 'orange', 
              'flag32ndUp': 'green', 'gClef': 'blue', 'beam': 'purple'}
    
    plt.scatter(x_positions, y_positions, c=[colors.get(label, 'gray') for label in labels], alpha=0.7)
    
    # Add labels for important elements
    for i, (x, y, label) in enumerate(zip(x_positions, y_positions, labels)):
        if label in ['rest8th', 'rest32nd', 'flag32ndUp']:
            plt.annotate(label, (x, y), fontsize=8)
    
    # Highlight beam groups
    for i, group in enumerate(beam_groups):
        group_x = [note['bbox']['center_x'] for note in group]
        group_y = [note['bbox']['center_y'] for note in group]
        plt.plot(group_x, group_y, 'r-', alpha=0.3)
        plt.annotate(f"Beam Group {i+1}", (np.mean(group_x), np.mean(group_y)), 
                    fontsize=10, color='red')
    
    plt.title('Musical Symbol Analysis')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True, alpha=0.3)
    plt.savefig('symbol_analysis.png', dpi=300)
    print("Visualization saved as symbol_analysis.png")

# Analysis report
report = ["PATTERN-SPECIFIC RHYTHM ANALYSIS", "==============================\n"]

# Analyze the specific rhythmic pattern
report.append("BEAM GROUP ANALYSIS:")
for i, group in enumerate(beam_groups):
    group_notes = len(group)
    report.append(f"Group {i+1}: {group_notes} notes")
    for note in sorted(group, key=lambda n: n['bbox']['center_x']):
        report.append(f"  - Note at x={note['bbox']['center_x']:.1f}")

# Find 32nd flag
flag32 = next((s for s in symbols if 'flag32nd' in s['class_name']), None)
if flag32:
    report.append(f"\nFound 32nd flag at x={flag32['bbox']['center_x']:.1f}")
    
    # Find the closest note to the flag
    noteheads = [s for s in symbols if s['class_name'] == 'noteheadBlack']
    distances = []
    for note in noteheads:
        x1, y1 = flag32['bbox']['center_x'], flag32['bbox']['center_y']
        x2, y2 = note['bbox']['center_x'], note['bbox']['center_y']
        dist = ((x2-x1)**2 + (y2-y1)**2)**0.5
        distances.append((note, dist))
    
    # Sort by distance
    distances.sort(key=lambda x: x[1])
    if distances:
        closest_note = distances[0][0]
        report.append(f"Closest notehead is at x={closest_note['bbox']['center_x']:.1f}")

# Find all rests
rests = [s for s in symbols if 'rest' in s['class_name']]
report.append("\nREST ANALYSIS:")
for rest in sorted(rests, key=lambda r: r['bbox']['center_x']):
    report.append(f"- {rest['class_name']} at x={rest['bbox']['center_x']:.1f}")

# Now identify the specific pattern in your data
# Based on the images, we expect groups of beamed notes (32nd or 16th notes)
# interspersed with rests

# Create a sequence of events based on x-position
all_events = []

# Add notes (differentiate between regular and beamed notes)
for note in [s for s in symbols if s['class_name'] == 'noteheadBlack']:
    # Check if it's in a beam group
    in_beam = False
    for group in beam_groups:
        if note in group:
            in_beam = True
            break
    
    # Check if it has the 32nd flag
    has_32nd_flag = False
    if flag32:
        x1, y1 = flag32['bbox']['center_x'], flag32['bbox']['center_y']
        x2, y2 = note['bbox']['center_x'], note['bbox']['center_y']
        if ((x2-x1)**2 + (y2-y1)**2)**0.5 < 50:  # If close to the flag
            has_32nd_flag = True
    
    # Determine duration
    if has_32nd_flag:
        duration = 60  # 32nd note
        note_type = '32nd note'
    elif in_beam:
        duration = 120  # 16th note (beamed)
        note_type = '16th note (beamed)'
    else:
        duration = 120  # Default to 16th note
        note_type = '16th note'
    
    all_events.append({
        'type': 'note',
        'x_pos': note['bbox']['center_x'],
        'duration': duration,
        'note_value': 69,  # A4
        'description': note_type
    })

# Add rests
for rest in rests:
    if rest['class_name'] == 'rest32nd':
        duration = 60  # 32nd rest
        rest_type = '32nd rest'
    elif rest['class_name'] == 'rest8th':
        duration = 240  # 8th rest
        rest_type = '8th rest'
    else:
        duration = 120  # Default
        rest_type = 'default rest'
    
    all_events.append({
        'type': 'rest',
        'x_pos': rest['bbox']['center_x'],
        'duration': duration,
        'description': rest_type
    })

# Sort events by x position
all_events.sort(key=lambda e: e['x_pos'])

# Generate report for the event sequence
report.append("\nEVENT SEQUENCE:")
for i, event in enumerate(all_events):
    if event['type'] == 'note':
        report.append(f"{i+1}. {event['description']} at x={event['x_pos']:.1f}, duration={event['duration']}")
    else:
        report.append(f"{i+1}. {event['description']} at x={event['x_pos']:.1f}, duration={event['duration']}")

# Generate MIDI events
midi_events = []
position = 0

report.append("\nMIDI SEQUENCE:")
for event in all_events:
    if event['type'] == 'note':
        midi_events.append((position, 'note_on', event['note_value']))
        midi_events.append((position + event['duration'], 'note_off', event['note_value']))
        report.append(f"Note at tick {position}, duration {event['duration']}")
    else:  # rest
        report.append(f"Rest at tick {position}, duration {event['duration']}")
    
    position += event['duration']

# Sort MIDI events by time
midi_events.sort(key=lambda e: e[0])

# Convert to MIDI with delta times
last_time = 0
for time, event_type, note in midi_events:
    delta = time - last_time
    
    if event_type == 'note_on':
        track.append(Message('note_on', note=note, velocity=80, time=delta))
    else:  # note_off
        track.append(Message('note_off', note=note, velocity=0, time=delta))
    
    last_time = time

# End of track
track.append(mido.MetaMessage('end_of_track', time=0))

# Save the MIDI file
mid.save(args.output)

# Save analysis report
with open('pattern_analysis.txt', 'w') as f:
    f.write('\n'.join(report))

print(f"MIDI file created: {args.output}")
print(f"Pattern analysis saved to: pattern_analysis.txt")