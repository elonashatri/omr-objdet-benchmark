import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import networkx as nx
from sklearn.cluster import DBSCAN

class MusicSymbolLinker:
    def __init__(self, detection_file_path, staff_line_tolerance=10):
        """
        Initialize the Music Symbol Linker
        
        Args:
            detection_file_path: Path to the JSON file containing detection data
            staff_line_tolerance: Vertical tolerance (in pixels) for considering symbols 
                                  to be on the same staff line
        """
        self.detection_file_path = detection_file_path
        self.staff_line_tolerance = staff_line_tolerance
        self.detections = self._load_detections()
        self.staff_systems = []
        self.linked_symbols = defaultdict(list)
        
        # Define musical symbol categories
        self.symbol_categories = {
            'staff': ['staff', 'staff_line'],
            'clefs': ['g_clef', 'f_clef', 'c_clef'],
            'notes': ['notehead_filled', 'notehead_empty', 'rest_quarter', 'rest_eighth', 
                     'rest_sixteenth', 'rest_whole', 'rest_half'],
            'stems': ['stem'],
            'beams': ['beam'],
            'flags': ['flag_eighth', 'flag_sixteenth'],
            'accidentals': ['accidental_sharp', 'accidental_flat', 'accidental_natural'],
            'time_sig': ['time_signature_4', 'time_signature_3', 'time_signature_2', 
                        'time_signature_C', 'time_signature_cutC'],
            'key_sig': ['key_signature'],
            'barlines': ['barline', 'barline_double', 'barline_end'],
            'dynamics': ['dynamic_p', 'dynamic_f', 'dynamic_m', 'dynamic_s', 
                        'dynamic_z', 'dynamic_r'],
            'articulations': ['dot', 'tenuto', 'staccato', 'accent']
        }
        
        # Define relationship types
        self.relationship_types = {
            'belongs_to_staff': 0,
            'belongs_to_measure': 1,
            'belongs_to_voice': 2,
            'belongs_to_chord': 3,
            'belongs_to_beam_group': 4,
            'has_accidental': 5,
            'has_dot': 6,
            'has_articulation': 7
        }
    
    def _load_detections(self):
        """Load detections from JSON file"""
        with open(self.detection_file_path, 'r') as f:
            data = json.load(f)
        return data['detections']
    
    def identify_staves(self):
        """Identify staff systems in the score"""
        # Extract staff line detections
        staff_lines = [d for d in self.detections if 'staff' in d['class_name'].lower()]
        
        if not staff_lines:
            print("No staff lines detected. Cannot proceed with staff identification.")
            return
        
        # Extract y-coordinates of staff lines
        y_coords = np.array([line['bbox']['center_y'] for line in staff_lines])
        
        # Use DBSCAN to cluster staff lines into staff systems
        clustering = DBSCAN(eps=self.staff_line_tolerance*3, min_samples=3).fit(y_coords.reshape(-1, 1))
        
        # Group staff lines by cluster
        systems = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            if label >= 0:  # Ignore noise points (-1)
                systems[label].append(staff_lines[i])
        
        # Sort each staff system by y-coordinate
        for label in systems:
            systems[label] = sorted(systems[label], key=lambda x: x['bbox']['center_y'])
        
        self.staff_systems = list(systems.values())
        print(f"Identified {len(self.staff_systems)} staff systems")
        
        return self.staff_systems
    
    def assign_symbols_to_staves(self):
        """Assign detected symbols to their respective staves"""
        if not self.staff_systems:
            self.identify_staves()
            if not self.staff_systems:
                return
        
        # For each detected symbol, find the closest staff system
        for symbol in self.detections:
            if 'staff' in symbol['class_name'].lower():
                continue  # Skip staff lines as they're already assigned
            
            symbol_y = symbol['bbox']['center_y']
            min_distance = float('inf')
            assigned_staff = None
            
            # Find the closest staff system
            for staff_system in self.staff_systems:
                # Calculate distance to the staff system (min and max y values)
                staff_min_y = min(staff['bbox']['y1'] for staff in staff_system)
                staff_max_y = max(staff['bbox']['y2'] for staff in staff_system)
                
                # Check if symbol is within or close to the staff system
                if staff_min_y <= symbol_y <= staff_max_y:
                    # Symbol is within staff lines
                    distance = 0
                else:
                    # Symbol is outside staff lines, calculate distance to nearest boundary
                    distance = min(abs(symbol_y - staff_min_y), abs(symbol_y - staff_max_y))
                
                # Update closest staff
                if distance < min_distance:
                    min_distance = distance
                    assigned_staff = staff_system
            
            # Assign symbol to staff
            if assigned_staff:
                staff_id = self.staff_systems.index(assigned_staff)
                if 'staff_assignment' not in symbol:
                    symbol['staff_assignment'] = staff_id
                    # Add to linked symbols
                    self.linked_symbols[f'staff_{staff_id}'].append(symbol)
    
    def identify_measures(self):
        """Identify measures based on barline positions"""
        if not self.staff_systems:
            self.assign_symbols_to_staves()
            if not self.staff_systems:
                return
        
        # Process each staff system separately
        for staff_idx, staff_system in enumerate(self.staff_systems):
            # Get all symbols assigned to this staff
            staff_symbols = self.linked_symbols[f'staff_{staff_idx}']
            
            # Find barlines
            barlines = [s for s in staff_symbols 
                       if any(b in s['class_name'].lower() for b in ['barline', 'bar_line'])]
            
            # Sort barlines by x position
            barlines = sorted(barlines, key=lambda x: x['bbox']['center_x'])
            
            # Create measures (regions between barlines)
            if barlines:
                # Add start (x=0) as implied barline
                implied_start = {
                    'bbox': {'center_x': 0},
                    'implied': True
                }
                
                # Get page width from rightmost detection
                max_x = max(s['bbox']['x2'] for s in self.detections)
                implied_end = {
                    'bbox': {'center_x': max_x},
                    'implied': True
                }
                
                # Create complete barline list with implied start/end
                complete_barlines = [implied_start] + barlines + [implied_end]
                
                # Create measures
                for i in range(len(complete_barlines) - 1):
                    start_x = complete_barlines[i]['bbox']['center_x']
                    end_x = complete_barlines[i+1]['bbox']['center_x']
                    measure_id = f'staff_{staff_idx}_measure_{i}'
                    
                    # Assign symbols to this measure
                    for symbol in staff_symbols:
                        symbol_x = symbol['bbox']['center_x']
                        if start_x <= symbol_x < end_x:
                            if 'measure_assignment' not in symbol:
                                symbol['measure_assignment'] = measure_id
                                # Add to linked symbols
                                self.linked_symbols[measure_id].append(symbol)
    
    def link_noteheads_to_stems(self):
        """Link noteheads to stems"""
        # Process each measure
        for measure_id, symbols in self.linked_symbols.items():
            if not measure_id.startswith('staff_'):
                continue  # Skip non-staff collections
                
            # Find noteheads and stems
            noteheads = [s for s in symbols if 'notehead' in s['class_name'].lower()]
            stems = [s for s in symbols if 'stem' in s['class_name'].lower()]
            
            # For each notehead, find the closest stem
            for notehead in noteheads:
                nh_x = notehead['bbox']['center_x']
                nh_y = notehead['bbox']['center_y']
                nh_width = notehead['bbox']['width']
                
                # Calculate distance to each stem
                min_distance = float('inf')
                closest_stem = None
                
                for stem in stems:
                    stem_x = stem['bbox']['center_x']
                    stem_y = stem['bbox']['center_y']
                    
                    # Horizontal distance is most important
                    h_dist = abs(stem_x - nh_x)
                    
                    # If stem is within reasonable distance horizontally (e.g., half a notehead width)
                    if h_dist <= nh_width * 1.5:
                        # Use combined distance (weighted more toward horizontal)
                        dist = h_dist * 3 + abs(stem_y - nh_y) * 0.5
                        
                        if dist < min_distance:
                            min_distance = dist
                            closest_stem = stem
                
                # Link the notehead to the stem
                if closest_stem:
                    if 'linked_symbols' not in notehead:
                        notehead['linked_symbols'] = []
                    
                    if 'linked_symbols' not in closest_stem:
                        closest_stem['linked_symbols'] = []
                    
                    # Add bidirectional links
                    notehead['linked_symbols'].append({
                        'id': closest_stem.get('id', symbols.index(closest_stem)),
                        'type': 'has_stem'
                    })
                    
                    closest_stem['linked_symbols'].append({
                        'id': notehead.get('id', symbols.index(notehead)),
                        'type': 'has_notehead'
                    })
    
    def link_notes_to_beams(self):
        """Link notes (stems) to beams"""
        # Process each measure
        for measure_id, symbols in self.linked_symbols.items():
            if not measure_id.startswith('staff_'):
                continue  # Skip non-staff collections
                
            # Find stems and beams
            stems = [s for s in symbols if 'stem' in s['class_name'].lower()]
            beams = [s for s in symbols if 'beam' in s['class_name'].lower()]
            
            # For each beam, find connected stems
            for beam in beams:
                beam_x1 = beam['bbox']['x1']
                beam_x2 = beam['bbox']['x2']
                beam_y = beam['bbox']['center_y']
                
                # Initialize beam's linked symbols list
                if 'linked_symbols' not in beam:
                    beam['linked_symbols'] = []
                
                # Find stems that intersect with the beam
                for stem in stems:
                    stem_x = stem['bbox']['center_x']
                    stem_y1 = stem['bbox']['y1']
                    stem_y2 = stem['bbox']['y2']
                    
                    # Check if stem intersects with beam
                    if (beam_x1 <= stem_x <= beam_x2 and 
                        min(stem_y1, stem_y2) <= beam_y <= max(stem_y1, stem_y2)):
                        
                        # Initialize stem's linked symbols list
                        if 'linked_symbols' not in stem:
                            stem['linked_symbols'] = []
                        
                        # Add bidirectional links
                        beam['linked_symbols'].append({
                            'id': stem.get('id', symbols.index(stem)),
                            'type': 'connected_to_stem'
                        })
                        
                        stem['linked_symbols'].append({
                            'id': beam.get('id', symbols.index(beam)),
                            'type': 'connected_to_beam'
                        })
    
    def identify_chords(self):
        """Identify chord structures (multiple noteheads on the same stem)"""
        # Process each measure
        for measure_id, symbols in self.linked_symbols.items():
            if not measure_id.startswith('staff_'):
                continue  # Skip non-staff collections
                
            # Group noteheads by their connected stem
            stems_with_noteheads = defaultdict(list)
            
            for symbol in symbols:
                if 'linked_symbols' in symbol and 'notehead' in symbol['class_name'].lower():
                    for link in symbol['linked_symbols']:
                        if link['type'] == 'has_stem':
                            stems_with_noteheads[link['id']].append(symbol)
            
            # Identify chords (stems with multiple noteheads)
            chord_id = 0
            for stem_id, noteheads in stems_with_noteheads.items():
                if len(noteheads) > 1:
                    # This is a chord
                    chord_name = f"{measure_id}_chord_{chord_id}"
                    chord_id += 1
                    
                    # Assign noteheads to this chord
                    for notehead in noteheads:
                        if 'chord_assignment' not in notehead:
                            notehead['chord_assignment'] = chord_name
                            # Add to linked symbols
                            self.linked_symbols[chord_name].append(notehead)
                    
                    # Find the stem and add it to the chord too
                    for symbol in symbols:
                        if symbol.get('id') == stem_id or symbols.index(symbol) == stem_id:
                            symbol['chord_assignment'] = chord_name
                            self.linked_symbols[chord_name].append(symbol)
    
    def identify_note_groups(self):
        """Identify groups of notes connected by beams"""
        # Process each measure
        for measure_id, symbols in self.linked_symbols.items():
            if not measure_id.startswith('staff_'):
                continue  # Skip non-staff collections
            
            # Build a graph of connected stems and beams
            G = nx.Graph()
            
            # Add all stems and beams as nodes
            for i, symbol in enumerate(symbols):
                if 'stem' in symbol['class_name'].lower() or 'beam' in symbol['class_name'].lower():
                    symbol_id = symbol.get('id', i)
                    G.add_node(symbol_id, symbol=symbol)
            
            # Add edges for connections
            for i, symbol in enumerate(symbols):
                if 'linked_symbols' in symbol:
                    symbol_id = symbol.get('id', i)
                    for link in symbol['linked_symbols']:
                        if link['type'] in ['connected_to_beam', 'connected_to_stem']:
                            G.add_edge(symbol_id, link['id'])
            
            # Find connected components (groups of connected notes)
            connected_components = list(nx.connected_components(G))
            
            # Assign symbols to note groups
            for group_idx, component in enumerate(connected_components):
                group_name = f"{measure_id}_group_{group_idx}"
                
                # Get symbols in this component
                for node_id in component:
                    for symbol in symbols:
                        symbol_id = symbol.get('id', symbols.index(symbol))
                        if symbol_id == node_id:
                            if 'note_group_assignment' not in symbol:
                                symbol['note_group_assignment'] = group_name
                                # Add to linked symbols
                                self.linked_symbols[group_name].append(symbol)
                                
                                # Also add any connected noteheads
                                if 'linked_symbols' in symbol and 'stem' in symbol['class_name'].lower():
                                    for link in symbol['linked_symbols']:
                                        if link['type'] == 'has_notehead':
                                            for nh in symbols:
                                                nh_id = nh.get('id', symbols.index(nh))
                                                if nh_id == link['id']:
                                                    if 'note_group_assignment' not in nh:
                                                        nh['note_group_assignment'] = group_name
                                                        self.linked_symbols[group_name].append(nh)
    
    def link_accidentals_to_notes(self):
        """Link accidentals to their respective notes"""
        # Process each measure
        for measure_id, symbols in self.linked_symbols.items():
            if not measure_id.startswith('staff_'):
                continue  # Skip non-staff collections
                
            # Find accidentals and noteheads
            accidentals = [s for s in symbols if 'accidental' in s['class_name'].lower()]
            noteheads = [s for s in symbols if 'notehead' in s['class_name'].lower()]
            
            # Sort by x-position
            accidentals.sort(key=lambda x: x['bbox']['center_x'])
            noteheads.sort(key=lambda x: x['bbox']['center_x'])
            
            # For each accidental, find the closest notehead to the right
            for accidental in accidentals:
                acc_x = accidental['bbox']['center_x']
                acc_y = accidental['bbox']['center_y']
                
                # Find noteheads to the right and at similar y-position
                candidate_noteheads = [
                    nh for nh in noteheads 
                    if nh['bbox']['center_x'] > acc_x and 
                    abs(nh['bbox']['center_y'] - acc_y) < self.staff_line_tolerance
                ]
                
                if candidate_noteheads:
                    # Find the closest one
                    closest = min(candidate_noteheads, 
                                 key=lambda nh: nh['bbox']['center_x'] - acc_x)
                    
                    # Link them if they're close enough
                    if closest['bbox']['center_x'] - acc_x < closest['bbox']['width'] * 3:
                        if 'linked_symbols' not in accidental:
                            accidental['linked_symbols'] = []
                        
                        if 'linked_symbols' not in closest:
                            closest['linked_symbols'] = []
                        
                        # Add bidirectional links
                        accidental['linked_symbols'].append({
                            'id': closest.get('id', symbols.index(closest)),
                            'type': 'modifies_note'
                        })
                        
                        closest['linked_symbols'].append({
                            'id': accidental.get('id', symbols.index(accidental)),
                            'type': 'has_accidental'
                        })
    
    def visualize_links(self, output_path=None):
        """Visualize the linked symbols"""
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot all symbols
        for detection in self.detections:
            x1, y1 = detection['bbox']['x1'], detection['bbox']['y1']
            x2, y2 = detection['bbox']['x2'], detection['bbox']['y2']
            
            # Draw bounding box
            plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'b-', linewidth=0.5)
            
            # Color based on category
            color = 'gray'  # default
            for category, classes in self.symbol_categories.items():
                if any(cls.lower() in detection['class_name'].lower() for cls in classes):
                    if category == 'staff':
                        color = 'lightgray'
                    elif category == 'clefs':
                        color = 'red'
                    elif category == 'notes':
                        color = 'blue'
                    elif category == 'stems':
                        color = 'green'
                    elif category == 'beams':
                        color = 'orange'
                    elif category == 'accidentals':
                        color = 'purple'
                    elif category == 'barlines':
                        color = 'brown'
                    else:
                        color = 'black'
                    break
            
            # Fill with light color
            plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], color, alpha=0.2)
        
        # Draw links if they exist
        for detection in self.detections:
            if 'linked_symbols' in detection:
                src_x = detection['bbox']['center_x']
                src_y = detection['bbox']['center_y']
                
                for link in detection['linked_symbols']:
                    # Find target symbol
                    for target in self.detections:
                        target_id = target.get('id', self.detections.index(target))
                        if target_id == link['id']:
                            tgt_x = target['bbox']['center_x']
                            tgt_y = target['bbox']['center_y']
                            
                            # Draw line based on relationship type
                            if link['type'] == 'has_stem' or link['type'] == 'has_notehead':
                                plt.plot([src_x, tgt_x], [src_y, tgt_y], 'g-', linewidth=1)
                            elif link['type'] == 'connected_to_beam' or link['type'] == 'connected_to_stem':
                                plt.plot([src_x, tgt_x], [src_y, tgt_y], 'orange', linewidth=1)
                            elif link['type'] == 'has_accidental' or link['type'] == 'modifies_note':
                                plt.plot([src_x, tgt_x], [src_y, tgt_y], 'purple', linewidth=1)
                            else:
                                plt.plot([src_x, tgt_x], [src_y, tgt_y], 'k--', linewidth=0.5)
        
        # Invert y-axis as image coordinates have origin at top-left
        plt.gca().invert_yaxis()
        
        # Set title and labels
        plt.title('Musical Symbol Linking Visualization')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
    
    def save_linked_data(self, output_path):
        """Save the linked symbol data to a JSON file"""
        # Prepare data structure
        linked_data = {
            'symbols': self.detections,
            'staff_systems': [
                {
                    'id': f'staff_{i}',
                    'symbols': [self.detections.index(s) for s in staff]
                }
                for i, staff in enumerate(self.staff_systems)
            ],
            'groupings': {
                group_id: [self.detections.index(s) if hasattr(s, 'index') else s for s in symbols]
                for group_id, symbols in self.linked_symbols.items()
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(linked_data, f, indent=2)
        
        print(f"Saved linked data to {output_path}")
    
    def process(self, output_dir=None):
        """Process all linking steps"""
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Execute linking pipeline
        self.identify_staves()
        self.assign_symbols_to_staves()
        self.identify_measures()
        self.link_noteheads_to_stems()
        self.link_notes_to_beams()
        self.identify_chords()
        self.identify_note_groups()
        self.link_accidentals_to_notes()
        
        # Save visualization if output directory is provided
        if output_dir:
            base_name = Path(self.detection_file_path).stem.replace('_detections', '')
            visualization_path = os.path.join(output_dir, f"{base_name}_linked_visualization.png")
            self.visualize_links(visualization_path)
            
            # Save linked data
            linked_data_path = os.path.join(output_dir, f"{base_name}_linked_data.json")
            self.save_linked_data(linked_data_path)
        
        return self.linked_symbols


def main():
    """Main function to process detection files"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Link detected music symbols")
    parser.add_argument("--detection-file", type=str, required=True, 
                        help="Path to detection JSON file")
    parser.add_argument("--output-dir", type=str, default="results/linked",
                        help="Output directory for visualization and linked data")
    parser.add_argument("--staff-line-tolerance", type=int, default=10,
                        help="Vertical tolerance for considering symbols on the same staff line")
    
    args = parser.parse_args()
    
    # Create linker and process
    linker = MusicSymbolLinker(args.detection_file, args.staff_line_tolerance)
    linker.process(args.output_dir)


if __name__ == "__main__":
    main()