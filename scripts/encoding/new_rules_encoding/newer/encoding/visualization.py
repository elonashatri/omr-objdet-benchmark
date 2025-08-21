import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# Try to import from music_elements, but have a fallback for class checking
try:
    from music_elements import Note, Rest, Accidental, Clef, Barline, Beam, Flag, TimeSignatureElement
except ImportError:
    # Define dummy classes for isinstance checks if imports fail
    class Note: pass
    class Rest: pass
    class Accidental: pass
    class Clef: pass
    class Barline: pass
    class Beam: pass
    class Flag: pass
    class TimeSignatureElement: pass



def visualize_accidentals(processor, output_path=None):
    """
    Create a specialized visualization focusing on accidentals and their placement.
    This helps verify that accidentals are correctly connected to notes.
    
    Args:
        processor: An OMRProcessor instance with processed music elements
        output_path: Path to save the visualization (optional)
    """
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Define colors for different elements
    colors = {
        'staff_line': 'lightgray',
        'note': 'black', 
        'sharp': 'red',
        'flat': 'blue',
        'natural': 'green',
        'key_sig': 'purple',
        'connection': 'gray',
        'alignment': 'magenta'
    }
    
    # Draw staff lines
    for system in processor.staff_systems:
        for line_num, y in system.lines.items():
            ax.axhline(y=y, color=colors['staff_line'], linestyle='-', alpha=0.7)
    
    # Draw notes (simplified)
    for note in processor.notes:
        # Simple circle for each note
        circle = plt.Circle((note.x, note.y), 5, color=colors['note'], alpha=0.7)
        ax.add_patch(circle)
        
        # Add pitch label with altered state
        alter_symbol = '#' if note.alter == 1 else 'b' if note.alter == -1 else ''
        pitch_label = f"{note.step}{alter_symbol}{note.octave}"
        ax.text(note.x, note.y - 15, pitch_label, 
               fontsize=5, ha='center', color=colors['note'])
    
    # Draw accidentals with special focus
    for acc in processor.accidentals:
        # Choose color based on accidental type
        if acc.is_key_signature:
            color = colors['key_sig']
            label = f"{acc.type} (key)"
        else:
            color = colors[acc.type] if acc.type in colors else 'gray'
            label = acc.type
        
        # Draw rectangle around accidental
        rect = patches.Rectangle(
            (acc.bbox['x1'], acc.bbox['y1']),
            acc.width, acc.height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(acc.x, acc.y - 10, label, fontsize=5, ha='center', color=color)
        
        # Add alignment reference point
        if acc.type == 'flat':
            # Mark the alignment reference point for flats
            reference_y = acc.y + (acc.height * 0.3)
            ax.plot(acc.x, reference_y, 'o', color=colors['alignment'], markersize=4)
            ax.text(acc.x - 15, reference_y, "align", fontsize=6, ha='right', color=colors['alignment'])
        
        # Connect to affected note with a fancier arrow
        if acc.affected_note:
            # For flats, use the reference point for the arrow
            if acc.type == 'flat':
                reference_y = acc.y + (acc.height * 0.3)
                start_point = (acc.x, reference_y)
            else:
                start_point = (acc.x, acc.y)
                
            # Draw arrow from accidental to note
            arrow = patches.FancyArrowPatch(
                start_point,
                (acc.affected_note.x, acc.affected_note.y),
                connectionstyle="arc3,rad=0.1",
                arrowstyle="->",
                color=color,
                alpha=0.7,
                linewidth=1
            )
            ax.add_patch(arrow)
            
            # Calculate and show horizontal distance
            distance = acc.affected_note.x - acc.x
            midpoint_x = (acc.x + acc.affected_note.x) / 2
            midpoint_y = (acc.y + acc.affected_note.y) / 2 + 10
            ax.text(midpoint_x, midpoint_y, f"{distance:.1f}px", 
                   fontsize=7, ha='center', color=colors['connection'])
    
    # Draw measure boundaries
    for measure in processor.measures:
        system = measure.staff_system
        if system.lines:
            top_y = min(system.lines.values()) - system.line_spacing
            bottom_y = max(system.lines.values()) + system.line_spacing
            
            # Measure separator lines
            ax.axvline(x=measure.start_x, ymin=top_y, ymax=bottom_y, 
                      color='lightgray', linestyle='--', alpha=0.5)
            ax.axvline(x=measure.end_x, ymin=top_y, ymax=bottom_y, 
                      color='lightgray', linestyle='--', alpha=0.5)
    
    # Set axis limits
    all_elements = processor.notes + processor.accidentals
    all_x = [e.x for e in all_elements]
    all_y = [e.y for e in all_elements]
    
    if all_elements:
        margin = 50
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(max(all_y) + margin, min(all_y) - margin)  # Invert y-axis
    
    # Add title and labels
    ax.set_title('Accidental Placement Analysis')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=colors['sharp'], label='Sharp'),
        patches.Patch(facecolor='none', edgecolor=colors['flat'], label='Flat'),
        patches.Patch(facecolor='none', edgecolor=colors['natural'], label='Natural'),
        patches.Patch(facecolor='none', edgecolor=colors['key_sig'], label='Key Signature'),
        patches.Patch(facecolor='none', edgecolor=colors['note'], label='Note'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['alignment'], 
                 markersize=8, label='Flat Alignment Point')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def visualize_overlay(processor, image_path, output_path=None):
    """
    Overlay detections and linking information on the original music score image.
    """
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img)

    colors = {
        'note': 'lime',
        'rest': 'cyan',
        'accidental': 'blue',
        'clef': 'purple',
        'barline': 'red',
        'beam': 'orange',
        'flag': 'magenta',
        'time_sig': 'brown',
        'chord': 'green',
        'link': 'yellow',  # Changed to yellow
        'staff_line': 'pink',
    }

    def draw_bbox(element, color, label=None):
        rect = patches.Rectangle(
            (element.bbox['x1'], element.bbox['y1']),
            element.width, element.height,
            linewidth=1, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        if label:
            ax.text(
                element.bbox['x1'] + element.width / 2,
                element.bbox['y1'] - 2,
                label,
                fontsize=4,  # Consistent font size of 4
                ha='center',
                va='bottom',
                color=color
            )

    for system in processor.staff_systems:
        for line_num, y in system.lines.items():
            ax.axhline(y=y, color=colors['staff_line'], linestyle='-', alpha=0.5)
            ax.text(
                system.elements[0].x - 30 if system.elements else 100,
                y,
                f"L{line_num}",
                fontsize=5,  # Consistent font size of 4
                ha='right',
                va='center',
                color=colors['staff_line']
            )

    # Notes
    for note in processor.notes:
        draw_bbox(note, colors['note'], label=note.pitch if note.pitch else None)

    for rest in processor.rests:
        draw_bbox(rest, colors['rest'], label=rest.duration_type)

    for acc in processor.accidentals:
        draw_bbox(acc, colors['accidental'], label=acc.type)
        if acc.affected_note:
            ax.plot([acc.x, acc.affected_note.x], [acc.y, acc.affected_note.y],
                linestyle='-', color=colors['link'], alpha=1.0, linewidth=0.5)

    for clef in processor.clefs:
        draw_bbox(clef, colors['clef'], label=clef.type)

    for barline in processor.barlines:
        draw_bbox(barline, colors['barline'])

    for beam in processor.beams:
        draw_bbox(beam, colors['beam'])
        for note in beam.connected_notes:
            ax.plot([beam.x, note.x], [beam.y, note.y],
                    linestyle='--', color=colors['beam'], alpha=0.5)

    for flag in processor.flags:
        draw_bbox(flag, colors['flag'])
        if flag.connected_note:
            ax.plot([flag.x, flag.connected_note.x], [flag.y, flag.connected_note.y],
                    linestyle='--', color=colors['flag'], alpha=0.5)

    for time_sig in processor.time_signature_elements:
        draw_bbox(time_sig, colors['time_sig'], label=str(time_sig.value))

    for note in processor.notes:
        if note.is_chord_member and note == note.chord[0]:
            chord_points = [(n.x, n.y) for n in note.chord]
            xs, ys = zip(*chord_points)
            ax.plot(xs, ys, color=colors['chord'], linestyle='-', alpha=0.1)

    ax.set_title("Overlay of Detected Elements on Original Score", fontsize=3)  # Consistent font size
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def visualize_score(processor, output_path=None):
    """
    Visualize the music score with detected elements and relationships.
    
    Args:
        processor: An OMRProcessor instance with processed music elements
        output_path: Path to save the visualization (optional)
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Define colors for different element types
    colors = {
        'staff_line': 'black',
        'note': 'green',
        'rest': 'cyan',
        'accidental': 'blue',
        'barline': 'red',
        'clef': 'purple',
        'beam': 'orange',
        'flag': 'magenta',
        'time_sig': 'brown',
        'chord': 'lime',
        'measure': 'gray'
    }
    
    # Draw staff lines
    for system in processor.staff_systems:
        for line_num, y in system.lines.items():
            ax.axhline(y=y, color=colors['staff_line'], linestyle='-', alpha=0.5)
            
            # Add line number
            ax.text(system.elements[0].x - 30 if system.elements else 100, 
                   y, f"Line {line_num}", fontsize=5, ha='right', va='center')
    
    # Draw measures
    for measure in processor.measures:
        # Get staff system bounds
        system = measure.staff_system
        if system.lines:
            top_y = min(system.lines.values()) - system.line_spacing
            bottom_y = max(system.lines.values()) + system.line_spacing
            
            # Draw measure rectangle
            rect = patches.Rectangle(
                (measure.start_x, top_y),
                measure.end_x - measure.start_x, bottom_y - top_y,
                linewidth=1, edgecolor=colors['measure'], facecolor='none',
                linestyle='--', alpha=0.3
            )
            ax.add_patch(rect)
            
            # Add measure number
            if measure in system.measures:
                measure_num = system.measures.index(measure) + 1
                ax.text(measure.start_x + 10, top_y - 10, f"M{measure_num}",
                      fontsize=5, color=colors['measure'])
    
    # Draw notes
    for note in processor.notes:
        rect = patches.Rectangle(
            (note.bbox['x1'], note.bbox['y1']),
            note.width, note.height,
            linewidth=1, edgecolor=colors['note'], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add pitch label
        if note.pitch:
            ax.text(note.x, note.y, note.pitch,
                    fontsize=5, ha='center', va='center', color=colors['note'])
        
        # Add duration
        if note.duration_type:
            ax.text(note.x, note.bbox['y2'] + 5, note.duration_type,
                  fontsize=7, ha='center', va='bottom', color=colors['note'])
    
    # Draw rests
    for rest in processor.rests:
        rect = patches.Rectangle(
            (rest.bbox['x1'], rest.bbox['y1']),
            rest.width, rest.height,
            linewidth=1, edgecolor=colors['rest'], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add rest type
        if rest.duration_type:
            ax.text(rest.x, rest.y, rest.duration_type,
                  fontsize=5, ha='center', va='center', color=colors['rest'])
    
    # Draw accidentals
    for acc in processor.accidentals:
        rect = patches.Rectangle(
            (acc.bbox['x1'], acc.bbox['y1']),
            acc.width, acc.height,
            linewidth=1, edgecolor=colors['accidental'], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add accidental type
        if acc.type:
            ax.text(acc.x, acc.y - 10, acc.type,
                  fontsize=5, ha='center', color=colors['accidental'])
        
        # Connect to affected note
        if acc.affected_note:
            ax.plot([acc.x, acc.affected_note.x], [acc.y, acc.affected_note.y],
                  'b-', alpha=0.5)
        
        # Mark key signature accidentals differently
        if acc.is_key_signature:
            ax.text(acc.x, acc.bbox['y1'] - 5, "K",
                  fontsize=5, ha='center', color='purple', fontweight='bold')
    
    # Draw clefs
    for clef in processor.clefs:
        rect = patches.Rectangle(
            (clef.bbox['x1'], clef.bbox['y1']),
            clef.width, clef.height,
            linewidth=1, edgecolor=colors['clef'], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add clef type
        if clef.type:
            ax.text(clef.x, clef.y - 15, f"{clef.type}-clef",
                  fontsize=5, ha='center', color=colors['clef'])
    
    # Draw barlines
    for barline in processor.barlines:
        rect = patches.Rectangle(
            (barline.bbox['x1'], barline.bbox['y1']),
            barline.width, barline.height,
            linewidth=1, edgecolor=colors['barline'], facecolor='none'
        )
        ax.add_patch(rect)
    
    # Draw beams
    for beam in processor.beams:
        rect = patches.Rectangle(
            (beam.bbox['x1'], beam.bbox['y1']),
            beam.width, beam.height,
            linewidth=1, edgecolor=colors['beam'], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Connect beam to notes
        for note in beam.connected_notes:
            ax.plot([note.x, note.x], [note.y, beam.y],
                  color=colors['beam'], linestyle='--', alpha=0.5)
    
    # Draw flags
    for flag in processor.flags:
        rect = patches.Rectangle(
            (flag.bbox['x1'], flag.bbox['y1']),
            flag.width, flag.height,
            linewidth=1, edgecolor=colors['flag'], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Connect flag to note
        if flag.connected_note:
            ax.plot([flag.x, flag.connected_note.x], [flag.y, flag.connected_note.y],
                  color=colors['flag'], linestyle='--', alpha=0.5)
    
    # Draw time signature elements
    for time_sig in processor.time_signature_elements:
        rect = patches.Rectangle(
            (time_sig.bbox['x1'], time_sig.bbox['y1']),
            time_sig.width, time_sig.height,
            linewidth=1, edgecolor=colors['time_sig'], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add time sig value
        ax.text(time_sig.x, time_sig.y, str(time_sig.value),
              fontsize=10, ha='center', va='center', color=colors['time_sig'])
    
    # Highlight chords
    for note in processor.notes:
        if note.is_chord_member and note == note.chord[0]:  # Only process first note in chord
            # Draw chord connector
            chord_points = [(n.x, n.y) for n in note.chord]
            xs, ys = zip(*chord_points)
            ax.plot(xs, ys, color=colors['chord'], linestyle='-', alpha=0.7)
    
    # Set axis limits
    all_elements = (processor.notes + processor.rests + processor.accidentals + processor.clefs + 
                   processor.barlines + processor.beams + processor.flags + processor.time_signature_elements)
    
    if all_elements:
        all_x = [e.x for e in all_elements]
        all_y = [e.y for e in all_elements]
        
        margin = 50
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(max(all_y) + margin, min(all_y) - margin)  # Invert y-axis
    
    # Add title and labels
    ax.set_title('Music Score Analysis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add legend
    legend_elements = [patches.Patch(facecolor='none', edgecolor=color, label=name)
                     for name, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()