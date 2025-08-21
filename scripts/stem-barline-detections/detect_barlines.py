import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def detect_barlines_computer_vision(image_path, staff_info):
    """
    Detect barlines using computer vision techniques
    
    Args:
        image_path: Path to score image
        staff_info: Dictionary with staff line information
    
    Returns:
        List of detected barline dictionaries
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Cannot load image from {image_path}")
        return []
    
    # Get image dimensions
    height, width = img.shape
    
    # Preprocess image to enhance vertical lines
    # Threshold the image to get binary
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Create kernels for morphological operations
    kernel_vertical = np.ones((31, 1), np.uint8)  # Adjust size based on image resolution
    
    # Enhance vertical lines
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)
    
    # Get all staff systems
    staff_systems = staff_info.get('staff_systems', [])
    
    detected_barlines = []
    
    # Process each staff system
    for system in staff_systems:
        system_id = system.get('id')
        
        # Get staff lines for this system
        staff_lines = [line for line in staff_info.get('detections', []) 
                      if line.get('staff_system') == system_id]
        
        if not staff_lines:
            continue
        
        # Calculate staff boundaries
        top_line = min(staff_lines, key=lambda x: x['bbox']['y1'])
        bottom_line = max(staff_lines, key=lambda x: x['bbox']['y2'])
        
        staff_top = top_line['bbox']['y1'] - 10  # Add margin
        staff_bottom = bottom_line['bbox']['y2'] + 10  # Add margin
        staff_left = min(line['bbox']['x1'] for line in staff_lines)
        staff_right = max(line['bbox']['x2'] for line in staff_lines)
        
        # Calculate staff height
        staff_height = staff_bottom - staff_top
        
        # Crop to staff region with margins
        margin = staff_height * 0.5  # Adjust as needed
        roi_top = max(0, int(staff_top - margin))
        roi_bottom = min(height, int(staff_bottom + margin))
        roi_left = max(0, int(staff_left))
        roi_right = min(width, int(staff_right))
        
        # Extract ROI
        staff_roi = vertical_lines[roi_top:roi_bottom, roi_left:roi_right]
        
        # Create vertical projection profile
        vertical_projection = np.sum(staff_roi, axis=0)
        
        # Normalize projection
        if np.max(vertical_projection) > 0:
            vertical_projection = vertical_projection / np.max(vertical_projection)
        
        # Detect peaks in the projection (potential barlines)
        peaks, _ = find_peaks(vertical_projection, 
                             height=0.5,         # Minimum height of the peak
                             distance=30,        # Minimum distance between peaks
                             prominence=0.3)     # Minimum prominence of peak
        
        # Filter peaks by checking the vertical extent of the lines
        for peak_idx in peaks:
            # Convert peak index to original image coordinates
            x_pos = roi_left + peak_idx
            
            # Check vertical extent
            column = staff_roi[:, peak_idx]
            non_zero = np.nonzero(column)[0]
            
            if len(non_zero) > staff_height * 0.7:  # At least 70% of staff height
                # Get the actual vertical extent
                y_start = roi_top + non_zero[0]
                y_end = roi_top + non_zero[-1]
                
                # Create barline detection
                barline = {
                    'class_id': 8,  # Barline class ID
                    'class_name': 'barline',
                    'confidence': 0.9,
                    'bbox': {
                        'x1': x_pos - 2,  # Adjust thickness as needed
                        'y1': y_start,
                        'x2': x_pos + 2,
                        'y2': y_end,
                        'width': 4,
                        'height': y_end - y_start,
                        'center_x': x_pos,
                        'center_y': (y_start + y_end) / 2
                    },
                    'staff_system': system_id,
                    'cv_detected': True
                }
                
                detected_barlines.append(barline)
    
    return detected_barlines

def analyze_chord_structure(noteheads, staff_info):
    """
    Analyze the structure of a chord to determine stem configurations
    
    Args:
        noteheads: List of notehead detections in the chord
        staff_info: Staff information dictionary
    
    Returns:
        Dictionary with chord analysis results
    """
    if not noteheads:
        return {'needs_stem': False}
    
    # Get staff system from first notehead
    staff_system = noteheads[0].get('staff_system')
    if staff_system is None:
        return {'needs_stem': False}
    
    # Get staff lines for this system
    staff_lines = [line for line in staff_info.get('detections', []) 
                  if line.get('staff_system') == staff_system]
    
    if not staff_lines:
        return {'needs_stem': False}
    
    # Get middle line position
    y_positions = [line['bbox']['center_y'] for line in staff_lines]
    middle_line_y = y_positions[len(y_positions) // 2] if y_positions else None
    
    if middle_line_y is None:
        return {'needs_stem': False}
    
    # Sort noteheads by vertical position (y-coordinate)
    noteheads_sorted = sorted(noteheads, key=lambda x: x['bbox']['center_y'])
    
    # Calculate pitch range
    pitch_range = 0
    if len(noteheads_sorted) > 1:
        pitch_range = noteheads_sorted[-1]['bbox']['center_y'] - noteheads_sorted[0]['bbox']['center_y']
    
    # Count noteheads above and below middle line
    above_middle = sum(1 for nh in noteheads if nh['bbox']['center_y'] < middle_line_y)
    below_middle = len(noteheads) - above_middle
    
    # Determine if there are potentially multiple voices
    # Criteria: more than 4 noteheads with significant pitch range
    multiple_voices = (len(noteheads) > 4 and 
                      pitch_range > 3 * (y_positions[1] - y_positions[0]))
    
    # Determine default stem direction
    # General rule: stems up if below middle line, down if above
    default_stem_direction = 'up' if below_middle > above_middle else 'down'
    
    # Define voice groups for complex chords
    voice_groups = {}
    
    if multiple_voices:
        # Attempt to separate into voices based on spacing
        clusters = []
        current_cluster = [noteheads_sorted[0]]
        
        # Simple clustering by vertical distance
        staff_spacing = (y_positions[1] - y_positions[0]) if len(y_positions) > 1 else 20
        threshold = staff_spacing * 1.5
        
        for i in range(1, len(noteheads_sorted)):
            current = noteheads_sorted[i]
            previous = noteheads_sorted[i-1]
            
            if current['bbox']['center_y'] - previous['bbox']['center_y'] > threshold:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [current]
            else:
                current_cluster.append(current)
        
        if current_cluster:
            clusters.append(current_cluster)
        
        # Assign stem directions to clusters
        for i, cluster in enumerate(clusters):
            avg_y = sum(nh['bbox']['center_y'] for nh in cluster) / len(cluster)
            direction = 'up' if avg_y > middle_line_y else 'down'
            voice_groups[f'voice_{i}'] = {
                'noteheads': cluster,
                'stem_direction': direction
            }
    else:
        # Single voice - all noteheads use the default stem direction
        voice_groups['main_voice'] = {
            'noteheads': noteheads,
            'stem_direction': default_stem_direction
        }
    
    return {
        'needs_stem': True,
        'default_stem_direction': default_stem_direction,
        'multiple_voices': multiple_voices,
        'voice_groups': voice_groups
    }

def infer_stems_for_chord(chord_analysis, staff_info):
    """
    Infer stems for a chord based on its structure analysis
    
    Args:
        chord_analysis: Dictionary with chord analysis results
        staff_info: Staff information dictionary
    
    Returns:
        List of inferred stem detections
    """
    if not chord_analysis.get('needs_stem', False):
        return []
    
    inferred_stems = []
    
    # Process each voice group
    for voice_id, voice_data in chord_analysis.get('voice_groups', {}).items():
        noteheads = voice_data.get('noteheads', [])
        if not noteheads:
            continue
        
        stem_direction = voice_data.get('stem_direction', 'up')
        
        # Get staff system from first notehead
        staff_id = noteheads[0].get('staff_system')
        
        # Calculate stem position
        if stem_direction == 'up':
            # Right side of rightmost notehead for up stems
            x_pos = max(nh['bbox']['x2'] - 3 for nh in noteheads)
            y_bottom = min(nh['bbox']['center_y'] for nh in noteheads)
            
            # Calculate appropriate stem height
            stem_height = STEM_HEIGHT_MEDIAN
            stem_height = min(STEM_HEIGHT_MEDIAN + STEM_HEIGHT_STD, 
                            max(STEM_HEIGHT_MEDIAN, stem_height))
            
            y_top = y_bottom - stem_height
        else:
            # Left side of leftmost notehead for down stems
            x_pos = min(nh['bbox']['x1'] + 3 for nh in noteheads)
            y_top = max(nh['bbox']['center_y'] for nh in noteheads)
            
            # Calculate appropriate stem height
            stem_height = STEM_HEIGHT_MEDIAN
            stem_height = min(STEM_HEIGHT_MEDIAN + STEM_HEIGHT_STD, 
                            max(STEM_HEIGHT_MEDIAN, stem_height))
            
            y_bottom = y_top + stem_height
        
        # Create the inferred stem
        stem = {
            'class_id': 5,  # Stem class ID
            'class_name': 'stem',
            'confidence': 0.85,
            'bbox': {
                'x1': x_pos - STEM_WIDTH/2,
                'y1': y_top,
                'x2': x_pos + STEM_WIDTH/2,
                'y2': y_bottom,
                'width': STEM_WIDTH,
                'height': y_bottom - y_top,
                'center_x': x_pos,
                'center_y': (y_top + y_bottom) / 2
            },
            'staff_system': staff_id,
            'inferred': True,
            'voice_id': voice_id
        }
        
        inferred_stems.append(stem)
    
    return inferred_stems

def visualize_computer_vision_process(image_path, staff_info, detected_barlines):
    """
    Visualize the computer vision process for debugging
    
    Args:
        image_path: Path to score image
        staff_info: Staff information dictionary
        detected_barlines: List of detected barlines
    
    Returns:
        Matplotlib figure
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image from {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a grayscale copy for processing
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Threshold the image to get binary
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Create kernels for morphological operations
    kernel_vertical = np.ones((31, 1), np.uint8)
    
    # Enhance vertical lines
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vertical)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Original Image')
    
    # Binary image
    axs[0, 1].imshow(binary, cmap='gray')
    axs[0, 1].set_title('Binary Image')
    
    # Vertical lines
    axs[1, 0].imshow(vertical_lines, cmap='gray')
    axs[1, 0].set_title('Vertical Lines')
    
    # Original with detected barlines
    axs[1, 1].imshow(img)
    axs[1, 1].set_title('Detected Barlines')
    
    # Draw detected barlines
    for barline in detected_barlines:
        bbox = barline['bbox']
        x1, y1 = bbox['x1'], bbox['y1']
        width, height = bbox['width'], bbox['height']
        
        rect = plt.Rectangle((x1, y1), width, height, 
                           linewidth=2, edgecolor='red', facecolor='none')
        axs[1, 1].add_patch(rect)
    
    # Display staff lines
    for line in staff_info.get('detections', []):
        bbox = line['bbox']
        x1, x2 = bbox['x1'], bbox['x2']
        y = bbox['center_y']
        
        axs[1, 1].plot([x1, x2], [y, y], 'g-', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    return fig