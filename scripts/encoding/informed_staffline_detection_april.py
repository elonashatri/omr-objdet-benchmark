import os
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth

class AdaptiveStaffDetector:
    
    def infer_staff_from_clef_box(clef_top, clef_height, clef_class_id):
        clef_mappings = {
            # class_id: (line index, offset ratio, spacing divisor)
            15: (1, 0.62, 4.8),     # gClef
            42: (1, 0.62, 4.8),     # gClef8vb
            88: (1, 0.62, 4.8),     # gClef8va
            22: (3, 0.267, 3.6),    # fClef
            63: (3, 0.267, 3.6),    # fClef8vb
            32: (2, 0.5, 4.2),      # cClef (e.g. alto clef)
            40: (2, 0.5, 4.0),      # unpitchedPercussionClef
        }

        if clef_class_id not in clef_mappings:
            return None  # unsupported clef

        line_index, offset_ratio, spacing_divisor = clef_mappings[clef_class_id]
        ref_line_y = clef_top + offset_ratio * clef_height
        spacing = clef_height / spacing_divisor
        top_line_y = ref_line_y - spacing * line_index

        return [top_line_y + i * spacing for i in range(5)], spacing

    """Staff line detector that automatically adapts to different score characteristics"""
    
    def __init__(self, line_merging_threshold=5):
        """
        Initialize the adaptive staff detector
        
        Args:
            line_merging_threshold: Maximum vertical distance to consider adjacent 
                                   rows as part of the same staff line
        """
        self.line_merging_threshold = line_merging_threshold
    
    def detect(self, image_path, detected_elements=None):
        """
        Detect staff lines in a music score image by trying multiple methods
        
        Args:
            image_path: Path to the music score image
            detected_elements: Optional dictionary of previously detected musical elements
                
        Returns:
            Dictionary containing staff line information
        """
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path
            
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Invert image if needed (staff lines should be black)
        if np.mean(img) > 127:
            img = 255 - img
            
        # Check if we have detected elements to use for staff inference
        have_detected_elements = (detected_elements is not None and 
                                 len(detected_elements.get("detections", [])) > 0)
        
        # If we have musical elements, try to use them first for more accurate staff detection
        if have_detected_elements:
            print("Using detected musical elements to assist staff line detection...")
            try:
                # Try regenerating staff lines from musical elements first
                staff_data = self._regenerate_staff_from_notation(img, detected_elements)
                
                # If we found at least one staff system with this method, we can return already
                if staff_data and len(staff_data["staff_systems"]) > 0:
                    print(f"Successfully generated staff lines using musical elements")
                    return staff_data
            except Exception as e:
                print(f"Staff regeneration from musical elements failed: {e}")
        
        # If regeneration didn't work or we don't have elements, try standard methods
        methods = [
            self._detect_run_length,
            self._detect_projection,
            self._detect_hough
        ]
        
        staff_data = None
        for method in methods:
            try:
                staff_data = method(img)
                # If we found at least one staff system, we're done
                if staff_data and len(staff_data["staff_systems"]) > 0:
                    print(f"Successfully detected staff lines using {method.__name__}")
                    break
            except Exception as e:
                print(f"Method {method.__name__} failed: {e}")
        
        # If standard methods worked but found incomplete staff systems (fewer than 5 lines),
        # try to refine them using detected elements
        if staff_data and len(staff_data["staff_systems"]) > 0 and have_detected_elements:
            # Check if any staff system has fewer than 5 lines
            incomplete_systems = any(
                len(system.get("lines", [])) < 5 
                for system in staff_data.get("staff_systems", [])
            )
            
            if incomplete_systems:
                print("Refining incomplete staff systems using detected musical elements...")
                try:
                    refined_data = self._refine_staff_systems(staff_data, detected_elements)
                    # Use the refined data if it looks better
                    if (refined_data and 
                        len(refined_data["staff_systems"]) >= len(staff_data["staff_systems"]) and
                        len(refined_data["detections"]) >= len(staff_data["detections"])):
                        staff_data = refined_data
                        print("Successfully refined staff systems")
                except Exception as e:
                    print(f"Staff refinement failed: {e}")
        
        # If no standard methods worked, try inference using musical elements
        if (staff_data is None or len(staff_data["staff_systems"]) < 1) and have_detected_elements:
            print("Standard detection methods failed. Trying to infer staff lines from detected musical elements...")
            try:
                staff_data = self._infer_from_musical_elements(img, detected_elements)
            except Exception as e:
                print(f"Staff inference from musical elements failed: {e}")
                
        # If still no staff data, return empty data
        if staff_data is None or len(staff_data["staff_systems"]) == 0:
            print("Failed to detect any staff lines with all methods.")
            return {"staff_systems": [], "detections": []}
            
        return staff_data
    
    def _detect_run_length(self, img):
        """
        Detect staff lines using run-length analysis
        Works well for clean, synthetic images
        """
        height, width = img.shape
        binary = img < 127
        
        # Find potential staff line heights by analyzing run-heights of black pixels
        vertical_runs = []
        for x in range(0, width, width // 10):  # Sample at intervals
            current_run = 0
            for y in range(height):
                if binary[y, x]:
                    current_run += 1
                else:
                    if current_run > 0:
                        vertical_runs.append(current_run)
                        current_run = 0
            # Handle run at the end
            if current_run > 0:
                vertical_runs.append(current_run)
        
        # If we don't have enough data, give up
        if len(vertical_runs) < 5:
            return {"staff_systems": [], "detections": []}
        
        # Find the most common height (staff line thickness)
        heights, counts = np.unique(vertical_runs, return_counts=True)
        # Filter out very short runs
        valid_indices = heights >= 1
        heights = heights[valid_indices]
        counts = counts[valid_indices]
        
        # If no valid heights, give up
        if len(heights) == 0:
            return {"staff_systems": [], "detections": []}
        
        # Find the most common height within a reasonable range
        common_heights = heights[counts > np.max(counts) * 0.2]
        if len(common_heights) == 0:
            staff_height = heights[np.argmax(counts)]
        else:
            # Take the middle value from common heights
            staff_height = common_heights[len(common_heights) // 2]
        
        # Now detect horizontal runs of that height
        staff_line_positions = []
        run_data = []
        
        # For each row, check if there's a line
        for y in range(height - staff_height + 1):
            # Check if there's a significant run of black pixels in this horizontal slice
            horizontal_slice = binary[y:y+staff_height, :]
            # A row is a staff line if most pixels (80%+) in at least one slice are black 
            row_is_line = False
            
            # Check consecutive columns for staff line presence
            for x_start in range(0, width - width//4, width//8):
                x_end = min(x_start + width//4, width)
                section = horizontal_slice[:, x_start:x_end]
                if np.mean(section) > 0.8:  # 80% of pixels are black
                    row_is_line = True
                    break
            
            if row_is_line:
                # Find start and end of line
                # Sum across height dimension to get a count of black pixels per column
                line_profile = np.sum(horizontal_slice, axis=0) >= staff_height/2
                # Find start and end of longest run
                runs = np.split(np.where(line_profile)[0], np.where(np.diff(np.where(line_profile)[0]) > 1)[0] + 1)
                if runs and len(runs) > 0:
                    longest_run = max(runs, key=len)
                    if len(longest_run) > width * 0.6:  # Line spans at least 60% of width
                        x1 = longest_run[0]
                        x2 = longest_run[-1]
                        staff_line_positions.append(y + staff_height // 2)
                        run_data.append((y + staff_height // 2, x1, x2))
        
        # If we don't have enough lines, give up
        if len(staff_line_positions) < 5:
            return {"staff_systems": [], "detections": []}
        
        # Merge adjacent staff line positions that are likely the same line
        merged_positions = []
        merged_run_data = []
        
        # Sort positions and run data
        indices = np.argsort(staff_line_positions)
        staff_line_positions = [staff_line_positions[i] for i in indices]
        run_data = [run_data[i] for i in indices]
        
        # Group by proximity
        current_group = [0]  # Start with first position
        
        for i in range(1, len(staff_line_positions)):
            # If this position is close to the previous one, consider it part of the same staff line
            if staff_line_positions[i] - staff_line_positions[current_group[-1]] <= self.line_merging_threshold:
                current_group.append(i)
            else:
                # Process the current group
                if current_group:
                    # Find the central position in this group
                    center_idx = current_group[len(current_group) // 2]
                    merged_positions.append(staff_line_positions[center_idx])
                    merged_run_data.append(run_data[center_idx])
                # Start a new group
                current_group = [i]
        
        # Process the last group
        if current_group:
            center_idx = current_group[len(current_group) // 2]
            merged_positions.append(staff_line_positions[center_idx])
            merged_run_data.append(run_data[center_idx])
        
        # Replace with merged data
        staff_line_positions = merged_positions
        run_data = merged_run_data
        
        # If we merged too much and don't have enough lines, give up
        if len(staff_line_positions) < 5:
            return {"staff_systems": [], "detections": []}
        
        # Calculate spacing between adjacent lines
        spacings = [staff_line_positions[i+1] - staff_line_positions[i] 
                  for i in range(len(staff_line_positions)-1)]
        
        # If no spacings, give up
        if len(spacings) == 0:
            return {"staff_systems": [], "detections": []}
        
        # Use DBSCAN to cluster spacings
        spacing_array = np.array(spacings).reshape(-1, 1)
        clustering = DBSCAN(eps=3, min_samples=2).fit(spacing_array)
        
        # Find most common spacing (staff line spacing)
        if len(np.unique(clustering.labels_)) <= 1:
            # If clustering failed, use median
            staff_spacing = np.median(spacings)
        else:
            # Use most common cluster
            unique_labels, counts = np.unique(clustering.labels_, return_counts=True)
            if -1 in unique_labels:  # Remove noise
                noise_idx = np.where(unique_labels == -1)[0]
                unique_labels = np.delete(unique_labels, noise_idx)
                counts = np.delete(counts, noise_idx)
            
            if len(counts) == 0:
                staff_spacing = np.median(spacings)
            else:
                best_cluster = unique_labels[np.argmax(counts)]
                staff_spacing = np.median([spacings[i] for i, label in enumerate(clustering.labels_) 
                                        if label == best_cluster])
        
        # Group staff lines into staff systems
        staff_systems = []
        current_staff = []
        
        for i, pos in enumerate(staff_line_positions):
            if i == 0:
                current_staff = [i]
            else:
                spacing = pos - staff_line_positions[i-1]
                
                # If this spacing is close to the expected staff line spacing
                if abs(spacing - staff_spacing) < staff_spacing * 0.25:
                    current_staff.append(i)
                    
                    # If we have 5 lines, it's a complete staff
                    if len(current_staff) == 5:
                        staff_systems.append(current_staff)
                        current_staff = []
                else:
                    # This is a line from a new staff system
                    if len(current_staff) >= 3:  # Partial staff needs at least 3 lines
                        staff_systems.append(current_staff)
                    current_staff = [i]
        
        # Add the last staff if it's not empty
        if len(current_staff) >= 3:
            staff_systems.append(current_staff)
        
        # Create staff line data in the expected format
        staff_line_data = []
        
        for system_idx, system in enumerate(staff_systems):
            for line_idx, idx in enumerate(system):
                pos, x1, x2 = run_data[idx]
                
                staff_line = {
                    "class_id": 0,
                    "class_name": "staff_line",
                    "confidence": 1.0,
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(pos - staff_height // 2),
                        "x2": float(x2),
                        "y2": float(pos + staff_height // 2),
                        "width": float(x2 - x1),
                        "height": float(staff_height),
                        "center_x": float((x1 + x2) / 2),
                        "center_y": float(pos)
                    },
                    "staff_system": system_idx,
                    "line_number": line_idx
                }
                
                staff_line_data.append(staff_line)
        
        return {
            "staff_systems": [
                {
                    "id": system_idx,
                    "lines": [
                        idx for idx, line in enumerate(staff_line_data) 
                        if line["staff_system"] == system_idx
                    ]
                }
                for system_idx in range(len(staff_systems))
            ],
            "detections": staff_line_data
        }
    
    def _detect_projection(self, img):
        """
        Detect staff lines using horizontal projection profiles
        Works well for clean images with consistent staff line thickness
        """
        height, width = img.shape
        
        # Create a horizontal projection profile
        projection = np.sum(img < 127, axis=1)
        
        # Smooth projection to reduce noise
        projection_smoothed = gaussian_filter1d(projection, sigma=1)
        
        # Find peaks in the projection
        peaks, properties = find_peaks(
            projection_smoothed, 
            height=np.max(projection_smoothed) * 0.3,  # Relative height threshold
            distance=2  # Reduced minimum distance to detect adjacent lines in same staff line
        )
        
        if len(peaks) < 5:
            return {"staff_systems": [], "detections": []}
        
        # Get peak prominence
        prominences = properties['prominences']
        
        # Filter by prominence
        min_prominence = np.max(prominences) * 0.3
        valid_peaks = peaks[prominences >= min_prominence]
        
        # If insufficient peaks, return empty result
        if len(valid_peaks) < 5:
            return {"staff_systems": [], "detections": []}
        
        # Group adjacent peaks that likely belong to the same staff line
        merged_peaks = []
        current_group = [valid_peaks[0]] if len(valid_peaks) > 0 else []
        
        for i in range(1, len(valid_peaks)):
            # If this peak is very close to the previous one (likely same staff line)
            if valid_peaks[i] - valid_peaks[i-1] <= self.line_merging_threshold:
                current_group.append(valid_peaks[i])
            else:
                # End of current group, calculate center of staff line
                if current_group:
                    # Use the position in the middle of the group as the center
                    center_peak = current_group[len(current_group) // 2]
                    merged_peaks.append(center_peak)
                    
                    # Start new group
                    current_group = [valid_peaks[i]]
        
        # Add the last group
        if current_group:
            center_peak = current_group[len(current_group) // 2]
            merged_peaks.append(center_peak)
        
        # Replace valid_peaks with merged_peaks
        valid_peaks = np.array(merged_peaks)
        
        # If we merged too much and don't have enough peaks, return empty result
        if len(valid_peaks) < 5:
            return {"staff_systems": [], "detections": []}
        
        # Calculate nearest line thickness based on the width of peaks
        line_thickness = []
        for peak in valid_peaks:
            # Find width at half height
            half_height = projection_smoothed[peak] / 2
            
            # Find points where projection crosses half height
            left = peak
            while left > 0 and projection_smoothed[left] > half_height:
                left -= 1
                
            right = peak
            while right < height-1 and projection_smoothed[right] > half_height:
                right += 1
            
            thickness = right - left
            line_thickness.append(thickness)
        
        # Get median thickness
        median_thickness = int(max(2, np.median(line_thickness)))
        
        # Get horizontal extents for each staff line
        staff_line_data = []
        
        for peak in valid_peaks:
            # Get row at this peak
            row = img[peak, :]
            
            # Find contiguous segments of staff line
            segments = []
            current_segment = []
            
            for x in range(width):
                if row[x] < 127:  # Black pixel
                    current_segment.append(x)
                else:  # White pixel
                    if len(current_segment) > width * 0.1:  # Min segment length
                        segments.append(current_segment)
                    current_segment = []
            
            # Handle last segment
            if len(current_segment) > width * 0.1:
                segments.append(current_segment)
            
            # Use longest segment as the staff line
            if segments:
                longest_segment = max(segments, key=len)
                x1 = longest_segment[0]
                x2 = longest_segment[-1]
                
                # Create staff line data
                y1 = max(0, peak - median_thickness // 2)
                y2 = min(height, peak + median_thickness // 2)
                
                staff_line_data.append({
                    "peak": peak,
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "thickness": median_thickness
                })
        
        # Sort staff lines by vertical position
        staff_line_data.sort(key=lambda x: x["peak"])
        
        # Calculate spacings between adjacent lines
        spacings = []
        for i in range(len(staff_line_data) - 1):
            spacing = staff_line_data[i+1]["peak"] - staff_line_data[i]["peak"]
            spacings.append(spacing)
        
        # Cluster spacings to find standard staff line spacing
        if len(spacings) > 1:
            spacing_array = np.array(spacings).reshape(-1, 1)
            clustering = DBSCAN(eps=3, min_samples=2).fit(spacing_array)
            
            # Find most common spacing cluster
            if len(np.unique(clustering.labels_)) <= 1:
                staff_spacing = np.median(spacings)
            else:
                unique_labels, counts = np.unique(clustering.labels_, return_counts=True)
                if -1 in unique_labels:  # Remove noise
                    noise_idx = np.where(unique_labels == -1)[0]
                    unique_labels = np.delete(unique_labels, noise_idx)
                    counts = np.delete(counts, noise_idx)
                
                if len(counts) == 0:
                    staff_spacing = np.median(spacings)
                else:
                    best_cluster = unique_labels[np.argmax(counts)]
                    staff_spacing = np.median([spacings[i] for i, label in enumerate(clustering.labels_) 
                                            if label == best_cluster])
        else:
            # Default spacing if we can't calculate
            staff_spacing = 20
        
        # Group staff lines into staff systems
        staff_systems = []
        current_staff = []
        
        for i, line in enumerate(staff_line_data):
            if i == 0:
                current_staff = [i]
            else:
                spacing = line["peak"] - staff_line_data[i-1]["peak"]
                
                # If this spacing is close to the expected staff line spacing
                if abs(spacing - staff_spacing) < staff_spacing * 0.25:
                    current_staff.append(i)
                    
                    # If we have 5 lines, it's a complete staff
                    if len(current_staff) == 5:
                        staff_systems.append(current_staff)
                        current_staff = []
                else:
                    # This is a line from a new staff system
                    if len(current_staff) >= 3:  # Partial staff needs at least 3 lines
                        staff_systems.append(current_staff)
                    current_staff = [i]
        
        # Add the last staff if it's not empty
        if len(current_staff) >= 3:
            staff_systems.append(current_staff)
        
        # Create final staff line data in the expected format
        final_staff_line_data = []
        
        for system_idx, system in enumerate(staff_systems):
            for line_idx, idx in enumerate(system):
                line = staff_line_data[idx]
                
                staff_line = {
                    "class_id": 0,
                    "class_name": "staff_line",
                    "confidence": 1.0,
                    "bbox": {
                        "x1": float(line["x1"]),
                        "y1": float(line["y1"]),
                        "x2": float(line["x2"]),
                        "y2": float(line["y2"]),
                        "width": float(line["x2"] - line["x1"]),
                        "height": float(line["thickness"]),
                        "center_x": float((line["x1"] + line["x2"]) / 2),
                        "center_y": float(line["peak"])
                    },
                    "staff_system": system_idx,
                    "line_number": line_idx
                }
                
                final_staff_line_data.append(staff_line)
        
        return {
            "staff_systems": [
                {
                    "id": system_idx,
                    "lines": [
                        idx for idx, line in enumerate(final_staff_line_data) 
                        if line["staff_system"] == system_idx
                    ]
                }
                for system_idx in range(len(staff_systems))
            ],
            "detections": final_staff_line_data
        }
    
    
    def _estimate_staff_line_parameters(self, elements_by_category, img_shape):
        """
        Estimate staff line parameters using categorized musical elements
        
        Args:
            elements_by_category: Dictionary with categorized musical elements
            img_shape: Tuple (height, width) of the image
                
        Returns:
            Dictionary with estimated staff parameters
        """
        height, width = img_shape
        print(f"Estimating staff parameters for image of size {width}x{height}")
        
        parameters = {
            "staff_line_thickness": 1,  # Default values
            "staff_line_spacing": 10,
            "staff_height": 40,
            "staff_positions": [],
            "staff_x_range": (0, width)
        }
        
        # Give highest priority to clefs for parameter estimation
        clefs = elements_by_category["clefs"]
        noteheads = elements_by_category["noteheads"]
        time_signatures = elements_by_category["time_signatures"]
        
        # Identify different clef types for better staff detection
        g_clefs = [c for c in clefs if c.get("class_id") == 15 or "gClef" in c.get("class_name", "")]
        f_clefs = [c for c in clefs if c.get("class_id") == 22 or "fClef" in c.get("class_name", "")]
        
        # Detect grand staff format (like piano music with both treble and bass clefs)
        grand_staff_detected = len(g_clefs) > 0 and len(f_clefs) > 0
        print(f"Detected {len(g_clefs)} G-clefs and {len(f_clefs)} F-clefs. Grand staff: {grand_staff_detected}")
        
        # First use clefs to estimate line spacing and thickness
        if clefs:
            # Clefs typically span the entire staff height
            clef_heights = [clef["bbox"]["height"] for clef in clefs]
            avg_clef_height = np.mean(clef_heights)
            
            # G clef (treble) typically spans about 3 staff spaces (4 lines)
            # F clef (bass) typically spans about 2 staff spaces (3 lines)
            
            if g_clefs:
                g_clef_heights = [clef["bbox"]["height"] for clef in g_clefs]
                avg_g_clef_height = np.mean(g_clef_heights)
                # Staff height is approximately the G clef height
                staff_height_from_g = avg_g_clef_height * 0.8
                parameters["staff_height"] = staff_height_from_g
            elif f_clefs:
                f_clef_heights = [clef["bbox"]["height"] for clef in f_clefs]
                avg_f_clef_height = np.mean(f_clef_heights)
                # Staff height is approximately 1.5x the F clef height
                staff_height_from_f = avg_f_clef_height * 1.2
                parameters["staff_height"] = staff_height_from_f
            else:
                # Generic clef height estimation
                parameters["staff_height"] = avg_clef_height * 0.9
            
            # Staff line spacing = staff height / 4 (5 lines = 4 spaces)
            parameters["staff_line_spacing"] = parameters["staff_height"] / 4.2
            
            # Estimate line thickness (typically 5-10% of spacing)
            parameters["staff_line_thickness"] = max(1, int(parameters["staff_line_spacing"] * 0.1))
            
            print(f"Based on clefs: staff height = {parameters['staff_height']:.2f}, " 
                f"spacing = {parameters['staff_line_spacing']:.2f}, "
                f"thickness = {parameters['staff_line_thickness']}")
        
        # If no clefs, try noteheads for spacing estimation
        elif noteheads:
            notehead_heights = [note["bbox"]["height"] for note in noteheads]
            avg_notehead_height = np.mean(notehead_heights)
            
            # Noteheads are typically about the same size as staff line spacing
            parameters["staff_line_spacing"] = avg_notehead_height
            parameters["staff_height"] = parameters["staff_line_spacing"] * 4  # 5 lines = 4 spaces
            parameters["staff_line_thickness"] = max(1, int(parameters["staff_line_spacing"] * 0.1))
            
            print(f"Based on noteheads: staff height = {parameters['staff_height']:.2f}, " 
                f"spacing = {parameters['staff_line_spacing']:.2f}, "
                f"thickness = {parameters['staff_line_thickness']}")
        
        # Find staff center positions
        # First, prioritize clefs for accurate vertical positioning
        if clefs:
            # Group clefs that likely belong to the same staff system
            clef_groups = []
            
            # Sort clefs from top to bottom
            sorted_clefs = sorted(clefs, key=lambda x: x["bbox"]["center_y"])
            
            # For grand staff, group pairs of clefs that are vertically close
            if grand_staff_detected:
                current_group = []
                vertical_threshold = parameters["staff_height"] * 3  # Maximum distance between grand staff clefs
                
                for clef in sorted_clefs:
                    if not current_group:
                        current_group.append(clef)
                    else:
                        last_clef = current_group[-1]
                        vertical_distance = abs(clef["bbox"]["center_y"] - last_clef["bbox"]["center_y"])
                        
                        # Check if this is likely part of the same grand staff
                        clef_type = "g" if clef.get("class_id") == 15 or "gClef" in clef.get("class_name", "") else "f"
                        last_type = "g" if last_clef.get("class_id") == 15 or "gClef" in last_clef.get("class_name", "") else "f"
                        
                        # If different clef types and close enough, they're likely the same grand staff
                        if clef_type != last_type and vertical_distance <= vertical_threshold:
                            current_group.append(clef)
                            clef_groups.append(current_group)
                            current_group = []
                        else:
                            # Either same type or too far apart, start new group
                            clef_groups.append([last_clef])
                            current_group = [clef]
                
                # Add the last group if not empty
                if current_group:
                    clef_groups.append(current_group)
            else:
                # For single staff systems, each clef gets its own group
                clef_groups = [[clef] for clef in sorted_clefs]
            
            # Process each clef group to determine staff positions
            for group in clef_groups:
                if not group:
                    continue
                    
                # For grand staff, calculate positions for both staves
                if len(group) > 1 and grand_staff_detected:
                    # Sort the group by y-position
                    group.sort(key=lambda x: x["bbox"]["center_y"])
                    
                    # Extract center positions for each clef
                    for clef in group:
                        parameters["staff_positions"].append(clef["bbox"]["center_y"])
                else:
                    # Single staff - use the clef's center position
                    parameters["staff_positions"].append(group[0]["bbox"]["center_y"])
        
        # If we don't have staff positions from clefs, try using other elements
        if not parameters["staff_positions"] and (noteheads or time_signatures):
            # Collect y-positions of all elements that could indicate staff positions
            y_positions = []
            
            # Prioritize time signatures, then noteheads and other elements
            for category in ["time_signatures", "noteheads", "rests"]:
                for element in elements_by_category[category]:
                    y_positions.append(element["bbox"]["center_y"])
            
            if y_positions:
                # Use clustering to find staff centers
                if len(y_positions) > 5:  # Need enough data points for clustering
                    y_array = np.array(y_positions).reshape(-1, 1)
                    clustering = DBSCAN(eps=parameters["staff_height"], min_samples=3).fit(y_array)
                    
                    # Get cluster centers
                    labels = clustering.labels_
                    unique_labels = set(labels) - {-1}  # Exclude noise
                    
                    # Extract center for each cluster
                    for label in unique_labels:
                        cluster_points = y_array[labels == label]
                        center = np.mean(cluster_points)
                        parameters["staff_positions"].append(float(center))
                else:
                    # Not enough points for clustering, use the mean
                    parameters["staff_positions"].append(float(np.mean(y_positions)))
        
        # If we still don't have any staff positions, make an educated guess
        if not parameters["staff_positions"]:
            # Guess based on typical layout - divide image into equal sections
            num_staves = max(1, min(5, height // 200))  # Limit based on image height
            
            for i in range(num_staves):
                position = height * (i + 1) / (num_staves + 1)
                parameters["staff_positions"].append(position)
            
            print(f"No reliable staff indicators found. Guessing {num_staves} staves.")
        
        # Determine horizontal extent (left to right boundaries) of staves
        all_elements = []
        for category in elements_by_category.values():
            all_elements.extend(category)
        
        if all_elements:
            # Calculate horizontal boundaries based on element positions
            x_positions = []
            for element in all_elements:
                if "bbox" in element:
                    x_positions.extend([element["bbox"]["x1"], element["bbox"]["x2"]])
            
            leftmost = max(0, min(x_positions) - parameters["staff_height"])
            rightmost = min(width, max(x_positions) + parameters["staff_height"])
            
            # For clefs, extend to the left
            if clefs:
                min_clef_x = min(clef["bbox"]["x1"] for clef in clefs)
                leftmost = min(leftmost, max(0, min_clef_x - parameters["staff_height"]))
        else:
            # No elements found, use full width
            leftmost = 0
            rightmost = width
        
        parameters["staff_x_range"] = (leftmost, rightmost)
        
        # Sort staff positions from top to bottom
        parameters["staff_positions"].sort()
        
        print(f"Estimated {len(parameters['staff_positions'])} staff positions at:")
        for idx, pos in enumerate(parameters["staff_positions"]):
            print(f"  Staff {idx}: y = {pos:.2f}")
        
        return parameters



    def _infer_from_musical_elements(self, img, detected_elements):
        """
        Infer staff lines using detected musical elements such as clefs, time signatures, and notes
        
        Args:
            img: Grayscale image
            detected_elements: Dictionary containing detected musical elements
            
        Returns:
            Dictionary containing inferred staff line information
        """
        height, width = img.shape
        staff_line_data = []
        staff_systems = []
        
        print(f"Attempting to infer staff lines from {len(detected_elements.get('detections', []))} musical elements")
        
        # Extract clefs, time signatures, and noteheads using predefined class IDs
        clefs = []
        time_signatures = []
        noteheads = []
        
        # Class ID ranges from your provided mapping
        clef_ids = [15, 22, 32, 42, 63, 88, 40]  # gClef, fClef, cClef, etc.
        timesig_ids = [37, 47, 48, 56, 60, 62, 65, 68, 75, 81, 87, 125]  # timeSig4, timeSig3, etc.
        notehead_ids = [1, 12, 27, 53, 61, 107, 111, 112, 176]  # noteheadBlack, noteheadHalf, etc.
        
        for element in detected_elements.get("detections", []):
            class_id = element.get("class_id", -1)
            class_name = element.get("class_name", "").lower()
            
            # Classify by ID if possible
            if class_id in clef_ids or "clef" in class_name:
                clefs.append(element)
            elif class_id in timesig_ids or "timesig" in class_name:
                time_signatures.append(element)
            elif class_id in notehead_ids or "note" in class_name:
                noteheads.append(element)
        
        print(f"Found {len(clefs)} clefs, {len(time_signatures)} time signatures, {len(noteheads)} noteheads")
        
        # Sort elements by their vertical position (y-coordinate)
        all_elements = clefs + time_signatures + noteheads
        all_elements.sort(key=lambda x: x.get("bbox", {}).get("center_y", 0))
        
        # Group elements that are likely on the same staff by vertical proximity
        element_groups = []
        current_group = []
        
        # Calculate typical element height as a reference
        if all_elements:
            element_heights = [element.get("bbox", {}).get("height", 0) for element in all_elements]
            median_element_height = np.median(element_heights) if element_heights else 20
            vertical_threshold = median_element_height * 3  # Threshold for grouping elements vertically
        else:
            vertical_threshold = 60  # Default threshold if no elements
        
        # Group elements by vertical proximity
        for i, element in enumerate(all_elements):
            if i == 0:
                current_group = [element]
            else:
                prev_y = all_elements[i-1].get("bbox", {}).get("center_y", 0)
                curr_y = element.get("bbox", {}).get("center_y", 0)
                
                if abs(curr_y - prev_y) <= vertical_threshold:
                    current_group.append(element)
                else:
                    if current_group:
                        element_groups.append(current_group)
                    current_group = [element]
        
        # Add the last group
        if current_group:
            element_groups.append(current_group)
        
        print(f"Grouped musical elements into {len(element_groups)} potential staff systems")
        
        # Process each group to infer a staff system
        for group_idx, group in enumerate(element_groups):
            if len(group) < 2:
                continue  # Need at least a few elements to infer staff lines
            
            # Use clefs if available, otherwise use the group's average position
            reference_elements = [e for e in group if e.get("class_id", -1) in clef_ids or "clef" in e.get("class_name", "").lower()]
            if not reference_elements:
                reference_elements = group
            
            # Extract y-positions
            y_positions = [e.get("bbox", {}).get("center_y", 0) for e in reference_elements]
            avg_y = np.mean(y_positions)
            
            # Estimate staff height based on element heights
            element_heights = [e.get("bbox", {}).get("height", 0) for e in group]
            avg_element_height = np.mean(element_heights)
            estimated_staff_height = avg_element_height * 2
            
            # Get typical width from the widest elements
            element_widths = [e.get("bbox", {}).get("width", 0) for e in group]
            element_widths.sort(reverse=True)
            avg_width = np.mean(element_widths[:min(3, len(element_widths))])
            
            # Find leftmost and rightmost position for staff width
            x_positions = [e.get("bbox", {}).get("center_x", 0) for e in group]
            leftmost = min(e.get("bbox", {}).get("x1", width/2) for e in group)
            rightmost = max(e.get("bbox", {}).get("x2", width/2) for e in group)
            
            # Extend staff beyond leftmost and rightmost elements
            staff_x1 = max(0, leftmost - avg_width)
            staff_x2 = min(width, rightmost + avg_width)
            
            # Estimate staff line spacing (typical staff has 4 spaces, 5 lines)
            line_spacing = estimated_staff_height / 4
            
            # Create 5 staff lines centered around the average y-position
            staff_lines = []
            line_thickness = max(1, int(line_spacing * 0.1))  # Estimate line thickness
            
            # Calculate the center position of the staff
            staff_center_y = avg_y
            
            # Calculate the top line position (2 spacings up from center)
            top_line_y = staff_center_y - line_spacing * 2
            
            # Create 5 staff lines
            for i in range(5):
                line_y = top_line_y + i * line_spacing
                
                # Create staff line data
                line_idx = len(staff_line_data)
                staff_line = {
                    "class_id": 0,
                    "class_name": "staff_line",
                    "confidence": 0.85,  # Lower confidence since it's inferred
                    "bbox": {
                        "x1": float(staff_x1),
                        "y1": float(line_y - line_thickness/2),
                        "x2": float(staff_x2),
                        "y2": float(line_y + line_thickness/2),
                        "width": float(staff_x2 - staff_x1),
                        "height": float(line_thickness),
                        "center_x": float((staff_x1 + staff_x2) / 2),
                        "center_y": float(line_y)
                    },
                    "staff_system": group_idx,
                    "line_number": i
                }
                
                staff_line_data.append(staff_line)
                staff_lines.append(line_idx)
            
            # Add this staff system
            staff_systems.append({"id": group_idx, "lines": staff_lines})
        
        print(f"Inferred {len(staff_systems)} staff systems with {len(staff_line_data)} total staff lines")
        
        return {
            "staff_systems": staff_systems,
            "detections": staff_line_data
        }

    def _refine_staff_systems(self, staff_data, detected_elements):
        """
        Refine existing staff systems using detected musical elements
        
        Args:
            staff_data: Existing staff line data
            detected_elements: Dictionary containing detected musical elements
            
        Returns:
            Updated staff line data
        """
        if not staff_data or "staff_systems" not in staff_data or not staff_data["staff_systems"]:
            return staff_data
        
        # Extract relevant musical elements using predefined class IDs
        clefs = []
        time_signatures = []
        noteheads = []
        
        # Class ID ranges from your provided mapping
        clef_ids = [15, 22, 32, 42, 63, 88, 40]  # gClef, fClef, cClef, etc.
        timesig_ids = [37, 47, 48, 56, 60, 62, 65, 68, 75, 81, 87, 125]  # timeSig4, timeSig3, etc.
        notehead_ids = [1, 12, 27, 53, 61, 107, 111, 112, 176]  # noteheadBlack, noteheadHalf, etc.
        
        for element in detected_elements.get("detections", []):
            class_id = element.get("class_id", -1)
            class_name = element.get("class_name", "").lower()
            
            # Classify by ID if possible
            if class_id in clef_ids or "clef" in class_name:
                clefs.append(element)
            elif class_id in timesig_ids or "timesig" in class_name:
                time_signatures.append(element)
            elif class_id in notehead_ids or "note" in class_name:
                noteheads.append(element)
        
        # If we don't have enough musical elements, return original data
        if len(clefs) + len(time_signatures) + len(noteheads) < 5:
            return staff_data
        
        # For each staff system, check if it has at least 5 lines
        updated_systems = []
        updated_detections = staff_data["detections"].copy()
        
        for system in staff_data["staff_systems"]:
            system_id = system["id"]
            
            # Get the lines for this system
            system_lines = [staff_data["detections"][line_idx] for line_idx in system["lines"]]
            
            # If we have fewer than 5 lines, try to infer the missing ones
            if len(system_lines) < 5:
                # Find elements that are likely associated with this staff system
                system_y_values = [line["bbox"]["center_y"] for line in system_lines]
                min_y = min(system_y_values) if system_y_values else 0
                max_y = max(system_y_values) if system_y_values else 0
                
                # Expand the range to ensure we capture all relevant elements
                staff_height = max_y - min_y
                min_y -= staff_height * 0.5
                max_y += staff_height * 0.5
                
                # Find musical elements within this vertical range
                system_elements = []
                for element in clefs + time_signatures + noteheads:
                    element_y = element.get("bbox", {}).get("center_y", 0)
                    if min_y <= element_y <= max_y:
                        system_elements.append(element)
                
                # If we have enough elements, try to infer the missing lines
                if len(system_elements) >= 2:
                    # Sort existing lines by y-position
                    system_lines.sort(key=lambda x: x["bbox"]["center_y"])
                    
                    # Calculate the average line spacing
                    if len(system_lines) >= 2:
                        line_spacings = []
                        for i in range(len(system_lines) - 1):
                            spacing = system_lines[i+1]["bbox"]["center_y"] - system_lines[i]["bbox"]["center_y"]
                            if spacing > 0:
                                line_spacings.append(spacing)
                        
                        avg_spacing = np.mean(line_spacings) if line_spacings else 10
                    else:
                        # Estimate from element heights
                        element_heights = [e.get("bbox", {}).get("height", 0) for e in system_elements]
                        avg_element_height = np.mean(element_heights) if element_heights else 40
                        avg_spacing = avg_element_height * 0.5
                    
                    # Determine the line thickness
                    line_thickness = max(1, int(avg_spacing * 0.1))
                    
                    # Get the x-range for the staff lines
                    x_values = []
                    for line in system_lines:
                        x_values.append(line["bbox"]["x1"])
                        x_values.append(line["bbox"]["x2"])
                    
                    for element in system_elements:
                        x_values.append(element.get("bbox", {}).get("x1", 0))
                        x_values.append(element.get("bbox", {}).get("x2", 0))
                    
                    min_x = min(x_values) if x_values else 0
                    max_x = max(x_values) if x_values else 1000
                    
                    # Determine which line positions we need to fill in
                    if system_lines:
                        # Create a complete sequence of 5 lines
                        if len(system_lines) == 1:
                            # If we only have one line, assume it's the middle line (line 2)
                            center_y = system_lines[0]["bbox"]["center_y"]
                            line_positions = [
                                center_y - 2 * avg_spacing,
                                center_y - avg_spacing,
                                center_y,
                                center_y + avg_spacing,
                                center_y + 2 * avg_spacing
                            ]
                        else:
                            # Use existing lines to extrapolate the full staff
                            first_y = system_lines[0]["bbox"]["center_y"]
                            last_y = system_lines[-1]["bbox"]["center_y"]
                            
                            # Calculate how many lines we need above and below
                            range_y = last_y - first_y
                            avg_gap = range_y / (len(system_lines) - 1) if len(system_lines) > 1 else avg_spacing
                            
                            # Generate all 5 line positions
                            if len(system_lines) < 3:
                                # Assume our lines are at the top or middle of the staff
                                center_idx = len(system_lines) // 2
                                center_y = system_lines[center_idx]["bbox"]["center_y"]
                                
                                line_positions = [
                                    center_y - (2 - center_idx) * avg_gap,
                                    center_y - (1 - center_idx) * avg_gap,
                                    center_y,
                                    center_y + (1 + center_idx) * avg_gap,
                                    center_y + (2 + center_idx) * avg_gap
                                ]
                            else:
                                # We have enough lines to make a good guess at the complete staff
                                full_range = 4 * avg_gap
                                top_y = first_y - (first_y - (last_y - full_range)) / 2
                                
                                line_positions = [
                                    top_y,
                                    top_y + avg_gap,
                                    top_y + 2 * avg_gap,
                                    top_y + 3 * avg_gap,
                                    top_y + 4 * avg_gap
                                ]
                    else:
                        # If we have no existing lines, use elements to estimate staff position
                        element_y_values = [e.get("bbox", {}).get("center_y", 0) for e in system_elements]
                        center_y = np.mean(element_y_values) if element_y_values else 500
                        
                        line_positions = [
                            center_y - 2 * avg_spacing,
                            center_y - avg_spacing,
                            center_y,
                            center_y + avg_spacing,
                            center_y + 2 * avg_spacing
                        ]
                    
                    # Create new lines for the missing positions
                    existing_positions = [line["bbox"]["center_y"] for line in system_lines]
                    new_line_indices = []
                    
                    for i, pos in enumerate(line_positions):
                        # Check if this position already has a line
                        has_line = False
                        for y in existing_positions:
                            if abs(y - pos) < avg_spacing * 0.3:
                                has_line = True
                                break
                        
                        if not has_line:
                            # Add a new line
                            new_line = {
                                "class_id": 0,
                                "class_name": "staff_line",
                                "confidence": 0.75,  # Lower confidence for inferred lines
                                "bbox": {
                                    "x1": float(min_x),
                                    "y1": float(pos - line_thickness/2),
                                    "x2": float(max_x),
                                    "y2": float(pos + line_thickness/2),
                                    "width": float(max_x - min_x),
                                    "height": float(line_thickness),
                                    "center_x": float((min_x + max_x) / 2),
                                    "center_y": float(pos)
                                },
                                "staff_system": system_id,
                                "line_number": i
                            }
                            
                            updated_detections.append(new_line)
                            new_line_indices.append(len(updated_detections) - 1)
                    
                    # Update the system's line indices
                    updated_line_indices = system["lines"] + new_line_indices
                    
                    # Sort the lines by vertical position and update line numbers
                    line_y_pairs = [(idx, updated_detections[idx]["bbox"]["center_y"]) for idx in updated_line_indices]
                    line_y_pairs.sort(key=lambda x: x[1])
                    
                    for i, (idx, _) in enumerate(line_y_pairs):
                        updated_detections[idx]["line_number"] = i
                    
                    updated_system = {
                        "id": system_id,
                        "lines": [idx for idx, _ in line_y_pairs]
                    }
                else:
                    # Not enough elements to infer missing lines
                    updated_system = system
            else:
                # System already has enough lines
                updated_system = system
            
            updated_systems.append(updated_system)
        
        return {
            "staff_systems": updated_systems,
            "detections": updated_detections
        }
    
    def _classify_musical_elements(self, detected_elements):
        """
        Classify detected musical elements into categories that can help with staff line inference
        
        Args:
            detected_elements: Dictionary containing detected musical elements
            
        Returns:
            Dictionary with categorized musical elements
        """
        categories = {
            "clefs": [],
            "time_signatures": [],
            "key_signatures": [],
            "noteheads": [],
            "rests": [],
            "barlines": [],
            "accidentals": [],
            "other": []
        }
        
        # Classification keywords for each category based on class names from your dataset
        keywords = {
            "clefs": ["clef", "gClef", "fClef", "cClef", "gClef8va", "gClef8vb", "fClef8vb", "unpitchedPercussionClef"],
            "time_signatures": ["timeSig", "timeSignatureComponent", "timeSigCommon", "timeSigCutCommon"],
            "noteheads": ["notehead", "noteheadBlack", "noteheadHalf", "noteheadWhole", "noteheadX"],
            "rests": ["rest", "restWhole", "restHalf", "restQuarter", "rest8th", "rest16th", "rest32nd", "rest64th", "rest128th", "rest256th"],
            "barlines": ["barline", "systemicBarline"],
            "accidentals": ["accidental", "accidentalFlat", "accidentalSharp", "accidentalNatural", "accidentalDoubleSharp", "accidentalDoubleFlat", "accidentalTripleSharp", "accidentalQuarterTone", "accidentalKomaSharp", "accidentalThreeQuarterTones"]
        }
        
        # Mapping of class IDs to categories based on your provided class mapping
        class_id_categories = {
            # Clefs
            15: "clefs",  # gClef
            22: "clefs",  # fClef
            32: "clefs",  # cClef
            42: "clefs",  # gClef8vb
            63: "clefs",  # fClef8vb
            88: "clefs",  # gClef8va
            40: "clefs",  # unpitchedPercussionClef
            
            # Time signatures
            37: "time_signatures",  # timeSig4
            47: "time_signatures",  # timeSig3
            48: "time_signatures",  # timeSig8
            56: "time_signatures",  # timeSig2
            60: "time_signatures",  # timeSigCommon
            62: "time_signatures",  # timeSig6
            65: "time_signatures",  # timeSig5
            68: "time_signatures",  # timeSignatureComponent
            75: "time_signatures",  # timeSigCutCommon
            81: "time_signatures",  # timeSig9
            87: "time_signatures",  # timeSig7
            125: "time_signatures",  # timeSig1
            
            # Noteheads
            1: "noteheads",   # noteheadBlack
            12: "noteheads",  # noteheadHalf
            27: "noteheads",  # noteheadWhole
            53: "noteheads",  # noteheadXBlack
            61: "noteheads",  # noteheadSlashVerticalEnds
            107: "noteheads", # noteheadDiamondBlack
            111: "noteheads", # mensuralNoteheadMinimaWhite
            112: "noteheads", # noteheadDoubleWholeSquare
            176: "noteheads", # noteheadXHalf
            
            # Rests
            8: "rests",   # restWhole
            9: "rests",   # rest8th
            16: "rests",  # restQuarter
            17: "rests",  # rest16th
            26: "rests",  # rest32nd
            30: "rests",  # restHalf
            76: "rests",  # rest
            78: "rests",  # rest128th
            116: "rests", # rest64th
            137: "rests", # rest256th
            
            # Barlines
            4: "barlines",  # barline
            10: "barlines", # systemicBarline
            
            # Accidentals
            6: "accidentals",  # accidentalFlat
            7: "accidentals",  # accidentalSharp
            13: "accidentals", # accidentalNatural
            58: "accidentals", # accidentalDoubleSharp
            64: "accidentals", # accidentalDoubleFlat
            85: "accidentals", # accidentalQuarterToneSharpStein
            90: "accidentals", # accidentalQuarterToneFlatStein
            98: "accidentals", # accidentalTripleSharp
            101: "accidentals", # accidentalThreeQuarterTonesSharpStein
            141: "accidentals", # accidentalKomaSharp
        }
        
        # Process each detected element
        for element in detected_elements.get("detections", []):
            class_id = element.get("class_id", -1)
            class_name = element.get("class_name", "").lower()
            
            # First try to categorize by class_id
            categorized = False
            if class_id in class_id_categories:
                category = class_id_categories[class_id]
                categories[category].append(element)
                categorized = True
            else:
                # Try to categorize by class_name
                for category, terms in keywords.items():
                    if any(term.lower() in class_name for term in terms):
                        categories[category].append(element)
                        categorized = True
                        break
            
            # If no category matches, add to "other"
            if not categorized:
                categories["other"].append(element)
        
        # Add some debugging information
        total_elements = len(detected_elements.get("detections", []))
        categorized_elements = sum(len(v) for k, v in categories.items() if k != "other")
        
        print(f"Classified {categorized_elements} of {total_elements} elements:")
        for category, elements in categories.items():
            print(f"  - {category}: {len(elements)}")
        
        return categories

    def _regenerate_staff_from_notation(self, img, detected_elements):
        """
        Regenerate complete staff systems based on detected notation elements
        
        Args:
            img: Grayscale image
            detected_elements: Dictionary containing detected musical elements
            
        Returns:
            Dictionary containing staff line information
        """
        height, width = img.shape
        print(f"Starting staff regeneration from detected elements in image of size {width}x{height}")
        
        # Classify detected elements
        elements_by_category = self._classify_musical_elements(detected_elements)
        
        # We'll prioritize clefs as the primary staff line indicators
        clefs = elements_by_category["clefs"]
        noteheads = elements_by_category["noteheads"]
        time_signatures = elements_by_category["time_signatures"]
        
        # Print counts of each category
        for category, elements in elements_by_category.items():
            print(f"  - {category}: {len(elements)}")
        
        # Determine if we have a grand staff (piano music with treble and bass clefs)
        g_clefs = [c for c in clefs if c.get("class_id") == 15 or "gClef" in c.get("class_name", "")]
        f_clefs = [c for c in clefs if c.get("class_id") == 22 or "fClef" in c.get("class_name", "")]
        
        grand_staff_detected = len(g_clefs) > 0 and len(f_clefs) > 0
        print(f"Detected {len(g_clefs)} G-clefs and {len(f_clefs)} F-clefs. Grand staff: {grand_staff_detected}")
        
        # Group clefs that should belong to the same vertical position
        clef_groups = []
        if clefs:
            # Sort clefs from top to bottom
            sorted_clefs = sorted(clefs, key=lambda x: x["bbox"]["center_y"])
            
            # Group clefs that are vertically close to each other (grand staff)
            current_group = [sorted_clefs[0]]
            clef_heights = [c["bbox"]["height"] for c in clefs]
            avg_clef_height = np.mean(clef_heights)
            vertical_threshold = avg_clef_height * 3  # Maximum distance for same system
            
            for clef in sorted_clefs[1:]:
                # Check if this clef is close to the last clef in the current group
                last_clef = current_group[-1]
                vertical_distance = abs(clef["bbox"]["center_y"] - last_clef["bbox"]["center_y"])
                
                if vertical_distance <= vertical_threshold and (
                        # For grand staff, we want one G and one F clef per group
                        (("gClef" in last_clef.get("class_name", "") and "fClef" in clef.get("class_name", "")) or
                        ("fClef" in last_clef.get("class_name", "") and "gClef" in clef.get("class_name", "")) or
                        # If clef types are the same, they might be for different instruments
                        vertical_distance <= avg_clef_height * 1.5)):
                    current_group.append(clef)
                else:
                    # Start a new group
                    clef_groups.append(current_group)
                    current_group = [clef]
            
            # Add the last group
            if current_group:
                clef_groups.append(current_group)
        
        print(f"Grouped clefs into {len(clef_groups)} staff systems")
        
        # Estimate staff parameters based on clefs and other elements
        staff_positions = []
        staff_line_spacing = 0
        staff_line_thickness = 1
        
        # Get spacing from noteheads if available
        if noteheads:
            notehead_heights = [note["bbox"]["height"] for note in noteheads]
            avg_notehead_height = np.mean(notehead_heights)
            # Staff line spacing is typically similar to notehead height
            staff_line_spacing = avg_notehead_height
            staff_line_thickness = max(1, int(staff_line_spacing * 0.1))
        
        # Refine with clef information if available
        if clefs:
            clef_heights = [clef["bbox"]["height"] for clef in clefs]
            avg_clef_height = np.mean(clef_heights)
            
            # Staff height is typically a bit smaller than clef height
            staff_height = avg_clef_height * 0.85
            
            # Override spacing calculation if clefs are available
            staff_line_spacing = staff_height / 4  # 5 lines = 4 spaces
            staff_line_thickness = max(1, int(staff_line_spacing * 0.1))
            
            # For each clef group, determine the center position of the staff
            for group in clef_groups:
                if not group:
                    continue
                    
                # Calculate the center position for this staff system
                g_clefs_in_group = [c for c in group if "gClef" in c.get("class_name", "")]
                f_clefs_in_group = [c for c in group if "fClef" in c.get("class_name", "")]
                
                if g_clefs_in_group and f_clefs_in_group:
                    # For grand staff, use the midpoint between G and F clefs
                    g_clef_y = g_clefs_in_group[0]["bbox"]["center_y"]
                    f_clef_y = f_clefs_in_group[0]["bbox"]["center_y"]
                    
                    # Add both staff positions
                    treble_staff_center = g_clef_y
                    bass_staff_center = f_clef_y
                    
                    staff_positions.append(treble_staff_center)
                    staff_positions.append(bass_staff_center)
                else:
                    # For single staff, use the clef position
                    group_y_positions = [clef["bbox"]["center_y"] for clef in group]
                    center_y = np.mean(group_y_positions)
                    staff_positions.append(center_y)
        
        # If we still don't have staff positions, try to infer from other elements
        if not staff_positions and (noteheads or time_signatures):
            # Use a clustering approach to find staff centers
            y_positions = []
            
            for category in ["time_signatures", "noteheads", "rests", "accidentals"]:
                for element in elements_by_category[category]:
                    y_positions.append(element["bbox"]["center_y"])
            
            if y_positions:
                # Use clustering to find staff centers
                y_array = np.array(y_positions).reshape(-1, 1)
                
                if len(y_positions) > 4:  # Need enough points for clustering
                    # Use DBSCAN for clustering
                    clustering = DBSCAN(eps=staff_line_spacing*3, min_samples=3).fit(y_array)
                    
                    # Get cluster centers
                    labels = clustering.labels_
                    unique_labels = set(labels)
                    
                    for label in unique_labels:
                        if label == -1:  # Skip noise
                            continue
                        
                        # Get center of this cluster
                        cluster_points = y_array[labels == label]
                        center = np.mean(cluster_points)
                        staff_positions.append(float(center))
                else:
                    # Not enough points for clustering, use mean
                    staff_positions.append(float(np.mean(y_positions)))
        
        # If we still don't have any staff positions, make a guess
        if not staff_positions:
            print("No reliable staff position indicators found. Making an educated guess.")
            num_staves = max(1, min(5, height // 200))
            
            for i in range(num_staves):
                position = height * (i + 1) / (num_staves + 1)
                staff_positions.append(position)
        
        # Check if we likely missed any staff systems
        expected_staff_count = len(clefs)
        if grand_staff_detected:
            # For grand staff, we expect half as many systems as clefs
            expected_staff_count = max(len(g_clefs), len(f_clefs))
        
        print(f"Found {len(staff_positions)} staff positions, expected around {expected_staff_count}")
        
        # If we have fewer positions than expected, check for missing systems
        if len(staff_positions) < expected_staff_count and len(staff_positions) >= 2:
            # Sort positions from top to bottom
            staff_positions.sort()
            
            # Calculate typical distance between staff systems
            staff_spacings = []
            for i in range(len(staff_positions) - 1):
                spacing = staff_positions[i+1] - staff_positions[i]
                staff_spacings.append(spacing)
            
            median_spacing = np.median(staff_spacings)
            print(f"Median spacing between detected staff systems: {median_spacing:.2f} pixels")
            
            # Look for large gaps that might indicate missing staff systems
            missing_positions = []
            for i in range(len(staff_positions) - 1):
                gap = staff_positions[i+1] - staff_positions[i]
                
                if gap > median_spacing * 1.7:  # Threshold for missing system
                    # How many systems could fit in this gap?
                    systems_in_gap = round(gap / median_spacing) - 1
                    
                    for j in range(1, systems_in_gap + 1):
                        # Place a new staff system in the gap
                        new_pos = staff_positions[i] + j * (gap / (systems_in_gap + 1))
                        missing_positions.append(new_pos)
                        print(f"Adding missing staff system at position {new_pos:.2f}")
            
            # Add the missing positions
            staff_positions.extend(missing_positions)
            staff_positions.sort()  # Re-sort
        
        # Determine horizontal extent of staves
        x_positions = []
        
        for category in elements_by_category.values():
            for element in category:
                if "bbox" in element:
                    x_positions.extend([element["bbox"]["x1"], element["bbox"]["x2"]])
        
        if x_positions:
            # Add margins
            min_x = max(0, min(x_positions) - staff_line_spacing)
            max_x = min(width, max(x_positions) + staff_line_spacing)
        else:
            # Default to full width if no elements
            min_x = 0
            max_x = width
        
        # Final check: verify staff positions against the image content
        refined_positions = []
        staff_height = staff_line_spacing * 4  # Height from top to bottom line
        
        for center_y in staff_positions:
            # Define search range around the estimated position
            search_range = int(staff_line_spacing)
            best_score = 0
            best_position = center_y
            
            # Search for the best alignment with actual dark lines in the image
            for offset in range(-search_range, search_range + 1):
                test_center = center_y + offset
                
                # Calculate positions of the 5 staff lines
                top_line = test_center - staff_line_spacing * 2
                line_positions = [top_line + i * staff_line_spacing for i in range(5)]
                
                # Score this alignment by checking for dark pixels at each line position
                score = 0
                for line_y in line_positions:
                    line_y_int = int(line_y)
                    if 0 <= line_y_int < height:
                        # Sample points along this horizontal line
                        sample_width = int((max_x - min_x) * 0.8)  # Use 80% of the width for sampling
                        sample_start = int(min_x + (max_x - min_x) * 0.1)  # Start 10% in from the left
                        sample_points = np.linspace(sample_start, sample_start + sample_width, 20, dtype=int)
                        
                        # Count dark pixels at sample points
                        dark_count = sum(1 for x in sample_points if x < width and img[line_y_int, x] < 127)
                        score += dark_count / len(sample_points)
                
                if score > best_score:
                    best_score = score
                    best_position = test_center
            
            # Use the refined position if it has a reasonable score
            if best_score > 0.3:  # Threshold may need tuning
                refined_positions.append(best_position)
                print(f"Refined staff position from {center_y:.2f} to {best_position:.2f}, score: {best_score:.2f}")
            else:
                # Keep the original position
                refined_positions.append(center_y)
                print(f"Keeping original staff position {center_y:.2f}, score too low: {best_score:.2f}")
        
        # Create staff lines for each staff position
        all_staff_lines = []
        staff_systems = []
        
        for staff_idx, center_y in enumerate(refined_positions):
            # Determine clef type for this staff
            clef_type = None
            clef_center_y = None
            
            # Check for clefs in the vicinity of this staff position
            for clef in clefs:
                clef_y = clef.get("bbox", {}).get("center_y", 0)
                if abs(clef_y - center_y) < staff_line_spacing * 3:
                    # Found a clef near this staff position
                    if clef.get("class_id") == 15 or "gClef" in clef.get("class_name", "").lower():
                        clef_type = 'g'
                        clef_center_y = clef_y
                        break
                    elif clef.get("class_id") == 22 or "fClef" in clef.get("class_name", "").lower():
                        clef_type = 'f'
                        clef_center_y = clef_y
                        break
            
            # If we found a clef, use its position instead of the estimated center
            if clef_center_y is not None:
                # Adjust center_y based on clef type for proper alignment
                if clef_type == 'g':
                    # For G-clef, line 2 is approximately 20px below clef center
                    # So we need to adjust center_y to compensate
                    center_y = clef_center_y + 20
                elif clef_type == 'f':
                    # For F-clef, line 4 is approximately 5px above clef center
                    center_y = clef_center_y - 5
            
            # Calculate the top line position (2 spacings up from center)
            top_line_y = center_y - staff_line_spacing * 3.5
            
            # Create 5 staff lines
            staff_lines = []
            
            for i in range(5):
                line_y = top_line_y + i * staff_line_spacing
                
                if 0 <= line_y < height:  # Ensure line is within image
                    staff_line = {
                        "class_id": 0,
                        "class_name": "staff_line",
                        "confidence": 0.9,  # High confidence for clef-based inference
                        "bbox": {
                            "x1": float(min_x),
                            "y1": float(line_y - staff_line_thickness/2),
                            "x2": float(max_x),
                            "y2": float(line_y + staff_line_thickness/2),
                            "width": float(max_x - min_x),
                            "height": float(staff_line_thickness),
                            "center_x": float((min_x + max_x) / 2),
                            "center_y": float(line_y)
                        },
                        "staff_system": staff_idx,
                        "line_number": i
                    }
                    
                    all_staff_lines.append(staff_line)
                    staff_lines.append(len(all_staff_lines) - 1)
            
            # Add this staff system
            if staff_lines:
                staff_systems.append({
                    "id": staff_idx,
                    "lines": staff_lines
                })
        # At the end of the _regenerate_staff_from_notation function add:
        print(f"Generated {len(staff_systems)} staff systems with {len(all_staff_lines)} total staff lines")
            
        return {
            "staff_systems": staff_systems,
            "detections": all_staff_lines
        }
                

 
  
    def _create_staff_lines_from_parameters(self, parameters, staff_idx, img_shape, clef_type=None, clefs=None):
        """
        Create staff lines based on estimated parameters with precise clef alignment

        Args:
            parameters: Dictionary with staff parameters
            staff_idx: Index of the staff system
            img_shape: Tuple (height, width) of the image
            clef_type: Type of clef to align with ('g', 'f', or None)
            clefs: List of clef detections to use for precise alignment

        Returns:
            List of staff line dictionaries
        """
        height, width = img_shape
        staff_lines = []

        if staff_idx < len(parameters["staff_positions"]):
            center_y = parameters["staff_positions"][staff_idx]
            spacing = parameters["staff_line_spacing"]
            thickness = parameters["staff_line_thickness"]
            x1, x2 = parameters["staff_x_range"]

            # If we have clefs and a clef type, use precise clef alignment
            if clefs:
                matching_clef = None
                for clef in clefs:
                    cid = clef.get("class_id")
                    if cid in [15, 42, 88, 22, 63, 32, 40]:  # Known clef IDs
                        matching_clef = clef
                        break

                if matching_clef:
                    clef_box = matching_clef["bbox"]
                    clef_top = clef_box["y1"]
                    clef_height = clef_box["height"]
                    clef_class_id = matching_clef["class_id"]

                    def infer_staff_from_clef_box(clef_top, clef_height, clef_class_id):
                        clef_mappings = {
                            15: (1, 0.62, 4.8),     # gClef
                            42: (1, 0.62, 4.8),     # gClef8vb
                            88: (1, 0.62, 4.8),     # gClef8va
                            22: (3, 0.267, 3.6),    # fClef
                            63: (3, 0.267, 3.6),    # fClef8vb
                            32: (2, 0.5, 4.2),      # cClef (e.g. alto clef)
                            40: (2, 0.5, 4.0),      # unpitchedPercussionClef
                        }
                        if clef_class_id not in clef_mappings:
                            return None, None
                        line_index, offset_ratio, spacing_divisor = clef_mappings[clef_class_id]
                        ref_line_y = clef_top + offset_ratio * clef_height
                        spacing = clef_height / spacing_divisor
                        top_line_y = ref_line_y - spacing * line_index
                        return [top_line_y + i * spacing for i in range(5)], spacing

                    line_positions, spacing = infer_staff_from_clef_box(clef_top, clef_height, clef_class_id)

                    if not line_positions:
                        top_line_y = center_y - spacing * 2
                        line_positions = [top_line_y + i * spacing for i in range(5)]
                else:
                    top_line_y = center_y - spacing * 2
                    line_positions = [top_line_y + i * spacing for i in range(5)]
            else:
                top_line_y = center_y - spacing * 2
                line_positions = [top_line_y + i * spacing for i in range(5)]

            for i, line_y in enumerate(line_positions):
                if 0 <= line_y < height:
                    staff_line = {
                        "class_id": 0,
                        "class_name": "staff_line",
                        "confidence": 0.9,
                        "bbox": {
                            "x1": float(x1),
                            "y1": float(line_y - thickness / 2),
                            "x2": float(x2),
                            "y2": float(line_y + thickness / 2),
                            "width": float(x2 - x1),
                            "height": float(thickness),
                            "center_x": float((x1 + x2) / 2),
                            "center_y": float(line_y)
                        },
                        "staff_system": staff_idx,
                        "line_number": 4 - i
                    }
                    staff_lines.append(staff_line)

        return staff_lines

            
    def visualize(self, image_path, staff_data, output_path=None, plot_details=True):
        """
        Visualize detected staff lines on the image
        
        Args:
            image_path: Path to the music score image
            staff_data: Staff line data from detect method
            output_path: Output path for the visualization image
            plot_details: Whether to include additional detail plots
        """
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path
            
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Create figure
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        
        # Colors for different staff systems
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        # Draw staff lines
        for i, system in enumerate(staff_data["staff_systems"]):
            color = colors[i % len(colors)]
            for line_idx in system["lines"]:
                line = staff_data["detections"][line_idx]
                x1 = line["bbox"]["x1"]
                y1 = line["bbox"]["center_y"]
                x2 = line["bbox"]["x2"]
                y2 = line["bbox"]["center_y"]
                
                # Plot staff line
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=2)
                
                # Draw bounding box to show thickness
                y_top = line["bbox"]["y1"]
                y_bottom = line["bbox"]["y2"]
                plt.plot([x1, x2, x2, x1, x1], [y_top, y_top, y_bottom, y_bottom, y_top], 
                        color=color, linestyle='--', linewidth=1, alpha=0.5)
                
                # Label staff system and line
                plt.text(x1 - 40, y1, f"S{i}L{line['line_number']}", 
                         color=color, fontsize=8, va='center')
        
        plt.title("Detected Staff Lines")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved staff line visualization to {output_path}")
        else:
            plt.show()
    
    def save_data(self, staff_data, output_path):
        """
        Save staff line data to a JSON file
        
        Args:
            staff_data: Staff line data
            output_path: Output path for the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(staff_data, f, indent=2)
        print(f"Saved staff line data to {output_path}")


def detect_and_save_staff_lines(image_path, output_dir="results/staff_lines", detected_elements=None):
    """
    Detect staff lines in an image and save the results
    
    Args:
        image_path: Path to the music score image
        output_dir: Output directory for staff line data and visualization
        detected_elements: Optional dictionary of previously detected musical elements
        
    Returns:
        Path to the saved staff line data JSON file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector with line merging threshold
    detector = AdaptiveStaffDetector(line_merging_threshold=5)
    
    # Check if image_path is a directory or a file
    if os.path.isdir(image_path):
        # Process all images in directory
        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(list(Path(image_path).glob(f"*{ext}")))
        
        json_paths = []
        for img_file in image_files:
            try:
                # Generate output paths
                json_output_path = os.path.join(output_dir, f"{img_file.stem}_staff_lines.json")
                visualization_path = os.path.join(output_dir, f"{img_file.stem}_staff_lines.png")
                
                # Get detected elements for this image if available
                img_elements = None
                if detected_elements and img_file.stem in detected_elements:
                    img_elements = detected_elements[img_file.stem]
                
                # Detect staff lines
                staff_data = detector.detect(str(img_file), img_elements)
                
                # Save data and visualization
                detector.save_data(staff_data, json_output_path)
                detector.visualize(str(img_file), staff_data, visualization_path)
                
                json_paths.append(json_output_path)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
        
        return json_paths
    else:
        # Process single image
        try:
            # Generate output paths
            img_name = os.path.basename(image_path)
            img_stem = os.path.splitext(img_name)[0]
            json_output_path = os.path.join(output_dir, f"{img_stem}_staff_lines.json")
            visualization_path = os.path.join(output_dir, f"{img_stem}_staff_lines.png")
            
            # Detect staff lines
            staff_data = detector.detect(image_path, detected_elements)
            
            # Save data and visualization
            detector.save_data(staff_data, json_output_path)
            detector.visualize(image_path, staff_data, visualization_path)
            
            return json_output_path
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None


def merge_detections(staff_line_file, symbol_detection_file, output_file):
    """
    Merge staff line detections with other symbol detections
    
    Args:
        staff_line_file: Path to the staff line detection JSON file
        symbol_detection_file: Path to the symbol detection JSON file
        output_file: Path to save the merged detections
        
    Returns:
        Path to the merged detections file
    """
    # Load staff line data
    with open(staff_line_file, 'r') as f:
        staff_data = json.load(f)
    
    # Load symbol detection data
    with open(symbol_detection_file, 'r') as f:
        symbol_data = json.load(f)
    
    # Merge detections
    merged_detections = staff_data["detections"] + symbol_data["detections"]
    
    # Save merged data
    merged_data = {"detections": merged_detections}
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged detections saved to {output_file}")
    return output_file


def main():
    """Main function to run the staff line detection pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect staff lines in music score images")
    parser.add_argument("--image", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--output-dir", type=str, default="results/staff_lines",
                        help="Output directory for staff line data and visualization")
    parser.add_argument("--line-merging-threshold", type=int, default=5,
                        help="Maximum vertical distance to consider adjacent rows as the same staff line")
    parser.add_argument("--object-detection", type=str, default=None,
                        help="Optional path to object detection JSON to assist staff line detection")
    
    args = parser.parse_args()
    
    # Load detected musical elements if provided
    detected_elements = None
    if args.object_detection and os.path.exists(args.object_detection):
        try:
            with open(args.object_detection, 'r') as f:
                detected_elements = json.load(f)
                print(f"Loaded detected elements from {args.object_detection}")
        except Exception as e:
            print(f"Error loading detected elements: {e}")
    
    # Run staff line detection
    detect_and_save_staff_lines(args.image, args.output_dir, detected_elements)


if __name__ == "__main__":
    main()