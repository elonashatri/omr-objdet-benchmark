import os
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN

class AdaptiveStaffDetector:
    """Staff line detector that automatically adapts to different score characteristics"""
    
    def __init__(self, line_merging_threshold=5):
        """
        Initialize the adaptive staff detector
        
        Args:
            line_merging_threshold: Maximum vertical distance to consider adjacent 
                                   rows as part of the same staff line
        """
        self.line_merging_threshold = line_merging_threshold
    
    def detect(self, image_path):
        """
        Detect staff lines in a music score image by trying multiple methods
        
        Args:
            image_path: Path to the music score image
            
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
            
        # Try multiple detection methods in order of reliability
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
        
        # If no methods worked, return empty data
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
    
    def _detect_hough(self, img):
        """
        Detect staff lines using Hough transform
        More robust for complex images but may be less precise
        """
        height, width = img.shape
        
        # Apply edge detection to highlight staff lines
        edges = cv2.Canny(img, 50, 150)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi/180, 
            threshold=width//3,  # Minimum line length
            minLineLength=width//2,
            maxLineGap=width//10
        )
        
        if lines is None or len(lines) < 5:
            return {"staff_systems": [], "detections": []}
        
        # Filter horizontal lines
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle to horizontal
            if x2 - x1 == 0:  # Vertical line
                continue
                
            angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            
            # If line is close to horizontal
            if angle < 5:
                # Calculate average y-position
                y_avg = (y1 + y2) // 2
                horizontal_lines.append({
                    "x1": min(x1, x2),
                    "y1": y_avg,
                    "x2": max(x1, x2),
                    "y2": y_avg,
                    "length": abs(x2 - x1)
                })
        
        # Sort by y-position
        horizontal_lines.sort(key=lambda x: x["y1"])
        
        # Group nearby horizontal lines that likely belong to the same staff line
        merged_lines = []
        
        if not horizontal_lines:
            return {"staff_systems": [], "detections": []}
            
        current_group = [horizontal_lines[0]]
        
        for i in range(1, len(horizontal_lines)):
            # If this line is close to the previous group
            if abs(horizontal_lines[i]["y1"] - current_group[-1]["y1"]) <= self.line_merging_threshold:
                current_group.append(horizontal_lines[i])
            else:
                # Process the current group
                if current_group:
                    # Merge the group into a single line
                    x1 = min(line["x1"] for line in current_group)
                    x2 = max(line["x2"] for line in current_group)
                    y_avg = sum(line["y1"] for line in current_group) // len(current_group)
                    
                    merged_lines.append({
                        "x1": x1,
                        "y1": y_avg,
                        "x2": x2,
                        "y2": y_avg,
                        "length": x2 - x1
                    })
                
                # Start a new group
                current_group = [horizontal_lines[i]]
        
        # Process the last group
        if current_group:
            x1 = min(line["x1"] for line in current_group)
            x2 = max(line["x2"] for line in current_group)
            y_avg = sum(line["y1"] for line in current_group) // len(current_group)
            
            merged_lines.append({
                "x1": x1,
                "y1": y_avg,
                "x2": x2,
                "y2": y_avg,
                "length": x2 - x1
            })
        
        # Replace with merged lines
        horizontal_lines = merged_lines
        
        # Require at least 5 lines
        if len(horizontal_lines) < 5:
            return {"staff_systems": [], "detections": []}
        
        # Calculate spacings between adjacent lines
        spacings = []
        for i in range(len(horizontal_lines) - 1):
            spacing = horizontal_lines[i+1]["y1"] - horizontal_lines[i]["y1"]
            spacings.append(spacing)
        
        # Cluster spacings to find staff systems
        spacing_array = np.array(spacings).reshape(-1, 1)
        clustering = DBSCAN(eps=3, min_samples=2).fit(spacing_array)
        
        # Find typical staff line spacing
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
        
        # Group lines into staff systems
        staff_systems = []
        current_staff = []
        
        for i, line in enumerate(horizontal_lines):
            if i == 0:
                current_staff = [i]
            else:
                spacing = line["y1"] - horizontal_lines[i-1]["y1"]
                
                # If spacing is close to expected staff line spacing
                if abs(spacing - staff_spacing) < staff_spacing * 0.25:
                    current_staff.append(i)
                    
                    # If we have 5 lines, it's a complete staff
                    if len(current_staff) == 5:
                        staff_systems.append(current_staff)
                        current_staff = []
                else:
                    # New staff system
                    if len(current_staff) >= 3:  # Require at least 3 lines
                        staff_systems.append(current_staff)
                    current_staff = [i]
        
        # Add the last staff if it has enough lines
        if len(current_staff) >= 3:
            staff_systems.append(current_staff)
        
        # Create staff line data in the expected format
        staff_line_data = []
        
        # Estimate staff line thickness
        line_thickness = max(2, int(staff_spacing * 0.15))  # Default if we can't determine
        
        for system_idx, system in enumerate(staff_systems):
            for line_idx, idx in enumerate(system):
                line = horizontal_lines[idx]
                
                # Estimate y1 and y2 based on line thickness
                y1 = line["y1"] - line_thickness // 2
                y2 = line["y1"] + line_thickness // 2
                
                staff_line = {
                    "class_id": 0,
                    "class_name": "staff_line",
                    "confidence": 1.0,
                    "bbox": {
                        "x1": float(line["x1"]),
                        "y1": float(y1),
                        "x2": float(line["x2"]),
                        "y2": float(y2),
                        "width": float(line["x2"] - line["x1"]),
                        "height": float(line_thickness),
                        "center_x": float((line["x1"] + line["x2"]) / 2),
                        "center_y": float(line["y1"])
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


def detect_and_save_staff_lines(image_path, output_dir="results/staff_lines"):
    """
    Detect staff lines in an image and save the results
    
    Args:
        image_path: Path to the music score image
        output_dir: Output directory for staff line data and visualization
        
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
                
                # Detect staff lines
                staff_data = detector.detect(str(img_file))
                
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
            staff_data = detector.detect(image_path)
            
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
    
    args = parser.parse_args()
    
    # Run staff line detection
    detect_and_save_staff_lines(args.image, args.output_dir)


if __name__ == "__main__":
    main()