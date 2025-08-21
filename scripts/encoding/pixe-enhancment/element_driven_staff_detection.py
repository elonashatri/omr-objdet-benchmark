import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict

class ElementDrivenStaffDetector:
    def __init__(self, debug=False):
        self.debug = debug

    def detect(self, image_path, detected_elements_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        with open(detected_elements_path, 'r') as f:
            detected_elements = json.load(f)

        elements_by_category = self._categorize_elements(detected_elements)
        staff_structure = self._determine_staff_structure(elements_by_category, img.shape)
        staff_data = self._generate_staff_lines(staff_structure, img)

        return staff_data

    def _categorize_elements(self, detected_elements):
        categories = defaultdict(list)

        for element in detected_elements.get("detections", []):
            class_name = element.get("class_name", "").lower()

            if "clef" in class_name:
                categories["clefs"].append(element)
            elif "notehead" in class_name:
                categories["noteheads"].append(element)
            else:
                categories["other"].append(element)

        if self.debug:
            for category, elements in categories.items():
                print(f"  {category}: {len(elements)}")

        return categories

    def _determine_staff_lines_from_clef(self, clef_data, noteheads, img_shape, global_spacing):
        clef_top = clef_data["bbox"]["y1"]
        clef_height = clef_data["bbox"]["height"]
        clef_name = clef_data.get("class_name", "").lower()

        is_g_clef = "gclef" in clef_name
        is_f_clef = "fclef" in clef_name

        if is_g_clef:
            top_line_y = clef_top + clef_height * 0.19
        elif is_f_clef:
            top_line_y = clef_top - clef_height * 0.01
        else:
            top_line_y = clef_top + clef_height * 0.25

        # line_positions = [top_line_y + i * global_spacing for i in range(5)]
        line_positions = [float(top_line_y + i * global_spacing) for i in range(5)]


        if self.debug:
            print(f"Clef {clef_name} at y={clef_top:.1f}, spacing={global_spacing:.2f}, lines={line_positions}")

        return line_positions

    def _determine_staff_structure(self, elements_by_category, img_shape):
        height, width = img_shape
        clefs = elements_by_category["clefs"]
        noteheads = elements_by_category["noteheads"]

        structure = {
            "staff_systems": [],
            "staff_line_thickness": 1,
            "staff_spacing": 10
        }

        if noteheads:
            notehead_heights = [n["bbox"]["height"] for n in noteheads if "noteheadblack" in n["class_name"].lower()]
            if len(notehead_heights) < 3:
                notehead_heights = [n["bbox"]["height"] for n in noteheads]
            median_notehead_height = np.median(notehead_heights)
            structure["staff_spacing"] = median_notehead_height / 1.14
            structure["staff_line_thickness"] = max(1, int(median_notehead_height * 0.05))

        if clefs:
            sorted_clefs = sorted(clefs, key=lambda c: c["bbox"]["center_y"])
            for clef_idx, clef in enumerate(sorted_clefs):
                clef_y = clef["bbox"]["center_y"]
                clef_height = clef["bbox"]["height"]
                search_margin = clef_height * 2
                nearby_noteheads = [n for n in noteheads if abs(n["bbox"]["center_y"] - clef_y) < search_margin]
                line_positions = self._determine_staff_lines_from_clef(
                    clef, nearby_noteheads, img_shape, structure["staff_spacing"]
                )
                x1 = max(0, clef["bbox"]["x1"] - 10)
                x2 = min(width, width - 10)
                staff_system = {
                    "id": clef_idx,
                    "reference_clef": clef,
                    "line_positions": line_positions,
                    "staff_spacing": structure["staff_spacing"],
                    "x_range": (x1, x2),
                    "y_range": (min(line_positions), max(line_positions))
                }
                structure["staff_systems"].append(staff_system)

        return structure

    def _generate_staff_lines(self, staff_structure, img):
        height, width = img.shape
        staff_systems = []
        staff_lines = []

        for system in staff_structure["staff_systems"]:
            system_id = system["id"]
            line_positions = system["line_positions"]
            x_range = system["x_range"]
            thickness = staff_structure["staff_line_thickness"]

            system_line_indices = []

            for i, y_pos in enumerate(line_positions):
                if y_pos < 0 or y_pos >= height:
                    continue

                y_center = round(y_pos)
                staff_line = {
                    "class_id": 0,
                    "class_name": "staff_line",
                    "confidence": 1.0,
                    "bbox": {
                        "x1": float(x_range[0]),
                        "y1": float(y_center - thickness/2),
                        "x2": float(x_range[1]),
                        "y2": float(y_center + thickness/2),
                        "width": float(x_range[1] - x_range[0]),
                        "height": float(thickness),
                        "center_x": float((x_range[0] + x_range[1]) / 2),
                        "center_y": float(y_center)
                    },
                    "staff_system": system_id,
                    "line_number": 4 - i
                }

                staff_lines.append(staff_line)
                system_line_indices.append(len(staff_lines) - 1)

            if system_line_indices:
                staff_systems.append({
                    "id": system_id,
                    "lines": system_line_indices
                })

        return {
            "staff_systems": staff_systems,
            "detections": staff_lines
        }


    def _verify_staff_lines(self, staff_data, img):
        """
        Verify and fine-tune staff lines against image content using pixel-perfect alignment
        
        Args:
            staff_data: Dictionary with staff line data
            img: Grayscale image
            
        Returns:
            Dictionary with verified staff line data
        """
        height, width = img.shape
        refined_data = staff_data.copy()
        refined_detections = []
        
        # Make sure image is properly oriented (staff lines should be dark)
        if np.mean(img) > 127:
            img_for_alignment = 255 - img
        else:
            img_for_alignment = img.copy()
        
        # Apply preprocessing to enhance staff lines
        enhanced_img = self._enhance_staff_lines(img_for_alignment)
        
        # Process each staff system
        for system in staff_data["staff_systems"]:
            system_id = system["id"]
            new_line_indices = []
            
            # Process each line in this system
            for line_idx in system.get("lines", []):
                line = staff_data["detections"][line_idx]
                initial_y = line["bbox"]["center_y"]
                x1 = int(line["bbox"]["x1"])
                x2 = int(line["bbox"]["x2"])
                thickness = line["bbox"]["height"]
                
                # Define search range - make it large enough to find the actual line
                search_range = max(5, int(thickness * 2))
                y_start = max(0, int(initial_y - search_range))
                y_end = min(height - 1, int(initial_y + search_range))
                
                # Skip if window is invalid
                if y_start >= y_end:
                    refined_detections.append(line)
                    new_line_indices.append(len(refined_detections) - 1)
                    continue
                
                # Extract window from enhanced image
                window = enhanced_img[y_start:y_end+1, x1:min(x2, width-1)]
                
                if window.size == 0 or window.shape[1] < 10:
                    refined_detections.append(line)
                    new_line_indices.append(len(refined_detections) - 1)
                    continue
                
                # Calculate darkness profile in window
                darkness = np.sum(window, axis=1)
                
                # Find position of maximum darkness
                if np.max(darkness) > 0:
                    # best_y = y_start + np.argmax(darkness)
                    delta = np.argmax(darkness) - search_range  # how far from original
                    if abs(delta) > 1:  # clamp max adjustment to ±1 pixel
                        delta = np.sign(delta)
                    best_y = int(initial_y + delta)

                    
                    # Create new line with adjusted position
                    new_line = line.copy()
                    new_line["bbox"] = line["bbox"].copy()
                    
                    # Update y-coordinates
                    new_line["bbox"]["y1"] = float(best_y - thickness/2)
                    new_line["bbox"]["y2"] = float(best_y + thickness/2)
                    new_line["bbox"]["center_y"] = float(best_y)
                    
                    refined_detections.append(new_line)
                else:
                    refined_detections.append(line)
                
                new_line_indices.append(len(refined_detections) - 1)
            
            # Update system with new line indices
            system["lines"] = new_line_indices
        
        refined_data["detections"] = refined_detections
        return refined_data
    
    def _enhance_staff_lines(self, img):
        """
        Enhance staff lines in the image for better detection
        
        Args:
            img: Grayscale image
            
        Returns:
            Enhanced image
        """
        # Apply bilateral filtering to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 4
        )
        
        # Create horizontal kernel for enhancing staff lines
        kernel_length = max(1, img.shape[1] // 80)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        
        # Apply morphological operations to enhance horizontal lines
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        
        # Dilate slightly to make staff lines more prominent
        horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=1)
        
        return horizontal
    
    def visualize(self, image_path, staff_data, output_path=None):
        """
        Visualize detected staff lines on the image
        
        Args:
            image_path: Path to the music score image
            staff_data: Staff line data from detect method
            output_path: Path to save the visualization image (optional)
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Create figure
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        
        # Colors for visualization
        colors = [
            (1, 0, 0),      # Red
            (0, 0.7, 0),    # Green
            (0, 0, 1),      # Blue
            (1, 0.5, 0),    # Orange
            (0.5, 0, 0.5),  # Purple
            (0, 0.7, 0.7),  # Cyan
            (1, 0, 1),      # Magenta
            (0.7, 0.7, 0)   # Yellow
        ]
        
        # Draw staff systems
        for system_idx, system in enumerate(staff_data["staff_systems"]):
            color = colors[system_idx % len(colors)]
            
            # Draw staff lines
            for line_idx in system["lines"]:
                if line_idx < len(staff_data["detections"]):
                    line = staff_data["detections"][line_idx]
                    x1 = line["bbox"]["x1"]
                    y1 = line["bbox"]["center_y"]
                    x2 = line["bbox"]["x2"]
                    y2 = line["bbox"]["center_y"]
                    
                    # Plot staff line
                    plt.plot([x1, x2], [y1, y2], color=color, linewidth=0.5)
                    
                    # Add label for first line only
                    if line_idx == system["lines"][0]:
                        plt.text(
                            x1 - 30, y1, f"Staff {system['id']}",
                            color=color, fontsize=10, va='center',
                            bbox=dict(facecolor='white', alpha=0.7, pad=1)
                        )
        
        plt.title("Detected Staff Lines")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()


def detect_staffs_from_elements(image_path, detected_elements_path, output_path=None):
    """
    Convenience function to detect staff lines from musical elements
    
    Args:
        image_path: Path to the music score image
        detected_elements_path: Path to the JSON file with YOLO detected elements
        output_path: Path to save staff line data (optional)
        
    Returns:
        Dictionary with staff line data
    """
    detector = ElementDrivenStaffDetector(debug=True)
    staff_data = detector.detect(image_path, detected_elements_path)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(staff_data, f, indent=2)
        
        # Save visualization
        viz_path = output_path.replace('.json', '_visualization.png')
        detector.visualize(image_path, staff_data, viz_path)
    
    return staff_data