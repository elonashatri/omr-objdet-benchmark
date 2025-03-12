import os
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class AdaptiveStaffDetector:
    """Staff line detector with improved peak detection and visualization."""

    def __init__(self, line_merging_threshold=5):
        self.line_merging_threshold = line_merging_threshold

    def detect(self, image_path):
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path

        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        inverted = cv2.bitwise_not(img)

        binary = cv2.adaptiveThreshold(
            inverted, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, blockSize=25, C=-2
        )

        projection = np.sum(binary, axis=1)
        projection_norm = (projection - projection.min()) / (projection.max() - projection.min())

        peaks, _ = find_peaks(projection_norm, height=0.5, distance=10)

        staff_systems = []
        current_staff = []
        spacing_threshold = np.median(np.diff(peaks)) * 0.3

        for peak in peaks:
            if not current_staff:
                current_staff.append(peak)
            elif abs(peak - current_staff[-1]) < spacing_threshold:
                current_staff.append(peak)
            else:
                if len(current_staff) >= 3:
                    staff_systems.append(current_staff)
                current_staff = [peak]

        if len(current_staff) >= 3:
            staff_systems.append(current_staff)

        height, width = binary.shape
        line_thickness = 2

        detections = []
        line_counter = 0
        for system_idx, system in enumerate(staff_systems):
            for line_idx, y in enumerate(system):
                detections.append({
                    "class_id": 0,
                    "class_name": "staff_line",
                    "confidence": 1.0,
                    "bbox": {
                        "x1": 0.0,
                        "y1": float(y - line_thickness // 2),
                        "x2": float(width),
                        "y2": float(y + line_thickness // 2),
                        "width": float(width),
                        "height": float(line_thickness),
                        "center_x": float(width / 2),
                        "center_y": float(y)
                    },
                    "staff_system": system_idx,
                    "line_number": line_counter
                })
                line_counter += 1

        return {
            "staff_systems": [
                {"id": idx, "lines": list(range(len(system)))}
                for idx, system in enumerate(staff_systems)
            ],
            "detections": detections
        }

    def visualize(self, image_path, staff_data, output_path=None):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 10))
        plt.imshow(img_rgb)

        for detection in staff_data["detections"]:
            y = detection["bbox"]["center_y"]
            plt.plot([0, img.shape[1]], [y, y], color='red', linewidth=2)

        plt.axis('off')
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def save_data(self, staff_data, output_path):
        with open(output_path, 'w') as f:
            json.dump(staff_data, f, indent=2)

def detect_and_save_staff_lines(image_path, output_dir="results/staff_lines"):
    os.makedirs(output_dir, exist_ok=True)
    detector = AdaptiveStaffDetector()

    if os.path.isdir(image_path):
        image_files = []
        for ext in [".jpg", ".jpeg", ".png"]:
            image_files.extend(list(Path(image_path).glob(f"*{ext}")))

        json_paths = []
        for img_file in image_files:
            json_output_path = os.path.join(output_dir, f"{img_file.stem}_staff_lines.json")
            visualization_path = os.path.join(output_dir, f"{img_file.stem}_staff_lines.png")

            staff_data = detector.detect(str(img_file))
            detector.save_data(staff_data, json_output_path)
            detector.visualize(str(img_file), staff_data, visualization_path)
            json_paths.append(json_output_path)

        return json_paths
    else:
        img_name = os.path.basename(image_path)
        img_stem = os.path.splitext(img_name)[0]
        json_output_path = os.path.join(output_dir, f"{img_stem}_staff_lines.json")
        visualization_path = os.path.join(output_dir, f"{img_stem}_staff_lines.png")

        staff_data = detector.detect(image_path)
        detector.save_data(staff_data, json_output_path)
        detector.visualize(image_path, staff_data, visualization_path)

        return json_output_path

def merge_detections(staff_line_file, symbol_detection_file, output_file):
    with open(staff_line_file, 'r') as f:
        staff_data = json.load(f)
    with open(symbol_detection_file, 'r') as f:
        symbol_data = json.load(f)

    merged_detections = staff_data["detections"] + symbol_data["detections"]
    merged_data = {"detections": merged_detections}

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    return output_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect staff lines in music score images")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/staff_lines")
    parser.add_argument("--line-merging-threshold", type=int, default=5)
    
    args = parser.parse_args()
    detect_and_save_staff_lines(args.image, args.output_dir)

if __name__ == "__main__":
    main()