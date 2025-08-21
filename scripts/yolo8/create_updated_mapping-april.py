import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def extract_class_frequency(annotation_dir, max_classes=None):
    annotation_dir = Path(annotation_dir)
    annotation_files = list(annotation_dir.glob("*.xml"))
    print(f"Found {len(annotation_files)} annotation files.")

    class_counts = defaultdict(int)

    for ann_file in tqdm(annotation_files, desc="Parsing XML annotations"):
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            for node in root.findall(".//Node"):
                cls = node.find("ClassName").text
                class_counts[cls] += 1
        except Exception as e:
            print(f"Error parsing {ann_file.name}: {e}")

    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    if max_classes:
        sorted_counts = sorted_counts[:max_classes]

    class_freq = {k: v for k, v in sorted_counts}
    class_mapping = {cls_name: idx + 1 for idx, (cls_name, _) in enumerate(sorted_counts)}

    return class_freq, class_mapping

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved to {path}")

if __name__ == "__main__":
    annotation_dir = "/homes/es314/omr-objdet-benchmark/data/annotations"
    output_dir = "/homes/es314/omr-objdet-benchmark/data"

    class_freq, class_mapping = extract_class_frequency(annotation_dir)

    save_json(class_freq, os.path.join(output_dir, "class_freq.json"))
    save_json(class_mapping, os.path.join(output_dir, "class_mapping.json"))
