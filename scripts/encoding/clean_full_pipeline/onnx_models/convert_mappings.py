import re
import json
import argparse
from pathlib import Path

def parse_frcnn_txt_mapping(txt_path):
    """Parses Faster R-CNN class mapping in .txt format"""
    with open(txt_path, 'r') as f:
        content = f.read()

    matches = re.findall(r'item\s*\{\s*id:\s*(\d+)\s*name:\s*\'([^\']+)\'\s*\}', content)
    id_to_name = {int(_id): name for _id, name in matches}
    name_to_id = {name: int(_id) for _id, name in matches}

    print(f"[✓] Loaded {len(id_to_name)} entries from FRCNN .txt mapping.")
    return id_to_name, name_to_id

def parse_yolo_json_mapping(json_path):
    """Parses YOLO class mapping in .json format"""
    with open(json_path, 'r') as f:
        name_to_id = json.load(f)
    id_to_name = {v: k for k, v in name_to_id.items()}
    print(f"[✓] Loaded {len(name_to_id)} entries from YOLO .json mapping.")
    return id_to_name, name_to_id

def merge_mappings(frcnn_map, yolo_map):
    """Create a unified name → ID mapping from both sources"""
    merged_name_to_id = {}
    merged_id_to_name = {}

    all_names = sorted(set(frcnn_map) | set(yolo_map))
    for new_id, name in enumerate(all_names, start=1):
        merged_name_to_id[name] = new_id
        merged_id_to_name[str(new_id)] = name

    print(f"[✓] Merged total of {len(merged_name_to_id)} unique class names.")
    return merged_name_to_id, merged_id_to_name

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")

def main():
    parser = argparse.ArgumentParser(description="Parse FRCNN/YOLO mappings and build a unified merged mapping")
    parser.add_argument("--frcnn_txt", required=True, help="Path to FRCNN .txt mapping file")
    parser.add_argument("--yolo_json", required=True, help="Path to YOLO .json mapping file")
    parser.add_argument("--out_dir", default="parsed_and_merged", help="Directory to save all outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse original mappings
    frcnn_id2name, frcnn_name2id = parse_frcnn_txt_mapping(args.frcnn_txt)
    yolo_id2name, yolo_name2id = parse_yolo_json_mapping(args.yolo_json)

    # Save original mappings
    save_json(frcnn_name2id, out_dir / "frcnn_name_to_id.json")
    save_json(frcnn_id2name, out_dir / "frcnn_id_to_name.json")
    save_json(yolo_name2id, out_dir / "yolo_name_to_id.json")
    save_json(yolo_id2name, out_dir / "yolo_id_to_name.json")

    # Merge mappings
    merged_name2id, merged_id2name = merge_mappings(frcnn_name2id, yolo_name2id)

    # Save merged mappings
    save_json(merged_name2id, out_dir / "merged_name_to_id.json")
    save_json(merged_id2name, out_dir / "merged_id_to_name.json")

    # Print overlap
    common_names = set(frcnn_name2id) & set(yolo_name2id)
    print(f"\n[✓] {len(common_names)} class names shared between YOLO and FRCNN:")
    for name in sorted(common_names):
        print(f"  {name}")

if __name__ == "__main__":
    main()

# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/convert_mappings.py \
#     --frcnn_txt /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex003/mapping.txt \
#     --yolo_json /homes/es314/omr-objdet-benchmark/data/class_mapping.json \
#     --out_dir /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models