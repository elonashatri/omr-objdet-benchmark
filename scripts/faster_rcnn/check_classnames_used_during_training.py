import json
import os
import glob
from xml.dom import minidom

# Paths
json_path = '/homes/es314/omr-objdet-benchmark/data/faster_rcnn_prepared_dataset/train/annotations.json'
xml_dir = '/homes/es314/omr-objdet-benchmark/data/annotations'
output_csv = '/homes/es314/omr-objdet-benchmark/data/faster_rcnn_prepared_dataset/train/class_stats.csv'

# Load JSON to get image filenames
with open(json_path, 'r') as f:
    data = json.load(f)

image_filenames = set(img['file_name'] for img in data['images'])

# Classnames for all pages in all matching XML files
all_classnames = set()

print('Beginning...')

# Iterate through XML files and match by image filename
for xml_file in glob.glob(os.path.join(xml_dir, '*.xml')):
    base_name = os.path.basename(xml_file).replace('.xml', '.png')
    if base_name in image_filenames:
        try:
            xmldoc = minidom.parse(xml_file)
            pages = xmldoc.getElementsByTagName('Page')
            for page in pages:
                nodes = page.getElementsByTagName('Node')
                for node in nodes:
                    class_tag = node.getElementsByTagName('ClassName')
                    if class_tag:
                        class_name = class_tag[0].firstChild.data.strip()
                        all_classnames.add(class_name)
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")

# Write to CSV
with open(output_csv, 'w') as out_file:
    for classname in sorted(all_classnames):
        out_file.write(f"{classname}\n")

print(f"Completed. {len(all_classnames)} unique classnames written to {output_csv}")
