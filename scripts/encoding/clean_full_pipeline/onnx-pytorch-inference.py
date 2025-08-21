import os
import cv2
import numpy as np
import onnxruntime
from PIL import Image
import json
import argparse
import matplotlib.pyplot as plt
import torch


class FasterRCNNOnnxDetector:
    def __init__(self, model_path, class_mapping_path, conf_threshold=0.2, max_detections=1000,
                 iou_threshold=0.5, min_box_area=0):
        self.model_path = model_path
        self.class_mapping_path = class_mapping_path
        self.conf_threshold = conf_threshold
        self.max_detections = max_detections
        self.iou_threshold = iou_threshold
        self.min_box_area = min_box_area

        print(f"Loading ONNX model from {model_path}")
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        print(f"Model loaded with input name: {self.input_name}")
        print(f"Output names: {self.output_names}")

        self.class_names = self.load_class_mapping(class_mapping_path)
        print(f"Loaded {len(self.class_names)} class mappings")

    def load_class_mapping(self, mapping_file):
        class_map = {}
        try:
            with open(mapping_file, 'r') as f:
                content = f.read()

            import re
            items = re.findall(r'item\{\s*id:\s*(\d+)\s*name:\s*\'([^\']+)\'\s*\}', content)
            for item_id, item_name in items:
                class_map[int(item_id)] = item_name
            return class_map
        except Exception as e:
            print(f"Error loading class names: {e}")
            return {}

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        return image_np_expanded, image.size

    @staticmethod
    def nms(boxes, scores, iou_threshold):
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        from torchvision.ops import nms
        keep = nms(boxes, scores, iou_threshold)

        return keep.numpy()

    def detect(self, image_path):
        image, (original_width, original_height) = self.preprocess_image(image_path)
        outputs = self.session.run(None, {self.input_name: image})

        boxes, scores, class_ids, num_detections = outputs[0], outputs[1], outputs[2], int(outputs[3][0])

        valid_indices = np.where(scores[0] >= self.conf_threshold)[0]
        if len(valid_indices) > self.max_detections:
            score_order = np.argsort(scores[0][valid_indices])[::-1][:self.max_detections]
            valid_indices = valid_indices[score_order]

        boxes_np = boxes[0][valid_indices]
        scores_np = scores[0][valid_indices]
        class_ids_np = class_ids[0][valid_indices]

        boxes_xyxy = np.array([
            [box[1], box[0], box[3], box[2]] for box in boxes_np
        ])

        keep_indices = self.nms(boxes_xyxy, scores_np, self.iou_threshold)

        filtered_boxes = boxes_np[keep_indices]
        filtered_scores = scores_np[keep_indices]
        filtered_class_ids = class_ids_np[keep_indices]

        if self.min_box_area > 0:
            final_indices = []
            for i, box in enumerate(filtered_boxes):
                h = (box[2] - box[0]) * original_height
                w = (box[3] - box[1]) * original_width
                if (h * w) >= self.min_box_area:
                    final_indices.append(i)
            filtered_boxes = filtered_boxes[final_indices]
            filtered_scores = filtered_scores[final_indices]
            filtered_class_ids = filtered_class_ids[final_indices]

        return {
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'classes': filtered_class_ids,
            'num_detections': len(filtered_boxes),
            'image_size': (original_height, original_width)
        }

    def visualize_detections(self, image_path, results, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image from {image_path}")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        boxes, scores, class_ids = results['boxes'], results['scores'], results['classes']

        unique_classes = np.unique(class_ids)
        colors = {}
        for i, cls_id in enumerate(unique_classes):
            hue = (i * 0.15) % 1.0
            rgb = plt.cm.hsv(hue)[:3]
            colors[cls_id] = tuple((np.array(rgb) * 255).astype(int).tolist())

        for box, score, cls_id in zip(boxes, scores, class_ids):
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            color = colors.get(cls_id, (0, 255, 0))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)

            class_name = self.class_names.get(int(cls_id), f"Unknown-{cls_id}")
            label = f"{class_name}: {score:.2f}"
            font_scale = 0.3
            thickness = 0
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            cv2.rectangle(image, (xmin, ymin - text_size[1] - 4), (xmin + text_size[0], ymin), color, -1)
            cv2.putText(image, label, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Detection visualization saved to {output_path}")

    def save_detection_data(self, results, image_name, output_dir):
        boxes, scores, class_ids = results['boxes'], results['scores'], results['classes']
        height, width = results['image_size']
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            ymin, xmin, ymax, xmax = box
            xmin_px = xmin * width
            xmax_px = xmax * width
            ymin_px = ymin * height
            ymax_px = ymax * height
            detection = {
                "class_id": int(class_id),
                "class_name": self.class_names.get(int(class_id), f"cls_{int(class_id)}"),
                "confidence": float(score),
                "bbox": {
                    "x1": float(xmin_px), "y1": float(ymin_px),
                    "x2": float(xmax_px), "y2": float(ymax_px),
                    "width": float(xmax_px - xmin_px),
                    "height": float(ymax_px - ymin_px),
                    "center_x": float((xmin_px + xmax_px) / 2),
                    "center_y": float((ymin_px + ymax_px) / 2)
                }
            }
            detections.append(detection)

        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{image_name}_detections.json")
        with open(json_path, 'w') as f:
            json.dump({"detections": detections}, f, indent=2)

        print(f"Detection data saved to {json_path}")
        return json_path


def main():
    parser = argparse.ArgumentParser(description="Run Faster R-CNN ONNX model for object detection")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--class_mapping", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--max_detections", type=int, default=1000)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--min_box_area", type=float, default=0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    detector = FasterRCNNOnnxDetector(
        args.model,
        args.class_mapping,
        conf_threshold=args.conf,
        max_detections=args.max_detections,
        iou_threshold=args.iou_threshold,
        min_box_area=args.min_box_area
    )

    results = detector.detect(args.image)
    img_name = os.path.splitext(os.path.basename(args.image))[0]
    detector.save_detection_data(results, img_name, args.output_dir)
    output_img_path = os.path.join(args.output_dir, f"{img_name}_onnx_detection.jpg")
    detector.visualize_detections(args.image, results, output_img_path)
    print("Processing complete.")


if __name__ == "__main__":
    main()


    # python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx-pytorch-inference.py \
    #     --model //homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex003/may_2023_ex003.onnx  \
    #     --image /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/12-Etudes-001.png  \
    #     --class_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/may_2023_ex003/mapping.txt  \
    #     --output_dir onnx_test_results \
    #     --conf 0.4 \
    #     --max_detections 1000
    #     --iou_threshold 0.5 
    
    
# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx-pytorch-inference.py \
#     --model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_resnet50-lr-0.003-classes-72-steps-100000-2475x3504-03-10-2020-004-train/Faster_R-CNN_resnet50-lr-0.003-classes-72-steps-100000-2475x3504-03-10-2020-004-train.onnx  \
#     --image /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/examples/12-Etudes-001.png  \
#     --class_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-lr-0.003-classes-72-steps-80000-2475x3504-01-10-2020-003-train/mapping_72.txt  \
#     --output_dir onnx_test_results \
#     --conf 0.4 \
#     --max_detections 1000
#     --iou_threshold 0.5


# /homes/es314/1-results-only-images/data/muscima-original-and-degraded_images/CVC-MUSCIMA_W-12_N-04_D-ideal_1-original.png


# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx-pytorch-inference.py \
#     --model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_resnet50-lr-0.003-classes-72-steps-100000-2475x3504-03-10-2020-004-train/Faster_R-CNN_resnet50-lr-0.003-classes-72-steps-100000-2475x3504-03-10-2020-004-train.onnx  \
#     --image /homes/es314/1-results-only-images/data/muscima-original-and-degraded_images/CVC-MUSCIMA_W-12_N-04_D-ideal_1-original.png  \
#     --class_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-lr-0.003-classes-72-steps-80000-2475x3504-01-10-2020-003-train/mapping_72.txt  \
#     --output_dir onnx_test_results \
#     --conf 0.4 \
#     --max_detections 1000
#     --iou_threshold 0.5


# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx-pytorch-inference.py \
#     --model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-lr-0.003-classes-72-steps-80000-2475x3504-01-10-2020-003-train/Faster_R-CNN_inception-lr-0.003-classes-72-steps-80000-2475x3504-01-10-2020-003-train.onnx  \
#     --image /homes/es314/1-results-only-images/data/muscima-original-and-degraded_images/CVC-MUSCIMA_W-12_N-04_D-ideal_1-original.png  \
#     --class_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-lr-0.003-classes-72-steps-80000-2475x3504-01-10-2020-003-train/mapping_72.txt  \
#     --output_dir onnx_test_results \
#     --conf 0.4 \
#     --max_detections 1000
#     --iou_threshold 0.5


# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx-pytorch-inference.py \
#     --model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-musicmma_pretrained-classes-88-steps-80000-2475x3504-09-10-2020-009-train/Faster_R-CNN_inception-musicmma_pretrained-classes-88-steps-80000-2475x3504-09-10-2020-009-train.onnx  \
#     --image /homes/es314/1-results-only-images/data/muscima-original-and-degraded_images/CVC-MUSCIMA_W-12_N-04_D-ideal_1-original.png  \
#     --class_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-musicmma_pretrained-classes-88-steps-80000-2475x3504-09-10-2020-009-train/mapping.txt  \
#     --output_dir onnx_test_results \
#     --conf 0.4 \
#     --max_detections 1000
#     --iou_threshold 0.5

# python /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx-pytorch-inference.py \
#     --model /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-musicmma_pretrained-classes-88-steps-80000-2475x3504-09-10-2020-009-train/Faster_R-CNN_inception-musicmma_pretrained-classes-88-steps-80000-2475x3504-09-10-2020-009-train.onnx  \
#     --image /homes/es314/music_detection_results/examples/images/2-solo-Christmas_Greeting-001.png \
#     --class_mapping /homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/Faster_R-CNN_inception-musicmma_pretrained-classes-88-steps-80000-2475x3504-09-10-2020-009-train/mapping.txt  \
#     --output_dir /homes/es314/evaluation_results/mask-rcnn-vs-faster/faster-rcnn-009 \
#     --conf 0.4 \
#     --max_detections 1000
#     --iou_threshold 0.5