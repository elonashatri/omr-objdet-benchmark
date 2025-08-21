import onnxruntime as ort
import numpy as np
from PIL import Image

# --- Paths ---
MODEL_PATH = "/homes/es314/omr-objdet-benchmark/scripts/encoding/clean_full_pipeline/onnx_models/mask-rcnn-best-model/mask_rcnn_doremi_0079.onnx"
IMAGE_PATH = "/homes/es314/omr-objdet-benchmark/scripts/encoding/testing_images/Abismo_de_Rosas__Canhoto_Amrico_Jacomino-002.png"

# --- Preprocessing ---
def load_and_preprocess(image_path, target_shape=(1024, 1024)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_shape)
    image_np = np.array(image).astype(np.float32)
    image_np -= np.array([123.7, 116.8, 103.9])
    image_np = np.expand_dims(image_np, axis=0)  # (1, H, W, 3)
    return image_np

def create_image_meta(batch_size=1):
    return np.zeros((batch_size, 84), dtype=np.float32)

# --- Anchor generation ---
def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride=1):
    anchors = []
    for scale, shape, stride in zip(scales, feature_shapes, feature_strides):
        for ratio in ratios:
            h = scale / np.sqrt(ratio)
            w = scale * np.sqrt(ratio)
            shifts_y = np.arange(0, shape[0]) * stride
            shifts_x = np.arange(0, shape[1]) * stride
            shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
            box_centers = np.stack([shift_y, shift_x], axis=-1).reshape(-1, 2)
            box_sizes = np.array([[h, w]])
            boxes = np.concatenate([
                box_centers - box_sizes / 2,
                box_centers + box_sizes / 2
            ], axis=1)
            anchors.append(boxes)
    return np.concatenate(anchors, axis=0)

def create_anchors(image_shape=(1024, 1024)):
    scales = [32, 64, 128, 256, 512]
    ratios = [0.5, 1, 2]
    strides = [4, 8, 16, 32, 64]
    feature_shapes = [(image_shape[0] // s, image_shape[1] // s) for s in strides]
    anchors = generate_pyramid_anchors(scales, ratios, feature_shapes, strides)
    anchors = anchors[np.newaxis, ...]  # shape becomes (1, 261888, 4)
    return anchors.astype(np.float32)  # shape: (261888, 4)





    return anchors

# --- Load model ---
session = ort.InferenceSession(MODEL_PATH)
print("Model inputs:")
for inp in session.get_inputs():
    print(f"{inp.name} : shape={inp.shape} type={inp.type}")

# --- Prepare input tensors ---
image_np = load_and_preprocess(IMAGE_PATH)
image_meta = create_image_meta()
anchors = create_anchors()

inputs = {
    "input_image:0": image_np,
    "input_image_meta:0": image_meta,
    "input_anchors:0": anchors
}

print("image_np:", image_np.shape)
print("image_meta:", image_meta.shape)
print("anchors:", anchors.shape)

# --- Run inference ---
output_names = ["mrcnn_class/Softmax:0", "mrcnn_mask/Sigmoid:0"]
outputs = session.run(output_names, inputs)

# --- Summary ---
classes = outputs[0]  # (1, 1000, 72)
masks = outputs[1]    # (1, 1000, 28, 28, 72)
print("✅ Inference successful!")
print("Classes shape:", classes.shape)
print("Masks shape:", masks.shape)
print("Model outputs:")
for out in session.get_outputs():
    print(f"{out.name} : shape={out.shape} dtype={out.type}")
