import onnx
import onnxruntime

# Load ONNX model
model_path = '/import/c4dm-05/elona/faster-rcnn-models-march-2025/yolo8runs/train-202-24classes-yolo-9654-data-splits/weights/81-best.onnx'
onnx_model = onnx.load(model_path)

# Print model metadata
print(f"ONNX Model IR Version: {onnx_model.ir_version}")
print(f"ONNX Model Opset Version: {onnx_model.opset_import[0].version}")

# Print input details
print("\nModel Inputs:")
for input in onnx_model.graph.input:
    print(f"  Name: {input.name}")
    print(f"  Shape: {[dim.dim_value if dim.dim_value else dim.dim_param for dim in input.type.tensor_type.shape.dim]}")
    print(f"  Type: {input.type.tensor_type.elem_type}")
    
# Print output details
print("\nModel Outputs:")
for output in onnx_model.graph.output:
    print(f"  Name: {output.name}")
    print(f"  Shape: {[dim.dim_value if dim.dim_value else dim.dim_param for dim in output.type.tensor_type.shape.dim]}")
    print(f"  Type: {output.type.tensor_type.elem_type}")

# Create an ONNX Runtime session
session = onnxruntime.InferenceSession(model_path)

# Get input details
inputs = session.get_inputs()
for i, input in enumerate(inputs):
    print(f"\nInput {i}:")
    print(f"  Name: {input.name}")
    print(f"  Shape: {input.shape}")
    print(f"  Type: {input.type}")

# Get output details
outputs = session.get_outputs()
for i, output in enumerate(outputs):
    print(f"\nOutput {i}:")
    print(f"  Name: {output.name}")
    print(f"  Shape: {output.shape}")
    print(f"  Type: {output.type}")