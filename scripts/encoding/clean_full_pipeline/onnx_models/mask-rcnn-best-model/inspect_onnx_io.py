import onnx

model_path = "mask_rcnn_doremi_0079.onnx"
model = onnx.load(model_path)
graph = model.graph

print("\n=== MODEL INPUTS ===")
for input_tensor in graph.input:
    dims = [d.dim_value if (d.dim_value > 0) else "?" for d in input_tensor.type.tensor_type.shape.dim]
    print(f"{input_tensor.name} : shape={dims}, dtype={input_tensor.type.tensor_type.elem_type}")

print("\n=== MODEL INITIALIZERS ===")
for init in graph.initializer:
    print(f"{init.name} : shape={list(init.dims)}, dtype={init.data_type}")

print("\n=== MODEL OUTPUTS ===")
for output_tensor in graph.output:
    dims = [d.dim_value if (d.dim_value > 0) else "?" for d in output_tensor.type.tensor_type.shape.dim]
    print(f"{output_tensor.name} : shape={dims}, dtype={output_tensor.type.tensor_type.elem_type}")
