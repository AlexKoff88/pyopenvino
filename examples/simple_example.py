import numpy as np

import torch
import torchvision.models as models

import openvino.runtime as ov
import pyopenvino as pyov

MODEL_LOCAL_PATH="mobilenet_v2.onnx"

def get_onnx_model(model):
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"] 
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, MODEL_LOCAL_PATH, verbose=False, input_names=input_names, output_names=output_names)

torch_model = models.mobilenet_v2(pretrained=True) 
torch_model.eval()

input = np.random.randint(255, size=(1,3,224,224), dtype=np.uint8).astype(float)
get_onnx_model(torch_model)

## Create from file
model = pyov.Model.from_file(MODEL_LOCAL_PATH)
result = model(input)
print(f"From file: {np.argmax(result['output'])}")

## Create from openvino.runtime.Model
core = ov.Core()
ov_model = core.read_model(MODEL_LOCAL_PATH)
model = pyov.Model(ov_model)
result = model(input)
print(f"From model: {np.argmax(result['output'])}")

## Select inference device and seetings
config = {"PERFORMANCE_HINT": "THROUGHPUT",
        "INFERENCE_PRECISION_HINT": "f32"}
model.to("CPU", config)

## Lower weights precision to FP16. GPU switches to FP16 inference
model.half()
result = model(input)
print(f"FP16 weights: {np.argmax(result['output'])}")

## Save to file
model.save("tmp.xml")