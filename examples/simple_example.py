import numpy as np

import torch
import torchvision.models as models

import pyopenvino as pyov

MODEL_LOCAL_PATH="mobilenet_v2.onnx"

def get_onnx_model(model):
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"] 
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, MODEL_LOCAL_PATH, verbose=True, input_names=input_names, output_names=output_names)

torch_model = models.mobilenet_v2(pretrained=True) 
torch_model.eval()

get_onnx_model(torch_model)

model = pyov.Model.from_file(MODEL_LOCAL_PATH)

input = np.random.randint(255, size=(1,3,224,224), dtype=np.uint8).astype(float)
result = model(input)

print(f"Class index: {np.argmax(result['output'])}")
