import numpy as np

import torch
import torchvision.models as models

import openvino.runtime as ov
import pyopenvino as pyov

MODEL_LOCAL_PATH="mobilenet_v2.onnx"
INFERENCE_NUMBER=10

def get_onnx_model(model):
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"] 
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, MODEL_LOCAL_PATH, verbose=False, input_names=input_names, output_names=output_names)

torch_model = models.mobilenet_v2(pretrained=True) 
torch_model.eval()

input = np.random.randint(255, size=(1,3,224,224), dtype=np.uint8).astype(float)
get_onnx_model(torch_model)

results = [False for _ in range(INFERENCE_NUMBER)] # container for results
def callback(request, userdata):
    print(f"Done! Number: {userdata}")
    results[userdata] = True

## Create Model from file
model = pyov.Model.from_file(MODEL_LOCAL_PATH)
model.workers = 4
model.callback = callback

## Run parallel inference
for i in range(INFERENCE_NUMBER):
    model.async_request(input, userdata=i)

model.wait()

assert all(results)