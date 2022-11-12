import numpy as np
import pytest
import tempfile

import torch
import torchvision.models as models

import openvino.runtime as ov
import pyopenvino as pyov


def get_onnx_model(model, file_name):
    dummy_input = torch.randn(1, 3, 224, 224)
    input_names = ["input"] 
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, file_name, verbose=False, input_names=input_names, output_names=output_names)

@pytest.mark.parametrize(("total_infers", "workers"), 
            [(10,2), (20,4)])
def test_async(total_infers, workers):
    input = np.random.randint(255, size=(1,3,224,224), dtype=np.uint8).astype(float)
    results = [False for _ in range(total_infers)] # container for results
    def callback(request, userdata):
        print(f"Done! Number: {userdata}")
        results[userdata] = True

    torch_model = models.mobilenet_v2(pretrained=True) 
    torch_model.eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        get_onnx_model(torch_model, tmp.name)
        ## Create Model from file
        model = pyov.Model.from_file(tmp.name)
    
    model.workers = workers
    model.callback = callback

    ## Run parallel inference
    for i in range(total_infers):
        model.async_request(input, userdata=i)

    model.wait()
    assert all(results)