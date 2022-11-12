import numpy as np
import pytest
import tempfile
import requests
import os

import torch
import torchvision.models as models

import openvino.runtime as ov
import pyopenvino as pyov

MODEL_PATH = tempfile.NamedTemporaryFile(suffix=".onnx").name

def download_model(file_name):
    URL = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    if not os.path.exists(file_name):
        r = requests.get(URL, allow_redirects=True)
        open(file_name, 'wb').write(r.content)

@pytest.mark.parametrize(("total_infers", "workers"), 
            [(10,2), (20,4)])
def test_async(total_infers, workers):
    input = np.random.randint(255, size=(1,3,224,224), dtype=np.uint8).astype(float)
    results = [False for _ in range(total_infers)] # container for results
    def callback(request, userdata):
        print(f"Done! Number: {userdata}")
        results[userdata] = True

    download_model(MODEL_PATH)
    ## Create Model from file
    model = pyov.Model.from_file(MODEL_PATH)
    
    model.workers = workers
    model.callback = callback

    ## Run parallel inference
    for i in range(total_infers):
        model.async_request(input, userdata=i)

    model.wait()
    assert all(results)

@pytest.fixture(scope='session', autouse=True)
def clean_up():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
