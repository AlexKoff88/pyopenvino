import os
import requests
import tempfile

from datasets import load_dataset

from pyopenvino.training.losses import L2Loss
import openvino.runtime as ov

import numpy as np

import pyopenvino as pyov

MODEL_PATH = tempfile.NamedTemporaryFile(suffix=".onnx").name

def download_model(file_name):
    URL = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    if not os.path.exists(file_name):
        r = requests.get(URL, allow_redirects=True)
        open(file_name, 'wb').write(r.content)

def test_l2_loss_inference():
    loss = L2Loss([1,1000])
    '''loss.save("l2loss.xml")

    print(f"Inputs: {loss.get_parameters()}")'''
    
    x = np.random.rand(1,1000)
    result = loss({"input":x, "target": x})

    print(f"result: {result}")
    assert result["output0"] == 0

def test_l2_loss_attach():
    download_model(MODEL_PATH)
    model = pyov.Model.from_file(MODEL_PATH)
    result_name = model.get_result().get_friendly_name()
    
    loss = L2Loss([-1,1000])
    new_model = loss.attach(model, result_name)
    assert new_model

def test_e2e():
    download_model(MODEL_PATH)
    model = pyov.Model.from_file(MODEL_PATH)
    result_name = model.get_result().get_friendly_name()
    
    loss = L2Loss([-1,1000])
    new_model = loss.attach(model, result_name)

    dataset = load_dataset("food101", split="validation[:100]")




