import os
import requests
import tempfile

import torch
from torchvision import transforms
from torchvision import datasets

from pyopenvino.training.losses import L2Loss
import openvino.runtime as ov

import numpy as np

import pyopenvino as pyov

MODEL_PATH = tempfile.NamedTemporaryFile(suffix=".onnx").name
DATASET_PATH = "./dataset/"

def download_model(file_name):
    #URL = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    URL = "https://huggingface.co/AlexKoff88/mobilenet_v2_food101/resolve/main/mobilenet_v2_food101.onnx"
    if not os.path.exists(file_name):
        r = requests.get(URL, allow_redirects=True)
        open(file_name, 'wb').write(r.content)

def test_l2_loss_inference():
    loss = L2Loss([1,100])
    '''loss.save("l2loss.xml")

    print(f"Inputs: {loss.get_parameters()}")'''
    
    x = np.random.rand(1,100)
    result = loss({"input":x, "target": x})

    print(f"result: {result}")
    assert result["output0"] == 0

def test_l2_loss_attach():
    download_model(MODEL_PATH)
    model = pyov.Model.from_file(MODEL_PATH)
    result_name = model.get_result().get_friendly_name()
    
    loss = L2Loss([-1,100])
    new_model = loss.attach(model, result_name)
    assert new_model

def test_e2e():
    download_model(MODEL_PATH)
    model = pyov.Model.from_file(MODEL_PATH)
    result_name = model.get_result().get_friendly_name()
    
    loss = L2Loss([-1,100])
    new_model = loss.attach(model, result_name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.Food101(
        root="DATASET_PATH",
        split = 'test', 
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
        download = True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0)

    for i, (data, target) in enumerate(val_loader):
        model_input = {"input": data.numpy(), "target": target.numpy()}
        loss_val = new_model(model_input)
        print(f"Loss: {loss_val}")
        if i>10:
            break



