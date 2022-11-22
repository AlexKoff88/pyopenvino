import os
import requests
import tempfile

import torch
from torchvision import transforms
from torchvision import datasets

from pyopenvino.training.losses import L2Loss, CELoss
import openvino.runtime as ov

import numpy as np

import pyopenvino as pyov

MODEL_PATH = tempfile.NamedTemporaryFile(suffix=".onnx").name
DATASET_PATH = "./dataset/"
NUM_CLASSES = 101

def download_model(file_name):
    #URL = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    URL = "https://huggingface.co/AlexKoff88/mobilenet_v2_food101/resolve/main/mobilenet_v2_food101.onnx"
    if not os.path.exists(file_name):
        r = requests.get(URL, allow_redirects=True)
        open(file_name, 'wb').write(r.content)

def test_ce_loss_inference():
    loss = CELoss([1,NUM_CLASSES])
    loss.save("celoss.xml")

    #print(f"Inputs: {loss.get_parameters()}")
    
    x = np.random.rand(1,NUM_CLASSES)
    target = np.array([1], dtype=int)
    one_hot = np.full((2, 2), np.inf) #np.zeros((1,NUM_CLASSES))
    one_hot[0][target] = 1
    result = loss({"input":x, "target": one_hot})
    print(f"One hot: {one_hot}")

    ref_loss = torch.nn.CrossEntropyLoss(reduction="sum")
    ref_x = torch.from_numpy(x)
    ref_target = torch.from_numpy(target)
    ref_result = ref_loss(ref_x, ref_target)

    print(f"Target: {target}, Ref target: {ref_target}, Results: {result}, Ref result: {ref_result}")

    #print(f"result: {result}")
    assert result["output0"] == ref_result.numpy()

def test_l2_loss_inference():
    loss = L2Loss([1,NUM_CLASSES])
    loss.save("l2loss.xml")

    #print(f"Inputs: {loss.get_parameters()}")
    
    x = np.random.rand(1,NUM_CLASSES)
    result = loss({"input":x, "target": x})

    #print(f"result: {result}")
    assert result["output0"] == 0

def test_l2_loss_attach():
    download_model(MODEL_PATH)
    model = pyov.Model.from_file(MODEL_PATH)
    result_name = model.get_result().get_friendly_name()
    
    loss = L2Loss([1,NUM_CLASSES])
    new_model = loss.attach(model, result_name)

    x = np.random.rand(1,3,224,224)
    label = np.random.rand(1,NUM_CLASSES)
    model_input = {"input": x, "target": label}
    result = new_model(model_input)
    assert result

def test_e2e():
    download_model(MODEL_PATH)
    model = pyov.Model.from_file(MODEL_PATH)
    result_name = model.get_result().get_friendly_name()
    
    loss = L2Loss([1,NUM_CLASSES])
    new_model = loss.attach(model, result_name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.Food101(
        root=DATASET_PATH,
        split = "test", 
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

    for i, (data, label) in enumerate(val_loader):
        #print(f"Data shape: {data.shape}, target shape: {label.shape}")
        target = np.zeros((data.size(0),NUM_CLASSES))
        target[0][0] = 1 #target[0][label.item()] = 1
        print(f"Label: {label.item()}, Target: {target}")
        model_input = {"input": data.numpy(), "target": target}
        loss_val = new_model(model_input)
        print(f"Loss: {loss_val['output0'][0]}")
        if i>1:
            break



