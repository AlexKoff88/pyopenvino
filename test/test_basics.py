from genericpath import exists
import numpy as np
import pytest
import tempfile
import requests
import os

import openvino.runtime as ov
import pyopenvino as pyov

MODEL_PATH = tempfile.NamedTemporaryFile(suffix=".onnx").name

def download_model(file_name):
    URL = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    if not os.path.exists(file_name):
        r = requests.get(URL, allow_redirects=True)
        open(file_name, 'wb').write(r.content)

def test_from_file():
    download_model(MODEL_PATH)

    model = pyov.Model.from_file(MODEL_PATH)
    assert model

def test_from_ovmodel():
    download_model(MODEL_PATH)

    ov_model = ov.Core().read_model(MODEL_PATH)
    model = pyov.Model(ov_model)
    assert model

def test_inference():
    input = np.random.randint(255, size=(1,3,224,224), dtype=np.uint8).astype(float)
    download_model(MODEL_PATH)

    model = pyov.Model.from_file(MODEL_PATH)
    result = model(input)
    class_label = np.argmax(result['output'])
    assert class_label >=0 and class_label < 1000

def test_to_device():
    input = np.random.randint(255, size=(1,3,224,224), dtype=np.uint8).astype(float)
    download_model(MODEL_PATH)

    model = pyov.Model.from_file(MODEL_PATH)
    config = {"PERFORMANCE_HINT": "THROUGHPUT",
        "INFERENCE_PRECISION_HINT": "f32"}
    model.to("CPU", config)

    result = model(input)
    class_label = np.argmax(result['output'])
    assert class_label >=0 and class_label < 1000

def test_fp16():
    input = np.random.randint(255, size=(1,3,224,224), dtype=np.uint8).astype(float)
    download_model(MODEL_PATH)

    model = pyov.Model.from_file(MODEL_PATH)
    model.half()
    result = model(input)
    class_label = np.argmax(result['output'])
    assert class_label >=0 and class_label < 1000

def test_save():
    download_model(MODEL_PATH)
    model = pyov.Model.from_file(MODEL_PATH)
    with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
        model.save(tmp.name)
        assert os.path.exists(tmp.name)


@pytest.fixture(scope='session', autouse=True)
def clean_up():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)