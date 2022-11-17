from datasets import load_dataset

from pyopenvino.training.losses import L2Loss
import openvino.runtime as ov

import numpy as np

def test_l2_loss():
    loss = L2Loss(1000)
    loss.save("l2loss.xml")

    print(f"Inputs: {loss.get_parameters()}")
    
    x = np.random.rand(1,1000)
    result = loss({"input":x, "target": x})

    print(f"result: {result}")
    assert result["output0"] == 0

def test_e2e():
    dataset = load_dataset("food101", split="validation[:100]") 