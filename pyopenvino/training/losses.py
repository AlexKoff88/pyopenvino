from abc import ABC, abstractmethod
from typing import List
import copy

import openvino.runtime as ov
from openvino.runtime import opset9
from openvino.runtime.utils import replace_node

from pyopenvino import Model

class BaseLoss(ABC, Model):
    @abstractmethod
    def attach(self, model: Model, ouput_names):
        pass

class L2Loss(BaseLoss):
    def __init__(self, target_shape):
        super(L2Loss, self).__init__()

        # Create subgraph for L2
        input = opset9.parameter(target_shape, name="input")
        target = opset9.parameter(target_shape, name="target")
        l2 = opset9.squared_difference(input, target, name="l2")
        shapeof = opset9.shape_of(l2, name="shape")
        shape_of_shape = opset9.shape_of(shapeof, name="shape_of_shape")
        axis = opset9.constant(0)
        axis_end = opset9.squeeze(shape_of_shape, axis, name="squeeze")
        axis_start = opset9.constant(1)
        step = opset9.constant(1)
        shape = opset9.range(axis_start, axis_end, step, name="shape")

        mean = opset9.reduce_mean(l2, shape, name="mean")
        result = opset9.result(mean, name="l2_loss")

        self._model = ov.Model([result], [input, target])

    def attach(self, model: Model, result_name: str) -> Model:
        param_node = self.get_input("input")
        result_node = model.get_output(result_name)
        param_node.output(0).replace(result_node.input(0).get_source_output())

        new_params = list(model.get_parameters()) + [self.get_operation("target")]
        new_results = [self.get_operation("l2_loss")]

        ov_model = ov.Model(new_results, new_params)
        return Model(ov_model)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        elif attr in dir(self._model):
            self._compiled_model = None
            return getattr(self._model, attr)
        elif self._compiled_model is not None:
            return getattr(self._compiled_model, attr)
        raise ValueError(f"Unknown attribute: {attr}")

