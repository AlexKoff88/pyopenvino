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
    def __init__(self, num_classes):
        super(L2Loss, self).__init__()

        # Create subgraph for L2
        input = opset9.parameter([1, num_classes], name="input")
        target = opset9.parameter([1, num_classes], name="target")
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
        resulted_model = copy.deepcopy(model)
        source_model = resulted_model.native_model()

        result = resulted_model.get_operation(result_name)
        input_node_output = result.inputs()[0].get_source_output()

        loss_input = self.get_operation("input")
        for target_in in loss_input.outputs()[0].get_target_inputs():
            target_in.replace_source_output(input_node_output)

        for sig_output in sigmoid_node.outputs():
            for target_in in sig_output.get_target_inputs():
                target_in.replace_source_output(input_node_output)



        source_ouput_input = source_ouput_node.get_target_inputs(0)
        l2 = self.get_operation("l2")
        l2_param_input = l2.get_target_inputs(0)
        l2_param_input.replace_source_output()
        target_in = source_ouput_input.replace_source_output(l2)


        for sig_output in source_model.outputs():
            for target_in in sig_output.get_target_inputs():
                target_in.replace_source_output(input_node_output)

