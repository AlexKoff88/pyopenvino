from __future__ import annotations
from typing import Any, List, Union

import openvino.runtime as ov
from openvino.offline_transformations import compress_model_transformation

class Model():
    """
    A helper class that provides a simple API to use OpenVINO for DL inference
    Features:
        - Load/save model
        - Move to device and apply inference options
        - Lower precision to FP16 (GPU inference)
        - Infer asynchcrounously to improve the throughput
    See "examples" for usage details.
    """
    def __init__(self, model: ov.Model=None, workers=1):
        self._model = model
        self._workers = workers
        self._compiled_model = None
        self._core = ov.Core()
        self._device = "CPU"
        self._config = None
        self._queue = None
        self._callback = None
        self._outputs_names_map = None

    @classmethod
    def from_file(cls, file_name: str) -> Model:
        """
        Creates Model from file 

        Arguments:
            file_name (`str`):
                File with model (.xml, .onnx, .pb)
        """
        core = ov.Core()
        model = core.read_model(model=file_name)
        return cls(model)

    def __call__(self, inputs: dict) -> dict:
        self._compile()

        infer_result = self._compiled_model(inputs)
        
        if self._outputs_names_map is None:
            self._outputs_names_map = {}
            for i, (output, _) in enumerate(infer_result.items()):
                self._outputs_names_map[output] = f"output{i}" if len(output.names) == 0 else next(iter(output.names))

        result = {self._outputs_names_map[output]: value for (output, value) in infer_result.items()}
        return result

    def async_request(self, inputs: Any, userdata=Any):
        """
        Send the asynchrounous inference request

        Arguments:
            inputs (dict, numpy.Array, torch.Tensor):
                Device name, e.g. "CPU" or "GPU"
            userdata:
                User-defined object to track the request
        """
        if not self._callback:
            raise RuntimeError("Cannot run async inference. Callback is empty.")
        self._compile()
        self._create_queue()
        self._queue.start_async(inputs, userdata=userdata)

    def wait(self):
        """
        Blocks the execution until all the asyncrounous requests are being processed
        """
        if self._queue:
            self._queue.wait_all()
    
    def native_model(self):
        """
        Accessor to ov.Model object

        Returns:
            ov.Model: ov.Model object
        """
        return self._model

    def workers(self, value):
        """
        Number of parallel thread for asynchrounous inference processing
        """
        self._workers = value
        self._queue = None # recreate queue

    @property
    def callback(self):
        """
        User-defined callback function to process results of 
        asynchrounous inference
        """
        return self._callback

    def get_operation(self, name):
        """
        Search for operation by its name

        Arguments:
            name (str):
                Operation name
        Returns:
            Node: node in the openvino graph (pybind object)
        """
        result = None
        for operation in self._model.get_ops():
            if operation.get_friendly_name() == name:
                result = operation
                break
        return result

    def get_output_names(self) -> List[str]:
        """
        Returns names of Result operations in the model

        Returns:
            List: names of Results
        """
        result = [r.get_friendly_name() for r in self._model.get_results()]
        return result

    def get_output(self, name):
        """
        Search for result by its name

        Arguments:
            name (str):
                Result name
        Returns:
            Node: node in the openvino graph (pybind object)
        """
        result = None
        for operation in self._model.get_results():
            if operation.get_friendly_name() == name:
                result = operation
                break
        return result

    def get_input_names(self) -> List[str]:
        """
        Returns names of Parameter operations in the model

        Returns:
            List: names of Parameters
        """
        result = [p.get_friendly_name() for p in self._model.get_parameters()]
        return result

    def get_input(self, name):
        """
        Search for parameter (input node) by its name

        Arguments:
            name (str):
                Result name
        Returns:
            Node: node in the openvino graph (pybind object)
        """
        result = None
        for operation in self._model.get_parameters():
            if operation.get_friendly_name() == name:
                result = operation
                break
        return result

    @callback.setter
    def callback(self, value):
        self._callback = value
        if self._queue:
            self._queue.set_callback(self._callback)
    
    def to(self, device: str, config: dict=None) -> Model:
        """
        Select inference device and settings

        Arguments:
            device (`str`):
                Device name, e.g. "CPU" or "GPU"
            config (dict):
                OpenVINO config with inference options
        """
        self._device = device
        self._config = config
        self._compiled_model = None
        return self

    def half(self) -> Model:
        """
        Lower weights precision to FP16. GPU switches to FP16 inference
        """
        compress_model_transformation(self._model)
        self._compiled_model = None 
        return self

    def save(self, file_name: str):
        """
        Save model to file

        Arguments:
            file_name (`str`):
                File to store the model. Should contain .xml extension.
        """
        ov.serialize(self._model, file_name)

    def _compile(self) -> ov.CompiledModel:
        if not self._compiled_model:
            self._compiled_model = self._core.compile_model(self._model, self._device, self._config)
            self._outputs_names_map = None
        return self._compiled_model

    def _create_queue(self):
        if not self._queue:
            self._queue = ov.AsyncInferQueue(self._compiled_model, self._workers)
            if self._callback:
                self._queue.set_callback(self._callback)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        elif attr in dir(self._model):
            self._compiled_model = None
            return getattr(self._model, attr)
        elif self._compiled_model is not None:
            return getattr(self._compiled_model, attr)
        raise ValueError(f"Unknown attribute: {attr}")