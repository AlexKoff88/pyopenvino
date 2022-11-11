from __future__ import annotations

import openvino.runtime as ov
from openvino.offline_transformations import compress_model_transformation

class Model():
    def __init__(self, model: ov.Model=None):
        self._model = model
        self._compiled_model = None
        self._core = ov.Core()
        self._device = "CPU"
        self._config = None

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
        if not self._compiled_model:
            self._compiled_model = self._core.compile_model(self._model, self._device, self._config)

        infer_result = self._compiled_model(inputs)
        result = {next(iter(output.names)): value for (output, value) in infer_result.items()}
        return result
    
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

    def compile(self) -> ov.CompiledModel:
        if not self._compiled_model:
            self._compiled_model = self._core.compile_model(self._model, self._device, self._config)
        return self._compiled_model

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        elif attr in self._model.__dict__:
            self._compiled_model = None
            return getattr(self._model, attr)
        return getattr(self._compiled_model, attr)