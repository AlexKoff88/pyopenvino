import openvino.runtime as ov

class Model():
    def __init__(self, model: ov.Model=None):
        self._model = model
        self._compiled_model = None
        self._core = ov.Core()

    @classmethod
    def from_file(cls, file_name: str):
        core = ov.Core()
        model = core.read_model(model=file_name)
        return cls(model)

    def __call__(self, inputs) -> dict:
        if not self._compiled_model:
            self._compiled_model = self._core.compile_model(self._model)
            self.outputs_map = {next(iter(output.names)): output for output in self._compiled_model.outputs}

        infer_result = self._compiled_model(inputs)
        result = {next(iter(output.names)): value for (output, value) in infer_result.items()}
        return result

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        elif attr in self._model.__dict__:
            self._compiled_model = None
            return getattr(self._model, attr)
        return getattr(self._compiled_model, attr)