from typing import Any
import openvino.runtime as ov
from pyopenvino import Model

class InferQueue():
    def __init__(self, model:Model, callback:callable, workers:int=1):
        """
        Creates a queue for asyncronous inference

        Arguments:
            model (pyopenvino.Model):
                Model object for inference
            callback:
                Callback function to process inference results
            workers:
                Number of parallel executors
        """
        self._model = model
        self._queue = ov.AsyncInferQueue(self._model.compile(), workers)
        self._callback = callback
        self._queue.set_callback(self._callback)

    def send_request(self, inputs: dict, userdata=Any):
        self._queue.start_async(inputs, userdata=userdata)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._queue, attr)