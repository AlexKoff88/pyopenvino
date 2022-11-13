# pyopenvino
A Torch-style simplified Python API for OpenVINO Toolkit. 

## Features
- Load/save model
- Move to device and apply inference options
- Lower precision to FP16 (GPU inference)
- Infer asynchcrounously to improve the throughput.

See "examples" folder for usage details.

## Installation
```python
python setup.py install
```

## Run Example
```python
python examples/simple_example.py
```

## API Reference
For more information about API, please refer to [API documentation](https://alexkoff88.github.io/pyopenvino/pyopenvino/index.html).
