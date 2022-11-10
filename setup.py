from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="pyopenvino",
    version="0.1",
    install_requires=required,
    packages=["pyopenvino"],
)