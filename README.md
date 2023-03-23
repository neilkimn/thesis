# thesis

## Installation

**Method 1:** setuptools

```shell
$ pip install -U setuptools

$ pip install -e .
```

**Method 2:** conda

```shell
$ conda env create -f environment.yaml

$ conda activate thesis

$ pip install -e . # need to install as package anyway
```

Depending on CUDA version, update dependencies accordingly (e.g. [CuPy](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi))