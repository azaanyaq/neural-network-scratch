"""
EduNet — a small, from-scratch neural network library built for learning:
every activation, cost function, and diagnostic is written out in the
open, so you can read (and swap) every piece.
"""

from .network import (
    NeuralNetworkBinary,
    Explainable,
    Sigmoid,
    ReLU,
    Tanh,
    Identity,
    BinaryCrossEntropy,
    MeanSquaredError,
    GradientCheck,
)

# Controls exactly what `from EduNet import *` pulls in — without this,
# it would also leak network's own imports (numpy, textwrap) into the
# caller's namespace.
__all__ = [
    "NeuralNetworkBinary",
    "Explainable",
    "Sigmoid",
    "ReLU",
    "Tanh",
    "Identity",
    "BinaryCrossEntropy",
    "MeanSquaredError",
    "GradientCheck",
    "demo_vis",
    "make_blobs_dataset",
    "make_xor_dataset",
    "make_circles_dataset",
    "make_moons_dataset",
    "TrainingRecorder",
    "load_csv",
    "handle_missing",
    "encode_categorical",
    "load_dataset",
]

# name -> which submodule actually defines it, imported lazily on first
# access. Keeps `import EduNet` (and anything using just
# NeuralNetworkBinary) free of a hard matplotlib/pandas dependency — each
# submodule is only imported the moment one of its own names is touched.
_LAZY_MODULES = {
    "demo_vis": "visualizer",
    "make_blobs_dataset": "visualizer",
    "make_xor_dataset": "visualizer",
    "make_circles_dataset": "visualizer",
    "make_moons_dataset": "visualizer",
    "TrainingRecorder": "visualizer",
    "load_csv": "data_utils",
    "handle_missing": "data_utils",
    "encode_categorical": "data_utils",
    "load_dataset": "data_utils",
}


def __getattr__(name):
  # PEP 562: only reached for names not already bound above.
  module_name = _LAZY_MODULES.get(name)
  if module_name is None:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  import importlib
  module = importlib.import_module(f".{module_name}", __name__)
  return getattr(module, name)
