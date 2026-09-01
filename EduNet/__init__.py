"""
EduNet — a small, from-scratch neural network library built for learning:
every activation, cost function, and diagnostic is written out in the
open, so you can read (and swap) every piece.

Importing the package pulls in everything needed for a full pipeline —
no need to import from nn_lib directly:

    import EduNet

    net = EduNet.NeuralNetworkBinary(
        n=[2, 20, 20, 1],
        hidden_activation=EduNet.ReLU,
        cost_fn=EduNet.BinaryCrossEntropy,
    )
    X_train, X_test, y_train, y_test = EduNet.NeuralNetworkBinary.train_test_split(X, y)

Or, to use bare names instead of the EduNet.* prefix:

    from EduNet import *

    net = NeuralNetworkBinary(n=[2, 20, 20, 1], hidden_activation=ReLU)

The interactive visualizer is also available from the package directly —
EduNet.visualize(...) or `from EduNet import visualize` — but it's only
actually imported (and matplotlib along with it) the moment you touch it,
not just from `import EduNet`. See visualize.__doc__ for its arguments.

    EduNet.visualize(dataset=EduNet.make_blobs_dataset, architecture=[2, 4, 4, 1], epochs=150, alpha=0.6)

Same story for loading a real dataset from a CSV/URL — EduNet.load_dataset(...)
imports pandas only the moment you call it, not just from `import EduNet`:

    X, y = EduNet.load_dataset("https://example.com/data.csv", target_column="Label")
"""

from .nn_lib import (
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
# it would also leak nn_lib's own imports (numpy, textwrap) into the
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
    "visualize",
    "make_blobs_dataset",
    "make_xor_dataset",
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
    "visualize": "nn_visualizer",
    "make_blobs_dataset": "nn_visualizer",
    "make_xor_dataset": "nn_visualizer",
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
