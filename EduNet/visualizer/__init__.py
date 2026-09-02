from ..network import Sigmoid, BinaryCrossEntropy
from .training_capture import train_and_capture, TrainingRecorder
from .app import _launch_visualizer
from .datasets import make_blobs_dataset, make_xor_dataset, make_circles_dataset, make_moons_dataset, PRESET_DATASETS


def demo_vis(architecture, epochs, alpha, seed=None,
             hidden_activation=Sigmoid, output_activation=Sigmoid, cost_fn=BinaryCrossEntropy):
  """
  Trains a NeuralNetworkBinary on a built-in preset dataset and opens the
  interactive visualizer window — click the dataset buttons in the GUI to
  retrain live on a different preset (Blobs/XOR/Circles/Moons).

  architecture, epochs, alpha, seed: same meaning as NeuralNetworkBinary /
           train_and_capture above.
  hidden_activation, output_activation, cost_fn: same swappable components
           as NeuralNetworkBinary itself (Sigmoid/ReLU/Tanh/Identity,
           BinaryCrossEntropy/MeanSquaredError).
  """
  label, dataset_fn = PRESET_DATASETS[0]
  X, y = dataset_fn()
  data = train_and_capture(architecture, X, y, epochs, alpha, seed=seed,
                            hidden_activation=hidden_activation,
                            output_activation=output_activation, cost_fn=cost_fn)
  train_kwargs = dict(architecture=architecture, epochs=epochs, alpha=alpha, seed=seed,
                       hidden_activation=hidden_activation, output_activation=output_activation,
                       cost_fn=cost_fn)
  return _launch_visualizer(data, presets=PRESET_DATASETS, train_kwargs=train_kwargs, active_label=label)
