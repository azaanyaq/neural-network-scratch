import numpy as np

from .explainable import Explainable

class Sigmoid(Explainable):
  _formula = "A = 1 / (1 + e^-Z)"
  _why_used = (
      "The classic activation function — smooth, and its output is bounded "
      "in (0, 1), so it reads naturally as a probability. That's why it's "
      "still the standard choice for an output layer doing binary "
      "classification, even when it's not used in the hidden layers."
  )
  _why_not = (
      "Vanishing-gradient prone: its derivative maxes out at 0.25 and "
      "shrinks fast away from Z=0, so a deep stack of sigmoid hidden "
      "layers barely learns in its earliest layers. Modern networks "
      "mostly reserve it for the output layer and use ReLU (or similar) "
      "for hidden layers instead."
  )

  @staticmethod
  def forward(Z):
    return 1 / (1 + np.exp(-1 * Z))

  @staticmethod
  def backward(A):
    return A * (1 - A)


class ReLU(Explainable):
  _formula = "A = max(0, Z)"
  _why_used = (
      "Cheap to compute, and doesn't suffer vanishing gradient for "
      "positive inputs — the default choice for hidden layers in most "
      "modern networks."
  )
  _why_not = (
      "\"Dying ReLU\": a neuron whose weights push it permanently negative "
      "outputs exactly 0 forever and stops learning entirely. It's also "
      "unbounded, so it's a poor choice for an output layer that needs to "
      "represent a probability."
  )

  @staticmethod
  def forward(Z):
    return np.maximum(0, Z)

  @staticmethod
  def backward(A):
    return (A > 0).astype(A.dtype)


class Tanh(Explainable):
  _formula = "A = tanh(Z)"
  _why_used = (
      "Zero-centered output (unlike Sigmoid, which is always positive), "
      "which tends to help gradient descent converge better in hidden "
      "layers."
  )
  _why_not = (
      "Still vanishing-gradient prone, just less severely than Sigmoid — "
      "in deep networks, ReLU is generally preferred for hidden layers "
      "for that reason."
  )

  @staticmethod
  def forward(Z):
    return np.tanh(Z)

  @staticmethod
  def backward(A):
    return 1 - A ** 2


class Identity(Explainable):
  """Linear/no-op activation — pairs with MeanSquaredError for regression."""

  _formula = "A = Z"
  _why_used = (
      "Predicting an unbounded real number (a price, a temperature, a "
      "count) — regression — needs an output layer that isn't squashed "
      "into any particular range, so no activation at all is the correct "
      "choice."
  )
  _why_not = (
      "Never useful in a hidden layer — stacking linear layers with no "
      "nonlinearity between them collapses mathematically into a single "
      "linear layer, so the network gains no extra representational power "
      "from the extra depth."
  )

  @staticmethod
  def forward(Z):
    return Z

  @staticmethod
  def backward(A):
    return np.ones_like(A)
