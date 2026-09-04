import numpy as np

from .explainable import Explainable

class Sigmoid(Explainable):
  _formula = "A = 1 / (1 + e^-Z)"
  _why_used = (
      "A smooth function whose output is bounded between 0 and 1, so it "
      "reads naturally as a probability. This makes it the standard choice "
      "for a binary classification output layer, even in networks that use "
      "a different activation for hidden layers."
  )
  _why_not = (
      "Prone to vanishing gradients. Its derivative maxes out at 0.25 and "
      "shrinks quickly away from Z=0, so a deep stack of sigmoid hidden "
      "layers barely learns in its earliest layers. Modern networks "
      "mostly use ReLU or similar for hidden layers instead."
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
      "Cheap to compute and does not suffer vanishing gradients for "
      "positive inputs. This makes it the default choice for hidden "
      "layers in most modern networks."
  )
  _why_not = (
      "A neuron whose weights push it permanently negative outputs zero "
      "forever and stops learning. This is called a dying ReLU. It is "
      "also unbounded, so it is a poor choice for an output layer that "
      "needs to represent a probability."
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
      "Zero-centered output, unlike Sigmoid, which is always positive. "
      "This tends to help gradient descent converge faster in hidden "
      "layers."
  )
  _why_not = (
      "Still prone to vanishing gradients, just less severely than "
      "Sigmoid. In deep networks, ReLU is usually preferred for hidden "
      "layers instead."
  )

  @staticmethod
  def forward(Z):
    return np.tanh(Z)

  @staticmethod
  def backward(A):
    return 1 - A ** 2


class Identity(Explainable):
  """Linear/no-op activation. Pairs with MeanSquaredError for regression."""

  _formula = "A = Z"
  _why_used = (
      "Regression tasks predict an unbounded real number, like a price "
      "or a temperature. The output layer should not be squashed into "
      "any particular range, so no activation at all is the right choice."
  )
  _why_not = (
      "Never useful in a hidden layer. Stacking linear layers with no "
      "nonlinearity between them collapses into a single linear layer, "
      "so the extra depth adds no representational power."
  )

  @staticmethod
  def forward(Z):
    return Z

  @staticmethod
  def backward(A):
    return np.ones_like(A)
