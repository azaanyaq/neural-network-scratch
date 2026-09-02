import numpy as np

from .explainable import Explainable

class BinaryCrossEntropy(Explainable):
  """Pair with a Sigmoid output layer (assumes y_hat in (0, 1))."""

  _formula = "C = -mean( y*log(y_hat) + (1-y)*log(1-y_hat) )"
  _why_used = (
      "The standard loss for binary classification. Paired with a Sigmoid "
      "output, its gradient stays useful even when the network is "
      "confidently wrong — exactly when you need the strongest learning "
      "signal."
  )
  _why_not = (
      "Breaks numerically if y_hat ever hits exactly 0 or 1 (log(0) is "
      "undefined) — needs an output activation that stays strictly inside "
      "(0, 1), like Sigmoid. Also the wrong tool for regression or "
      "multi-class problems (those need MeanSquaredError or categorical "
      "cross-entropy respectively)."
  )

  @staticmethod
  def forward(y_hat, y):
    # 1. Losses is a n^L x m matrix
    losses = -((y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat))

    m = y_hat.size  # Calculates total number of predictions that make up y_hat

    # 2. Summing across axis = 1 means we sum across rows, making this a n^L x 1 matrix
    summed_losses = (1 / m) * np.sum(losses, axis=1)

    return np.sum(summed_losses)

  @staticmethod
  def backward(y_hat, y, m):
    return (1 / m) * (-(y / y_hat) + (1 - y) / (1 - y_hat))


class MeanSquaredError(Explainable):
  """Pair with an Identity output layer for regression."""

  _formula = "C = mean( (y_hat - y)^2 )"
  _why_used = (
      "The natural fit for regression — \"average squared distance from "
      "the true value.\" What ordinary least-squares fitting is built on."
  )
  _why_not = (
      "If paired with a Sigmoid output for classification — a very "
      "natural first instinct — the sigmoid's saturation makes MSE's "
      "gradient go nearly flat exactly when predictions are very wrong, "
      "so learning stalls right when it matters most. That's the actual "
      "reason BinaryCrossEntropy exists as a separate loss rather than "
      "everyone just using MSE for classification too."
  )

  @staticmethod
  def forward(y_hat, y):
    losses = (y_hat - y) ** 2
    m = y_hat.size
    summed_losses = (1 / m) * np.sum(losses, axis=1)
    return np.sum(summed_losses)

  @staticmethod
  def backward(y_hat, y, m):
    return (2 / m) * (y_hat - y)
