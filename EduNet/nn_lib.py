import textwrap
import numpy as np

# ============================================================
# Class version of the handcoded network in nn.py — sigmoid activations +
# binary cross-entropy by default, but both are now swappable: pass your
# own activation function(s) and cost function into the constructor.
#
# Every network piece is its own method (feed_forward, backprop_layer,
# cost, prepare_data, train) so you can call them individually to build
# your own training loop, rather than being locked into train().
#
# Nothing runs on import — only when you actually construct a
# NeuralNetworkBinary and call its methods.
#
# Example, from a new file in this same directory:
#
#   from nn_lib import NeuralNetworkBinary, ReLU, MeanSquaredError
#
#   net = NeuralNetworkBinary(
#       n=[2, 20, 20, 1],
#       hidden_activation=ReLU,       # default: Sigmoid
#       cost_fn=MeanSquaredError,     # default: BinaryCrossEntropy
#   )
#   A0, Y, m = net.prepare_data(X, y)
#   costs = net.train(A0, Y, m, epochs=1000, alpha=0.01)
# ============================================================


# ---------------- self-documentation ----------------

class Explainable:
  """
  Mixin giving every activation/cost class a shared `.explain()` — call it
  on the class itself, e.g. `ReLU.explain()`, no instance needed. Content
  lives in the three class attributes below; this base class only handles
  formatting, so adding a new activation/cost just means setting those
  three strings, not writing a new print routine.
  """
  _formula = ""
  _why_used = ""
  _why_not = ""

  @classmethod
  def explain(cls):
    width = 64
    bar = "=" * width
    wrap = lambda text: textwrap.fill(text, width=width)

    print(bar)
    print(cls.__name__)
    print(bar)
    print("Formula:")
    print(textwrap.indent(wrap(cls._formula), "  "))
    print()
    print("Why it's used:")
    print(textwrap.indent(wrap(cls._why_used), "  "))
    print()
    print("Why it's avoided:")
    print(textwrap.indent(wrap(cls._why_not), "  "))
    print(bar)


# ---------------- activation functions ----------------
#
# Each one is forward(Z) -> A, and backward(A) -> dA/dZ (the derivative,
# expressed in terms of the already-computed activation A rather than Z —
# matches how the original handcoded backprop_layer worked, and avoids
# needing to cache Z separately).

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


# ---------------- cost functions ----------------
#
# Each one is forward(y_hat, y) -> scalar cost, and backward(y_hat, y, m)
# -> dC/dy_hat (elementwise, same shape as y_hat). `m` is the sample count
# (same m used everywhere else in backprop) — the two coincide with
# forward()'s own normalization exactly when the output layer has 1 node,
# which is the case for every network this library is designed for.

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


# ---------------- diagnostics ----------------
#
# Not a math component like the activations/costs above — this documents
# the NeuralNetworkBinary.gradient_check() method the same way, so
# `GradientCheck.explain()` works consistently with every other technique
# in this library.

class GradientCheck(Explainable):
  _formula = (
      "relative_diff = ||analytical - numerical|| / (||analytical|| + ||numerical||),  "
      "where numerical[i] ~= (cost(theta_i + eps) - cost(theta_i - eps)) / (2*eps)"
  )
  _why_used = (
      "The only way to verify backprop is actually correct, rather than "
      "just 'runs without crashing.' A subtle sign error, wrong transpose, "
      "or wrong axis in backprop_layer can still train something, "
      "decrease the cost, and look completely fine — gradient checking is "
      "a fully independent, brute-force way (finite differences directly "
      "on the cost function) to confirm the analytical gradient is right. "
      "Especially worth running right after adding a new activation or "
      "cost function, since a wrong derivative there would otherwise be "
      "very hard to notice."
  )
  _why_not = (
      "Slow — two full forward passes per parameter checked — so it's "
      "never run during actual training, only as a one-off correctness "
      "check (pass num_checks to sample a subset on a larger network). It "
      "also only verifies the math is internally consistent, not that "
      "the architecture or hyperparameters are a good choice for your "
      "problem."
  )


# ---------------- network ----------------

class NeuralNetworkBinary:

  def __init__(self, n, hidden_activation=Sigmoid, output_activation=Sigmoid, cost_fn=BinaryCrossEntropy):
    self.n = n  # Layer size list, e.g. [2, 20, 20, 1]
    self.L = len(n) - 1  # Number of layers (excluding input layer)
    self.hidden_activation = hidden_activation  # used for layers 1..L-1
    self.output_activation = output_activation  # used for layer L
    self.cost_fn = cost_fn
    self.params = {}  # Dictionary to hold W1..WL and b1..bL

    for l in range(1, self.L + 1):  # l in ranges 1 to (and including) L
      self.params[f"W{l}"] = np.random.randn(n[l], n[l - 1])  # Weight matrix follows n^[l] x n^[l-1]
      self.params[f"b{l}"] = np.random.randn(n[l], 1)  # Bias matrix follows n^[l] x 1

  def summary(self):
    """Prints per-layer shapes, which activation each layer uses, and the
    total parameter count — a quick sanity check before training,
    especially for catching a typo'd n list."""
    col = "{:<10}{:<14}{:<12}{:<12}{:<10}"
    width = 58

    print("=" * width)
    print(f"NeuralNetworkBinary — {self.L} layer{'s' if self.L != 1 else ''}, input size {self.n[0]}")
    print("=" * width)
    print(col.format("Layer", "W shape", "b shape", "Activation", "Params"))
    print("-" * width)

    total_params = 0
    for l in range(1, self.L + 1):
      W = self.params[f"W{l}"]
      b = self.params[f"b{l}"]
      activation = self.output_activation if l == self.L else self.hidden_activation
      layer_params = W.size + b.size
      total_params += layer_params
      label = f"{l} (out)" if l == self.L else str(l)
      print(col.format(label, str(W.shape), str(b.shape), activation.__name__, layer_params))

    print("-" * width)
    print(f"Cost function: {self.cost_fn.__name__}")
    print(f"Total params:  {total_params}")
    print("=" * width)

  def cost(self, y_hat, y):  # Both y_hat and y should be a n^L x m matrix
    return self.cost_fn.forward(y_hat, y)

  @staticmethod
  def train_test_split(X, y, test_size=0.2, seed=None):
    """Shuffles X and y together, then splits off `test_size` fraction as
    a held-out test set. Returns (X_train, X_test, y_train, y_test).
    Doesn't need an instance — call as NeuralNetworkBinary.train_test_split(...)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

  def prepare_data(self, X, y):
    # X: Matrix of raw samples
    # y: Array of training labels

    # Stored so predict() can standardize new data the same way later,
    # instead of recomputing mean/std from whatever it's given — see
    # predict()'s docstring for why that distinction matters.
    self.X_mean = X.mean(axis=0)
    self.X_std = X.std(axis=0)
    X = (X - self.X_mean) / self.X_std  # Standardising (Z-score normalisation) each feature (column) to mean 0, std 1

    m = X.shape[0]  # Number of training samples
    A0 = X.T  # Transposes the matrix, obtaining A^[0] in shape n^[0] x m
    Y = y.reshape(self.n[-1], m)  # Reshaping training labels to fit output layer

    return A0, Y, m

  def predict(self, X, threshold=0.5):
    """
    Predicts on new data, standardizing it using the training set's mean/std
    (captured by prepare_data) rather than X's own — see the chat
    discussion this was built from for why recomputing per-call is wrong.

    threshold: values >= threshold become 1, else 0 — the right default
    for classification (Sigmoid output). Pass threshold=None to get the
    raw y_hat instead (e.g. for a regression setup using Identity output).
    """
    if not hasattr(self, "X_mean"):
      raise RuntimeError("predict() needs prepare_data() to have been called on training data first")

    X_std = (X - self.X_mean) / self.X_std
    A0 = X_std.T
    y_hat, _ = self.feed_forward(A0)

    if threshold is None:
      return y_hat
    return (y_hat >= threshold).astype(int)

  def predict_grid(self, xs, ys):
    """
    Like predict(), but evaluates over every point of a coordinate grid
    instead of a list of samples — for plotting a 2D decision boundary.
    xs/ys are raw-space coordinates (same units as your original X);
    standardized internally the same way predict() does.

    Returns predictions reshaped to (len(ys), len(xs)), ready for
    matplotlib's imshow/contour.
    """
    if not hasattr(self, "X_mean"):
      raise RuntimeError("predict_grid() needs prepare_data() to have been called on training data first")

    XX, YY = np.meshgrid(xs, ys)
    grid_raw = np.stack([XX.ravel(), YY.ravel()], axis=1)
    grid_std = (grid_raw - self.X_mean) / self.X_std
    y_hat, _ = self.feed_forward(grid_std.T)
    return y_hat[0].reshape(XX.shape)

  def feed_forward(self, A0):
    cache = {"A0": A0}  # Creates a cache dictionary with A0 first entry
    A = A0  # Initialise value A (firstly as A0)

    for l in range(1, self.L + 1):  # l in range 1 to (and including) L
      W = self.params[f"W{l}"]  # Grabs weights for layer l
      b = self.params[f"b{l}"]  # Grabs biases for layer l
      Z = W @ A + b  # Matrix multiplication and addition to find pre-activation value
      activation = self.output_activation if l == self.L else self.hidden_activation
      A = activation.forward(Z)  # Find post-activation value
      cache[f"A{l}"] = A  # Add to the cache dictionary

    y_hat = A

    return y_hat, cache

  def backprop_layer(self, l, cache, m, Y, propagator_dC_dA):  # l is what layer gradients are being computed
    A_l = cache[f"A{l}"]  # Extracting A value of this layer
    A_prev = cache[f"A{l - 1}"]  # Extracting A value of previous layer
    W_l = self.params[f"W{l}"]  # Extracting weights of this layer

    if l == self.L:  # Output layer: chain rule through the cost fn, then the output activation
      dC_dA = self.cost_fn.backward(A_l, Y, m)
      dA_dZ = self.output_activation.backward(A_l)
      dC_dZ = dC_dA * dA_dZ
    else:  # Every other layer
      dA_dZ = self.hidden_activation.backward(A_l)
      dC_dZ = propagator_dC_dA * dA_dZ  # Calculates dC/dZ from propogator handed down
    assert dC_dZ.shape == (self.n[l], m)

    dC_dW = dC_dZ @ A_prev.T
    assert dC_dW.shape == (self.n[l], self.n[l - 1])

    dC_db = np.sum(dC_dZ, axis=1, keepdims=True)
    assert dC_db.shape == (self.n[l], 1)

    dC_dA_prev = W_l.T @ dC_dZ  # Propagator for the layer below
    assert dC_dA_prev.shape == (self.n[l - 1], m)

    return dC_dW, dC_db, dC_dA_prev

  def train(self, A0, Y, m, epochs=1000, alpha=0.01):
    costs = []  # Create an empty list for costs (to be appended later)

    for e in range(epochs):  # Going through loop for each epoch

      y_hat, cache = self.feed_forward(A0)  # Feed forward (outputting prediction y_hat and intermediate layers A)

      error = self.cost(y_hat, Y)  # Calculating cost for each epoch (compares y_hat with Y)
      costs.append(error)  # Appending these individual costs to the empty list

      grads = {}  # Creating empty dictionary for gradients
      propagator = None  # Starts as none as no layer after L (therefore no inherited propogator)

      for l in range(self.L, 0, -1):  # Looping backwards from L -> 1
        dC_dW, dC_db, propagator = self.backprop_layer(  # Calculating gradients (PDs) of that layer
            l, cache, m, Y=Y, propagator_dC_dA=propagator
        )
        grads[f"W{l}"] = dC_dW  # Weight gradients added to dictionary
        grads[f"b{l}"] = dC_db  # Bias gradients added to dictionary

      for l in range(1, self.L + 1):  # Looping from 1 -> L
        self.params[f"W{l}"] = self.params[f"W{l}"] - (alpha * grads[f"W{l}"])  # Weights updated using weight gradients of resp layers
        self.params[f"b{l}"] = self.params[f"b{l}"] - (alpha * grads[f"b{l}"])  # Biases updated using bias gradients of resp layers

      if e % 20 == 0:  # Every 20 epochs, print the current cost
        print(f"epoch {e}: cost = {error:4f}")

    return costs

  def plot_cost(self, costs):
    """Plots the cost list returned by train() — the same handful of
    matplotlib lines every example file was repeating, as a one-liner.
    Imports matplotlib lazily, so just importing nn_lib doesn't require it."""
    import matplotlib.pyplot as plt

    plt.plot(range(len(costs)), costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.show()

  def gradient_check(self, A0, Y, m, epsilon=1e-7, tolerance=1e-7, num_checks=None, verbose=True):
    """
    See GradientCheck.explain() for what this does and why.

    Verifies backprop_layer's analytical gradients against a numerical
    estimate — nudge each parameter by +-epsilon, see how much the cost
    actually moves, and compare that to what backprop claims the gradient
    is. If they agree, backprop is very likely implemented correctly; if
    they don't, something in feed_forward/backprop_layer is wrong.

    Only meaningful for a single-output-node network (n[-1] == 1) — the
    same assumption backprop_layer's output-layer shortcut already relies
    on (see the cost_fn.backward docstring above).

    num_checks: check every parameter by default. For a large network
    that's a lot of forward passes (two per parameter) — pass a number to
    randomly sample that many parameters instead, which is much faster and
    still a statistically meaningful check.
    """
    # 1. One real forward + backward pass, to get the analytical gradients
    y_hat, cache = self.feed_forward(A0)
    grads = {}
    propagator = None
    for l in range(self.L, 0, -1):
      dW, db, propagator = self.backprop_layer(l, cache, m, Y, propagator)
      grads[f"W{l}"] = dW
      grads[f"b{l}"] = db

    keys = []
    for l in range(1, self.L + 1):
      keys += [f"W{l}", f"b{l}"]

    # 2. Every individual parameter position, as (key, flat_index) pairs
    all_positions = [(k, i) for k in keys for i in range(self.params[k].size)]

    if num_checks is not None and num_checks < len(all_positions):
      rng = np.random.default_rng()
      chosen = rng.choice(len(all_positions), size=num_checks, replace=False)
      positions = [all_positions[i] for i in chosen]
    else:
      positions = all_positions

    analytical = np.zeros(len(positions))
    numerical = np.zeros(len(positions))

    # 3. For each chosen parameter: nudge it +-epsilon, recompute cost each
    # time, and estimate the gradient as the resulting slope.
    for idx, (k, i) in enumerate(positions):
      analytical[idx] = grads[k].reshape(-1)[i]

      original_shape = self.params[k].shape
      flat = self.params[k].reshape(-1)
      original_value = flat[i]

      flat[i] = original_value + epsilon
      self.params[k] = flat.reshape(original_shape)
      cost_plus = self.cost(self.feed_forward(A0)[0], Y)

      flat[i] = original_value - epsilon
      self.params[k] = flat.reshape(original_shape)
      cost_minus = self.cost(self.feed_forward(A0)[0], Y)

      numerical[idx] = (cost_plus - cost_minus) / (2 * epsilon)

      flat[i] = original_value  # restore — gradient_check must not leave params changed
      self.params[k] = flat.reshape(original_shape)

    # 4. Compare the two gradient vectors as one relative difference
    numerator = np.linalg.norm(analytical - numerical)
    denominator = np.linalg.norm(analytical) + np.linalg.norm(numerical)
    relative_difference = numerator / denominator if denominator > 0 else 0.0
    passed = relative_difference < tolerance

    if verbose:
      status = "PASSED" if passed else "FAILED"
      print(f"Gradient check {status} — checked {len(positions)}/{len(all_positions)} "
            f"parameters, relative difference: {relative_difference:.2e} "
            f"(tolerance: {tolerance:.0e})")

    return passed


if __name__ == "__main__":
  # Demo — only runs when this file is executed directly
  # (`python3 nn_lib.py`), not when imported as a library.
  import matplotlib.pyplot as plt

  X = np.array([  # Weight (lbs) and height (inches) training values
      [150, 70],
      [254, 73],
      [312, 68],
      [120, 60],
      [154, 61],
      [212, 65],
      [216, 67],
      [145, 67],
      [184, 64],
      [130, 69],
  ])
  y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])  # 1 = at risk, 0 = not at risk

  net = NeuralNetworkBinary(n=[2, 20, 20, 1])  # default: Sigmoid everywhere, binary cross-entropy
  A0, Y, m = net.prepare_data(X, y)
  costs = net.train(A0, Y, m, epochs=1000, alpha=0.01)

  plt.plot(range(len(costs)), costs)
  plt.xlabel("Iterations")
  plt.ylabel("Cost")
  plt.title("Cost vs Iterations")
  plt.show()
