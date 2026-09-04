import numpy as np

from .explainable import Explainable
from .activations import Sigmoid
from .costs import BinaryCrossEntropy

# Diagnostics

class GradientCheck(Explainable):
  _formula = (
      "relative_diff = ||analytical - numerical|| / (||analytical|| + ||numerical||),  "
      "where numerical[i] ~= (cost(theta_i + eps) - cost(theta_i - eps)) / (2*eps)"
  )
  _why_used = (
      "The only real way to verify backprop is correct, not just that it "
      "runs without crashing. A subtle sign error or wrong axis in "
      "backprop_layer can still train something and decrease the cost "
      "while being completely wrong. Gradient checking compares the "
      "analytical gradient against an independent, brute-force estimate "
      "using finite differences. Run it especially after adding a new "
      "activation or cost function, since a wrong derivative there is "
      "otherwise hard to notice."
  )
  _why_not = (
      "Slow: two full forward passes per parameter checked, so it is "
      "never run during actual training. Use num_checks to sample a "
      "subset on a larger network. It also only verifies the math is "
      "internally consistent, not that the architecture or "
      "hyperparameters are a good choice for your problem."
  )


# Networks

# Binary Neural Network

class NeuralNetworkBinary:

  def __init__(self, n, hidden_activation=Sigmoid, output_activation=Sigmoid, cost_fn=BinaryCrossEntropy):
    if len(n) < 2:
      raise ValueError(f"n must have at least 2 entries (input + output layer), got {n!r}")
    for size in n:
      if not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"every layer size in n must be a positive integer, got {size!r} in n={n!r}")
    if n[-1] != 1:
      raise ValueError(
          f"n[-1] (output layer size) must be 1 — got {n[-1]} in n={n!r}. "
          f"NeuralNetworkBinary only supports single-output binary classification "
          f"(BinaryCrossEntropy's cost normalization assumes it)."
      )

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
    # Prints per-layer shapes, activation, and param count -- quick sanity check before training
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
    # Shuffles X, y together then splits off test_size fraction as a held-out test set
    rng = np.random.default_rng(seed)  # rng = random number generator, seeded for reproducibility
    indices = rng.permutation(len(X))  # Shuffled index order
    split = int(len(X) * (1 - test_size))  # Index where the train/test split happens
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
    zero_std = self.X_std == 0  # constant column (or only 1 training sample) -- std is 0
    if np.any(zero_std):
      cols = np.where(zero_std)[0].tolist()
      print(f"prepare_data(): column(s) {cols} have zero variance (constant "
            f"value, or only 1 training sample) — standardizing to 0 instead "
            f"of dividing by zero.")
    self.X_std = np.where(zero_std, 1, self.X_std)  # avoid divide-by-zero; numerator is already 0 there
    X = (X - self.X_mean) / self.X_std  # Standardising (Z-score normalisation) each feature (column) to mean 0, std 1

    m = X.shape[0]  # Number of training samples
    A0 = X.T  # Transposes the matrix, obtaining A^[0] in shape n^[0] x m
    Y = y.reshape(self.n[-1], m)  # Reshaping training labels to fit output layer

    return A0, Y, m

  def predict(self, X, threshold=0.5):
    # Standardises new data using the TRAINING set's mean/std (from prepare_data), not X's own
    # threshold: values >= threshold become 1, else 0. threshold=None returns the raw y_hat instead (e.g. regression)
    if not hasattr(self, "X_mean"):
      raise RuntimeError("predict() needs prepare_data() to have been called on training data first")
    if X.shape[1] != self.X_mean.shape[0]:
      raise ValueError(
          f"predict() got {X.shape[1]} feature column(s), but prepare_data() "
          f"was trained on {self.X_mean.shape[0]} — X must have the same "
          f"features (same columns, same order) as the training data."
      )

    X_std = (X - self.X_mean) / self.X_std  # Standardise new data the same way as training data
    A0 = X_std.T
    y_hat, _ = self.feed_forward(A0)

    if threshold is None:
      return y_hat
    return (y_hat >= threshold).astype(int)

  def predict_grid(self, xs, ys):
    # Like predict(), but evaluates every point of a coordinate grid instead of a list of samples -- for plotting a 2D decision boundary
    # xs/ys are raw-space coordinates (same units as X); standardized internally the same way predict() does
    # Returns predictions reshaped to (len(ys), len(xs)), ready for matplotlib's imshow/contour
    if not hasattr(self, "X_mean"):
      raise RuntimeError("predict_grid() needs prepare_data() to have been called on training data first")
    if self.X_mean.shape[0] != 2:
      raise ValueError(
          f"predict_grid() only works for a network trained on exactly 2 "
          f"features (it evaluates a 2D grid) — this network was trained "
          f"on {self.X_mean.shape[0]}."
      )

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
    # Plots cost vs iterations -- same matplotlib lines from the bottom of nn_variable.py, as a one-liner
    import matplotlib.pyplot as plt  # Imported lazily so importing network alone doesn't require matplotlib

    plt.plot(range(len(costs)), costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.show()

  def gradient_check(self, A0, Y, m, epsilon=1e-7, tolerance=1e-7, num_checks=None, verbose=True):
    # See GradientCheck.explain() for what this does and why.
    # Nudges each parameter by +-epsilon, sees how much the cost actually moves, and compares
    # that to what backprop claims the gradient is -- if they agree, backprop is very likely correct.
    # num_checks: check every parameter by default, or pass a number to randomly sample that many instead (faster on a large network)

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
