import numpy as np

from ..network import NeuralNetworkBinary, Sigmoid, BinaryCrossEntropy
from .datasets import make_grid_axes
from .app import _launch_visualizer


def train_and_capture(n, X, y, epochs, alpha, seed=None,
                       hidden_activation=Sigmoid, output_activation=Sigmoid, cost_fn=BinaryCrossEntropy,
                       grid_res=40, grid_pad=1.0):
  np.random.seed(seed)  # np.random.seed(None) re-seeds from OS entropy - a valid "unseeded" run
  net = NeuralNetworkBinary(n, hidden_activation, output_activation, cost_fn)
  A0, Y, m = net.prepare_data(X, y)  # also captures net.X_mean/net.X_std, used by predict_grid below

  # The heatmap/contour view is a 2D grid technique -- it has no meaning for
  # a network with more than 2 input features. Skip building the grid (and
  # every predict_grid() call below) entirely rather than crashing on
  # predict_grid()'s hardcoded 2-column grid_raw; _launch_visualizer shows
  # an explanatory message in that panel instead.
  supports_2d_view = X.shape[1] == 2
  # Grid built in raw units (predict_grid standardizes internally, same
  # convention as predict()).
  grid_x, grid_y = make_grid_axes(X, pad=grid_pad, res=grid_res) if supports_2d_view else (None, None)

  print("Training network...")
  snapshots = []
  for e in range(epochs):
    y_hat, cache = net.feed_forward(A0)
    error = net.cost(y_hat, Y)

    grads = {}   # keys "W{l}"/"b{l}" — used below to actually update net.params
    dWs = {}     # keys "dW{l}" — just the weight gradients, for the snapshot
    dZs = {}     # keys "dZ{l}"
    propagator = None
    for l in range(net.L, 0, -1):
      A_l = cache[f"A{l}"]
      # backprop_layer doesn't return dZ (its public signature can't change —
      # train()/gradient_check()/every example script depend on the current
      # 3-value return) — recompute it here with the same formula it uses
      # internally, so we can capture it for the backward-view coloring.
      if l == net.L:
        dZ = net.cost_fn.backward(A_l, Y, m) * net.output_activation.backward(A_l)
      else:
        dZ = propagator * net.hidden_activation.backward(A_l)
      dZs[f"dZ{l}"] = dZ

      dW, db, propagator = net.backprop_layer(l, cache, m, Y, propagator)
      grads[f"W{l}"] = dW
      grads[f"b{l}"] = db
      dWs[f"dW{l}"] = dW

    for l in range(1, net.L + 1):
      net.params[f"W{l}"] -= alpha * grads[f"W{l}"]
      net.params[f"b{l}"] -= alpha * grads[f"b{l}"]

    boundary = net.predict_grid(grid_x, grid_y) if supports_2d_view else None  # uses weights AFTER this step's update
    snapshots.append({
        "epoch": e,
        "cost": error,
        "weights": {f"W{l}": net.params[f"W{l}"].copy() for l in range(1, net.L + 1)},
        "activations": {f"A{l}": cache[f"A{l}"].copy() for l in range(1, net.L + 1)},
        "dW": dWs,
        "dZ": dZs,
        "boundary": boundary,
    })
  if not snapshots:
    raise RuntimeError("train_and_capture() got epochs=0 — nothing to visualize, pass epochs >= 1")
  print(f"  final cost: {snapshots[-1]['cost']:.4f}")

  return {
      "n": n, "L": net.L,
      "X_raw": X, "X_std": A0.T, "y": y,
      "grid_x": grid_x, "grid_y": grid_y,
      "snapshots": snapshots,
      "supports_2d_view": supports_2d_view,
  }


class TrainingRecorder:
  """
  Capture per-epoch snapshots from your OWN manual training loop, instead
  of letting visualize() own training end to end. Call .capture(...) once
  per epoch, right after you update net.params, then .show() when done.

  capture_every: only record every Nth epoch instead of all of them — the
  expensive part of a capture is predict_grid (one extra forward pass over
  a grid_res x grid_res grid), so this is the direct lever if capturing
  every single epoch is too slow for your architecture/epoch count.
  """

  def __init__(self, net, X, Y, m, grid_res=40, grid_pad=1.0, capture_every=1):
    if capture_every < 1:
      raise ValueError(f"capture_every must be >= 1 (1 = capture every epoch), got {capture_every}")
    self.net, self.X, self.Y, self.m = net, X, Y, m
    self.capture_every = capture_every
    # The heatmap/contour view is a 2D grid technique -- meaningless for a
    # network with more than 2 input features. Skip building the grid (and
    # every predict_grid() call in capture()) entirely rather than crashing
    # on predict_grid()'s hardcoded 2-column grid; _launch_visualizer shows
    # an explanatory message in that panel instead.
    self.supports_2d_view = X.shape[1] == 2
    self.grid_x, self.grid_y = (
        make_grid_axes(X, pad=grid_pad, res=grid_res) if self.supports_2d_view else (None, None)
    )
    self.snapshots = []
    self._epoch = 0
    # Weights as they were BEFORE the caller's next params update -- i.e.
    # the weights that actually produced whatever cache/grads capture() is
    # about to be handed. Documented usage calls capture() AFTER the
    # caller's own params-update loop, so net.params is already one step
    # ahead by then; recomputing dZ's propagator from net.params directly
    # would silently use the wrong (next-step) weights. Refreshed
    # unconditionally at the end of every capture() call (not just captured
    # epochs) since net.params changes every epoch regardless of
    # capture_every.
    self._prev_params = {k: v.copy() for k, v in net.params.items()}

  def capture(self, cache, grads, error):
    if self._epoch % self.capture_every == 0:
      net, Y, m, L = self.net, self.Y, self.m, self.net.L
      # backprop_layer doesn't return dZ (its public signature can't
      # change — train()/gradient_check()/every example script depend on
      # the current 3-value return) — recompute it here the same way
      # train_and_capture does, using the caller's own cache/grads.
      dZs, propagator = {}, None
      for l in range(L, 0, -1):
        A_l = cache[f"A{l}"]
        if l == L:
          dZ = net.cost_fn.backward(A_l, Y, m) * net.output_activation.backward(A_l)
        else:
          dZ = propagator * net.hidden_activation.backward(A_l)
        dZs[f"dZ{l}"] = dZ
        propagator = self._prev_params[f"W{l}"].T @ dZ

      self.snapshots.append({
          "epoch": self._epoch,
          "cost": error,
          "weights": {f"W{l}": net.params[f"W{l}"].copy() for l in range(1, L + 1)},
          "activations": {f"A{l}": cache[f"A{l}"].copy() for l in range(1, L + 1)},
          "dW": {f"dW{l}": grads[f"W{l}"] for l in range(1, L + 1)},
          "dZ": dZs,
          "boundary": net.predict_grid(self.grid_x, self.grid_y) if self.supports_2d_view else None,
      })
    self._prev_params = {k: v.copy() for k, v in self.net.params.items()}
    self._epoch += 1

  def show(self):
    """Opens the same interactive window demo_vis() does, using whatever
    history .capture() has recorded so far. No dataset-switcher buttons here --
    those only apply to demo_vis()'s built-in presets, not a dataset you supplied
    yourself."""
    if not self.snapshots:
      raise RuntimeError("no snapshots captured — call .capture(...) at least once before .show()")
    data = {
        "n": self.net.n, "L": self.net.L,
        "X_raw": self.X, "X_std": self._X_std_from_net(), "y": self.Y.ravel(),
        "grid_x": self.grid_x, "grid_y": self.grid_y,
        "snapshots": self.snapshots,
        "supports_2d_view": self.supports_2d_view,
    }
    return _launch_visualizer(data)

  def _X_std_from_net(self):
    return (self.X - self.net.X_mean) / self.net.X_std
