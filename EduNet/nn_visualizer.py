import numpy as np
import matplotlib

try:
  matplotlib.use("TkAgg")
except ImportError:
  pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap

# Dual import: works both when this file is run/imported directly as a
# top-level module (main.py/nn_test.py do `from nn_visualizer import ...`
# with EduNet/ itself on sys.path — needs the plain absolute import) and
# when it's imported as EduNet.nn_visualizer, e.g. via `import EduNet;
# EduNet.visualize(...)` from outside the package (EduNet/ itself isn't on
# sys.path there — needs the relative import instead).
try:
  from nn_lib import (
      NeuralNetworkBinary,
      Sigmoid, ReLU, Tanh, Identity,
      BinaryCrossEntropy, MeanSquaredError,
  )
except ImportError:
  from .nn_lib import (
      NeuralNetworkBinary,
      Sigmoid, ReLU, Tanh, Identity,
      BinaryCrossEntropy, MeanSquaredError,
  )


### Dataset generators ###

def make_blobs_dataset(n_per_class=10, seed=1):
  rng = np.random.RandomState(seed)
  c0 = rng.randn(n_per_class, 2) * 0.6 + np.array([-1.3, -1.0])
  c1 = rng.randn(n_per_class, 2) * 0.6 + np.array([1.3, 1.0])
  X = np.vstack([c0, c1])
  y = np.array([0] * n_per_class + [1] * n_per_class)
  idx = rng.permutation(len(X))
  return X[idx], y[idx]


def make_xor_dataset(n_per_quadrant=8, seed=2):
  rng = np.random.RandomState(seed)
  centers = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
  labels = [0, 1, 1, 0]  # same-sign quadrants -> 0, opposite-sign -> 1
  Xs, ys = [], []
  for (cx, cy), lab in zip(centers, labels):
    pts = rng.randn(n_per_quadrant, 2) * 0.35 + np.array([cx, cy])
    Xs.append(pts)
    ys.append(np.full(n_per_quadrant, lab))
  X = np.vstack(Xs)
  y = np.concatenate(ys)
  idx = rng.permutation(len(X))
  return X[idx], y[idx]


def make_grid_axes(X, pad=1.0, res=40):
  # pad is in units of that column's own std deviation — keeps the same
  # meaning ("1.0 = pad by 1 std dev") whether X is raw or standardized:
  # for already-standardized data (std=1) this reduces to the old behavior.
  x_std, y_std = X[:, 0].std(), X[:, 1].std()
  x_min, x_max = X[:, 0].min() - pad * x_std, X[:, 0].max() + pad * x_std
  y_min, y_max = X[:, 1].min() - pad * y_std, X[:, 1].max() + pad * y_std
  return np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res)


def train_and_capture(n, X, y, epochs, alpha, seed=None,
                       hidden_activation=Sigmoid, output_activation=Sigmoid, cost_fn=BinaryCrossEntropy,
                       grid_res=40, grid_pad=1.0):
  np.random.seed(seed)  # np.random.seed(None) re-seeds from OS entropy - a valid "unseeded" run
  net = NeuralNetworkBinary(n, hidden_activation, output_activation, cost_fn)
  A0, Y, m = net.prepare_data(X, y)  # also captures net.X_mean/net.X_std, used by predict_grid below

  # Grid built in raw units (predict_grid standardizes internally, same
  # convention as predict()).
  grid_x, grid_y = make_grid_axes(X, pad=grid_pad, res=grid_res)

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

    boundary = net.predict_grid(grid_x, grid_y)  # uses weights AFTER this step's update
    snapshots.append({
        "epoch": e,
        "cost": error,
        "weights": {f"W{l}": net.params[f"W{l}"].copy() for l in range(1, net.L + 1)},
        "activations": {f"A{l}": cache[f"A{l}"].copy() for l in range(1, net.L + 1)},
        "dW": dWs,
        "dZ": dZs,
        "boundary": boundary,
    })
  print(f"  final cost: {snapshots[-1]['cost']:.4f}")

  return {
      "n": n, "L": net.L,
      "X_raw": X, "X_std": A0.T, "y": y,
      "grid_x": grid_x, "grid_y": grid_y,
      "snapshots": snapshots,
  }


def visualize(dataset, architecture, epochs, alpha, seed=None,
              hidden_activation=Sigmoid, output_activation=Sigmoid, cost_fn=BinaryCrossEntropy):
  """
  Trains a NeuralNetworkBinary and opens the interactive visualizer window.

  dataset: a callable returning (X, y) — e.g. make_blobs_dataset,
           make_xor_dataset, or your own function in the same shape.
  architecture, epochs, alpha, seed: same meaning as NeuralNetworkBinary /
           train_and_capture above.
  hidden_activation, output_activation, cost_fn: same swappable components
           as NeuralNetworkBinary itself (Sigmoid/ReLU/Tanh/Identity,
           BinaryCrossEntropy/MeanSquaredError).
  """
  X, y = dataset()
  data = train_and_capture(architecture, X, y, epochs, alpha, seed=seed,
                            hidden_activation=hidden_activation,
                            output_activation=output_activation, cost_fn=cost_fn)
  _launch_visualizer(data)


class TrainingRecorder:
  """
  Capture per-epoch snapshots from your OWN manual training loop (like
  main.py's), instead of letting visualize() own training end to end.
  Call .capture(...) once per epoch, right after you update net.params —
  everything else about your loop stays exactly as it is:

      recorder = TrainingRecorder(net, X, Y, m)

      for e in range(epochs):
          y_hat, cache = net.feed_forward(A0)
          error = net.cost(y_hat, Y)

          grads = {}
          propagator = None
          for l in range(net.L, 0, -1):
              dW, db, propagator = net.backprop_layer(l, cache, m, Y, propagator)
              grads[f"W{l}"] = dW
              grads[f"b{l}"] = db

          for l in range(1, net.L + 1):
              net.params[f"W{l}"] -= alpha * grads[f"W{l}"]
              net.params[f"b{l}"] -= alpha * grads[f"b{l}"]

          recorder.capture(cache, grads, error)   # <- the one line to add

      recorder.show()   # opens the same interactive window as visualize()

  capture_every: only record every Nth epoch instead of all of them — the
  expensive part of a capture is predict_grid (one extra forward pass over
  a grid_res x grid_res grid), so this is the direct lever if capturing
  every single epoch is too slow for your architecture/epoch count.
  """

  def __init__(self, net, X, Y, m, grid_res=40, grid_pad=1.0, capture_every=1):
    self.net, self.X, self.Y, self.m = net, X, Y, m
    self.capture_every = capture_every
    self.grid_x, self.grid_y = make_grid_axes(X, pad=grid_pad, res=grid_res)
    self.snapshots = []
    self._epoch = 0

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
        propagator = net.params[f"W{l}"].T @ dZ

      self.snapshots.append({
          "epoch": self._epoch,
          "cost": error,
          "weights": {f"W{l}": net.params[f"W{l}"].copy() for l in range(1, L + 1)},
          "activations": {f"A{l}": cache[f"A{l}"].copy() for l in range(1, L + 1)},
          "dW": {f"dW{l}": grads[f"W{l}"] for l in range(1, L + 1)},
          "dZ": dZs,
          "boundary": net.predict_grid(self.grid_x, self.grid_y),
      })
    self._epoch += 1

  def show(self):
    """Opens the same interactive window visualize() does, using whatever
    history .capture() has recorded so far."""
    if not self.snapshots:
      raise RuntimeError("no snapshots captured — call .capture(...) at least once before .show()")
    data = {
        "n": self.net.n, "L": self.net.L,
        "X_raw": self.X, "X_std": self._X_std_from_net(), "y": self.Y.ravel(),
        "grid_x": self.grid_x, "grid_y": self.grid_y,
        "snapshots": self.snapshots,
    }
    _launch_visualizer(data)

  def _X_std_from_net(self):
    return (self.X - self.net.X_mean) / self.net.X_std


### Colors ###

NODE_FWD_LOW, NODE_FWD_HIGH = (8, 8, 8), (255, 255, 255)

W_POS, W_NEG = (50, 150, 255), (255, 40, 40) # weight sign: blue / red
G_POS, G_NEG = (80, 230, 120), (185, 90, 230) # gradient sign: green / purple

# Decision-boundary heatmap uses its own separate blue scale — unrelated to
# the node-diagram colors above (that panel isn't the "network skeleton").
HEATMAP_LOW, HEATMAP_HIGH = (30, 35, 50), (79, 163, 255)


def clamp01(t):
  return max(0.0, min(1.0, t))


def mix(c1, c2, t):
  t = clamp01(t)
  return tuple((c1[i] + (c2[i] - c1[i]) * t) / 255 for i in range(3))


def forward_node_color(v, layer_min, layer_max):
  # Normalizes relative to this layer's own observed range (across all
  # samples, at the current epoch) rather than assuming a fixed 0-1 range —
  # Sigmoid naturally sits near [0,1], but ReLU/Tanh/Identity don't, so a
  # fixed assumption would saturate everything to solid white or black.
  span = layer_max - layer_min
  t = 0.5 if span <= 1e-12 else (v - layer_min) / span  # flat layer (e.g. a dead ReLU) -> neutral mid-gray
  return mix(NODE_FWD_LOW, NODE_FWD_HIGH, t)


def input_node_color(v):
  t = 1 / (1 + np.exp(-v))  # squash raw standardized feature purely for display
  return mix(NODE_FWD_LOW, NODE_FWD_HIGH, t)


def _exaggerate(t):
  # Pushes mid-low magnitudes up so more edges read as vividly colored
  # rather than washed out — an intentional exaggeration, not a literal
  # linear magnitude scale.
  return clamp01(t) ** 0.6


def weight_edge_style(w, max_abs):
  t = _exaggerate(abs(w) / max_abs if max_abs > 0 else 0)
  color = W_POS if w >= 0 else W_NEG
  return tuple(c / 255 for c in color), 0.25 + 0.70 * t


def grad_node_color(v, max_abs):
  t = _exaggerate(abs(v) / max_abs if max_abs > 0 else 0)
  return mix(NODE_FWD_LOW, NODE_FWD_HIGH, t)  # grayscale, same as forward view


def grad_edge_style(g, max_abs):
  t = _exaggerate(abs(g) / max_abs if max_abs > 0 else 0)
  color = G_POS if g >= 0 else G_NEG
  return tuple(c / 255 for c in color), 0.25 + 0.70 * t


FWD_CMAP = LinearSegmentedColormap.from_list(
    "fwd", [tuple(c / 255 for c in HEATMAP_LOW), tuple(c / 255 for c in HEATMAP_HIGH)]
)


### Diagram geometry ###

def build_node_positions(n):
  L = len(n) - 1
  xs = np.linspace(1, 9, L + 1) if L > 0 else [5.0]
  positions = []
  for l in range(L + 1):
    count = n[l]
    ys = [5.0] if count == 1 else list(np.linspace(1, 9, count))
    positions.append([(xs[l], y) for y in ys])
  return positions


def build_diagram(ax, n):
  ax.cla()
  ax.set_xlim(0, 10)
  ax.set_ylim(0, 10)
  ax.set_aspect("equal")  # so Circle patches render as actual circles, not ovals
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_color("#555555")

  L = len(n) - 1
  positions = build_node_positions(n)

  edge_lines = [None]  # edge_lines[l] = edges INTO layer l (l = 1..L)
  for l in range(1, L + 1):
    layer_edges = []
    for i in range(n[l]):
      row = []
      for j in range(n[l - 1]):
        x1, y1 = positions[l - 1][j]
        x2, y2 = positions[l][i]
        line, = ax.plot([x1, x2], [y1, y2], color="#3296ff", alpha=0.3, linewidth=1.2, zorder=1)
        row.append(line)
      layer_edges.append(row)
    edge_lines.append(layer_edges)

  node_patches = []
  for l in range(L + 1):
    row = []
    for (x, y) in positions[l]:
      circle = Circle((x, y), 0.35, facecolor="#141414", edgecolor="#e6e6e6", linewidth=1.3, zorder=2)
      ax.add_patch(circle)
      row.append(circle)
    node_patches.append(row)

  return node_patches, edge_lines


### Visualizer app ###

# Everything below is internal machinery for visualize() and isn't meant
# to be called directly. It's wrapped in one function so nothing here
# executes at import time — only when visualize() calls it with trained data.

def _launch_visualizer(data):
  plt.rcParams.update({
      "figure.facecolor": "black",
      "axes.facecolor": "black",
      "savefig.facecolor": "black",
      "text.color": "#e0e0e0",
      "axes.edgecolor": "#555555",
      "axes.labelcolor": "#e0e0e0",
      "xtick.color": "#888888",
      "ytick.color": "#888888",
      "font.family": "monospace",
  })

  state = {
      "epoch": 0,
      "selected": 0,
      "playing": False,
      "boundary_mode": "heatmap", # "heatmap" or "contour"
      "speed": 1.0, # multiplier on playback speed, 0.1x - 5x
  }
  diagram = {}
  widgets = {}

  fig = plt.figure(figsize=(11.5, 6.8))
  fig.suptitle("NEURAL NETWORK VISUALIZER", color="#e0e0e0", family="monospace", fontsize=13, y=0.98)

  ax_net = fig.add_axes([0.04, 0.16, 0.50, 0.78])
  ax_scatter = fig.add_axes([0.60, 0.50, 0.36, 0.42])
  ax_info = fig.add_axes([0.60, 0.30, 0.36, 0.17])
  ax_legend = fig.add_axes([0.60, 0.05, 0.36, 0.22])
  ax_slider = fig.add_axes([0.08, 0.105, 0.33, 0.025])
  ax_play = fig.add_axes([0.46, 0.10, 0.07, 0.035])
  ax_speed_slider = fig.add_axes([0.08, 0.045, 0.33, 0.025])
  ax_boundary_toggle = fig.add_axes([0.60, 0.935, 0.22, 0.04])

  for ax in (ax_info, ax_legend):
    ax.axis("off")

  ### Rendering ###

  def update_scatter_and_boundary():
    snap = data["snapshots"][state["epoch"]]

    ax_scatter.cla()
    xs, ys_ = data["grid_x"], data["grid_y"]
    extent = [xs.min(), xs.max(), ys_.min(), ys_.max()]

    if state["boundary_mode"] == "heatmap":
      ax_scatter.imshow(snap["boundary"], extent=extent, origin="lower",
                         cmap=FWD_CMAP, vmin=0, vmax=1, aspect="auto", zorder=0)
    else:
      # Contour-only view: just the p=0.5 decision boundary line, no shading,
      # so it reads as a clean line rather than a gradient.
      XX, YY = np.meshgrid(xs, ys_)
      ax_scatter.contour(XX, YY, snap["boundary"], levels=[0.5],
                          colors=["#e0e0e0"], linewidths=1.8, zorder=1)

    ax_scatter.set_xlim(extent[0], extent[1])
    ax_scatter.set_ylim(extent[2], extent[3])

    # X_raw, not X_std — extent/xlim/ylim above are in raw coordinate space
    # (matching grid_x/grid_y, which predict_grid also expects in raw
    # units), so the scatter has to plot in that same space to line up.
    X_raw, y = data["X_raw"], data["y"]
    colors = ["#4fa3ff" if lab == 1 else "#ff8a5c" for lab in y]
    # Every point gets a black outline so it stays visible against the
    # heatmap even when the point's own fill color matches the background
    # (e.g. a class-1 point sitting in a strongly class-1-predicted region).
    # The selected point additionally gets a thicker white outline on top.
    sizes = [90 if i == state["selected"] else 32 for i in range(len(y))]
    edgecolors = ["white" if i == state["selected"] else "black" for i in range(len(y))]
    linewidths = [2.2 if i == state["selected"] else 1.0 for i in range(len(y))]
    ax_scatter.scatter(X_raw[:, 0], X_raw[:, 1], c=colors, s=sizes,
                        edgecolors=edgecolors, linewidths=linewidths, zorder=2)

    ax_scatter.set_xticks([])
    ax_scatter.set_yticks([])
    for spine in ax_scatter.spines.values():
      spine.set_color("#555555")

  def update_info_panel():
    snap = data["snapshots"][state["epoch"]]
    sample_idx = state["selected"]
    raw = data["X_raw"][sample_idx]
    true_label = data["y"][sample_idx]
    label_color = "#4fa3ff" if true_label == 1 else "#ff8a5c"

    ax_info.cla()
    ax_info.axis("off")

    # Training-state block (changes as the epoch slider moves)
    training_text = (f"EPOCH    {snap['epoch']}\n"
                      f"COST     {snap['cost']:.4f}")
    ax_info.text(0, 1.0, training_text, va="top", ha="left",
                 family="monospace", fontsize=9, color="#e0e0e0")

    ax_info.text(0, 0.52, "-" * 30, va="top", ha="left",
                 family="monospace", fontsize=9, color="#444444")

    # Selected-sample block (fixed fact about the data, doesn't change with
    # epoch) — visually separated from the training block above, and colored
    # to match the sample's true-label color used everywhere else.
    ax_info.text(0, 0.42, "SELECTED SAMPLE", va="top", ha="left",
                 family="monospace", fontsize=8, color="#888888")
    sample_text = (f"POINT    ({raw[0]:.2f}, {raw[1]:.2f})\n"
                   f"TRUE     {true_label}")
    ax_info.text(0, 0.28, sample_text, va="top", ha="left",
                 family="monospace", fontsize=9, color=label_color)

  def draw_legend():
    ax_legend.cla()
    ax_legend.axis("off")
    lines = [
        ("FORWARD VIEW", "#e0e0e0"),
        ("  node color = activation (black->white)", "#cccccc"),
        ("  edge blue = +weight", "#3296ff"),
        ("  edge red = -weight", "#ff2828"),
        ("BACKWARD VIEW (during Play)", "#e0e0e0"),
        ("  node/edge green = +gradient", "#50e678"),
        ("  node/edge purple = -gradient", "#b95ae6"),
        ("", "#e0e0e0"),
        ("click a dataset point to inspect it", "#888888"),
    ]
    y = 1.0
    for label, color in lines:
      ax_legend.text(0, y, label, family="monospace", fontsize=8, color=color, va="top")
      y -= 0.115

  def render():
    n, L = data["n"], data["L"]
    snap = data["snapshots"][state["epoch"]]
    sample_idx = state["selected"]

    x_std_sample = data["X_std"][sample_idx]
    for i, v in enumerate(x_std_sample):
      diagram["node_patches"][0][i].set_facecolor(input_node_color(v))

    for l in range(1, L + 1):
      A_l = snap["activations"][f"A{l}"]
      layer_min, layer_max = A_l.min(), A_l.max()
      for i in range(n[l]):
        diagram["node_patches"][l][i].set_facecolor(forward_node_color(A_l[i, sample_idx], layer_min, layer_max))

      W_l = snap["weights"][f"W{l}"]
      max_abs = np.max(np.abs(W_l))
      for i in range(n[l]):
        for j in range(n[l - 1]):
          color, alpha = weight_edge_style(W_l[i, j], max_abs)
          diagram["edge_lines"][l][i][j].set_color(color)
          diagram["edge_lines"][l][i][j].set_alpha(alpha)

    update_scatter_and_boundary()
    update_info_panel()
    fig.canvas.draw()
    fig.canvas.flush_events()

  def render_transition_frame(next_snap, phase):
    # Only touches the network diagram (ax_net) — used for the brief forward/
    # backward flashes during Play. The scatter/boundary/info panel only ever
    # update on settle (render()), so they don't flicker mid-transition.
    n, L = data["n"], data["L"]
    sample_idx = state["selected"]

    if phase == "forward":
      # Edges are left as-is (still showing the PREVIOUS epoch's weights,
      # which are exactly the weights these new activations were computed
      # from). Only the nodes update, to the new activations.
      x_std_sample = data["X_std"][sample_idx]
      for i, v in enumerate(x_std_sample):
        diagram["node_patches"][0][i].set_facecolor(input_node_color(v))
      for l in range(1, L + 1):
        A_l = next_snap["activations"][f"A{l}"]
        layer_min, layer_max = A_l.min(), A_l.max()
        for i in range(n[l]):
          diagram["node_patches"][l][i].set_facecolor(forward_node_color(A_l[i, sample_idx], layer_min, layer_max))

    elif phase == "backward":
      for l in range(1, L + 1):
        dZ_l = next_snap["dZ"][f"dZ{l}"]
        max_abs_z = np.max(np.abs(dZ_l))
        for i in range(n[l]):
          diagram["node_patches"][l][i].set_facecolor(grad_node_color(dZ_l[i, sample_idx], max_abs_z))

        dW_l = next_snap["dW"][f"dW{l}"]
        max_abs_w = np.max(np.abs(dW_l))
        for i in range(n[l]):
          for j in range(n[l - 1]):
            color, alpha = grad_edge_style(dW_l[i, j], max_abs_w)
            diagram["edge_lines"][l][i][j].set_color(color)
            diagram["edge_lines"][l][i][j].set_alpha(alpha)

    fig.canvas.draw()
    fig.canvas.flush_events()

  ### Widget callbacks ###

  def create_slider(max_epoch, init_epoch):
    ax_slider.cla()
    s = Slider(ax_slider, "EPOCH", 0, max_epoch, valinit=init_epoch, valstep=1, color="#4fa3ff")
    s.label.set_color("#e0e0e0")
    s.valtext.set_color("#e0e0e0")
    s.on_changed(on_slider_change)
    widgets["slider"] = s

  def on_slider_change(val):
    state["playing"] = False
    widgets["play_button"].label.set_text("Play")
    state["epoch"] = int(round(val))
    render()

  def on_speed_change(val):
    state["speed"] = val

  def on_boundary_toggle_click(event):
    state["boundary_mode"] = "contour" if state["boundary_mode"] == "heatmap" else "heatmap"
    widgets["boundary_toggle"].label.set_text(f"View: {state['boundary_mode'].capitalize()}")
    update_scatter_and_boundary()
    fig.canvas.draw()
    fig.canvas.flush_events()

  def on_click(event):
    if event.inaxes != ax_scatter:
      return
    pts = data["X_raw"]  # ax_scatter is displayed in raw coordinate space — see update_scatter_and_boundary
    dists = np.hypot(pts[:, 0] - event.xdata, pts[:, 1] - event.ydata)
    state["selected"] = int(np.argmin(dists))
    render()

  def step_forward_animated():
    n_snaps = len(data["snapshots"])
    next_idx = (state["epoch"] + 1) % n_snaps
    next_snap = data["snapshots"][next_idx]

    render_transition_frame(next_snap, "forward")
    plt.pause(0.12 / state["speed"])
    if not state["playing"]:
      state["epoch"] = next_idx
      render()
      return

    render_transition_frame(next_snap, "backward")
    plt.pause(0.12 / state["speed"])

    state["epoch"] = next_idx
    render() # "update" phase: settle into the new weights + activations
    plt.pause(0.05 / state["speed"])

    s = widgets["slider"]
    s.eventson = False
    s.set_val(state["epoch"])
    s.eventson = True

  def run_play_loop():
    while state["playing"]:
      step_forward_animated()

  def on_play_click(event):
    state["playing"] = not state["playing"]
    widgets["play_button"].label.set_text("Pause" if state["playing"] else "Play")
    if state["playing"]:
      run_play_loop()

  ### Build + show ###

  diagram["node_patches"], diagram["edge_lines"] = build_diagram(ax_net, data["n"])
  draw_legend()

  create_slider(len(data["snapshots"]) - 1, state["epoch"])

  speed_slider = Slider(ax_speed_slider, "SPEED", 0.1, 5.0, valinit=state["speed"],
                         valstep=0.1, color="#ffd25a", valfmt="%.1fx")
  speed_slider.label.set_color("#e0e0e0")
  speed_slider.valtext.set_color("#e0e0e0")
  speed_slider.on_changed(on_speed_change)
  widgets["speed_slider"] = speed_slider

  play_button = Button(ax_play, "Play", color="#232a3f", hovercolor="#35406a")
  play_button.label.set_color("#e0e0e0")
  widgets["play_button"] = play_button
  play_button.on_clicked(on_play_click)

  boundary_toggle_button = Button(ax_boundary_toggle, "View: Heatmap", color="#232a3f", hovercolor="#35406a")
  boundary_toggle_button.label.set_color("#e0e0e0")
  boundary_toggle_button.label.set_fontsize(9)
  widgets["boundary_toggle"] = boundary_toggle_button
  boundary_toggle_button.on_clicked(on_boundary_toggle_click)

  fig.canvas.mpl_connect("button_press_event", on_click)

  render()
  plt.show()


if __name__ == "__main__":
  # Demo — only runs when this file is executed directly
  # (`python3 nn_visualizer.py`), not when imported as a library.
  visualize(
      dataset=make_blobs_dataset,
      architecture=[2, 100, 4, 4, 4, 1],
      epochs=150,
      alpha=0.6,
      seed=10,
  )