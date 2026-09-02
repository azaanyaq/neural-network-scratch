import numpy as np
import matplotlib

try:
  matplotlib.use("TkAgg")
except ImportError:
  pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle

from .colors import (
    FWD_CMAP,
    forward_node_color, input_node_color, weight_edge_style,
    grad_node_color, grad_edge_style,
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

def _launch_visualizer(data, presets=None, train_kwargs=None, active_label=None):
  # presets/train_kwargs/active_label: only passed by demo_vis() -- enables the in-GUI
  # dataset-switcher button row + live retrain. TrainingRecorder.show() passes neither
  # (it's showing one already-completed manual run, with no dataset to switch to), so
  # presets stays None and the button row simply doesn't get built.
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
      "active_dataset": active_label, # None when presets is None (TrainingRecorder path)
  }
  diagram = {}
  widgets = {}
  cost_widgets = {}

  fig = plt.figure(figsize=(15, 8.85))
  fig.text(0.015, 0.99, "NEURAL NETWORK VISUALIZER", color="#888888",
           family="monospace", fontsize=9, ha="left", va="top")

  ax_net = fig.add_axes([0.04, 0.16, 0.50, 0.78])
  ax_scatter = fig.add_axes([0.60, 0.50, 0.36, 0.37 if presets else 0.42])
  ax_info = fig.add_axes([0.60, 0.40, 0.36, 0.09])
  ax_legend = fig.add_axes([0.60, 0.21, 0.36, 0.21])
  ax_cost = fig.add_axes([0.60, 0.05, 0.36, 0.14])
  ax_slider = fig.add_axes([0.08, 0.105, 0.33, 0.025])
  ax_play = fig.add_axes([0.46, 0.10, 0.07, 0.035])
  ax_speed_slider = fig.add_axes([0.08, 0.045, 0.33, 0.025])
  ax_boundary_toggle = fig.add_axes([0.60, 0.935, 0.22, 0.04])
  ax_dataset_buttons = None
  if presets:
    dataset_width = (0.96 - 0.60 - 0.03 * (len(presets) - 1)) / len(presets)
    ax_dataset_buttons = []
    for i in range(len(presets)):
      left = 0.60 + i * (dataset_width + 0.03)
      ax_dataset_buttons.append(fig.add_axes([left, 0.878, dataset_width, 0.04]))

  for ax in (ax_info, ax_legend):
    ax.axis("off")

  ### Rendering ###

  def update_scatter_and_boundary():
    snap = data["snapshots"][state["epoch"]]

    ax_scatter.cla()

    if not data["supports_2d_view"]:
      # Heatmap/contour is a 2D grid technique -- nothing meaningful to plot
      # for a network with more than 2 input features. Leave the panel
      # blank with an explanation instead of attempting a misleading
      # 2-of-N-column projection.
      ax_scatter.set_xticks([])
      ax_scatter.set_yticks([])
      for spine in ax_scatter.spines.values():
        spine.set_color("#555555")
      ax_scatter.text(0.5, 0.5,
          f"Heatmap/contour view only supports 2 input features\n"
          f"(this network has {data['n'][0]})",
          transform=ax_scatter.transAxes, ha="center", va="center",
          family="monospace", fontsize=10, color="#888888", wrap=True)
      return

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

    # make_grid_axes() deliberately stays anchored to the bulk of the data
    # (rejecting statistical outliers via median/MAD, not true min/max) so
    # 1-2 far outliers can't blow the whole view out and hide the real
    # decision-boundary detail -- but that means a genuine outlier can
    # render outside the visible grid entirely. Surface that instead of
    # silently dropping the point off-screen.
    outside = ((X_raw[:, 0] < extent[0]) | (X_raw[:, 0] > extent[1]) |
               (X_raw[:, 1] < extent[2]) | (X_raw[:, 1] > extent[3]))
    n_hidden = int(np.sum(outside))
    if n_hidden:
      label = "point" if n_hidden == 1 else "points"
      ax_scatter.text(0.02, 0.97, f"Warning: {n_hidden} outlier {label} not shown",
                       transform=ax_scatter.transAxes, ha="left", va="top",
                       family="monospace", fontsize=8, color="#ffd25a", zorder=3)

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

    # Training-state block (changes as the epoch slider moves), left column
    training_text = (f"EPOCH    {snap['epoch']}\n"
                      f"COST     {snap['cost']:.4f}")
    ax_info.text(0, 1.0, training_text, va="top", ha="left",
                 family="monospace", fontsize=10, color="#e0e0e0")

    # Selected-sample block (fixed fact about the data, doesn't change with
    # epoch) — right column, same row height as the block above (both
    # anchored at y=1.0), colored to match the sample's true-label color
    # used everywhere else for visual distinction instead of a header.
    sample_text = (f"POINT    ({raw[0]:.2f}, {raw[1]:.2f})\n"
                   f"TRUE     {true_label}")
    ax_info.text(0.42, 1.0, sample_text, va="top", ha="left",
                 family="monospace", fontsize=10, color=label_color)

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
    ]
    y = 1.0
    for label, color in lines:
      ax_legend.text(0, y, label, family="monospace", fontsize=11, color=color, va="top")
      y -= 0.15

  def build_cost_curve():
    # The curve itself is static (drawn once, here) — only the "you are
    # here" marker moves, updated every render() from update_cost_marker().
    ax_cost.cla()
    epochs = [s["epoch"] for s in data["snapshots"]]
    costs = [s["cost"] for s in data["snapshots"]]
    ax_cost.plot(epochs, costs, color="#e0e0e0", linewidth=1)

    x_pad = (epochs[-1] - epochs[0]) * 0.05 or 0.5  # single-snapshot run (epochs[0] == epochs[-1]) still gets visible padding
    ax_cost.set_xlim(epochs[0] - x_pad, epochs[-1] + x_pad)
    finite_costs = [c for c in costs if np.isfinite(c)]
    if finite_costs:
      cost_min, cost_max = min(finite_costs), max(finite_costs)
      pad = (cost_max - cost_min) * 0.1 or 0.05  # flat curve (e.g. 1-snapshot run) still gets visible padding
      ax_cost.set_ylim(cost_min - pad, cost_max + pad)
    # else: every cost is NaN/Inf (e.g. diverging training) -- leave ylim on
    # matplotlib's own autoscale rather than crashing on set_ylim(nan, nan)

    ax_cost.tick_params(labelsize=7, colors="#888888")
    for spine in ax_cost.spines.values():
      spine.set_color("#555555")

    marker, = ax_cost.plot([], [], marker="o", markersize=6, color="#4fa3ff", zorder=3)
    cost_widgets["marker"] = marker

  def update_cost_marker():
    snap = data["snapshots"][state["epoch"]]
    cost_widgets["marker"].set_data([snap["epoch"]], [snap["cost"]])

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
    update_cost_marker()
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
    s.label.set_fontsize(10)
    s.valtext.set_color("#e0e0e0")
    s.valtext.set_fontsize(10)
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

  DATASET_ACTIVE_COLOR = "#3f6fae"
  DATASET_INACTIVE_COLOR = "#232a3f"

  def restyle_dataset_buttons():
    for lbl, btn in widgets["dataset_buttons"].items():
      btn.color = DATASET_ACTIVE_COLOR if lbl == state["active_dataset"] else DATASET_INACTIVE_COLOR
      btn.ax.patch.set_facecolor(btn.color)  # Button only repaints its own patch on hover -- set both

  def switch_dataset(label):
    if label == state["active_dataset"]:
      return  # already showing this dataset -- avoid a pointless retrain

    state["playing"] = False
    widgets["play_button"].label.set_text("Play")

    # Retraining is a blocking numpy loop -- paint a status message before the freeze so it isn't silent
    ax_info.cla()
    ax_info.axis("off")
    ax_info.text(0.5, 0.5, f"Training on {label}...", transform=ax_info.transAxes,
                 ha="center", va="center", family="monospace", fontsize=11, color="#ffd25a")
    fig.canvas.draw()
    fig.canvas.flush_events()

    from .training_capture import train_and_capture  # local import -- avoids a circular import with training_capture.py, which imports _launch_visualizer from this module

    dataset_fn = dict(presets)[label]
    X, y = dataset_fn()
    new_data = train_and_capture(train_kwargs["architecture"], X, y,
                                  train_kwargs["epochs"], train_kwargs["alpha"],
                                  seed=train_kwargs["seed"],
                                  hidden_activation=train_kwargs["hidden_activation"],
                                  output_activation=train_kwargs["output_activation"],
                                  cost_fn=train_kwargs["cost_fn"])

    old_n = data["n"]
    data.clear()
    data.update(new_data)

    state["epoch"] = 0
    state["selected"] = 0
    state["active_dataset"] = label

    if data["n"] != old_n:  # defensive only -- never triggers with today's all-2-feature presets
      diagram["node_patches"], diagram["edge_lines"] = build_diagram(ax_net, data["n"])

    build_cost_curve()
    create_slider(len(data["snapshots"]) - 1, state["epoch"])
    restyle_dataset_buttons()

    render()
    fig.canvas.draw()
    fig.canvas.flush_events()

  def on_click(event):
    if event.inaxes != ax_scatter:
      return
    if not data["supports_2d_view"]:
      return  # panel is just an explanatory message -- nothing plotted to select
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
  build_cost_curve()

  create_slider(len(data["snapshots"]) - 1, state["epoch"])

  speed_slider = Slider(ax_speed_slider, "SPEED", 0.1, 5.0, valinit=state["speed"],
                         valstep=0.1, color="#ffd25a", valfmt="%.1fx")
  speed_slider.label.set_color("#e0e0e0")
  speed_slider.label.set_fontsize(10)
  speed_slider.valtext.set_color("#e0e0e0")
  speed_slider.valtext.set_fontsize(10)
  speed_slider.on_changed(on_speed_change)
  widgets["speed_slider"] = speed_slider

  play_button = Button(ax_play, "Play", color="#232a3f", hovercolor="#35406a")
  play_button.label.set_color("#e0e0e0")
  play_button.label.set_fontsize(10)
  widgets["play_button"] = play_button
  play_button.on_clicked(on_play_click)

  boundary_toggle_button = Button(ax_boundary_toggle, "View: Heatmap", color="#232a3f", hovercolor="#35406a")
  boundary_toggle_button.label.set_color("#e0e0e0")
  boundary_toggle_button.label.set_fontsize(10)
  widgets["boundary_toggle"] = boundary_toggle_button
  boundary_toggle_button.on_clicked(on_boundary_toggle_click)

  if presets:
    widgets["dataset_buttons"] = {}
    for (label, _fn), ax in zip(presets, ax_dataset_buttons):
      btn = Button(ax, label, color=DATASET_INACTIVE_COLOR, hovercolor="#35406a")
      btn.label.set_color("#e0e0e0")
      btn.label.set_fontsize(9)
      btn.on_clicked(lambda event, lbl=label: switch_dataset(lbl))  # default-arg capture avoids late binding
      widgets["dataset_buttons"][label] = btn
    restyle_dataset_buttons()

  fig.canvas.mpl_connect("button_press_event", on_click)

  render()

  try:
    fig.canvas.manager.window.resizable(False, False)
  except Exception:
    pass  # non-Tk backend (e.g. TkAgg failed to load earlier) — not critical

  plt.show()  # blocks until the window closes in real interactive use

  # Only reached in headless testing (plt.show monkeypatched to a no-op) -- lets a
  # test script drive switch_dataset()/state/widgets without a real display.
  return {"fig": fig, "state": state, "data": data, "switch_dataset": switch_dataset, "widgets": widgets}
