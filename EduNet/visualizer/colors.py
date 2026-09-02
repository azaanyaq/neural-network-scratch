import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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
