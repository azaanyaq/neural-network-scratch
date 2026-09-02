import numpy as np

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


def make_circles_dataset(n_per_class=25, seed=3):
  rng = np.random.RandomState(seed)
  theta_outer = rng.uniform(0, 2 * np.pi, n_per_class)
  r_outer = 1.6 + rng.randn(n_per_class) * 0.12
  outer = np.column_stack([r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)])
  theta_inner = rng.uniform(0, 2 * np.pi, n_per_class)
  r_inner = 0.7 + rng.randn(n_per_class) * 0.12
  inner = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
  X = np.vstack([outer, inner])
  y = np.array([0] * n_per_class + [1] * n_per_class)  # outer ring -> 0, inner ring -> 1
  idx = rng.permutation(len(X))
  return X[idx], y[idx]


def make_moons_dataset(n_per_class=25, seed=4):
  rng = np.random.RandomState(seed)
  theta0 = rng.uniform(0, np.pi, n_per_class)
  moon0 = np.column_stack([np.cos(theta0), np.sin(theta0)])
  theta1 = rng.uniform(0, np.pi, n_per_class)
  moon1 = np.column_stack([1 - np.cos(theta1), 1 - np.sin(theta1) - 0.5])
  X = np.vstack([moon0, moon1]) * 1.8  # scale to roughly match blobs/xor/circles spatial extent
  X += rng.randn(*X.shape) * 0.12
  y = np.array([0] * n_per_class + [1] * n_per_class)
  idx = rng.permutation(len(X))
  return X[idx], y[idx]


def make_grid_axes(X, pad=1.0, res=40):
  # True min/max moves with a single outlier point, which used to let 1-2
  # far-flung points blow the whole grid out to cover them, squeezing all
  # the boundary detail near the real cluster into ~1 grid cell. A fixed
  # median +/- spread window fixes that but has the opposite problem --
  # it also clips ordinary tail points of a ordinary spread-out (but
  # outlier-free) dataset, which should never be hidden.
  #
  # So: flag a point as an outlier via the "modified z-score" (median +
  # MAD, threshold 3.5 -- the standard robust rule, Iglewicz & Hoaglin),
  # then size the grid from the min/max of the remaining (non-outlier)
  # points, padded exactly like the original min/max-based bounds always
  # were. A dataset with no real outliers gets every point included in
  # full, same as before; only genuine outliers get excluded from the
  # view. Tradeoff: a flagged outlier can render outside the visible
  # grid -- update_scatter_and_boundary() checks for and warns about this.
  def bounds_for(values):
    med = np.median(values)
    robust_std = 1.4826 * np.median(np.abs(values - med))
    if robust_std == 0:  # most values identical -- nothing meaningful to flag
      inliers = values
    else:
      inliers = values[np.abs(values - med) / robust_std <= 3.5]
    spread = inliers.std() or values.std() or 1.0
    return inliers.min() - pad * spread, inliers.max() + pad * spread

  x_min, x_max = bounds_for(X[:, 0])
  y_min, y_max = bounds_for(X[:, 1])
  return np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res)


# label -> generator, shared by demo_vis()'s default pick and its in-GUI dataset buttons
PRESET_DATASETS = [
    ("Blobs", make_blobs_dataset),
    ("XOR", make_xor_dataset),
    ("Circles", make_circles_dataset),
    ("Moons", make_moons_dataset),
]
