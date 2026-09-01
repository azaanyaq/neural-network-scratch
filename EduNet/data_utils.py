import pandas as pd

# ============================================================
# Turns a real, messy CSV (local path or URL — mixed numeric/categorical
# columns, missing values) into a clean (X, y) numpy pair ready for
# NeuralNetworkBinary.prepare_data. Each step is its own function, plus
# load_dataset() chains all of them for the one-call version — same
# "call each piece yourself, or use the shortcut" philosophy as
# NeuralNetworkBinary itself.
#
# Nothing runs on import — only when you actually call these functions.
# ============================================================


def load_csv(path_or_url):
  """pd.read_csv handles both a local path and a URL natively."""
  return pd.read_csv(path_or_url)


def handle_missing(df, strategy="drop"):
  """
  strategy="drop" (default): drop any row with a missing value anywhere.
  strategy="mean": fill missing values in numeric columns with that
  column's mean; rows still missing a non-numeric value are dropped
  (a mean doesn't make sense for a category).
  """
  if strategy == "drop":
    return df.dropna()
  if strategy == "mean":
    numeric_cols = df.select_dtypes(include="number").columns
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df.dropna()
  raise ValueError(f"unknown strategy: {strategy!r} (expected 'drop' or 'mean')")


def encode_categorical(df, columns=None):
  """One-hot encodes categorical columns (auto-detected via dtype if
  columns=None). Every category becomes its own 0/1 column — no
  drop_first, since multicollinearity isn't the concern for a neural
  net that it would be for linear regression."""
  if columns is None:
    columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
  return pd.get_dummies(df, columns=columns)


def load_dataset(path_or_url, target_column, positive_label=None,
                  missing_strategy="drop", categorical_columns=None, drop_columns=None):
  """
  The one-call version: load_csv -> drop_columns -> handle_missing ->
  split off the target -> encode_categorical on the remaining features ->
  map the target to 0/1. Returns (X, y) ready for
  NeuralNetworkBinary.prepare_data.

  drop_columns: column names to exclude entirely before anything else —
  e.g. an ID column. Without this, a text identifier column (unique per
  row) gets auto-detected as "categorical" the same as a genuine category
  and one-hot encoded into one column per row, which is never what you
  want — pass it here instead of feeding it to the network.

  positive_label: which target value becomes 1. If the target isn't
  already 0/1 and this is omitted, the mapping is inferred (alphabetical)
  and printed — never silently guessed without telling you.
  """
  df = load_csv(path_or_url)
  if drop_columns:
    df = df.drop(columns=list(drop_columns))
  df = handle_missing(df, strategy=missing_strategy)

  y_raw = df[target_column]
  X_df = encode_categorical(df.drop(columns=[target_column]), columns=categorical_columns)
  X = X_df.values.astype(float)

  unique_vals = sorted(y_raw.unique(), key=str)
  if len(unique_vals) != 2:
    raise ValueError(f"target column {target_column!r} has {len(unique_vals)} unique "
                      f"values {unique_vals} — NeuralNetworkBinary only supports binary "
                      f"classification (exactly 2)")
  if positive_label is not None:
    if positive_label not in unique_vals:
      raise ValueError(f"positive_label={positive_label!r} not in target values {unique_vals}")
    neg_val = [v for v in unique_vals if v != positive_label][0]
    pos_val = positive_label
  else:
    neg_val, pos_val = unique_vals
    if set(unique_vals) != {0, 1}:  # already-clean binary targets don't need an announced mapping
      print(f"Target mapping (pass positive_label=... to control this): {neg_val!r} -> 0, {pos_val!r} -> 1")

  y = (y_raw == pos_val).astype(int).values
  return X, y
