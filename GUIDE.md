# EduNet Guide

This is the full reference for EduNet — every public class, every method, every
activation and cost function, and a complete worked example. The
[README](README.md) is the pitch; this is the manual.

## Contents

- [Installation](#installation)
- [Core concepts](#core-concepts)
- [`NeuralNetworkBinary` — full API](#neuralnetworkbinary--full-api)
- [Activation functions](#activation-functions)
- [Cost functions](#cost-functions)
- [Verifying correctness: gradient checking](#verifying-correctness-gradient-checking)
- [Loading real-world data](#loading-real-world-data)
- [Visualizing training](#visualizing-training)
- [Full example: build, customize, train, and test a network](#full-example-build-customize-train-and-test-a-network)

## Installation

```bash
pip install pyedunet
```

`import EduNet` gives you `NeuralNetworkBinary` and every activation/cost
function immediately — that path only depends on NumPy. Anything that needs
matplotlib (the visualizer) or pandas (the CSV loader) is imported lazily the
first time you actually touch it, so you never pay for either unless you use
that part.

## Core concepts

A network is described by one list, `n` — the size of every layer, input
included. `n=[2, 5, 5, 1]` means: 2 input features, two hidden layers of 5
neurons each, and 1 output neuron (this library only supports single-output
binary classification — `n[-1]` must always be `1`).

Training data flows through the library in a consistent shape:

- `X` — your raw feature matrix, `(m, n_features)`: one row per sample.
- `y` — your raw labels, `(m,)`: one label per sample, 0 or 1.
- `A0` — `X` transposed and standardized (z-score normalized), `(n_features, m)`. This is what actually gets fed into the network.
- `Y` — `y` reshaped to `(1, m)` to match the output layer.
- `m` — the sample count, used throughout backprop's averaging.

`net.prepare_data(X, y)` does the `X`/`y` → `A0`/`Y`/`m` conversion for you,
and — importantly — **remembers** the mean/std it standardized with
(`net.X_mean`, `net.X_std`). `predict()` and `predict_grid()` reuse those
stored values rather than recomputing fresh statistics from whatever data
they're given, because standardizing new data with its *own* mean/std would
silently distort every prediction. Always call `prepare_data()` on your
**training** set only.

## `NeuralNetworkBinary` — full API

### `NeuralNetworkBinary(n, hidden_activation=Sigmoid, output_activation=Sigmoid, cost_fn=BinaryCrossEntropy)`

Builds the network and randomly initializes every weight matrix (`W1..WL`)
and bias vector (`b1..bL`). Every component is swappable — see
[Activation functions](#activation-functions) and
[Cost functions](#cost-functions) below for what's available.

Raises `ValueError` immediately if `n` is invalid: fewer than 2 entries,
a non-positive or non-integer layer size, or `n[-1] != 1`.

```python
from EduNet import NeuralNetworkBinary, ReLU, BinaryCrossEntropy

net = NeuralNetworkBinary(
    n=[2, 16, 16, 1],
    hidden_activation=ReLU,
    output_activation=Sigmoid,
    cost_fn=BinaryCrossEntropy,
)
```

### `net.summary()`

Prints a per-layer table — weight/bias shapes, which activation each layer
uses, and the total parameter count. A quick sanity check before training,
and the fastest way to catch a typo'd `n`.

### `NeuralNetworkBinary.train_test_split(X, y, test_size=0.2, seed=None)`

A **staticmethod** — no instance needed. Shuffles `X`/`y` together and splits
off `test_size` fraction as a held-out test set. `seed` makes the split
reproducible; leave it `None` for a different split each run.

```python
X_train, X_test, y_train, y_test = NeuralNetworkBinary.train_test_split(X, y, test_size=0.2, seed=42)
```

### `net.prepare_data(X, y)` → `(A0, Y, m)`

Standardizes `X`, stores the mean/std for later, transposes it into `A0`,
and reshapes `y` into `Y`. Call this once, on your training data, before
anything else.

### `net.predict(X, threshold=0.5)`

Standardizes `X` using the **training set's** stored mean/std, runs it
through the network, and thresholds the result. Pass `threshold=None` to get
the raw `y_hat` probabilities instead (useful for a regression setup with an
`Identity` output). Raises `RuntimeError` if `prepare_data()` hasn't been
called yet, and `ValueError` if `X`'s feature count doesn't match training.

### `net.predict_grid(xs, ys)`

Like `predict()`, but evaluates every point on a 2D coordinate grid instead
of a list of samples — this is what powers the visualizer's decision-boundary
heatmap. Only works for a network trained on exactly 2 features. `xs`/`ys`
are raw-space coordinates (not standardized); returns predictions reshaped
to `(len(ys), len(xs))`.

### `net.feed_forward(A0)` → `(y_hat, cache)`

Runs one forward pass. `cache` holds every layer's activation (`A0..AL`) —
you need it to run `backprop_layer` afterward.

### `net.backprop_layer(l, cache, m, Y, propagator_dC_dA)` → `(dC_dW, dC_db, dC_dA_prev)`

Computes one layer's gradients via the chain rule. Call it in a loop from
the output layer (`l = net.L`) back to the first hidden layer (`l = 1`),
passing each call's `dC_dA_prev` in as the next call's `propagator_dC_dA`
(pass `None` for the very first call, at the output layer). This is exactly
what `train()` does internally — see the [full example](#full-example-build-customize-train-and-test-a-network)
below for the loop written out by hand.

### `net.train(A0, Y, m, epochs=1000, alpha=0.01)` → `costs`

The full training loop: forward pass, cost, backprop every layer, update
every parameter, repeat. Prints the cost every 20 epochs. Returns the list
of per-epoch costs.

### `net.plot_cost(costs)`

A one-line matplotlib plot of cost vs. iterations, given the list `train()`
returns.

### `net.gradient_check(A0, Y, m, epsilon=1e-7, tolerance=1e-7, num_checks=None, verbose=True)`

Verifies that `backprop_layer`'s analytical gradients are actually correct,
by comparing them against an independent numerical estimate. See
[Verifying correctness](#verifying-correctness-gradient-checking) below.

## Activation functions

Every activation is a class with a `forward(Z)` and `backward(A)` (the
derivative, expressed in terms of the already-computed activation `A`
rather than `Z`). Pass the class itself — not an instance — as
`hidden_activation`/`output_activation`. Call `.explain()` on any of them
at any time (`ReLU.explain()`) to print the same summary shown below.

### `Sigmoid`

`A = 1 / (1 + e^-Z)`

A smooth function whose output is bounded between 0 and 1, so it reads
naturally as a probability — the standard choice for a binary classification
output layer, even in networks that use a different activation for hidden
layers.

Prone to vanishing gradients: its derivative maxes out at 0.25 and shrinks
quickly away from `Z=0`, so a deep stack of sigmoid hidden layers barely
learns in its earliest layers. Modern networks mostly use ReLU (or similar)
for hidden layers instead.

### `ReLU`

`A = max(0, Z)`

Cheap to compute and doesn't suffer vanishing gradients for positive inputs
— the default choice for hidden layers in most modern networks.

A neuron whose weights push it permanently negative outputs zero forever and
stops learning ("dying ReLU"). It's also unbounded, so it's a poor choice
for an output layer that needs to represent a probability.

### `Tanh`

`A = tanh(Z)`

Zero-centered output, unlike Sigmoid (which is always positive) — this tends
to help gradient descent converge faster in hidden layers.

Still prone to vanishing gradients, just less severely than Sigmoid. In deep
networks, ReLU is usually preferred for hidden layers instead.

### `Identity`

`A = Z` (linear / no-op)

Regression tasks predict an unbounded real number (a price, a temperature) —
the output layer shouldn't be squashed into any particular range, so no
activation at all is the right choice. Pair with `MeanSquaredError`.

Never useful in a hidden layer — stacking linear layers with no nonlinearity
between them collapses into a single linear layer, so the extra depth adds
no representational power.

## Cost functions

Every cost is a class with `forward(y_hat, y)` (returns the scalar cost) and
`backward(y_hat, y, m)` (returns `dC/dy_hat`). Pass the class itself as
`cost_fn`.

### `BinaryCrossEntropy`

`C = -mean( y*log(y_hat) + (1-y)*log(1-y_hat) )`

The standard loss for binary classification. Paired with a Sigmoid output,
its gradient stays strong even when the network is confidently wrong —
exactly when the strongest learning signal is needed.

Breaks numerically if `y_hat` ever reaches exactly 0 or 1 (`log(0)` is
undefined) — needs an output activation that stays strictly between 0 and 1,
like Sigmoid. Also the wrong tool for regression or multi-class problems.

### `MeanSquaredError`

`C = mean( (y_hat - y)^2 )`

The natural fit for regression — the average squared distance from the true
value. The same idea ordinary least-squares fitting is built on. Pair with
an `Identity` output.

Pairing this with a Sigmoid output for classification seems natural at
first, but the sigmoid's saturation makes the gradient go nearly flat
exactly when predictions are very wrong — learning stalls right when it
matters most. This is the actual reason `BinaryCrossEntropy` exists as a
separate loss rather than everyone just using MSE for classification too.

## Verifying correctness: gradient checking

`gradient_check()` answers one question: is your hand-coded backprop math
actually right, or does it just *look* like it's working? A subtle bug (a
wrong transpose, a sign flip) can still train something and drop the cost —
gradient descent is forgiving enough that a somewhat-wrong gradient often
still points roughly downhill. Watching the cost curve alone can't catch
that.

Instead, `gradient_check()` verifies backprop against something that
doesn't depend on backprop at all: for each parameter, it nudges the value
by `epsilon`, recomputes the cost twice (once nudged up, once down), and
estimates the gradient as the slope between those two costs — a direct,
brute-force numerical derivative. It then compares that against what
`backprop_layer` claims the gradient is.

```python
A0, Y, m = net.prepare_data(X_train, y_train)
passed = net.gradient_check(A0, Y, m, num_checks=50)
```

- `num_checks` — check every parameter by default, which is `O(2 × params)`
  forward passes. Pass a number to randomly sample that many instead — much
  faster on a larger network, and still a statistically meaningful check.
- Run this **after adding or changing any activation, cost function, or
  backprop math** — a wrong derivative is exactly the kind of bug that's
  hard to notice by just watching training happen.
- Never call it during actual training — it's a one-off correctness check,
  not something to run every epoch.

## Loading real-world data

`EduNet.load_dataset(...)` turns a messy real CSV (local path or URL, mixed
numeric/categorical columns, missing values) into a clean `(X, y)` pair
ready for `prepare_data()`.

```python
from EduNet import load_dataset

X, y = load_dataset(
    "https://example.com/titanic.csv",
    target_column="Survived",
    drop_columns=["PassengerId", "Name", "Ticket"],
    missing_strategy="mean",
)
```

- `drop_columns` — exclude columns entirely before anything else. Matters in
  practice: a text identifier column (unique per row) gets auto-detected as
  "categorical" the same as a real category and one-hot encoded into one
  column *per row* unless you exclude it here.
- `missing_strategy` — `"drop"` (default) drops any row with a missing
  value; `"mean"` fills missing numeric values with that column's mean
  (rows missing a non-numeric value are still dropped — a mean doesn't make
  sense for a category). A missing *target* value is always dropped first,
  regardless of this setting.
- `positive_label` — which target value becomes `1`. If the target isn't
  already 0/1 and this is omitted, the mapping is inferred alphabetically
  and **printed** — never silently guessed.
- `categorical_columns` — which columns to one-hot encode; auto-detected by
  dtype if omitted.

If you'd rather call the pipeline steps yourself instead of the one-call
version, `load_csv`, `handle_missing`, and `encode_categorical` are all
available individually and importable the same way.

## Visualizing training

### `demo_vis(architecture, epochs, alpha, seed=None, hidden_activation=Sigmoid, output_activation=Sigmoid, cost_fn=BinaryCrossEntropy)`

The fastest way to see a network train. Trains on a built-in preset dataset
and opens an interactive window — no dataset argument needed:

```python
from EduNet import demo_vis

demo_vis(architecture=[2, 5, 5, 1], epochs=200, alpha=0.6)
```

Inside the window: scrub the epoch slider, hit Play to watch the forward and
backward pass animate, toggle between a heatmap and contour view of the
decision boundary, click any data point to inspect it, and click between
four built-in datasets (Blobs, XOR, Circles, Moons) — each click retrains
live, in place.

### `TrainingRecorder` — visualizing your own training loop

If you're writing your own manual training loop (like the
[full example](#full-example-build-customize-train-and-test-a-network) below)
instead of using `demo_vis()`, `TrainingRecorder` lets you visualize it too:

```python
from EduNet.visualizer import TrainingRecorder

recorder = TrainingRecorder(net, X_train, Y, m, capture_every=1)

for e in range(epochs):
    # ... your training loop ...
    recorder.capture(cache, grads, error)   # Call once per epoch, after updating net.params

recorder.show()   # Opens the same interactive window as demo_vis()
```

`capture_every` skips epochs to control cost — the expensive part of a
capture is `predict_grid` (one extra forward pass over a grid), so this is
the lever to pull if capturing every single epoch is too slow. Note that
this window has no dataset-switcher buttons, since you've already supplied
your own data — there's nothing built in to switch to.

## Full example: build, customize, train, and test a network

This puts everything above together: a custom architecture, swapped-in
activation and cost functions, a train/test split, a correctness check
before trusting the results, training, and evaluating real test accuracy.

```python
import numpy as np
from EduNet import (
    NeuralNetworkBinary,
    ReLU, Sigmoid,
    BinaryCrossEntropy,
)
from EduNet.visualizer import make_moons_dataset

# 1. Get some data; swap this for your own (X, y), or EduNet.load_dataset(...)
X, y = make_moons_dataset(n_per_class=60)

# 2. Create a train and test split so it isn't tested on known data
X_train, X_test, y_train, y_test = NeuralNetworkBinary.train_test_split(
    X, y, test_size=0.2, seed=42
)

# 3. Build a network with a custom architecture and swapped-in components.

# ReLU hidden layers + a Sigmoid output is the standard modern pairing for
# binary classification; BinaryCrossEntropy is the matching cost function.
net = NeuralNetworkBinary(
    n=[2, 16, 16, 1],
    hidden_activation=ReLU,
    output_activation=Sigmoid,
    cost_fn=BinaryCrossEntropy,
)
net.summary()

# 4. Standardize the training data (this also stores the mean/std that
# predict() will reuse later, so test data is standardized consistently)
A0, Y, m = net.prepare_data(X_train, y_train)

# 5. Verify the backprop math is actually correct before trusting anything
# that follows - see "Verifying correctness" above for why this matters
passed = net.gradient_check(A0, Y, m, num_checks=50)
assert passed, "gradient check failed — don't trust the training below"

# 6. Train, and plot the cost curve
costs = net.train(A0, Y, m, epochs=500, alpha=0.1)
net.plot_cost(costs)

# 7. Evaluate on the held-out test set
test_preds = net.predict(X_test)
test_accuracy = (test_preds == y_test.reshape(1, -1)).mean()
print(f"Test accuracy: {test_accuracy:.2%}")

# 8. Inquire as to why ReLU and BinaryCrossEntropy were used
ReLU.explain()
BinaryCrossEntropy.explain()
```

Things worth trying from here: swap `hidden_activation=ReLU` for `Tanh` or
`Sigmoid` and see how the cost curve changes; swap the dataset for
`make_circles_dataset()` or `make_xor_dataset()` and see which architectures
struggle with which shapes; or wire this exact loop into a
`TrainingRecorder` (see [Visualizing training](#visualizing-training) above)
to watch it happen instead of just reading the final accuracy number.
