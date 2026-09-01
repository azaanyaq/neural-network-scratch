import numpy as np
import matplotlib.pyplot as plt

from nn_lib import NeuralNetworkBinary
from nn_lib import GradientCheck
from nn_visualizer import TrainingRecorder

net = NeuralNetworkBinary(n=[2, 4, 4, 1])

net.summary()

X = np.array([
    [202, 65], [279, 73], [192, 70], [114, 72], [206, 70],
    [171, 66], [288, 72], [120, 70], [202, 58], [221, 64],
    [310, 66], [314, 58], [174, 69], [302, 65], [187, 68],
    [216, 74], [199, 65], [203, 60], [251, 60], [230, 58],
    [249, 62], [152, 67], [101, 64], [187, 66], [257, 64],
    [137, 66], [229, 65], [291, 69], [287, 59], [120, 58],
    [260, 73], [303, 62], [157, 60], [121, 69], [188, 65],
    [148, 60], [318, 58], [158, 60], [269, 62], [319, 72],
    [287, 71], [307, 60], [114, 58], [289, 62], [289, 71],
    [274, 64], [289, 66], [150, 72], [207, 72], [154, 67],
    [163, 70], [230, 64], [150, 74], [234, 61], [120, 62],
    [172, 64], [266, 70], [117, 72], [231, 68], [188, 61],
    [159, 70], [113, 64], [108, 59], [189, 67], [152, 70],
    [229, 63], [183, 69], [191, 69], [210, 68], [287, 64],
    [298, 58], [271, 58], [107, 70], [274, 66], [134, 60],
    [305, 64], [180, 63], [263, 65], [149, 66], [203, 62],
    [231, 58], [101, 67], [233, 69], [153, 72], [205, 66],
    [103, 74], [153, 74], [290, 69], [245, 64], [317, 59],
    [143, 60], [261, 74], [301, 62], [289, 74], [113, 74],
    [194, 74], [147, 59], [114, 59], [299, 62], [305, 58],
])

y = np.array([1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,1,1,0,1,1,
              1,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,1,1,0,0,1,0,1,0,1,0,1,1,0,1,0,1,0,0,0,1,1,
              0,1,1,1,0,1,0,0,1,1])

# --- Train/test split — held-out data so predict() means something ---
X_train, X_test, y_train, y_test = NeuralNetworkBinary.train_test_split(X, y, test_size=0.2, seed=0)

A0, Y, m = net.prepare_data(X_train, y_train)

epochs = 1000
alpha = 0.6
costs = []  # track cost per epoch so the training loop's progress is visible

recorder = TrainingRecorder(net, X_train, Y, m, capture_every=5)  # see the visualizer at the end

for e in range(epochs):
    y_hat, cache = net.feed_forward(A0)
    error = net.cost(y_hat, Y)
    costs.append(error)

    grads = {}
    propagator = None

    for l in range(net.L, 0, -1):
        dW, db, propagator = net.backprop_layer(l, cache, m, Y, propagator)
        grads[f"W{l}"] = dW
        grads[f"b{l}"] = db

    for l in range(1, net.L + 1):
        net.params[f"W{l}"] -= alpha * grads[f"W{l}"]
        net.params[f"b{l}"] -= alpha * grads[f"b{l}"]

    recorder.capture(cache, grads, error)

    if e % 20 == 0:  # same convention as train() in nn_lib.py
        print(f"epoch {e}: cost = {error:.4f}")

net.plot_cost(costs)

predictions = net.predict(X_test).ravel()
accuracy = (predictions == y_test).mean()

print()
print(f"Test set ({len(y_test)} samples):")

for i in range(len(y_test)):
    mark = "correct" if predictions[i] == y_test[i] else "wrong"
    print(f"  weight={X_test[i][0]:<4} height={X_test[i][1]:<3} "
          f"predicted={predictions[i]}  actual={y_test[i]}  [{mark}]")

print(f"Test accuracy: {accuracy:.2%}")

recorder.show()  # opens the interactive visualizer on the training run above