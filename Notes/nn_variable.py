import numpy as np
import matplotlib.pyplot as plt

# Neurel Network: sigmoid output + binary cross entropy

def init_params(n):
  L = len(n) - 1 # Number of layers (excluding input layer)
  params = {"n": n, "L": L} # Creating dictionary with 2 entries

  for l in range(1, L + 1): # l in ranges 1 to (and including) L, adding entries to dictionary
    params[f"W{l}"] = np.random.randn(n[l], n[l - 1]) # Weight matrix follows n^[l] x n^[l-1]
    params[f"b{l}"] = np.random.randn(n[l], 1) # Bias matrix follows n^[l] x 1

  """
  Example for n = [3, 5, 4, 2] and L = 3:
  {
    "n": [3, 5, 4, 2], 
    "L": 3,
    "W1": <5x3 array>, "b1": <5x1 array>, 
    "W2": <4x5 array>, "b2": <4x1 array>,
    "W3": <2x4 array>, "b3": <2x1 array>,
  }
  """
  return params

def sigmoid(arr): # Activation function g(z) (Sigmoid)
    return 1 / (1 + np.exp(-1 * arr))

def cost(y_hat, y): # Both y_hat and y should be a n^L x m matrix

    # 1. Losses is a n^L x m matrix
    losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )

    m = y_hat.size # Calculates total number of predictions that make up y_hat

    # 2. Summing across axis = 1 means we sum across rows, making this a n^L x 1 matrix
    summed_losses = (1 / m) * np.sum(losses, axis = 1)

    return np.sum(summed_losses)

def prepare_data(X, y, n):
  # X: Matrix of raw samples
  # y: Array of training labels

  X = (X - X.mean(axis=0)) / X.std(axis=0) # Standardising (Z-score normalisation) each feature (column) to mean 0, std 1

  m = X.shape[0] # Number of training samples
  A0 = X.T # Transposes the matrix using the .T function, obtaining A^[0] in shape n^[0] x m
  Y = y.reshape(n[-1], m) # Reshaping training lables to fit output layer (-1 index works for anything)

  return A0, Y, m

def feed_forward(A0, params):
  L = params["L"] # Extracts layer count list
  n = params["n"] # Extracts layer size list

  cache = {"A0": A0} # Creates a cache dictionary with A0 first entry
  A = A0 # Initialise value A (firstly as A0)

  for l in range(1, L + 1): # l in range 1 to (and including) L 
    W = params[f"W{l}"] # Grabs weights for layer l
    b = params[f"b{l}"] # Grabs biases for layer l
    Z = W @ A + b # Matrix multiplication and addition to find pre-activation value
    A = sigmoid(Z) # Find post-activation value
    cache[f"A{l}"] = A # Add to the cache dictionary

  y_hat = A 

  return y_hat, cache

def backprop_layer(l, params, cache, m, Y, propagator_dC_dA): # l is what layer gradients are being computed
  L = params["L"] # Extract layer count list
  n = params["n"] # Extract layer size list

  A_l = cache[f"A{l}"] # Extracting A value of this layer
  A_prev = cache[f"A{l - 1}"] # Extracting A value of previous layer
  W_l = params[f"W{l}"] # Extracting weights of this layer

  if l == L: # Output layer
    dC_dZ = (1 / m) * (A_l - Y) # Calculates dC/dZ directly from Y
  else: # Every other layer
    dA_dZ = A_l * (1 - A_l) 
    dC_dZ = propagator_dC_dA * dA_dZ # Calculates dC/dZ from propogator handed down
  assert dC_dZ.shape == (n[l], m) 

  dC_dW = dC_dZ @ A_prev.T
  assert dC_dW.shape == (n[l], n[l - 1])

  dC_db = np.sum(dC_dZ, axis=1, keepdims=True)
  assert dC_db.shape == (n[l], 1)

  dC_dA_prev = W_l.T @ dC_dZ  # Propagator for the layer below
  assert dC_dA_prev.shape == (n[l - 1], m)

  return dC_dW, dC_db, dC_dA_prev

def train(params, A0, Y, m, epochs=1000, alpha=0.01):
  L = params["L"] # Extract L value
  costs = [] # Create an empty list for costs (to be appended later)

  for e in range(epochs): # Going through loop for each epoch

    y_hat, cache = feed_forward(A0, params) # Feed forward (outputing prediction y_hat and intermediate layers A)

    error = cost(y_hat, Y) # Calculating cost for each epoch (compares y_hat with Y)
    costs.append(error) # Appending these individual costs to the empty list

    grads = {} # Creating empty dictionary for gradients
    propagator = None # Starts as none as no layer after L (therefore no inherited propogator)

    for l in range(L, 0, -1): # Looping backwards from L -> 1
      dC_dW, dC_db, propagator = backprop_layer( # Calculating gradients (PDs) of that layer
          l, params, cache, m, Y=Y, propagator_dC_dA=propagator
      )
      grads[f"W{l}"] = dC_dW # Weight gradients added to dictionary
      grads[f"b{l}"] = dC_db # Bias gradients added to dictionary

    for l in range(1, L + 1): # Looping from 1 -> L
      params[f"W{l}"] = params[f"W{l}"] - (alpha * grads[f"W{l}"]) # Weights updated using weight gradients of resp layers
      params[f"b{l}"] = params[f"b{l}"] - (alpha * grads[f"b{l}"]) # Biases updated using bias gradients of resp layers

    if e % 20 == 0: # Every 20 epochs, print the current cost
      print(f"epoch {e}: cost = {error:4f}")

  return costs

### Configuring + Running the network ###

# Swap X and y for any dataset

# X must be a 2D array with: 1 row per sample and 1 column per feature
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

# y must be a 1D array with exactly 1 label per sample
y = np.array([1,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,1,1,0,0,1,1,0,0,1,0,1,0,1,1,1,1,0,1,1,
              1,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,2,1,0,0,1,0,1,0,1,0,1,1,0,1,0,1,0,4,0,1,1,
              0,1,1,1,0,1,0,0,1,1])

n = [2, 50, 50, 50, 50, 1] # n[0] must match X's number of columns (features)

params = init_params(n)
A0, Y, m = prepare_data(X, y, n)

costs = train(params, A0, Y, m, epochs=50001, alpha=0.001)

plt.plot(range(len(costs)), costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()