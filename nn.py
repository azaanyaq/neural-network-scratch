import numpy as np
import matplotlib.pyplot as plt

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

def cost(y_hat, y): # both y_hat and y should be a n^L x m matrix

    # 1. Losses is a n^L x m matrix
    losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )

    m = y_hat.size # Calculates total number of predictions that make up y_hat

    # 2. Summing across axis = 1 means we sum across rows, making this a n^L x 1 matrix
    summed_losses = (1 / m) * np.sum(losses, axis = 1)

    return np.sum(summed_losses)

def prepare_data(n):
  X = np.array([ # Weight (lbs) and height (inches) training values
      [150, 70], 
      [254, 73],
      [312, 68],
      [120, 60],
      [154, 61],
      [212, 65],
      [216, 67],
      [145, 67],
      [184, 64],
      [130, 69]
    ])

  y = np.array([0,1,1,0,0,1,1,0,1,0]) # Training labels: 1 for at risk, 0 for not at risk

  m = 10 # Number of training samples
  A0 = X.T # Transposes the matrix via .T function, obtaining A^[0] in shape n^[0] x m
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
