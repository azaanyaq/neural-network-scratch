import numpy as np
import matplotlib.pyplot as plt

def init_params(n):
  
  L = len(n) - 1 # Number of layers (excluding input layer)
  params = {"n": n, "L": L} # Creating dictionary with 2 entries

  for l in range(1, L + 1): # l in ranges 1 to (and including) L
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