import numpy as np
import matplotlib.pyplot as plt

L = 3
n = [2, 20, 20, 1] # Number of nodes in each layer

W1 = np.random.randn(n[1], n[0]) # Np function to create a weight matrix of random values
W2 = np.random.randn(n[2], n[1]) # Arguments are rows and columns of matrix 
W3 = np.random.randn(n[3], n[2]) # Can tell is of the form n^[l] x n^[l-1]

b1 = np.random.randn(n[1], 1) # Np function to create a bias matrix of random values
b2 = np.random.randn(n[2], 1) # Can tell is of the form n^[l] x 1
b3 = np.random.randn(n[3], 1)

def prepare_data():
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
  Y = y.reshape(n[L], m) # Reshaping training lables to fit 

  return A0, Y, m

def cost(y_hat, y): # both y_hat and y should be a n^L x m matrix

    # 1. Losses is a n^L x m matrix
    losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )

    m = y_hat.size # Calculates total number of predictions that make up y_hat

    # 2. Summing across axis = 1 means we sum across rows, making this a n^L x 1 matrix
    summed_losses = (1 / m) * np.sum(losses, axis = 1)

    return np.sum(summed_losses)

def sigmoid(arr): # Activation function g(z) (Sigmoid)
    return 1 / (1 + np.exp(-1 * arr))

def feed_forward(A0): # A0 an argument as is the input layer

  # Layer 1 calculations

  Z1 = W1 @ A0 + b1 # The @ means matrix multiplication
  A1 = sigmoid(Z1)

  # Layer 2 calculations (same thing)

  Z2 = W2 @ A1 + b2
  A2 = sigmoid(Z2)

  # Layer 3 calculations (same thing)

  Z3 = W3 @ A2 + b3
  A3 = sigmoid(Z3) # A 1 x 10 column vector

  y_hat = A3 # y_hat is essentially the prediction of the model

  cache = { # Created a dictionary for A0 - A2
    "A0": A0,
    "A1": A1,
    "A2": A2
  }
  
  return A3, cache # Return A0 - A3 as need them for backprop calcualtions

# Layer L (layer 3) calculations

def backprop_layer_3(y_hat, Y, m, A2, W3):
  A3 = y_hat

  # 1. Calculate dC/dZ3 using shorthand: dC/dZ3 = dC/dA3 * dA3/dZ3

  dC_dZ3 = (1/m) * (A3 - Y)
  assert dC_dZ3.shape == (n[3], m) # Checking shape of PD

  # 2. Calculate dC/dW3 = dC/dZ3 * dZ3/dW3, matrix multiply dC/dZ3 with (dZ3/dW3)^T

  dZ3_dW3 = A2
  assert dZ3_dW3.shape == (n[2], m)

  dC_dW3 = dC_dZ3 @ dZ3_dW3.T # @ symbol is for matrix multiplication
  assert dC_dW3.shape == (n[3], n[2])

  # 3. Calculate dC/db3 = np.sum(dC/dZ3, axis=1, keepdims=True)

  dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True) # Compressing to a n^[l]x1 
  assert dC_db3.shape == (n[3], 1)

  # 4. Calculate the propogator dC/dA2 = dC/dZ3 * dZ3/dA2
  dZ3_dA2 = W3 
  dC_dA2 = W3.T @ dC_dZ3
  assert dC_dA2.shape == (n[2], m)

  return dC_dW3, dC_db3, dC_dA2

# Layer 2 calculations

def backprop_layer_2(propagator_dC_dA2, m, A1, A2, W2):

  # 1. Calculate dC/dZ2 = dC/dA2 * dA2/dZ2

  # Use sigmoid derivation to arrive at this answer:
  # Sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
  # And if a = sigmoid(z), then sigmoid'(z) = a * (1 - a)
  
  dA2_dZ2 = A2 * (1 - A2)
  dC_dZ2 = propagator_dC_dA2 * dA2_dZ2
  assert dC_dZ2.shape == (n[2], m)

  # 2. Calculate dC/dW2 = dC/dZ2 * dZ2/dW2 
  
  dZ2_dW2 = A1
  assert dZ2_dW2.shape == (n[1], m)

  dC_dW2 = dC_dZ2 @ dZ2_dW2.T
  assert dC_dW2.shape == (n[2], n[1])

  # 3. Calculate dC/db2 = np.sum(dC/dZ2, axis=1, keepdims=True)
  
  dC_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)
  assert dC_db2.shape == (n[2], 1)

  # 4. Calculate propagator dC/dA1 = dC/dZ2 * dZ2/dA1
  
  dZ2_dA1 = W2
  dC_dA1 = dZ2_dA1.T @ dC_dZ2
  assert dC_dA1.shape == (n[2], m)

  return dC_dW2, dC_db2, dC_dA1

# Layer 1 calculations

def backprop_layer_1(propagator_dC_dA1, m, A1, A0, W1):

  # 1. calculate dC/dZ1 = dC/dA1 * dA1/dZ1

  # Use sigmoid derivation to arrive at this answer:
  # Sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
  # And if a = sigmoid(z), then sigmoid'(z) = a * (1 - a)
  
  dA1_dZ1 = A1 * (1 - A1)
  dC_dZ1 = propagator_dC_dA1 * dA1_dZ1
  assert dC_dZ1.shape == (n[1], m)

  # 2. calculate dC/dW1 = dC/dZ1 * dZ1/dW1 
  
  dZ1_dW1 = A0
  assert dZ1_dW1.shape == (n[0], m)

  dC_dW1 = dC_dZ1 @ dZ1_dW1.T
  assert dC_dW1.shape == (n[1], n[0])

  # 3. calculate dC/db1 = np.sum(dC/dZ1, axis=1, keepdims=True)
  
  dC_db1 = np.sum(dC_dZ1, axis=1, keepdims=True)
  assert dC_db1.shape == (n[1], 1)

  return dC_dW1, dC_db1

A0, Y, m = prepare_data()

def train():

  # Must use global keywords to modify global variables
  global W3, W2, W1, b3, b2, b1

  epochs = 1001 # Setting number of iterations
  alpha = 0.01 # Setting learning rate
  costs = [] # List to store costs

  for e in range(epochs):

    # 1. Feed forward (calcaulates all A1, A2, A3)
    
    y_hat, cache = feed_forward(A0)

    # 2. Cost calculation
    
    error = cost(y_hat, Y)
    costs.append(error)

    # 3. Back prop calculations
    
    dC_dW3, dC_db3, dC_dA2 = backprop_layer_3(
        y_hat, 
        Y, 
        m, 
        A2= cache["A2"], 
        W3=W3
    )

    dC_dW2, dC_db2, dC_dA1 = backprop_layer_2(
        propagator_dC_dA2=dC_dA2, 
        m=m,
        A1=cache["A1"],
        A2=cache["A2"],
        W2=W2
    )

    dC_dW1, dC_db1 = backprop_layer_1(
        propagator_dC_dA1=dC_dA1, 
        m=m,
        A1=cache["A1"],
        A0=cache["A0"],
        W1=W1
    )

    # 4. Update weights
    
    W3 = W3 - (alpha * dC_dW3)
    W2 = W2 - (alpha * dC_dW2)
    W1 = W1 - (alpha * dC_dW1)

    b3 = b3 - (alpha * dC_db3)
    b2 = b2 - (alpha * dC_db2)
    b1 = b1 - (alpha * dC_db1)

    if e % 20 == 0:
       print(f"epoch {e}: cost = {error:4f}")

  return costs

cost = train()

plt.plot(range(len(cost)), cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost vs Iterations")
plt.show()