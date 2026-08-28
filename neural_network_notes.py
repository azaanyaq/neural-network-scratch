"""
| General Notes 1 - 7 |

Article link: https://blog.stackademic.com/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc

- Machines learn from data like humans do
- All machine learnign consists of is trying to minimise the cost metric as much as possible

- Vectors are 1 dimensional tensors
- Matrices are 2 dimensional tensors
- Matrices are classified by rows x columns
- Matrix multiplication is possible if axb and bxc -> axc (row of 1st and column of 2nd)

- Transpose a matrix by switching its rows and columns

- Derrivatives represent the gradient of a tanegnt of a curve at a cetrain point
- PDs allow us to find how any variable affects the cost function and tweak it accordingly
- Each PD of the cost function acts as a dial where turning it can reduce the cost

- Each node has a numeric value and we connect nodes using weights, acting as multipliers
- A layer in a network has 4 rules:
    - Each node in a layer must connect to every single other node in the next layer using connections
      called weights
    - Every weigth has a predetermined value that we choose at random (don't know values of nodes)
    - Only 1st layer of nodes start off having values (input layer)
    - Every subsequent layer gets calculated based on the previous layer and weights connecting them
- A bias is a number added to a node's value at the end of the weight vector calculation:
    - We apply biases to each node except for the input layer nodes
    - Every node (barring input layer) has their own randomly assinged bias of our choosing
- Biases are improtant as to avoid 0 values for nodes (which would propogate and spread till 
  the model was all 0)
- Can adjust a network via only weights and biases, which the computer keeps track of and varies
  in order to minimise the cost function

Steps for running a neural network:
1. Provide the input data to the network as an input layer, and then the network uses those values
   from the input layer and the weigths to connect it to the second layer to compute values for the 
   second layer
2. Process propogates through all layers, until each node in the network has a value
3. Last layer will output the prediction, which is then compared to labeled data to provide a cost 
   metric for the network
4. Based on this cost, we calculate the gradient with respect to cost (delta C) and update the weights
   and biases accordingly
5. Repeat 1-4 to cost is minimused as much as possible

- Neural networks are comprised of: input layer -> 'hidden' layers -> output layer
- Count no. of layers using L (where input layer is the 0th layer)
- Input -> 2 hidden -> output is therefore L=3 (0, 1, 2, 3)
- Represent the no. of nodes in each layer using n^x (superscript)
- n^0 = 2, n^1 = 3, n^2 = 3, n^3 = 1
- For any layer L, there will be n^(L) * n^(L-1) weights connecting layer L-1 to L

- Represent each layer of the network as a matrix (reshaped from a vector)
- Represent each set of weights in between 2 layers as a matrix

- Row vector: a vector with n elements reshaped to 1 x n (single row)
- Column vector: a vector with n elements reshaped to n x 1 (single column)

- In matrix multiplication, weights matrix ^T (transposed) * column matrix (node values)
- W^(L) has dimensions n^(L) x n^(L-1) if n^(L) is the number of nodes in layer L and
  n^(L-1) is the number of numbers in the previous layer L-1 (so if layer 1 has 3 nodes
  and layer 0 has 2 nodes, weight matrix will be 3 x 2)
- Can then find out that W^(1) = 3x2, W^(2) = 3x3 and W^(3) = 1x3

- For any layer L, the bias column vector matrix b^L has dimensions n^(L) x 1 (column)

- Represent the column vector matrix of nodes in a layer as z^(L) for some layer L
- We know that z^(0) = 2x1, z^(1) = 2x1, W^(1) = 2x2, b^(1) = 2x1
- W^(1) z^(0) + b^(1) = z^(1)
- To get node values of next layer, we matrix multiply the weight matrix by the previous node values 
  and then add the bias

- Activation functions squish the output of the equation above into something our networks
  can deal with - notated as g(z)
- Sigmoid function is commonly used as g(z) as it ranges between 0 and 1
- Passing the z values into g(z) we will obtain the actual value for nodes in a layer, denoted
  as a^(L) for a given layer L

- Feedforward process: W^(L) z^(L-1) + b^(L) = z^(L)
                       a^(L) = g( z^(L) )

- Notate the matrix of input data as X: X^(i) for the i'th training sample
- Notate the matrix of output data as Y
- Can designate X as a^(0)
- Notate the output of a model as ŷ
- We compare ŷ (predicted) to y (actual) to create an error metric

- By vectorising our training data, we do the feed foward on all training data at once rather
  than looping over each every single data value 

- Input layer to our network is no longer a column vector n^(0) x 1, but instead a matrix
  n^(0) x m where n = number of features and m = number of training samples
- This means that all intermediary values and activation layers will vectorise

- X or A^(0): the n^(0) x m matrix of training data (input layer)

- Represent the vectorised layer of activations across all training samples for a layer L as
  A^(L), which will always ahve m (number of training samples) columns
- Their number of rows is equal to the number of nodes in that layer n^(L)

- Broadcasting is when you stretch the bias matrix to have m columns  

***

Z^[L] = W^[L] A^[L-1] + b^[L]
A^[L] = g( Z[L] )

Where:

- n^[L] is the number of nodes in layer L
- m = number of training samples
- Z^[L]: pre-activation values for layer L, vectorised across all training samples. Has
  dimensions n^[L] x m
- A^[L]: post-actvation values etc. Has dimensions n^[L] x m
- A^[L-1]: node values for layer L-1. Has dimensions n^[L] x m
- W^[L]: matrix of weights between later L-1 to L, with dimensions n^[L] x n^[L-1]

"""
import numpy as np

# 1. Create network architecture

L = 3
n = [2, 3, 3, 1] # Number of nodes in each layer

# 2. Create weights and biases

W1 = np.random.randn(n[1], n[0]) # Np function to create a weight matrix of random values
W2 = np.random.randn(n[2], n[1]) # Arguments are rows and columns of matrix 
W3 = np.random.randn(n[3], n[2]) # Can tell is of the form n^[l] x n^[l-1]

b1 = np.random.randn(n[1], 1) # Np function to create a bias matrix of random values
b2 = np.random.randn(n[2], 1) # Can tell is of the form n^[l] x 1
b3 = np.random.randn(n[3], 1)

# 3. Create training data and labels

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

# 4. Create activation function
def sigmoid(arr): # Activation function g(z) (Sigmoid)
    return 1 / (1 + np.exp(-1 * arr))

# 5. Create feed forward process

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

"""
| General notes 8 |

- The success of a neural network is determined by how much it can minimise the cost function
- Convex cost functions (ones with 1 global miminum) are easy to minimise as no other minima
- Non convex cost functions have multiple local minima so can get stuck in these and not be
  able to reduce cost further
- Using a binary cross entropy loss function:

    L(ŷ, y) = -(y ln(ŷ) + (1-y) ln(1-ŷ))
    C = 1/m * sum (i=1 -> m) (L (ŷ, y) )

    y = the training label poutput for the i'th training sample
    ŷ = the model's prediction for the i'th training sample
    L(...) = the loss function (error for a single training sample)
    C = the cost function, summing up all losses calculated for all training samples and then
        averaging them
"""

def cost(y_hat, y): # both y_hat and y should be a n^L x m matrix

    # 1. Losses is a n^L x m matrix
    losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )

    m = y_hat.size # Calculates total number of predictions that make up y_hat

    # 2. Summing across axis = 1 means we sum across rows, making this a n^L x 1 matrix
    summed_losses = (1 / m) * np.sum(losses, axis = 1)

    return np.sum(summed_losses)

"""
| General notes 9 - 10 |

- Partial derrivatives are used in back propogation to change weights for reducing cost
- delC / delW^[L], meaning how much individual weight in W^[L] matrix affecrs the final cost c
- Backprop is jsut finding the derriavtive of all equations from feed forward
- W^[L] + b^[L] -> Z^[L] -> A^[L] -> C
- Can find delC / delW^[L] via chain rule (from computation graph above)
- Matrix of gradient of cost with respect to weights in alayer should have the same
  matrix dimensions of the weight matrix itself
- After finding the PDE expressions for all, we can obtain the expression for:

  delC / delW[L] = 1/m * (A^[L] - y) * (A^[L-1])^T
  n^[L] x n^[L-1] matrix

- Can also find backprop expression relating to biases:

  delC / delb^[L] = 1/m * sum (i=1 -> m) (a^[L(i)] - y^[i])
  Sum a n^L x m matrix by its rows (compress column dimension) and thus n^L x 1 matrix

- At each other layer, we calculate almost the exact same set of PDs
- The generalised backprop PDEs are:

  delC / delW^[L] = delC/delA^[L] * delA^[L]/delZ^[L] * delZ^[L]/delW^[L]
  delC / delb^[L] = delC/delA^[L] * delA^[L]/delZ^[L] * delZ^[L]/delb^[L]
  delC / delA^[L-1] = delC/delA^[L] * delA^[L]/delZ^[L] * delZ^[L]/delA^[L-1]

- Backprop algorithm is:
  1. Calculate delC/delW^[L] and delC/delb^[L] for the final layer L
  2. Calcualte the propogater for the penultimate layer L-1 by finding delC/delA^[L-1]
  3. For all layers l starting from l = L-1, and going until the first layer l=1,
  calculate delC/delW^[l], delC/delb^[l], and the propogator for the next layer
  delC/delA^[l-1]

- delC is just the gradients of the weights and biases from each layer

- Gradient descent is just using the negative value of delC to find the global minima 
  of the cost function
- Learning rate regualtes how big or small are 'steps' are
- We multiply our learning rate (alpha) by the negative gradient and add that to 
  our parameters to update them:
  
  W^[L] = W^[L] - alpha * delC/delW^[L]
  b^[L] = b^[L] - alpha * delC/delb^[L]

- The general training algorithm therefore is:

  1. Use feed forward to calculate the network output ŷ
  2. Calculate the cost C from network output ŷ and training label y
  3. Save the cost to a list of costs. This is useful in diagnosing if our model
    is converging to a minimum cost correctly
  4. Run backprop to compute delta C ( delC/delW^[l] and delC/db^[l] for each 
     layer l)
  5. Update the parameters using the learning rate (default to alpha = 0.01)
  6. Repeat steps 1-5 x amount of times

"""

import numpy as np
import matplotlib.pyplot as plt

L = 3
n = [2, 3, 3, 1] # Number of nodes in each layer

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

  epochs = 10001 # Setting number of iterations
  alpha = 0.001 # Setting learning rate
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