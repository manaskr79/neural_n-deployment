import numpy as np
import math

# Define the configuration parameters
NUM_LAYERS = 3
layer_sizes = [2, 2, 1]

# Define the theta0 and theta lists
theta0 = [None]
theta = [None]

# Function to initialize layer biases
def initialize_layer_biases(num_units):
    return np.random.uniform(low=-1, high=1, size=(1, num_units))

# Function to initialize layer weights
def initialize_layer_weights(num_units_l_1, num_units_l):
    return np.random.uniform(low=-1, high=1, size=(num_units_l_1, num_units_l))

# Function to initialize parameters
def initialize_parameters():
    global theta0, theta
    for l in range(1, NUM_LAYERS - 1):
        theta0.append(initialize_layer_biases(layer_sizes[l]) / math.sqrt(layer_sizes[l - 1]))
        theta.append(initialize_layer_weights(layer_sizes[l - 1], layer_sizes[l]) / math.sqrt(layer_sizes[l - 1]))

    theta0.append(initialize_layer_biases(layer_sizes[l]) / math.sqrt(layer_sizes[l - 1]))
    theta.append(initialize_layer_weights(layer_sizes[l - 1], layer_sizes[l]) / math.sqrt(layer_sizes[l - 1]))
