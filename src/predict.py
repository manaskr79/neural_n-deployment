import numpy as np
from config import config
import pickle
from preprocessing.data_management import load_model
import pipeline as pl
# Defining the  XOR data Sample
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_train = np.array([[0], [1], [1], [0]])

# Initializing the configuration
NUM_LAYERS = config.NUM_LAYERS
epsilon = 10**-7
tol = 10**-8
batch_size = 2

# Initializing  placeholders for various parameters
z = [None] * NUM_LAYERS
h = [None] * NUM_LAYERS

del_fl_by_del_z = [None] * NUM_LAYERS
del_hl_by_del_theta0 = [None] * NUM_LAYERS
del_hl_by_del_theta = [None] * NUM_LAYERS
del_L_by_del_h = [None] * NUM_LAYERS
del_L_by_del_theta0 = [None] * NUM_LAYERS
del_L_by_del_theta = [None] * NUM_LAYERS

theta0 = [None] * NUM_LAYERS
theta = [None] * NUM_LAYERS

# Definining the neural network functions
def initialize_parameters():
    global theta0, theta
    for l in range(1, NUM_LAYERS):
        theta0[l] = np.random.randn(1, config.layer_sizes[l])
        theta[l] = np.random.randn(config.layer_sizes[l-1], config.layer_sizes[l])

def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):
    return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs, current_layer_neurons_weights)

def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):
    if current_layer_neurons_activation_function == "linear":
        return current_layer_neurons_weighted_sums
    elif current_layer_neurons_activation_function == "sigmoid":
        return 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "tanh":
        return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums)) / \
               (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))
    elif current_layer_neurons_activation_function == "relu":
        return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)

def del_layer_neurons_outputs_wrt_weighted_sums(current_layer_neurons_activation_function, current_layer_neurons_weighted_sums):
    if current_layer_neurons_activation_function == "linear":
        return np.ones_like(current_layer_neurons_weighted_sums)
    elif current_layer_neurons_activation_function == "sigmoid":
        current_layer_neurons_outputs = 1 / (1 + np.exp(-current_layer_neurons_weighted_sums))
        return current_layer_neurons_outputs * (1 - current_layer_neurons_outputs)
    elif current_layer_neurons_activation_function == "tanh":
        return (2 / (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))) ** 2
    elif current_layer_neurons_activation_function == "relu":
        return (current_layer_neurons_weighted_sums > 0)

def del_layer_neurons_outputs_wrt_biases(current_layer_neurons_outputs_dels):
    return current_layer_neurons_outputs_dels

def del_layer_neurons_outputs_wrt_weights(previous_layer_neurons_outputs, current_layer_neurons_outputs_dels):
    return np.matmul(previous_layer_neurons_outputs.T, current_layer_neurons_outputs_dels)

def train_neural_network():
    epoch_counter = 0
    mse = 1
    loss_per_epoch = list()
    loss_per_epoch.append(mse)
    initialize_parameters()

    while True:
        mse = 0
        for batch_start in range(0, X_train.shape[0], batch_size):
            batch_end = batch_start + batch_size
            X_batch = X_train[batch_start:batch_end]
            Y_batch = Y_train[batch_start:batch_end]
            
            batch_grad_theta0 = [np.zeros_like(theta0[l]) for l in range(NUM_LAYERS)]
            batch_grad_theta = [np.zeros_like(theta[l]) for l in range(NUM_LAYERS)]
            
            for i in range(X_batch.shape[0]):
                h[0] = X_batch[i].reshape(1, X_batch.shape[1])

                for l in range(1, NUM_LAYERS):
                    z[l] = layer_neurons_weighted_sum(h[l - 1], theta0[l], theta[l])
                    h[l] = layer_neurons_output(z[l], config.f[l])

                    del_fl_by_del_z[l] = del_layer_neurons_outputs_wrt_weighted_sums(config.f[l], z[l])
                    del_hl_by_del_theta0[l] = del_layer_neurons_outputs_wrt_biases(del_fl_by_del_z[l])
                    del_hl_by_del_theta[l] = del_layer_neurons_outputs_wrt_weights(h[l - 1], del_fl_by_del_z[l])

                Y_batch[i] = Y_batch[i].reshape(Y_batch[i].shape[0], 1)
                L = (1 / 2) * (Y_batch[i][0] - h[NUM_LAYERS - 1][0, 0]) ** 2
                mse += L

                del_L_by_del_h[NUM_LAYERS - 1] = (h[NUM_LAYERS - 1] - Y_batch[i])
                for l in range(NUM_LAYERS - 2, 0, -1):
                    del_L_by_del_h[l] = np.matmul(del_L_by_del_h[l + 1], (del_fl_by_del_z[l + 1] * theta[l + 1]).T)

                for l in range(1, NUM_LAYERS):
                    batch_grad_theta0[l] += del_L_by_del_h[l] * del_hl_by_del_theta0[l]
                    batch_grad_theta[l] += del_L_by_del_h[l] * del_hl_by_del_theta[l]

            for l in range(1, NUM_LAYERS):
                theta0[l] -= (epsilon * batch_grad_theta0[l] / batch_size)
                theta[l] -= (epsilon * batch_grad_theta[l] / batch_size)

        mse /= X_train.shape[0]
        epoch_counter += 1
        loss_per_epoch.append(mse)

        print(f"Epoch # {epoch_counter}, Loss = {mse}")

        if abs(loss_per_epoch[epoch_counter] - loss_per_epoch[epoch_counter - 1]) < tol:
            break

    save_model(theta0, theta, "xor_model.pkl")

def save_model(theta0, theta, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump((theta0, theta), file)

def load_model(file_name):
    with open(file_name, 'rb') as file:
        theta0, theta = pickle.load(file)
    return theta0, theta

def predict(X):
    z = [None] * NUM_LAYERS
    h = [None] * NUM_LAYERS

    h[0] = X.reshape(1, X.shape[0])
    for l in range(1, NUM_LAYERS):
        z[l] = layer_neurons_weighted_sum(h[l - 1], theta0[l], theta[l])
        h[l] = layer_neurons_output(z[l], config.f[l])
    
     
    # Now Applying threshold to convert decimal output to binary output
    binary_output = (h[NUM_LAYERS - 1] >= 0.5).astype(int)
    return binary_output



if __name__ == "__main__":
    # Training the model
    train_neural_network()

    # Loading the model
    theta0, theta = load_model("xor_model.pkl")

    # Example input (XOR inputs)
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for x in X_test:
        prediction = predict(x)
        print(f"Input: {x}, Prediction: {prediction}")
    
    