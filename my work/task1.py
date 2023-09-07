import csv
import numpy as np
import sys

# Read the weights in the csv file into weights list
def load_weights(file_path):
    weights = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader: 
            weights.append(row)
    return weights

# Read the biases in the csv file into biases list
def load_biases(file_path):
    biases = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader: 
            biases.append(row)
    return biases

# Insert weights and biases into parameters dictionary
def get_neural_network_parameters(weights, biases, num_layers, layer_sizes):    
    parameters = {}
    row_counter = 0
    
    for l in range(num_layers):
        parameters[f"Weight{l + 1}"] = []
        for i in range(layer_sizes[l]):
            weight_sub = np.asarray(weights[row_counter][1:], dtype=np.float32)
            parameters[f"Weight{l + 1}"].append(weight_sub)
            row_counter += 1
        parameters[f"Weight{l + 1}"] = np.array(parameters[f"Weight{l + 1}"])

    for l in range(num_layers):
        parameters[f"Bias{l + 1}"] = np.asarray(biases[l][1:], dtype=np.float32) 
    return parameters

# Activation function for all layers excluding the last layer
def relu(z):
    return np.maximum(0, z)

# For taking the derivative of relu function
def relu_derivative(Z):
    return np.where(Z > 0, 1, 0)

# Softmax function is applied on the last layer
def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)

# For forward propagation
def forward_propagation(X, parameters, num_layers):
    store = {}
    A = X.T 
    
    for l in range(num_layers - 1):
        Z = parameters[f"Weight{l + 1}"].T.dot(A) + parameters[f"Bias{l + 1}"]
        A = relu(Z)

        store[f"Activation{l + 1}"] = A 
        store[f"Weight{l + 1}"] = parameters[f"Weight{l + 1}"]
        store[f"Z{l + 1}"] = Z

    Z = parameters[f"Weight{num_layers}"].T.dot(A) + parameters[f"Bias{num_layers}"]
    A = softmax(Z)
    store[f"Activation{num_layers}"] = A
    store[f"Weight{num_layers}"] = parameters[f"Weight{num_layers}"]
    store[f"Z{num_layers}"] = Z

    return A, store

# For computing the derivative of loss function with respect to weights and biases in each layer
def backward_propagation(X, Y, store, num_layers):
    derivatives = {}
    Y = np.asarray(Y, dtype=int)
    store[f"Activation0"] = X
    
    dZ = store[f"Activation{num_layers}"] - Y
    dW = np.outer(store[f"Activation{num_layers - 1}"], dZ)
    dAPrev = store[f"Weight{num_layers}"].dot(dZ)

    derivatives[f"dWeight{num_layers}"] = dW
    derivatives[f"dBias{num_layers}"] = dZ

    for l in range(num_layers - 1, 0, -1):
        relder = relu_derivative(store[f"Z{l}"])
        dZ = dAPrev * relder
        dW = np.outer(dZ, store[f"Activation{l - 1}"])
        if l > 1:
            dAPrev = store[f"Weight{l}"].dot(dZ)

        derivatives[f"dWeight{l}"] = dW.T
        derivatives[f"dBias{l}"] = dZ

    return derivatives

# Write the derivatives with respect to weights into dw csv file
def write_dW_to_csv(derivatives):
    for i in range(num_layers):
        file = open(r'dw.csv', 'a+', newline ='') 
        with file: 
            write = csv.writer(file) 
            array = derivatives[f"dWeight{i+1}"] 
            write.writerows(array) 

# Write the derivatives with respect to biases into db csv file
def write_dB_to_csv(derivatives):
    array = []
    for i in range(num_layers):
        array.append(derivatives[f"dBias{i+1}"]) 

    file = open(r'db.csv', 'w+', newline ='') 
    with file: 
        write = csv.writer(file) 
        write.writerows(array) 

def main(weights_file, biases_file, num_layers, layer_sizes):
    # Load weights and biases
    weights = load_weights(weights_file)
    biases = load_biases(biases_file)
    
    # Get neural network parameters
    parameters = get_neural_network_parameters(weights, biases, num_layers, layer_sizes)
    
    # Define your input data X and target output Y here
    X = [-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]
    X = np.asarray(X, dtype=int)
    Y = [0, 0, 0, 1]
    
    # Perform forward propagation
    A, store = forward_propagation(X, parameters, num_layers)
    
    # Perform backward propagation to compute derivatives
    derivatives = backward_propagation(X, Y, store, num_layers)
    
    # Write the derivatives with respect to weights into dw.csv file
    write_dW_to_csv(derivatives)
    
    # Write the derivatives with respect to biases into db.csv file
    write_dB_to_csv(derivatives)

if __name__ == "__main__":
    weights_file = 'w-100-40-4.csv'
    biases_file = 'b-100-40-4.csv'
    num_layers = 3
    layer_sizes = [14, 100, 40, 4]

    main(weights_file, biases_file, num_layers, layer_sizes)