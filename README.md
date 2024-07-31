# Graph Neural Network

Implements a Graph Neural Network in Keras on the Cora dataset: https://paperswithcode.com/dataset/cora

Further information can be found in the following blog post:

https://nathanbaileyw.medium.com/implementing-a-graph-neural-network-in-keras-91e8300f1ba4

### Code:
The main code is located in the following files:
* main.py - Main entry file for training the network
* model.py - Implements the Graph Neural Network
* model_building_blocks.py - Implements a Graph Convolutional Layer
* helper_functions.py - Function for compiling and training the model
* graph_conv_layer_binary_adjacency_matrix.py - Implements a Graph Convolutional Layer that uses an NxN adjacency matrix
* model_binary_adjacency_matrix.py - The Graph Neural Network but uses an NxN adjacency matrix
* main_binary_adjacency_matrix.py - Main entry file but uses an NxN adjacency matrix
* plot.py - Plots the training and accuracy loss
* lint.sh - runs linters on the code
