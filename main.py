import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from helper_functions import compile_and_train_model
from model import GNNNodeClassifier
from plot import display_learning_curves

zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)

DATA_DIR = os.path.join(os.path.dirname(zip_file), "cora")
print(DATA_DIR)

# Citation Data
citations = pd.read_csv(
    os.path.join(DATA_DIR, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)

# Papers Data
# There exists binary entry for 1433 words for each paper
column_names = (
    ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
)
papers = pd.read_csv(
    os.path.join(DATA_DIR, "cora.content"),
    sep="\t",
    header=None,
    names=column_names,
)

# Create zero-based id values for the data
class_values = sorted(papers["subject"].unique())
paper_values = sorted(papers["paper_id"].unique())
class_idx = {name: idx for idx, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(paper_values)}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

# Visualize the citation graph data
plt.figure(figsize=(10, 10))
cora_graph = nx.from_pandas_edgelist(citations.sample(n=1500))
# Get the subjects for the nodes in the graph
subjects = list(
    papers[papers["paper_id"].isin(list(cora_graph.nodes))]["subject"]
)
# Plot and color based on subjects
nx.draw_spring(cora_graph, node_size=15, node_color=subjects)
plt.savefig("graph_data.png")

train_data, test_data = [], []
# Get the papers from each class (subject)
for _, group_data in papers.groupby("subject"):
    # Create random numbers between 0 and 1 for each data point
    # Use it to randomly select data for the train and test data
    # Roughly 50% are selected for train data and 50% for test data
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

# Shuffle the data
train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

feature_names = list(set(papers.columns) - {"paper_id", "subject"})
num_features = len(feature_names)
num_classes = len(class_idx)
print(f"Number of Features: {num_features}")
print(f"Number of Classes: {num_classes}")

x_train = train_data.paper_id.to_numpy()
x_test = test_data.paper_id.to_numpy()
y_train = train_data["subject"]
y_test = test_data["subject"]

# Source paper cites target papers
# Edges are laid out like:
# [0, 1, 1, 0, 0]
# [5, 1, 8, 9, 7]
# Neighbours of 0 are 5, 9, 7
# Neighbours of 1 are 1, 8
edges = citations[["source", "target"]].to_numpy().T
# Set to ones, as no weights needed here
edge_weights = tf.ones(shape=edges.shape[1])

node_features = tf.cast(
    papers.sort_values("paper_id")[feature_names].to_numpy(),
    dtype=tf.dtypes.float32,
)

graph_info = (node_features, edges, edge_weights)

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256


gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
)


print("GNN output shape:", gnn_model([1, 10, 100]))
gnn_model.summary()

# Compile and train model
history = compile_and_train_model(
    gnn_model, x_train, y_train, learning_rate, num_epochs, batch_size
)
# Plot learning curves
display_learning_curves(history)

x_test = test_data.paper_id.to_numpy()
_, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
