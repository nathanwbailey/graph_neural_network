from typing import Any

import tensorflow as tf
from numpy.typing import NDArray
from tensorflow import keras

from model_building_blocks import GraphConvLayer, create_feed_forward_layer


class GNNNodeClassifier(keras.models.Model):  # type: ignore[misc]
    """Graph Neural Network Model."""

    def __init__(
        self,
        graph_info: tuple[tf.Tensor, NDArray[Any], tf.Tensor | None],
        num_classes: int,
        hidden_units: list[int],
        aggregation_type: str = "sum",
        combination_type: str = "concat",
        dropout_rate: float = 0.2,
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights

        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])

        self.edge_weights /= tf.math.reduce_sum(self.edge_weights)

        self.preprocess = create_feed_forward_layer(
            hidden_units, dropout_rate
        )

        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
        )
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
        )

        self.postprocess = create_feed_forward_layer(
            hidden_units, dropout_rate
        )
        self.compute_logits = keras.layers.Dense(units=num_classes)

    def call(self, input_node_indices: NDArray[Any]) -> tf.Tensor:
        """Model Forward Pass."""
        x = self.preprocess(self.node_features)
        x1 = self.conv1((x, self.edges, self.edge_weights))
        x = x1 + x
        x2 = self.conv2((x, self.edges, self.edge_weights))
        x = x2 + x
        x = self.postprocess(x)
        # Get the final node embeddings for the batch of  node IDs we passed in
        node_embeddings = tf.gather(x, input_node_indices)
        # Compute the logits for classification
        return self.compute_logits(node_embeddings)
