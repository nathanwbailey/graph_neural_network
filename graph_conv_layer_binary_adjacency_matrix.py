from typing import Any

import tensorflow as tf
from numpy.typing import NDArray
from tensorflow import keras
from model_building_blocks import create_feed_forward_layer


class GraphConvLayer(keras.layers.Layer):  # type: ignore[misc]
    """Creates a Graph Convolutional Layer."""

    def __init__(
        self,
        hidden_units: list[int],
        dropout_rate: float = 0.2,
        combination_type: str = "concat",
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        self.combination_type = combination_type

        self.ffn_prepare = create_feed_forward_layer(
            hidden_units, dropout_rate
        )
        self.update_fn = create_feed_forward_layer(hidden_units, dropout_rate)

    def aggregate(
        self,
        neighbour_messages: tf.Tensor,
        adjacency_matrix: tf.Tensor,
    ) -> tf.Tensor:
        """Aggregate Messages from Neighbours."""
        aggregated_message = tf.matmul(adjacency_matrix, neighbour_messages)
        return aggregated_message

    def update(
        self, node_representations: tf.Tensor, aggregated_messages: tf.Tensor
    ) -> tf.Tensor:
        """Update node representations based on the incomoing messages."""
        if self.combination_type == "concat":
            h = tf.concat([node_representations, aggregated_messages], axis=1)
        else:
            h = node_representations + aggregated_messages

        node_embeddings = self.update_fn(h)
        return node_embeddings

    def call(
        self, inputs: tuple[tf.Tensor, NDArray[Any]]
    ) -> tf.Tensor:
        """Forward Pass."""
        node_representations, adjacency_matrix = inputs
        neighbour_messages = self.ffn_prepare(node_representations)

        aggregated_messages = self.aggregate(
            neighbour_messages, adjacency_matrix,
        )
        # Perform the update
        return self.update(node_representations, aggregated_messages)
