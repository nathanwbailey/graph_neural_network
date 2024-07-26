import numpy as np
import tensorflow as tf
from tensorflow import keras


def create_feed_forward_layer(
    hidden_units: list[int], dropout_rate: float
) -> keras.Sequential:
    """Create a feed forward network."""
    ffn_layers = []
    for units in hidden_units:
        ffn_layers.append(keras.layers.BatchNormalization())
        ffn_layers.append(keras.layers.Dropout(dropout_rate))
        ffn_layers.append(keras.layers.Dense(units))
        ffn_layers.append(keras.layers.Activation("gelu"))

    return keras.Sequential(ffn_layers)


class GraphConvLayer(keras.layers.Layer):
    """Creates a Graph Convolutional Layer."""

    def __init__(
        self,
        hidden_units: list[int],
        dropout_rate: float = 0.2,
        aggregration_type="mean",
        combination_type="concat",
    ) -> None:
        """Init variables and layers."""
        super().__init__()
        self.aggregation_type = aggregration_type
        self.combination_type = combination_type

        self.ffn_prepare = create_feed_forward_layer(
            hidden_units, dropout_rate
        )
        self.update_fn = create_feed_forward_layer(hidden_units, dropout_rate)

    def prepare(
        self,
        node_representations: tf.Tensor,
        weights: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """Pass Neighbour features through a NN to produce messages."""
        messages = self.ffn_prepare(node_representations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(
        self,
        node_indices: np.ndarray,
        neighbour_messages: tf.Tensor,
        node_representations: tf.Tensor,
    ) -> tf.Tensor:
        """Aggregate Messages from Neighbours."""

        num_nodes = node_representations.shape[0]

        # Aggregate the messages corresponding to the node neighbours
        # Messages matching the node index will be summed
        # E.g. neighbour_messages = [5, 1, 8, 9, 7]
        # Node Indices = [0, 1, 1, 0, 0]
        # Result is [21, 9]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )

        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )

        else:
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )

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
        self, inputs: tuple[tf.Tensor, tf.Tensor, np.ndarray | None]
    ) -> tf.Tensor:
        """Forward Pass."""
        node_representations, edges, edge_weights = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        # Expand the representations so we create a copy of the neighbour for each node that links to it
        # We pass each neighbour representation through the same weight matrix, so same result for same representation
        # Allows us to share the weight for the layer
        neighbour_representations = tf.gather(
            node_representations, neighbour_indices
        )
        neighbour_messages = self.prepare(
            neighbour_representations, edge_weights
        )
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_representations
        )
        # Perform the update
        return self.update(node_representations, aggregated_messages)
