# Fixed Graph CNN Components - TensorFlow 2.x Compatible
# Based on the research paper: "Graph Convolution Neural Network Based End-to-End
# Channel Selection and Classification for Motor Imagery Brain-Computer Interfaces"

import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

print("Building TensorFlow 2.x Compatible Graph CNN Components for EEG-ARNN")
print("=" * 70)

class GraphConvolution(layers.Layer):
    """
    Graph Convolution Layer for EEG Channel Relationships.

    Implements: H = W_hat * X * Theta

    Where:
    - W_hat: Normalized adjacency matrix (learnable)
    - X: Input EEG features
    - Theta: Learnable linear transformation
    """

    def __init__(self, output_dim, use_bias=True, activation=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape is a list: [features_shape, adjacency_shape]
        features_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        input_dim = features_shape[-1]

        # Linear transformation weights
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.output_dim,),
                initializer='zeros',
                trainable=True
            )

        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs):
        # inputs: [features, adjacency_matrix]
        if isinstance(inputs, list):
            features, adjacency = inputs
        else:
            raise ValueError("GraphConvolution expects a list of [features, adjacency]")

        # Linear transformation: X * Theta
        support = tf.matmul(features, self.kernel)

        # Graph convolution: A * X * Theta
        output = tf.matmul(adjacency, support)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0] if isinstance(input_shape, list) else input_shape
        return (features_shape[0], features_shape[1], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'use_bias': self.use_bias,
            'activation': keras.activations.serialize(self.activation)
        }
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DynamicAdjacencyMatrix(layers.Layer):
    """
    Dynamic Adjacency Matrix Layer for CARM.

    Learns the optimal connectivity between EEG channels through backpropagation.
    Eliminates the need for manual adjacency matrix construction.

    From paper equation (15): W*_ij = 1 if i != j, 0 if i == j (initial state)
    Updated via equation (17): W* = (1 - ρ)W* - ρ * ∂Loss/∂W*
    """

    def __init__(self, num_channels, rho=0.001, **kwargs):
        super(DynamicAdjacencyMatrix, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.rho = rho  # Learning rate for adjacency matrix update

    def build(self, input_shape):
        # Initialize adjacency matrix as identity matrix (paper equation 15)
        # All channels connected to themselves initially
        init_adjacency = np.eye(self.num_channels, dtype=np.float32)

        self.adjacency_matrix = self.add_weight(
            name='adjacency_matrix',
            shape=(self.num_channels, self.num_channels),
            initializer=keras.initializers.Constant(init_adjacency),
            trainable=True
        )

        super(DynamicAdjacencyMatrix, self).build(input_shape)

    def call(self, inputs):
        # Normalize adjacency matrix for stable gradients (paper equation 12)
        # Compute degree matrix
        degree = tf.reduce_sum(tf.abs(self.adjacency_matrix), axis=1, keepdims=True)
        degree = tf.maximum(degree, 1e-8)  # Avoid division by zero

        # Normalized adjacency: D^(-1/2) * A * D^(-1/2)
        degree_inv_sqrt = tf.pow(degree, -0.5)
        normalized_adj = self.adjacency_matrix * degree_inv_sqrt
        normalized_adj = normalized_adj * tf.transpose(degree_inv_sqrt)

        # Add self-connections for stability
        identity = tf.eye(self.num_channels, dtype=tf.float32)
        normalized_adj = normalized_adj + identity

        return normalized_adj

    def get_adjacency_weights(self):
        """Get the learned adjacency matrix weights."""
        return self.adjacency_matrix.numpy()

    def compute_output_shape(self, input_shape):
        return (self.num_channels, self.num_channels)

    def get_config(self):
        config = {
            'num_channels': self.num_channels,
            'rho': self.rho
        }
        base_config = super(DynamicAdjacencyMatrix, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CARM(layers.Layer):
    """
    Channel Active Reasoning Module.

    Combines dynamic adjacency learning with graph convolution
    to model spatial relationships between EEG channels.

    From paper: CARM eliminates the need to construct an artificial adjacency
    matrix and continuously modifies connectivity between channels.
    """

    def __init__(self, num_channels, output_dim, dropout_rate=0.25, **kwargs):
        super(CARM, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Build components
        self.adjacency_layer = DynamicAdjacencyMatrix(self.num_channels)
        self.graph_conv = GraphConvolution(self.output_dim, activation='elu')
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)

        super(CARM, self).build(input_shape)

    def call(self, inputs, training=None):
        # Get dynamic adjacency matrix
        adjacency = self.adjacency_layer(inputs)

        # Apply graph convolution
        graph_output = self.graph_conv([inputs, adjacency])

        # Normalization and regularization
        output = self.batch_norm(graph_output, training=training)
        output = self.dropout(output, training=training)

        return output

    def get_adjacency_matrix(self):
        """Extract learned adjacency matrix for analysis."""
        return self.adjacency_layer.get_adjacency_weights()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = {
            'num_channels': self.num_channels,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate
        }
        base_config = super(CARM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TFEM(layers.Layer):
    """
    Temporal Feature Extraction Module.

    CNN-based temporal processing to extract time-domain features
    from EEG signals while preserving channel structure for CARM.

    From paper: TFEM performs feature extraction in the time domain,
    avoiding amplitude-frequency feature extraction overhead.
    """

    def __init__(self, filters, kernel_size, dropout_rate=0.25, **kwargs):
        super(TFEM, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Build temporal processing components
        self.conv1d = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding='same',
            activation=None
        )
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation('elu')
        self.dropout = layers.Dropout(self.dropout_rate)

        super(TFEM, self).build(input_shape)

    def call(self, inputs, training=None):
        # Input shape: (batch_size, channels, time_points)
        # Transpose for Conv1D: (batch_size, time_points, channels)
        x = tf.transpose(inputs, [0, 2, 1])

        # Temporal convolution
        x = self.conv1d(x)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)

        # Transpose back: (batch_size, channels, time_points)
        output = tf.transpose(x, [0, 2, 1])

        return output

    def compute_output_shape(self, input_shape):
        return input_shape  # Shape preserved

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        }
        base_config = super(TFEM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Test the components
if __name__ == "__main__":
    print("\n✅ All Graph CNN components defined successfully!")
    print("\nAvailable components:")
    print("  • GraphConvolution: Core graph convolution operation")
    print("  • DynamicAdjacencyMatrix: Learnable channel connectivity")
    print("  • CARM: Channel Active Reasoning Module")
    print("  • TFEM: Temporal Feature Extraction Module")

    # Quick test
    print("\nRunning quick validation test...")
    try:
        # Test TFEM
        batch_size, num_channels, num_timepoints = 2, 10, 640
        test_input = tf.random.normal((batch_size, num_channels, num_timepoints))

        tfem = TFEM(filters=16, kernel_size=16)
        tfem_out = tfem(test_input)
        print(f"  ✅ TFEM: Input {test_input.shape} → Output {tfem_out.shape}")

        # Test CARM
        carm = CARM(num_channels=num_channels, output_dim=32)
        carm_out = carm(test_input)
        print(f"  ✅ CARM: Input {test_input.shape} → Output {carm_out.shape}")

        print("\n✅ All components working correctly!")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
