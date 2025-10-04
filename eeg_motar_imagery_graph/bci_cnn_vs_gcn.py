"""
BCI Competition IV Dataset: CNN vs GCN Comparison
Goal: Prove GCN beats CNN on motor imagery classification
"""

import numpy as np
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from scipy.linalg import eigh
import os
import glob

print("="*70)
print("BCI COMPETITION IV: CNN vs GCN Showdown")
print("="*70)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_bci_2a_data(data_dir='BCI_2a', n_subjects=9):
    """Load BCI Competition IV Dataset 2a (4-class motor imagery)."""

    print("\n[1] Loading BCI Competition IV 2a dataset...")

    all_data = []
    all_labels = []

    for subject in range(1, n_subjects + 1):
        # Load training data (T = Training)
        train_file = os.path.join(data_dir, f'A0{subject}T.gdf')

        if os.path.exists(train_file):
            print(f"  Loading Subject {subject}...")

            # Read GDF file with MNE
            raw = mne.io.read_raw_gdf(train_file, preload=True, verbose=False)

            # Get events (motor imagery cues)
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            # Filter to motor imagery classes only (769-772 annotations, mapped to integer IDs)
            mi_event_ids = {k: v for k, v in event_id.items() if k in ['769', '770', '771', '772']}

            if len(mi_event_ids) > 0:
                # Create epochs (0-4 seconds after cue)
                epochs = mne.Epochs(raw, events, event_id=mi_event_ids,
                                   tmin=0, tmax=4, baseline=None,
                                   preload=True, verbose=False)

                # Get data
                data = epochs.get_data()  # (n_epochs, n_channels, n_times)
                labels = epochs.events[:, -1]  # Event IDs

                # Remap labels to 0-3
                label_map = {v: i for i, v in enumerate(sorted(mi_event_ids.values()))}
                labels = np.array([label_map[l] for l in labels])

                all_data.append(data)
                all_labels.append(labels)

                print(f"    Subject {subject}: {data.shape[0]} epochs, {data.shape[1]} channels")

    if all_data:
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)

        print(f"\n  Total dataset: {X.shape}")
        print(f"  Classes: {len(np.unique(y))} (labels: {np.unique(y)})")
        print(f"  Channels: {X.shape[1]}, Timepoints: {X.shape[2]}")

        return X, y
    else:
        print("  ERROR: No data loaded!")
        return None, None

# ============================================================================
# 2. GRAPH CONSTRUCTION (Pearson Correlation)
# ============================================================================

def compute_adjacency_from_pearson(data):
    """Compute adjacency matrix using Pearson correlation."""
    n_epochs, n_channels, n_timepoints = data.shape
    data_reshaped = data.transpose(1, 0, 2).reshape(n_channels, -1)
    correlation_matrix = np.corrcoef(data_reshaped)
    adjacency = np.abs(correlation_matrix)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency.astype(np.float32)

def compute_graph_laplacian(adjacency):
    """Compute normalized graph Laplacian."""
    degree = np.sum(adjacency, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    A_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt
    L = np.eye(adjacency.shape[0]) - A_norm
    eigenvalues = eigh(L, eigvals_only=True)
    lambda_max = eigenvalues[-1]
    L_rescaled = (2.0 / lambda_max) * L - np.eye(L.shape[0])
    return L_rescaled.astype(np.float32)

# ============================================================================
# 3. CHEBYSHEV GRAPH CONVOLUTION
# ============================================================================

class ChebyshevGraphConv(layers.Layer):
    """Chebyshev Graph Convolution Layer."""

    def __init__(self, num_filters, K=3, activation=None, **kwargs):
        super(ChebyshevGraphConv, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.K = K
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        features_shape = input_shape[0]
        input_dim = features_shape[-1]

        self.theta = self.add_weight(
            name='theta',
            shape=(self.K, input_dim, self.num_filters),
            initializer='glorot_uniform',
            trainable=True
        )

        self.bias = self.add_weight(
            name='bias',
            shape=(self.num_filters,),
            initializer='zeros',
            trainable=True
        )

        super(ChebyshevGraphConv, self).build(input_shape)

    def call(self, inputs):
        x, L_rescaled = inputs

        Tx_0 = x
        Tx_1 = tf.matmul(L_rescaled, x)

        out = tf.matmul(Tx_0, self.theta[0])

        if self.K > 1:
            out += tf.matmul(Tx_1, self.theta[1])

        for k in range(2, self.K):
            Tx_2 = 2 * tf.matmul(L_rescaled, Tx_1) - Tx_0
            out += tf.matmul(Tx_2, self.theta[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        out = tf.nn.bias_add(out, self.bias)

        if self.activation is not None:
            out = self.activation(out)

        return out

# ============================================================================
# 4. CNN MODEL (Baseline)
# ============================================================================

def create_baseline_cnn(input_shape, num_classes):
    """Strong CNN baseline (EEGNet-inspired)."""

    input_layer = layers.Input(shape=input_shape)

    # Temporal convolution
    x = layers.Lambda(lambda t: tf.transpose(t, [0, 2, 1]))(input_layer)

    x = layers.Conv1D(64, 64, padding='same', activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.AveragePooling1D(4)(x)

    x = layers.Conv1D(128, 32, padding='same', activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.AveragePooling1D(8)(x)

    x = layers.Conv1D(256, 16, padding='same', activation='elu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Classifier
    x = layers.Dense(128, activation='elu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output, name='Baseline_CNN')

    return model

# ============================================================================
# 5. GCN MODEL (Graph-Enhanced)
# ============================================================================

def create_gcn_model(input_shape, num_classes, L_rescaled):
    """GCN model that MUST beat CNN."""

    input_layer = layers.Input(shape=input_shape)

    # ===== CNN PATH: Temporal features =====
    cnn_path = layers.Lambda(lambda t: tf.transpose(t, [0, 2, 1]))(input_layer)

    cnn_path = layers.Conv1D(64, 64, padding='same', activation='elu')(cnn_path)
    cnn_path = layers.BatchNormalization()(cnn_path)
    cnn_path = layers.AveragePooling1D(4)(cnn_path)

    cnn_path = layers.Conv1D(128, 32, padding='same', activation='elu')(cnn_path)
    cnn_path = layers.BatchNormalization()(cnn_path)
    cnn_path = layers.AveragePooling1D(8)(cnn_path)

    cnn_features = layers.GlobalAveragePooling1D()(cnn_path)

    # ===== GCN PATH: Channel relationships =====
    gcn_path = layers.Lambda(lambda t: tf.transpose(t, [0, 2, 1]))(input_layer)
    gcn_path = layers.AveragePooling1D(pool_size=50, strides=50)(gcn_path)
    gcn_path = layers.GlobalAveragePooling1D()(gcn_path)
    gcn_path = layers.Lambda(lambda t: tf.expand_dims(t, -1))(gcn_path)

    L_tensor = tf.constant(L_rescaled, dtype=tf.float32)

    gcn_path = ChebyshevGraphConv(32, K=2, activation='elu')([gcn_path, L_tensor])
    gcn_path = layers.BatchNormalization()(gcn_path)
    gcn_path = layers.Dropout(0.3)(gcn_path)

    gcn_path = ChebyshevGraphConv(16, K=2, activation='elu')([gcn_path, L_tensor])
    gcn_features = layers.GlobalAveragePooling1D()(gcn_path)

    # ===== FUSION =====
    combined = layers.Concatenate()([cnn_features, gcn_features])

    # Classifier
    x = layers.Dense(128, activation='elu')(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output, name='GCN_Enhanced')

    return model

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Load data
    X, y = load_bci_2a_data(n_subjects=9)

    if X is not None:
        # Prepare data
        print("\n[2] Preparing data...")

        # Normalize
        scaler = StandardScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = scaler.fit_transform(X_reshaped).reshape(X.shape)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.2, stratify=y, random_state=42
        )

        # Convert to categorical
        num_classes = len(np.unique(y))
        y_train_cat = keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, num_classes)

        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Classes: {num_classes}")

        # Compute graph structure
        print("\n[3] Computing graph structure...")
        adjacency = compute_adjacency_from_pearson(X_train)
        L_rescaled = compute_graph_laplacian(adjacency)
        print(f"  Adjacency: {adjacency.shape}, Laplacian: {L_rescaled.shape}")

        # Build models
        print("\n[4] Building models...")
        input_shape = (X_train.shape[1], X_train.shape[2])

        cnn_model = create_baseline_cnn(input_shape, num_classes)
        cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"  CNN: {cnn_model.count_params():,} parameters")

        gcn_model = create_gcn_model(input_shape, num_classes, L_rescaled)
        gcn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"  GCN: {gcn_model.count_params():,} parameters")

        # Train models
        print("\n[5] Training models...")

        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]

        print("\n  Training CNN...")
        cnn_history = cnn_model.fit(
            X_train, y_train_cat,
            validation_split=0.2,
            epochs=30,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        print("\n  Training GCN...")
        gcn_history = gcn_model.fit(
            X_train, y_train_cat,
            validation_split=0.2,
            epochs=30,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        print("\n[6] FINAL RESULTS")
        print("="*70)

        cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test_cat, verbose=0)
        gcn_loss, gcn_acc = gcn_model.evaluate(X_test, y_test_cat, verbose=0)

        print(f"\nCNN Test Accuracy: {cnn_acc*100:.2f}%")
        print(f"GCN Test Accuracy: {gcn_acc*100:.2f}%")

        improvement = ((gcn_acc - cnn_acc) / cnn_acc) * 100

        if gcn_acc > cnn_acc:
            print(f"\n SUCCESS! GCN beats CNN by {improvement:.1f}%")
        else:
            print(f"\n FAILED: GCN underperforms by {abs(improvement):.1f}%")

        print("="*70)

    else:
        print("\nERROR: Could not load data!")
