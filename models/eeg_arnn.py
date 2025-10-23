"""
EEG-ARNN: EEG Channel Active Inference Neural Network

Implementation of the EEG-ARNN model from:
Sun et al. (2023) - Graph Convolution Neural Network Based End-to-End
Channel Selection and Classification for Motor Imagery Brain-Computer Interfaces

The model consists of:
1. TFEM (Temporal Feature Extraction Module) - CNN-based
2. CARM (Channel Active Reasoning Module) - GCN-based
3. Three TFEM-CARM blocks + one final TFEM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalFeatureExtraction(nn.Module):
    """
    Temporal Feature Extraction Module (TFEM)

    CNN-based module for extracting temporal features from EEG signals.
    Does NOT convolve along the channel dimension to preserve physiological meaning.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 pool_size=None, dropout=0.25, is_first=False, is_last=False):
        super(TemporalFeatureExtraction, self).__init__()

        self.is_last = is_last

        # Temporal convolution (only along time dimension)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),  # Only convolve time, not channels
            stride=(1, stride),
            padding=(0, kernel_size // 2),
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        # Optional pooling (for intermediate layers)
        self.pool = None
        if pool_size is not None:
            self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_size))

        # For last TFEM: channel fusion (note: channels parameter needed)
        self.n_channels = None
        if is_last:
            # Will be set dynamically based on input
            self.channel_fusion = None

    def forward(self, x):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels_in, n_eeg_channels, time)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, channels_out, n_eeg_channels, time')
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.dropout(x)

        if self.pool is not None:
            x = self.pool(x)

        if self.is_last:
            # Create channel fusion layer dynamically if not exists
            if self.channel_fusion is None:
                n_eeg_channels = x.shape[2]
                self.channel_fusion = nn.Conv2d(
                    in_channels=x.shape[1],
                    out_channels=x.shape[1],
                    kernel_size=(n_eeg_channels, 1),
                    stride=(1, 1),
                    bias=False
                ).to(x.device)
                self.n_channels = n_eeg_channels

            x = self.channel_fusion(x)

        return x


class ChannelActiveReasoning(nn.Module):
    """
    Channel Active Reasoning Module (CARM)

    GCN-based module that learns the connectivity between EEG channels dynamically.
    Uses learnable adjacency matrix that is updated during training.
    """
    def __init__(self, n_channels, time_features, dropout=0.25, rho=0.001):
        super(ChannelActiveReasoning, self).__init__()

        self.n_channels = n_channels
        self.time_features = time_features
        self.rho = rho

        # Learnable adjacency matrix (initialized to connect all channels)
        # W*_ij = 1 for i != j, 0 for i == j
        adj_init = torch.ones(n_channels, n_channels) - torch.eye(n_channels)
        self.adj_matrix = nn.Parameter(adj_init, requires_grad=True)

        # Learnable transformation matrix for temporal features
        self.theta = nn.Parameter(torch.randn(time_features, time_features))

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass with graph convolution

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, features, n_channels, time)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, features, n_channels, time)
        """
        batch_size, features, n_channels, time_points = x.shape

        # Normalize adjacency matrix: D^(-1/2) * W * D^(-1/2)
        adj = self.adj_matrix

        # Compute degree matrix
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

        # D^(-1/2)
        D_inv_sqrt = torch.diag(degree_inv_sqrt)

        # Normalized adjacency: W_hat = D^(-1/2) * W * D^(-1/2)
        adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)

        # Reshape: (batch, features, channels, time) -> (batch*features, channels, time)
        x_reshaped = x.view(batch_size * features, n_channels, time_points)

        # Apply graph convolution: H = W_hat * X
        # (channels, channels) x (batch*features, channels, time) -> (batch*features, channels, time)
        h = torch.matmul(adj_normalized, x_reshaped)

        # Apply temporal transformation if dimensions match
        if self.theta.shape[0] == time_points:
            h = torch.matmul(h, self.theta)  # (batch*features, channels, time)

        # Reshape back: (batch*features, channels, time) -> (batch, features, channels, time)
        output = h.view(batch_size, features, n_channels, time_points)

        output = self.elu(output)
        output = self.dropout(output)

        return output


class EEG_ARNN(nn.Module):
    """
    EEG-ARNN: End-to-end model for MI-EEG classification with channel selection

    Architecture:
    - Input: (batch, 1, 64, 768) for 6-second epochs at 128 Hz
    - 3x TFEM-CARM blocks
    - 1x final TFEM
    - Fully connected layer
    - Output: (batch, 2) for binary classification
    """
    def __init__(self, n_channels=64, n_timepoints=768, n_classes=2, dropout=0.25):
        super(EEG_ARNN, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes

        # TFEM 1: Extract initial temporal features
        self.tfem1 = TemporalFeatureExtraction(
            in_channels=1,
            out_channels=16,
            kernel_size=16,
            stride=1,
            pool_size=None,
            dropout=dropout,
            is_first=True
        )

        # CARM 1: Learn channel connectivity
        self.carm1 = ChannelActiveReasoning(
            n_channels=n_channels,
            time_features=n_timepoints,
            dropout=dropout
        )

        # TFEM 2: Further temporal extraction with pooling
        self.tfem2 = TemporalFeatureExtraction(
            in_channels=16,
            out_channels=16,
            kernel_size=16,
            stride=1,
            pool_size=4,
            dropout=dropout
        )

        # CARM 2
        self.carm2 = ChannelActiveReasoning(
            n_channels=n_channels,
            time_features=n_timepoints // 4,
            dropout=dropout
        )

        # TFEM 3: More temporal features with pooling
        self.tfem3 = TemporalFeatureExtraction(
            in_channels=16,
            out_channels=16,
            kernel_size=16,
            stride=1,
            pool_size=4,
            dropout=dropout
        )

        # CARM 3
        self.carm3 = ChannelActiveReasoning(
            n_channels=n_channels,
            time_features=n_timepoints // 16,
            dropout=dropout
        )

        # TFEM 4: Final temporal features and channel fusion
        self.tfem4 = TemporalFeatureExtraction(
            in_channels=16,
            out_channels=32,
            kernel_size=1,
            stride=1,
            pool_size=None,
            dropout=dropout,
            is_last=True
        )

        # Calculate flattened size after all operations
        # After TFEM4: (batch, 32, 1, time//16)
        self.flatten_size = 32 * 1 * (n_timepoints // 16)

        # Fully connected layer
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def forward(self, x):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, n_channels, n_timepoints)

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, n_classes)
        """
        # TFEM-CARM block 1
        x = self.tfem1(x)
        x = self.carm1(x)

        # TFEM-CARM block 2
        x = self.tfem2(x)
        x = self.carm2(x)

        # TFEM-CARM block 3
        x = self.tfem3(x)
        x = self.carm3(x)

        # Final TFEM (includes channel fusion)
        x = self.tfem4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc(x)

        return x

    def get_adjacency_matrices(self):
        """
        Get the learned adjacency matrices from all CARM modules

        Returns
        -------
        dict
            Dictionary with adjacency matrices from each CARM
        """
        return {
            'carm1': self.carm1.adj_matrix.detach().cpu().numpy(),
            'carm2': self.carm2.adj_matrix.detach().cpu().numpy(),
            'carm3': self.carm3.adj_matrix.detach().cpu().numpy()
        }


def edge_selection(adj_matrix, k, ch_names=None):
    """
    Edge-Selection (ES) method for channel selection

    Selects channels based on the strongest edge connections in the adjacency matrix.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix of shape (n_channels, n_channels)
    k : int
        Number of edges to select (may result in fewer than k channels)
    ch_names : list, optional
        List of channel names

    Returns
    -------
    selected_channels : list
        Indices of selected channels
    """
    n_channels = adj_matrix.shape[0]

    # Compute edge weights: delta_ij = |f_ij| + |f_ji|
    edge_weights = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            weight = np.abs(adj_matrix[i, j]) + np.abs(adj_matrix[j, i])
            edge_weights.append((weight, i, j))

    # Sort edges by weight
    edge_weights.sort(reverse=True, key=lambda x: x[0])

    # Select top k edges and collect unique channels
    selected_channels = set()
    for weight, i, j in edge_weights[:k]:
        selected_channels.add(i)
        selected_channels.add(j)

    selected_channels = sorted(list(selected_channels))

    if ch_names is not None:
        selected_names = [ch_names[i] for i in selected_channels]
        return selected_channels, selected_names

    return selected_channels


def aggregation_selection(adj_matrix, k, ch_names=None):
    """
    Aggregation-Selection (AS) method for channel selection

    Selects channels based on their aggregated connectivity importance.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix of shape (n_channels, n_channels)
    k : int
        Number of channels to select
    ch_names : list, optional
        List of channel names

    Returns
    -------
    selected_channels : list
        Indices of selected channels
    """
    n_channels = adj_matrix.shape[0]

    # Compute degree (diagonal of D matrix)
    degree = np.abs(adj_matrix).sum(axis=1)

    # Compute node importance: tau_i = sum_j |f_ij| + |d_i|
    node_importance = np.abs(adj_matrix).sum(axis=1) + np.abs(degree)

    # Select top k channels
    selected_channels = np.argsort(node_importance)[-k:][::-1]
    selected_channels = sorted(selected_channels.tolist())

    if ch_names is not None:
        selected_names = [ch_names[i] for i in selected_channels]
        return selected_channels, selected_names

    return selected_channels


if __name__ == "__main__":
    # Test the model
    print("Testing EEG-ARNN model...")

    # Create model
    model = EEG_ARNN(n_channels=64, n_timepoints=768, n_classes=2)

    # Create dummy input
    batch_size = 8
    x = torch.randn(batch_size, 1, 64, 768)

    # Forward pass
    output = model(x)

    print(f"\nModel architecture:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test channel selection
    adj_matrices = model.get_adjacency_matrices()
    print(f"\nAdjacency matrices:")
    for name, adj in adj_matrices.items():
        print(f"  {name}: {adj.shape}")

    # Test ES and AS
    ch_names = [f"Ch{i+1}" for i in range(64)]

    es_channels, es_names = edge_selection(adj_matrices['carm1'], k=20, ch_names=ch_names)
    print(f"\nEdge-Selection (top 20 edges):")
    print(f"  Selected {len(es_channels)} channels: {es_names[:10]}...")

    as_channels, as_names = aggregation_selection(adj_matrices['carm1'], k=20, ch_names=ch_names)
    print(f"\nAggregation-Selection (top 20 nodes):")
    print(f"  Selected {len(as_channels)} channels: {as_names[:10]}...")

    print("\nModel test completed successfully!")
