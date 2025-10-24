"""
EEG-ARNN Model Architecture
Implements the 3-fold TFEM-CARM architecture from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TFEM(nn.Module):
    """
    Temporal Feature Extraction Module
    Uses 1D temporal convolution to extract time-domain features
    """
    def __init__(self, in_channels, out_channels, kernel_size=16, pool_size=2, use_pool=True):
        super(TFEM, self).__init__()
        self.use_pool = use_pool

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()

        if use_pool:
            self.pool = nn.AvgPool2d(kernel_size=(1, pool_size))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        if self.use_pool:
            x = self.pool(x)

        return x


class CARM(nn.Module):
    """
    Channel Active Reasoning Module
    Learns channel connectivity via graph convolution
    """
    def __init__(self, num_channels, hidden_dim=40):
        super(CARM, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        self.W = nn.Parameter(torch.FloatTensor(num_channels, num_channels))
        nn.init.xavier_uniform_(self.W)

        self.theta = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(hidden_dim)
        self.activation = nn.ELU()

    def forward(self, x):
        batch_size, channels, num_channels, time_steps = x.size()

        A = torch.sigmoid(self.W)
        A_sym = (A + A.t()) / 2

        x_reshaped = x.permute(0, 2, 1, 3)
        x_reshaped = x_reshaped.contiguous().view(batch_size * num_channels, channels, time_steps)
        x_reshaped = x_reshaped.unsqueeze(2)

        x_graph = torch.matmul(A_sym, x_reshaped.view(batch_size, num_channels, -1))
        x_graph = x_graph.view(batch_size, num_channels, channels, time_steps)
        x_graph = x_graph.permute(0, 2, 1, 3)

        theta_x = self.theta(x)
        phi_x = self.phi(x_graph)

        out = theta_x + phi_x
        out = self.bn(out)
        out = self.activation(out)

        return out, A_sym

    def get_adjacency_matrix(self):
        """Get the learned adjacency matrix"""
        with torch.no_grad():
            A = torch.sigmoid(self.W)
            A_sym = (A + A.t()) / 2
        return A_sym.cpu().numpy()


class EEGARNN(nn.Module):
    """
    Complete EEG-ARNN Architecture
    3-fold TFEM-CARM with final classification layer
    """
    def __init__(self, num_channels=64, num_timepoints=512, num_classes=4, hidden_dim=40):
        super(EEGARNN, self).__init__()

        self.num_channels = num_channels
        self.num_timepoints = num_timepoints
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.tfem1 = TFEM(in_channels=1, out_channels=hidden_dim, kernel_size=16, use_pool=False)
        self.carm1 = CARM(num_channels=num_channels, hidden_dim=hidden_dim)

        self.tfem2 = TFEM(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=16, pool_size=2, use_pool=True)
        self.carm2 = CARM(num_channels=num_channels, hidden_dim=hidden_dim)

        self.tfem3 = TFEM(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=16, pool_size=2, use_pool=True)
        self.carm3 = CARM(num_channels=num_channels, hidden_dim=hidden_dim)

        test_input = torch.zeros(1, 1, num_channels, num_timepoints)
        with torch.no_grad():
            out = self._forward_features(test_input)
            flattened_size = out.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

        self.adjacency_matrices = {}

    def _forward_features(self, x):
        x = self.tfem1(x)
        x, A1 = self.carm1(x)

        x = self.tfem2(x)
        x, A2 = self.carm2(x)

        x = self.tfem3(x)
        x, A3 = self.carm3(x)

        self.adjacency_matrices = {'A1': A1, 'A2': A2, 'A3': A3}

        return x

    def forward(self, x):
        x = self._forward_features(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_final_adjacency_matrix(self):
        """Get the final learned adjacency matrix from CARM3"""
        return self.carm3.get_adjacency_matrix()

    def get_all_adjacency_matrices(self):
        """Get all adjacency matrices from all CARM layers"""
        return {
            'A1': self.carm1.get_adjacency_matrix(),
            'A2': self.carm2.get_adjacency_matrix(),
            'A3': self.carm3.get_adjacency_matrix()
        }


class ChannelSelector:
    """
    Channel selection using learned adjacency matrix
    Implements both Edge Selection (ES) and Aggregation Selection (AS)
    """
    def __init__(self, adjacency_matrix, channel_names):
        """
        Parameters
        ----------
        adjacency_matrix : np.ndarray
            Learned adjacency matrix (num_channels, num_channels)
        channel_names : list
            List of channel names
        """
        self.adj_matrix = adjacency_matrix
        self.channel_names = np.array(channel_names)
        self.num_channels = len(channel_names)

    def edge_selection(self, k):
        """
        Edge Selection (ES) method
        Selects channels based on strongest pairwise connections

        Parameters
        ----------
        k : int
            Number of edges to select

        Returns
        -------
        selected_channels : list
            List of selected channel names
        selected_indices : np.ndarray
            Indices of selected channels
        """
        edges = []
        for i in range(self.num_channels):
            for j in range(i+1, self.num_channels):
                edge_importance = abs(self.adj_matrix[i, j]) + abs(self.adj_matrix[j, i])
                edges.append((i, j, edge_importance))

        sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)

        top_k_edges = sorted_edges[:k]

        selected_indices = set()
        for i, j, _ in top_k_edges:
            selected_indices.add(i)
            selected_indices.add(j)

        selected_indices = np.array(sorted(selected_indices))
        selected_channels = self.channel_names[selected_indices].tolist()

        return selected_channels, selected_indices

    def aggregation_selection(self, k):
        """
        Aggregation Selection (AS) method
        Selects channels based on aggregate connection strength

        Parameters
        ----------
        k : int
            Number of channels to select

        Returns
        -------
        selected_channels : list
            List of selected channel names
        selected_indices : np.ndarray
            Indices of selected channels
        """
        channel_scores = np.sum(np.abs(self.adj_matrix), axis=1)

        selected_indices = np.argsort(channel_scores)[-k:]
        selected_indices = np.sort(selected_indices)

        selected_channels = self.channel_names[selected_indices].tolist()

        return selected_channels, selected_indices

    def visualize_adjacency(self, save_path=None):
        """Visualize the adjacency matrix"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            self.adj_matrix,
            xticklabels=self.channel_names,
            yticklabels=self.channel_names,
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Connection Strength'},
            ax=ax
        )
        ax.set_title('Learned Channel Adjacency Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
