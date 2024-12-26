import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGCN(nn.Module):
    def __init__(self, n_variables, sequence_length, feature_dim, gcn_hidden_dim, temporal_hidden_dim):
        super(TemporalGCN, self).__init__()
        self.n_variables = n_variables
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

        # GCN for spatial relationships
        self.gcn1 = nn.Linear(feature_dim, gcn_hidden_dim)
        self.gcn2 = nn.Linear(gcn_hidden_dim, feature_dim)

        # Temporal encoding with GRU
        self.temporal_gru = nn.GRU(feature_dim, temporal_hidden_dim, batch_first=True)

        # Final linear layer
        self.fc = nn.Linear(temporal_hidden_dim, feature_dim)

    def forward(self, x, adj_matrix):
        """
        Args:
            x: Input of shape (batch_size, n_variables, sequence_length, feature_dim)
            adj_matrix: Adjacency matrix of shape (n_variables, n_variables)
        """
        batch_size = x.size(0)

        # Reshape to (batch_size * sequence_length, n_variables, feature_dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.sequence_length, self.n_variables, self.feature_dim)

        # Apply GCN for spatial relationships
        x = torch.matmul(adj_matrix, x)  # Shape: (batch_size * sequence_length, n_variables, feature_dim)
        x = F.relu(self.gcn1(x))  # First GCN layer
        x = self.gcn2(x)  # Second GCN layer

        # Reshape back to (batch_size, sequence_length, n_variables, feature_dim)
        x = x.view(batch_size, self.sequence_length, self.n_variables, self.feature_dim)

        # Temporal relationships
        x = x.permute(0, 2, 1, 3).reshape(batch_size * self.n_variables, self.sequence_length, self.feature_dim)
        _, temporal_features = self.temporal_gru(x)  # Output shape: (1, batch_size * n_variables, temporal_hidden_dim)
        temporal_features = temporal_features.squeeze(0)  # Shape: (batch_size * n_variables, temporal_hidden_dim)

        # Reshape to (batch_size, n_variables, temporal_hidden_dim)
        temporal_features = temporal_features.view(batch_size, self.n_variables, -1)

        # Final transformation
        output = self.fc(temporal_features)  # Shape: (batch_size, n_variables, feature_dim)
        return output


# Example usage
if __name__ == "__main__":
    batch_size = 32
    n_variables = 5
    sequence_length = 10
    feature_dim = 16
    gcn_hidden_dim = 32
    temporal_hidden_dim = 64

    # Input tensor: (batch_size, n_variables, sequence_length, feature_dim)
    x = torch.rand(batch_size, n_variables, sequence_length, feature_dim)

    # Adjacency matrix: (n_variables, n_variables)
    adj_matrix = torch.rand(n_variables, n_variables)

    # Model
    model = TemporalGCN(n_variables, sequence_length, feature_dim, gcn_hidden_dim, temporal_hidden_dim)
    output = model(x, adj_matrix)

    print("Output shape:", output.shape)  # Expected: (batch_size, n_variables, feature_dim)
