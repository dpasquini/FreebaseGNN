import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from gnn.model.HeteroGATLayer import HeteroGATLayer

class HeteroGAT(nn.Module):
    """
    A multi-layer Heterogeneous Graph Attention Network (HeteroGAT) for processing typed graphs with edge features.
    Each layer consists of a HeteroGATLayer that performs attention-based message passing across different node and edge types.
    """
    def __init__(
        self,
        metadata,
        input_dim=768,
        hidden_dim=2048,
        num_layers=2,
        n_heads=4,
        dropout_rate=0.2,
        edge_feature_dim=768,
        aggr='sum'
    ):

        super().__init__()
        node_types, _ = metadata
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.node_types = node_types
        self.hidden_dim = hidden_dim

        # Per-type input projection to hidden_dim
        self.input_proj = nn.ModuleDict({
            ntype: nn.Linear(input_dim, hidden_dim) for ntype in self.node_types
        })

        # Stack of hetero GAT layers (each keeps shape hidden_dim)
        self.gat_layers = nn.ModuleList([
            HeteroGATLayer(
                metadata=metadata,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                edge_feature_dim=edge_feature_dim,
                dropout=dropout_rate,
                aggr=aggr
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        This method performs a forward pass through the HeteroGAT model.
        1. It first projects the input features of each node type to a common hidden dimension using per-type linear layers.
        2. Then it applies a stack of HeteroGATLayer modules, which perform attention-based message passing across the heterogeneous graph.
        3. Finally, it returns a dictionary mapping each node type to its updated feature matrix of shape [num_nodes, hidden_dim].

        Parameters:
        - x_dict: A dictionary mapping node types to their feature matrices (shape [num_nodes, input_dim]).
        - edge_index_dict: A dictionary mapping edge types to their edge index tensors (shape [2, num_edges]).
        - edge_attr_dict: An optional dictionary mapping edge types to their edge attribute tensors (shape [num_edges, edge_feature_dim]).

        Returns:
        - A dictionary mapping node types to their updated feature matrices (shape [num_nodes, hidden_dim]).
        """
        
        # Project per-type inputs to hidden_dim
        h_dict = {}
        for ntype, x in x_dict.items():
            # Apply projection (ensure input tensor resides on correct device if manually passed)
            device = next(self.parameters()).device
            h = F.relu(self.input_proj[ntype](x.to(device)))
            h = self.dropout(h)
            h_dict[ntype] = h

        # Message passing
        for layer in self.gat_layers:
            h_dict = layer(h_dict, edge_index_dict, edge_attr_dict)

        # Return dict of node-type → [num_nodes, hidden_dim]
        return h_dict
