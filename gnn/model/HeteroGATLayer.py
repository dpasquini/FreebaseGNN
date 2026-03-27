import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv

class HeteroGATLayer(nn.Module):
    def __init__(self, metadata, hidden_dim, n_heads=4, edge_feature_dim=768, dropout=0.2, aggr='sum'):
        super().__init__()
        node_types, edge_types = metadata
        
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        self.convs = HeteroConv(
            {
                edge_type: GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // n_heads,
                    heads=n_heads,
                    add_self_loops=False, # Gestiamo noi i residui
                    edge_dim=edge_feature_dim
                )
                for edge_type in edge_types
            },
            aggr=aggr
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        This method performs a forward pass through the HeteroGATLayer.
        1. It first checks if edge attributes are provided; if not, it initializes them to None for all edge types.
        2. It then applies the HeteroConv layer, which computes the attention-based message passing across the heterogeneous graph. The output is a dictionary mapping node types to their updated feature matrices.
        3. Finally, it reintegrates any node types that did not receive messages (i.e., those not present in the output dictionary) by keeping their original features, 
        and applies a ReLU activation followed by dropout to the updated nodes. 
        This also includes a residual connection where the original features are added to the updated features for nodes that received messages.

        Parameters:
        - x_dict: A dictionary mapping node types to their feature matrices (shape [num_nodes, hidden_dim]).
        - edge_index_dict: A dictionary mapping edge types to their edge index tensors (shape [2, num_edges]).
        - edge_attr_dict: An optional dictionary mapping edge types to their edge attribute tensors (shape [num_edges, edge_feature_dim]).

        Returns:
        - A dictionary mapping node types to their updated feature matrices (shape [num_nodes, hidden_dim]).
        """
        if edge_attr_dict is None:
            edge_attr_dict = {et: None for et in edge_index_dict.keys()}

        # Apply the HeteroConv layer to get the updated node features for the types that received messages
        out = self.convs(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

        # Reintegrate node types that did not receive messages by keeping their original features
        # and apply activation + dropout to the updated nodes. Also add a residual connection for nodes that received messages.
        new_x_dict = {}
        for ntype, x_old in x_dict.items():
            if ntype in out:
                # Node type received messages: apply ReLU + Dropout to the updated features, then add the original features (residual connection)
                # x_new = Dropout(Relu(GAT_out)) + x_old
                updated = self.dropout(F.relu(out[ntype]))
                
                # Add residual connection only if dimensions match (they should, but we add a check just in case)
                if updated.shape == x_old.shape:
                    new_x_dict[ntype] = updated + x_old
                else:
                    new_x_dict[ntype] = updated # Fallback: if dimensions don't match, just use the updated features without residual
            else:
                # Node type did not receive messages: keep original features (this prevents KeyError and "vanishing node" issues)
                new_x_dict[ntype] = x_old

        return new_x_dict