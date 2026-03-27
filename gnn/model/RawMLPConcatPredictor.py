import torch.nn as nn
import torch.nn.functional as F
import torch

class RawMLPConcatPredictor(nn.Module):
    """
    Represents a simple MLP-based link predictor that concatenates the source and destination node features 
    for each edge and passes them through an MLP to produce a score. 
    This model does not perform any message passing or graph convolution, 
    making it a "no-GNN" baseline for link prediction tasks.
    """
    def __init__(self, in_dim=768, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward_logits(self, data, et, edge_index):
        src_t, _, dst_t = et
        x_src = data[src_t].x
        x_dst = data[dst_t].x
        s = edge_index[0]
        d = edge_index[1]
        h = torch.cat([x_src[s], x_dst[d]], dim=1)  # [E, 2*in_dim]
        return self.net(h).view(-1)                 # [E]