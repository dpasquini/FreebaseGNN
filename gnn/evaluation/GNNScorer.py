
import torch
from gnn.evaluation.BaselineScorer import BaselineScorer

class GNNScorer(BaselineScorer):
    """
    GNNScorer uses the trained GNN model to compute scores for edges.
    It implements the score method by forwarding the data through the GNN and applying the predictor to get edge scores.
    """

    def __init__(self, model):
        self.model = model

    def score(self, data, et, edge_index):
        self.model.eval()
        with torch.no_grad():
            x = self.model.encode(data)
            # decode returns raw logits, so we apply sigmoid to get probabilities
            logits = self.model.decode(x, edge_index, et).view(-1)
            return torch.sigmoid(logits)