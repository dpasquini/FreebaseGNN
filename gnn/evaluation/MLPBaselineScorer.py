import torch
from gnn.evaluation.BaselineScorer import BaselineScorer

class MLPBaselineScorer(BaselineScorer):
    """
    Wraps a trained MLP (e.g. RawMLPConcatPredictor) to produce probabilities
    compatible with the Evaluator interface.
    """

    def __init__(self, model):
        self.model = model

    def score(self, data, et, edge_index):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward_logits(data, et, edge_index)
            return torch.sigmoid(logits)
