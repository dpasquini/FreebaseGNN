from gnn.evaluation.BaselineScorer import BaselineScorer
import torch.nn.functional as F

class CosineBaselineScorer(BaselineScorer):
    """
    Stateless cosine similarity scorer.
    Score = 0.5 * (cosine(src, dst) + 1), mapped to [0, 1].
    """

    def score(self, data, et, edge_index):
        src_t, _, dst_t = et
        x_src = data[src_t].x.float()
        x_dst = data[dst_t].x.float()
        s, d = edge_index[0], edge_index[1]
        cos = F.cosine_similarity(x_src[s], x_dst[d], dim=1)
        return 0.5 * (cos + 1.0)