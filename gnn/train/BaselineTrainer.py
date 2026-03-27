import torch
import random
from utils.GraphUtils import GraphUtils

class BaselineTrainer:
    """
    Trains a baseline model (e.g., RawMLPConcatPredictor) on link prediction task.
    The model must implement a forward_logits(data, et, edge_index) method.
    """

    def __init__(self, model, edge_types, lr=1e-3, pos_weight=1.0):
        self.model = model
        self.edge_types = edge_types
        self.lr = lr
        self.pos_weight = pos_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_packs, epochs=20, neg_ratio=19):

        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], device=self.device)
        )

        for ep in range(1, epochs + 1):
            self.model.train()
            total_loss, used = 0.0, 0
            random.shuffle(train_packs)

            for (tr_data, _, _, full_pos) in train_packs:
                tr_data = tr_data.to(self.device)
                opt.zero_grad()
                loss_sum, n_terms = 0.0, 0

                for et in self.edge_types:
                    if et not in tr_data.edge_types or et not in full_pos:
                        continue

                    pos_edge_index = GraphUtils.get_pos_edges_from_split(tr_data, et)
                    if pos_edge_index is None:
                        continue
                    pos_edge_index = pos_edge_index.to(self.device)
                    num_pos = pos_edge_index.size(1)
                    if num_pos == 0:
                        continue

                    num_src = tr_data[et[0]].num_nodes
                    num_dst = tr_data[et[2]].num_nodes
                    max_neg = max(0, num_src * num_dst - full_pos[et].size(1))
                    target_neg = min(int(num_pos * neg_ratio), max_neg)
                    if target_neg <= 0:
                        continue

                    neg_edge_index = GraphUtils.sample_negatives_excluding_positives(
                        full_pos[et].to(self.device),
                        num_src,
                        num_dst,
                        target_neg,
                        (et[0] == et[2]),
                        self.device,
                        "sparse",
                    )
                    if neg_edge_index.size(1) == 0:
                        continue

                    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                    y = torch.cat(
                        [
                            torch.ones(num_pos, device=self.device),
                            torch.zeros(neg_edge_index.size(1), device=self.device),
                        ],
                        dim=0,
                    )
                    logits = self.model.forward_logits(tr_data, et, edge_index)
                    loss_sum += criterion(logits, y)
                    n_terms += 1

                if n_terms > 0:
                    loss_sum.backward()
                    opt.step()
                    total_loss += float(loss_sum.item())
                    used += 1

            print(
                f"[{self.model.__class__.__name__}] epoch {ep:03d} "
                f"loss={total_loss / max(1, used):.4f}"
            )

        return self.model
