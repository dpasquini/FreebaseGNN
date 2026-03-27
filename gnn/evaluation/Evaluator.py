

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.GraphUtils import GraphUtils

class Evaluator:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def metrics_from_scores(self, y_true, y_score, thr):
        """
        Converts predicted scores into binary classifications using the specified threshold 
        and computes evaluation metrics (accuracy, precision, recall, F1-score) for link prediction tasks.

        Parameters:
        - y_true: The ground truth binary labels (1 for positive edges, 0 for negative edges).
        - y_score: The predicted scores (e.g., probabilities) for the edges.
        - thr: The threshold to convert predicted scores into binary classifications (edges are classified as positive if their score is greater than or equal to the threshold, and negative otherwise).
        Returns:
        - A tuple containing the computed metrics: (accuracy, precision, recall, F1-score).
        """
        y_pred = (y_score >= thr).astype(int)
        return (
            (y_pred == y_true).mean(),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0)
        )

    def _collect_scores(self, packs, edge_types, neg_ratio, scorer, split="test", return_per_graph=False):
        """
        Iterates over packs, scores all edges, and returns concatenated labels and scores.
        """
        y_true_all, y_score_all = [], []
        graph_trues, graph_scores = [], [] # Used for macro metrics if return_per_graph is True

        for pack in packs:
            tr_data, va_data, te_data, full_pos = pack
            data = {"train": tr_data, "val": va_data, "test": te_data}[split]
            if self.device:
                data = data.to(self.device)

            gyt, gys = [], []

            for et in edge_types:
                if et not in data.edge_types or et not in full_pos:
                    continue

                pos_ei = GraphUtils.get_pos_edges_from_split(data, et)
                if pos_ei is None:
                    continue
                if self.device:
                    pos_ei = pos_ei.to(self.device)

                num_src = data[et[0]].num_nodes
                num_dst = data[et[2]].num_nodes
                max_neg = max(0, num_src * num_dst - full_pos[et].size(1))
                target_neg = min(int(pos_ei.size(1) * neg_ratio), max_neg)
                if target_neg <= 0:
                    continue

                neg_ei = GraphUtils.sample_negatives_excluding_positives(
                    full_pos[et].to(self.device) if self.device else full_pos[et],
                    num_src,
                    num_dst,
                    target_neg,
                    (et[0] == et[2]),
                    self.device,
                    "sparse",
                )
                if neg_ei.size(1) == 0:
                    continue

                edge_index = torch.cat([pos_ei, neg_ei], dim=1)
                scores = scorer.score(data, et, edge_index).view(-1)
                y = torch.cat(
                    [
                        torch.ones(pos_ei.size(1), device=self.device if self.device else None),
                        torch.zeros(neg_ei.size(1), device=self.device if self.device else None),
                    ],
                    dim=0,
                )
                lbl_cpu = y.cpu().numpy()
                sc_cpu = scores.cpu().numpy()
                
                y_true_all.append(lbl_cpu)
                y_score_all.append(sc_cpu)
                gyt.append(lbl_cpu)
                gys.append(sc_cpu)
                
            if return_per_graph and gyt:
                graph_trues.append(np.concatenate(gyt))
                graph_scores.append(np.concatenate(gys))

        if not y_true_all:
            if return_per_graph:
                return None, None, [], []
            return None, None
            
        if return_per_graph:
            return (
                np.concatenate(y_true_all).astype(int),
                np.concatenate(y_score_all).astype(float),
                graph_trues,
                graph_scores
            )
            
        return (
            np.concatenate(y_true_all).astype(int),
            np.concatenate(y_score_all).astype(float),
        )

    def find_best_threshold(self, val_packs, edge_types, neg_ratio, scorer):
        """Finds the threshold maximizing F1 on the validation set."""
        y, p = self._collect_scores(val_packs, edge_types, neg_ratio, scorer, split="val")
        if y is None:
            return 0.5

        thrs = np.unique(p)
        best_t, best_f = 0.5, -1.0
        step = max(1, len(thrs) // 500)
        
        for t in thrs[::step]:
            _, _, _, f = self.metrics_from_scores(y, p, t)
            if f > best_f:
                best_f, best_t = f, float(t)
        return best_t

    def evaluate_metrics_master(self, name, packs, edge_types, neg_ratio, scorer, threshold=0.5, split="test"):
        """
        Evaluate link prediction performance with comprehensive metrics:
        1. Micro Metrics: Global F1, Precision, Recall, Accuracy across all edge types and graphs.
        2. Macro Metrics: Average F1, Precision, Recall, Accuracy per graph (treating each graph equally).
        """
        y, p, graph_trues, graph_scores = self._collect_scores(
            packs, edge_types, neg_ratio, scorer, split=split, return_per_graph=True
        )
        if y is None:
            print(f"[{name} - Evaluation with threshold {threshold:.4f}] No edges were evaluated. Check if edge types and data splits are correct.")
            return

        # Micro global metrics
        acc, pr, re, f1 = self.metrics_from_scores(y, p, threshold)
        print(f"[{name} - Evaluation with threshold {threshold:.4f}] 1:{neg_ratio} Thr={threshold:.4f} MICRO: F1={f1:.4f} P={pr:.4f} R={re:.4f} Acc={acc:.4f}")

        # Macro graph-level metrics
        if graph_trues:
            graph_stats = {'f1': [], 'prec': [], 'rec': [], 'acc': []}
            for g_yt, g_ys in zip(graph_trues, graph_scores):
                g_acc, g_pr, g_re, g_f1 = self.metrics_from_scores(g_yt, g_ys, threshold)
                graph_stats['f1'].append(g_f1)
                graph_stats['prec'].append(g_pr)
                graph_stats['rec'].append(g_re)
                graph_stats['acc'].append(g_acc)
                
            if graph_stats['f1']:
                print(f"  MACRO GRAPH (Mean ± Std):")
                for m in ['f1', 'prec', 'rec', 'acc']:
                    mean_val = np.mean(graph_stats[m])
                    std_val = np.std(graph_stats[m])
                    print(f"    - {m.upper():<4}: {mean_val:.4f} ± {std_val:.4f}")