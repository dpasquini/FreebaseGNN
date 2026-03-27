from datetime import datetime
from typing import Optional

from utils import constants
import torch
from torch_geometric.utils import negative_sampling

class GraphUtils:
    """Shared utility methods for graph construction and enhancement."""

    @staticmethod
    def log_with_time(message: str) -> None:
        """Helper function to print messages with a timestamp."""
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

    @staticmethod
    def normalize_lang(lang: str) -> Optional[str]:
        """Normalize language codes to lowercase."""
        if not lang:
            return None
        return str(lang).lower()

    @staticmethod
    def is_mid(x: str) -> bool:
        """Check if a node ID is a Freebase MID/GUID."""
        return isinstance(x, str) and (x.startswith("m.") or x.startswith("g."))

    @staticmethod
    def is_meta_type(t: str) -> bool:
        """Check if a type ID is a meta-type."""
        return isinstance(t, str) and any(t.startswith(p) for p in constants.META_TYPE_PREFIXES)

    @staticmethod
    def is_meta_pred(p: str) -> bool:
        """Check if a predicate ID is a meta-predicate."""
        return isinstance(p, str) and any(p.startswith(pfx) for pfx in constants.META_PRED_PREFIXES)

    @staticmethod
    def get_pos_edges_from_split(split_data, et):
        """
        Helper function to extract the positive edge indices for a given edge type from a split graph.
        """
        store = split_data[et]
        if (not hasattr(store, "edge_label_index")) or (not hasattr(store, "edge_label")): return None
        ei, y = store.edge_label_index, store.edge_label
        pos_mask = (y > 0)
        return ei[:, pos_mask] if pos_mask.sum() > 0 else None

    @staticmethod
    def sample_negatives_excluding_positives(full_pos_edge_index, num_src, num_dst, target_neg, same_type, device=None, method="sparse"):
        """
        Samples negative edges while ensuring that they do not overlap with the provided positive edge indices.
        """
        if target_neg <= 0: return torch.empty((2, 0), dtype=torch.long, device=device)
        req = int(target_neg * 1.3) + 10
        neg = negative_sampling(full_pos_edge_index, num_nodes=(num_src, num_dst), num_neg_samples=req, method=method)
        if device: neg = neg.to(device)
        if same_type and neg.numel() > 0:
            mask = neg[0] != neg[1]
            neg = neg[:, mask]
        if neg.size(1) > target_neg: neg = neg[:, :target_neg]
        return neg