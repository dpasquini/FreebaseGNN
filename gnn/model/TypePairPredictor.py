import itertools
import torch
import torch.nn as nn
from typing import Dict, Tuple, Iterable
from torch_geometric.data import HeteroData

EdgeType = Tuple[str, str, str]  # (src_ntype, rel, dst_ntype)

class TypePairPredictor(nn.Module):
    """
    A link predictor that creates a separate MLP decoder for each pair of node types (src_type, dst_type).
    This is a "safety mode" predictor that ensures we have a decoder for any possible edge type, 
    preventing KeyErrors during decoding.
    The decoders are stored in a ModuleDict with keys formatted as "SrcType___DstType". 
    During decoding, we look up the appropriate decoder based on the source and destination node types of the edge being scored.
    """
    def __init__(self, gnn: nn.Module, hidden_dim: int, edge_types: Iterable[EdgeType]):
        super().__init__()
        self.gnn = gnn
        self.hidden_dim = hidden_dim

        node_types = sorted({src for src, _, _ in edge_types}.union({dst for _, _, dst in edge_types}))
        
        # MLP Dictionary: Key is "SrcType___DstType"
        self.type_pair_mlps = nn.ModuleDict()
        
        # Instead of looking at existing edges, 
        # we create decoders for all possible combinations of node types (Cartesian Product).
        # If we have 2 types, we create 4 decoders. If we have 3, we create 9.
        # Prevents any future KeyError.
        all_pairs = list(itertools.product(node_types, node_types))
        
        print(f"[TypePairPredictor] Safety Mode: Creating {len(all_pairs)} decoders for all possible pairs.")
        
        for src, dst in all_pairs:
            key = self._type_pair_key(src, dst)
            self.type_pair_mlps[key] = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _type_pair_key(self, src_type: str, dst_type: str) -> str:
        return f"{src_type}___{dst_type}"

    def _get_pair_mlp(self, src_type: str, dst_type: str) -> nn.Module:
        key = self._type_pair_key(src_type, dst_type)
        if key not in self.type_pair_mlps:
            raise KeyError(f"No decoder localized for type pair: ({src_type}, {dst_type})")
        return self.type_pair_mlps[key]

    def _collect_edge_inputs(self, data: HeteroData):
        dev = self.device
        x_dict = {nt: x.to(device=dev, dtype=torch.float32) for nt, x in data.x_dict.items()}
        edge_index_dict = {et: ei.to(dev) for et, ei in data.edge_index_dict.items()}
        edge_attr_dict = {}
        for et in data.edge_types:
            edge_attr_dict[et] = data[et].edge_attr.to(device=dev, dtype=torch.float32) \
                                 if hasattr(data[et], "edge_attr") else None
        return x_dict, edge_index_dict, edge_attr_dict

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        data = data.to(self.device)
        x_dict, edge_index_dict, edge_attr_dict = self._collect_edge_inputs(data)
        return self.gnn(x_dict, edge_index_dict, edge_attr_dict)

    def decode(self, node_embeddings: Dict[str, torch.Tensor], edge_label_index: torch.Tensor, edge_type: EdgeType) -> torch.Tensor:
        src_type, _, dst_type = edge_type
        decoder = self._get_pair_mlp(src_type, dst_type)

        device = node_embeddings[src_type].device
        edge_label_index = edge_label_index.to(device)
        x_src = node_embeddings[src_type]
        x_dst = node_embeddings[dst_type]

        src_indices = edge_label_index[0]
        dst_indices = edge_label_index[1]

        src_feats = x_src[src_indices]
        dst_feats = x_dst[dst_indices]
        
        cat_feats = torch.cat([src_feats, dst_feats], dim=1)
        return decoder(cat_feats).view(-1)

    def forward(self, data: HeteroData, edge_label_index: torch.Tensor, edge_type: EdgeType) -> torch.Tensor:
        node_embeddings = self.encode(data)
        return self.decode(node_embeddings, edge_label_index, edge_type)