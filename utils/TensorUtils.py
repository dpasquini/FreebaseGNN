import torch
from typing import Union

class TensorUtils:
    """Utility class for safe operations on PyTorch tensors."""

    @staticmethod
    def to_vec_32(x: Union[torch.Tensor, list, None], embedding_dim: int = 768) -> torch.Tensor:
        """Converts input (list, tensor, None) into a standard 1D float32 tensor."""
        if x is None:
            return torch.zeros(embedding_dim, dtype=torch.float32)
        
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().to(torch.float32)
        else:
            t = torch.as_tensor(x, dtype=torch.float32).cpu()
            
        # Squeeze leading batch if present
        if t.dim() == 2 and t.size(0) == 1:
            t = t.squeeze(0)
            
        return t

    @staticmethod
    def sanitize_rel(s: str) -> str:
        """Sanitizes relation keys to be usable in HeteroData edge types."""
        if not isinstance(s, str):
            return str(s)
        return (
            s.replace("#", "_HASH_")
            .replace(".", "_DOT_")
            .replace("/", "_SLASH_")
            .replace(" ", "_")
        )