import json
import torch
from pathlib import Path
from torch_geometric.data import HeteroData
from typing import Union, Optional, Dict

class HeteroDataIO:
    """
    Handles saving and loading of HeteroData structures and configurations.
    """

    def load_model(self, model, model_path: Union[str, Path], map_location: str | torch.device = None) -> torch.nn.Module:
        """Load a state dict into an instantiated model and return the model."""
        state_dict = torch.load(model_path, map_location=map_location, weights_only=False)
        model.load_state_dict(state_dict)
        return model
    
    def load_metadata(self, meta_path: Union[str, Path]) -> Dict:
        meta_path = Path(meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_heterodata(self, base_path: Union[str, Path], graph_id: str, file_pattern: str, map_location: str | torch.device = "cpu") -> HeteroData:
        """
        Loads a HeteroData object from the local filesystem based on the provided graph ID.
        
        Parameters:
            - base_path: Base directory where the graph files are stored.
            - graph_id: Unique identifier for the graph, used in filenames.
            - file_pattern: Pattern for the file names. Should include '{graph_id}' as a placeholder for the graph ID.
            - map_location: The device to map the loaded tensors to.
        """
        base_path = Path(base_path)
        formatted_filename = file_pattern.format(graph_id=graph_id)
        if not formatted_filename.endswith('.pt'):
            formatted_filename += '.pt'
        pt_path = base_path / formatted_filename
        return torch.load(pt_path, map_location=map_location, weights_only=False)

    @staticmethod
    def save_heterodata(
        data: HeteroData, 
        out_base: Union[str, Path], 
        graph_id: str, 
        save_node_ids: bool = True, 
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Saves a HeteroData object and its metadata to the local filesystem.
        
        Parameters:
            - data: The HeteroData object to save.
            - out_base: Base directory where the graph files will be stored.
            - graph_id: Unique identifier for the graph, used in filenames.
            - save_node_ids: Whether to save the mapping of node IDs to their original identifiers.
            - metadata: Optional dictionary of additional metadata to save alongside the graph.

        """
        out_base = Path(out_base)
        out_base.mkdir(parents=True, exist_ok=True)
        
        pt_path = out_base / f"hetero_graph.{graph_id}.pt"
        meta_path = out_base / f"hetero_graph.{graph_id}.meta.json"
        nid_path = out_base / f"hetero_graph.{graph_id}.nid.json"

        # Explicitly enforce moving to CPU before serialization
        data_cpu = data.to("cpu")
        torch.save(data_cpu, pt_path)

        if metadata:
            meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

        if save_node_ids:
            nid_dict = {}
            for ntype in data_cpu.node_types:
                if hasattr(data_cpu[ntype], "nid"):
                    nid_dict[ntype] = list(map(str, data_cpu[ntype].nid))
            nid_path.write_text(json.dumps(nid_dict, ensure_ascii=False))
