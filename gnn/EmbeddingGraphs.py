import os
import json
import torch
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
from glob import glob
from collections import defaultdict
from torch_geometric.data import HeteroData
from utils.TextToEncode import TextToEncode
from utils.TensorUtils import TensorUtils
from utils.HeteroDataIO import HeteroDataIO
from utils.IOOperations import IOOperations

class EmbeddingGraphs:
    """
    Orchestrates generation, translation, and persistence of Text/Graph data into HeteroData blocks.
    """
    
    def __init__(
        self, 
        input_graph_path: Union[str, Path], 
        output_graph_path: Union[str, Path], 
        file_pattern_nodes: str = "full_graph.nodes.{graph_id}.parquet", 
        file_pattern_edges: str = "full_graph.edges.{graph_id}.parquet",
        strategy: str = "classic"
    ):
        self.input_graph_path = Path(input_graph_path)
        self.output_graph_path = Path(output_graph_path)
        self.output_graph_path.mkdir(parents=True, exist_ok=True)
        self.file_pattern_nodes = file_pattern_nodes
        self.file_pattern_edges = file_pattern_edges
        self.text_to_encode = TextToEncode()
        self.strategy = strategy
    
    def convert_dfs_to_heterodata_singleton(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        question_embedding: torch.Tensor = None,
        *,
        graph_id: str = "graph",
        embedding_dim: int = 768,
        sanitize_keys: bool = True,
    ) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
        """
        Converts embedded Pandas DataFrames into PyTorch Geometric HeteroData structures,
        inferring the node type directly from the first element of the 'types' array feature.

        Parameters:
            - nodes_df: DataFrame containing node information with columns ['node', 'description', 'types', 'embedding'].
            - edges_df: DataFrame containing edge information with columns ['source', 'target', 'key', 'description', 'embedding'].
            - graph_id: Unique identifier for the graph, used in metadata.
            - embedding_dim: Expected dimensionality of the embedding vectors.
            - sanitize_keys: Whether to sanitize edge keys for compatibility with HeteroData edge types.
            - question_embedding: Optional tensor containing the question embedding. If provided, it will be saved alongside the graph embeddings for reference.
        """
        used_nodes = set(edges_df["source"].astype(str)).union(edges_df["target"].astype(str))
        nodes = nodes_df[nodes_df["node"].astype(str).isin(used_nodes)].copy()
        
        data = HeteroData()
        data.graph_id = graph_id
        if question_embedding is not None:
            data.question_embedding = TensorUtils.to_vec_32(question_embedding, embedding_dim)
        else:
            data.question_embedding = None

        def extract_ntype(r) -> str:
            types_val = r.get("types", None)
            if isinstance(types_val, list) and len(types_val) > 0:
                base_type = str(types_val[0])
            elif pd.notna(types_val) and not isinstance(types_val, list):
                # Handle cases where it might just be a string/scalar
                base_type = str(types_val).strip("[]'\"").split(',')[0].strip()
            else:
                base_type = "ntype_entity"

            sanitized_type = f"ntype_{TensorUtils.sanitize_rel(base_type)}"
            return sanitized_type

        # Processing nodes
        # For each node, determine its type from the 'types' column and group accordingly
        node_rows_by_type = defaultdict(list)
        for _, r in nodes.iterrows():
            ntype = extract_ntype(r)
            nid = str(r["node"])
            emb = TensorUtils.to_vec_32(r.get("embedding", None), embedding_dim)
            node_rows_by_type[ntype].append((nid, emb))

        # Create node features and ID mappings for each node type
        node_id_to_local = {}
        for ntype, lst in node_rows_by_type.items():
            # Unzip the list of tuples into separate lists for IDs and features
            ids = [nid for nid, _ in lst]
            feats = [emb for _, emb in lst]
            x = torch.stack(feats, dim=0) if feats else torch.zeros((0, embedding_dim), dtype=torch.float32)
            data[ntype].x = x
            data[ntype].nid = ids
            # Create a mapping from global node ID to local index for this node type
            node_id_to_local[ntype] = {nid: i for i, nid in enumerate(ids)}

        # Update lookup cache logic
        # This mapping allows us to quickly determine the node type for any given node ID when processing edges
        nid2type = {str(r["node"]): extract_ntype(r) for _, r in nodes.iterrows()}

        # 2. Processing edges
        edges = edges_df.copy()
        if sanitize_keys:
            edges["rel"] = edges["key"].astype(str).apply(TensorUtils.sanitize_rel)
        else:
            edges["rel"] = edges["key"].astype(str)

        edge_index_buckets = defaultdict(list)
        edge_attr_buckets = defaultdict(list)

        # Iterate over edges and bucket them by (source_type, relation, target_type)
        for _, e in edges.iterrows():
            s, t = str(e["source"]), str(e["target"])
            
            if s not in nid2type or t not in nid2type:
                continue
                
            st, tt = nid2type[s], nid2type[t]
            rel = e["rel"]
            et = (st, rel, tt)

            s_map = node_id_to_local.get(st, {})
            t_map = node_id_to_local.get(tt, {})
            
            if s not in s_map or t not in t_map:
                continue

            edge_index_buckets[et].append((s_map[s], t_map[t]))
            edge_attr_buckets[et].append(TensorUtils.to_vec_32(e.get("embedding", None), embedding_dim))

        # After bucketing edges, we can create the edge_index and edge_attr tensors for each edge type
        for et, pairs in edge_index_buckets.items():
            if not pairs:
                continue
                
            st, rel, tt = et
            src_idx, dst_idx = zip(*pairs)
            data[(st, rel, tt)].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            
            attrs = edge_attr_buckets[et]
            data[(st, rel, tt)].edge_attr = torch.stack(attrs, dim=0) if attrs else \
                torch.zeros((len(pairs), embedding_dim), dtype=torch.float32)

        return data, node_id_to_local

    def convert_dfs_to_heterodata_multihot(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        question_embedding: torch.Tensor = None,
        *,
        graph_id: str = "graph",
        embedding_dim: int = 768,
        sanitize_keys: bool = True,
        type_vocab: Optional[Dict[str, int]] = None
    ) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
        """
        Converts DataFrames into PyTorch Geometric HeteroData fusing textual embeddings
        and a multi-hot categorical encoding representing a node's full list of types.

        Parameters:
            - type_vocab: Optionally pass a global vocabulary mapping for consistent multihot dimensionality across graphs.
                          If None, it infers the vocabulary local to the current `nodes_df`.
            - question_embedding: Optional tensor containing the question embedding. This can be used to condition the node embeddings or saved alongside for reference.
        """
        used_nodes = set(edges_df["source"].astype(str)).union(edges_df["target"].astype(str))
        nodes = nodes_df[nodes_df["node"].astype(str).isin(used_nodes)].copy()
        
        data = HeteroData()
        data.graph_id = graph_id
        if question_embedding is not None:
            data.question_embedding = TensorUtils.to_vec_32(question_embedding, embedding_dim)
        else:
            data.question_embedding = None

        def parse_types(t_val) -> List[str]:
            if isinstance(t_val, list):
                return [str(v).strip() for v in t_val]
            if pd.notna(t_val):
                return [v.strip().strip("'\"") for v in str(t_val).strip("[]").split(',') if v.strip()]
            return []

        # Infer local vocabulary if a central one wasn't passed
        if type_vocab is None:
            all_types = set()
            for t_val in nodes["types"]:
                all_types.update(parse_types(t_val))
            type_vocab = {t: i for i, t in enumerate(sorted(list(all_types)))}
            
        num_types = len(type_vocab)
        
        # 1. Processing nodes
        node_rows_by_type = defaultdict(list)
        for _, r in nodes.iterrows():
            # Since nodes have overlapping types, safely generalize as Entity vs Sub-types locally
            ntype = "ntype_type" if bool(r.get("is_type", False)) else "ntype_entity"
            nid = str(r["node"])
            t_list = parse_types(r.get("types", None))
            
            text_emb = TensorUtils.to_vec_32(r.get("embedding", None), embedding_dim)
            
            # Generate multihot vector dimensions array
            multihot_emb = torch.zeros(num_types, dtype=torch.float32)
            for t in t_list:
                if t in type_vocab:
                    multihot_emb[type_vocab[t]] = 1.0
                    
            # Concat Base 768-D array with Sparse K-D array to ensure dimension is equal globally per iteration
            final_emb = torch.cat([text_emb, multihot_emb], dim=0)
            node_rows_by_type[ntype].append((nid, final_emb))

        node_id_to_local = {}
        target_dim = embedding_dim + num_types
        for ntype, lst in node_rows_by_type.items():
            ids = [nid for nid, _ in lst]
            feats = [emb for _, emb in lst]
            x = torch.stack(feats, dim=0) if feats else torch.zeros((0, target_dim), dtype=torch.float32)
            data[ntype].x = x
            data[ntype].nid = ids
            node_id_to_local[ntype] = {nid: i for i, nid in enumerate(ids)}

        nid2type = {str(r["node"]): ("ntype_type" if bool(r.get("is_type", False)) else "ntype_entity") for _, r in nodes.iterrows()}

        # 2. Processing edges 
        edges = edges_df.copy()
        if sanitize_keys:
            edges["rel"] = edges["key"].astype(str).apply(TensorUtils.sanitize_rel)
        else:
            edges["rel"] = edges["key"].astype(str)

        edge_index_buckets = defaultdict(list)
        edge_attr_buckets = defaultdict(list)

        for _, e in edges.iterrows():
            s, t = str(e["source"]), str(e["target"])
            
            if s not in nid2type or t not in nid2type:
                continue
                
            st, tt = nid2type[s], nid2type[t]
            rel = e["rel"]
            et = (st, rel, tt)

            s_map = node_id_to_local.get(st, {})
            t_map = node_id_to_local.get(tt, {})
            
            if s not in s_map or t not in t_map:
                continue

            edge_index_buckets[et].append((s_map[s], t_map[t]))
            edge_attr_buckets[et].append(TensorUtils.to_vec_32(e.get("embedding", None), embedding_dim))

        for et, pairs in edge_index_buckets.items():
            if not pairs:
                continue
                
            st, rel, tt = et
            src_idx, dst_idx = zip(*pairs)
            data[(st, rel, tt)].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            
            attrs = edge_attr_buckets[et]
            data[(st, rel, tt)].edge_attr = torch.stack(attrs, dim=0) if attrs else \
                torch.zeros((len(pairs), embedding_dim), dtype=torch.float32)

        return data, node_id_to_local

    def convert_dfs_to_heterodata_classic(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        question_embedding: torch.Tensor = None,
        *,
        graph_id: str = "graph",
        embedding_dim: int = 768,
        sanitize_keys: bool = True,
    ) -> Tuple[HeteroData, Dict[str, Dict[str, int]]]:
        """
        Converts embedded Pandas DataFrames into PyTorch Geometric HeteroData structures.
        Parameters:
            - nodes_df: DataFrame containing node information with columns ['node', 'description', 'is_type', 'embedding'].
            - edges_df: DataFrame containing edge information with columns ['source', 'target', 'key', 'description', 'embedding'].
            - graph_id: Unique identifier for the graph, used in metadata.
            - embedding_dim: Expected dimensionality of the embedding vectors.
            - sanitize_keys: Whether to sanitize edge keys for compatibility with HeteroData edge types.
            - question_embedding: Optional tensor containing the question embedding. If provided, it will be saved alongside the graph embeddings for reference.
        """
        # Clean nodes not present in the edges
        used_nodes = set(edges_df["source"].astype(str)).union(edges_df["target"].astype(str))
        nodes = nodes_df[nodes_df["node"].astype(str).isin(used_nodes)].copy()
        
        data = HeteroData()
        data.graph_id = graph_id
        if question_embedding is not None:
            data.question_embedding = TensorUtils.to_vec_32(question_embedding, embedding_dim)
        else:
            data.question_embedding = None

        # 1. Processing nodes
        node_rows_by_type = defaultdict(list)
        for _, r in nodes.iterrows():
            ntype = "ntype_type" if bool(r.get("is_type", False)) else "ntype_entity"
            nid = str(r["node"])
            emb = TensorUtils.to_vec_32(r.get("embedding", None), embedding_dim)
            node_rows_by_type[ntype].append((nid, emb))

        node_id_to_local = {}
        for ntype, lst in node_rows_by_type.items():
            ids = [nid for nid, _ in lst]
            feats = [emb for _, emb in lst]
            x = torch.stack(feats, dim=0) if feats else torch.zeros((0, embedding_dim), dtype=torch.float32)
            data[ntype].x = x
            data[ntype].nid = ids
            node_id_to_local[ntype] = {nid: i for i, nid in enumerate(ids)}

        # Ensure node -> type mapping exists for faster edge indexing
        nid2type = {str(r["node"]): ("ntype_type" if bool(r.get("is_type", False)) else "ntype_entity") for _, r in nodes.iterrows()}

        # 2. Processing edges
        edges = edges_df.copy()
        if sanitize_keys:
            edges["rel"] = edges["key"].astype(str).apply(TensorUtils.sanitize_rel)
        else:
            edges["rel"] = edges["key"].astype(str)

        edge_index_buckets = defaultdict(list)
        edge_attr_buckets = defaultdict(list)

        for _, e in edges.iterrows():
            s, t = str(e["source"]), str(e["target"])
            
            if s not in nid2type or t not in nid2type:
                continue
                
            st, tt = nid2type[s], nid2type[t]
            rel = e["rel"]
            et = (st, rel, tt)

            s_map = node_id_to_local.get(st, {})
            t_map = node_id_to_local.get(tt, {})
            
            if s not in s_map or t not in t_map:
                continue

            edge_index_buckets[et].append((s_map[s], t_map[t]))
            edge_attr_buckets[et].append(TensorUtils.to_vec_32(e.get("embedding", None), embedding_dim))

        for et, pairs in edge_index_buckets.items():
            if not pairs:
                continue
                
            st, rel, tt = et
            src_idx, dst_idx = zip(*pairs)
            data[(st, rel, tt)].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            
            attrs = edge_attr_buckets[et]
            data[(st, rel, tt)].edge_attr = torch.stack(attrs, dim=0) if attrs else \
                torch.zeros((len(pairs), embedding_dim), dtype=torch.float32)

        return data, node_id_to_local

    def load_split_graph(self, base_dir: Union[str, Path], graph_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads split graph artifacts ensuring schema and ordering matches internal blobs.
        
        """
        base = Path(base_dir)

        # 1. Load metadata with direct feather reads (pandas internally routes)
        nodes_meta = pd.read_feather(base / f"nodes_meta.{graph_id}.feather")
        edges_meta = pd.read_feather(base / f"edges_meta.{graph_id}.feather")

        # 2. Load embeddings using pytorch mapped directly to CPU
        nodes_blob = torch.load(base / f"nodes_emb.{graph_id}.pt", map_location="cpu")
        edges_blob = torch.load(base / f"edges_emb.{graph_id}.pt", map_location="cpu")
        question_embedding = torch.load(base / f"question_emb.{graph_id}.pt", map_location="cpu") if (base / f"question_emb.{graph_id}.pt").exists() else None

        node_ids_pt = list(map(str, nodes_blob["node_ids"]))
        edge_src = list(map(str, edges_blob["src"]))
        edge_dst = list(map(str, edges_blob["dst"]))
        edge_key = list(map(str, edges_blob["key"]))

        # 3. Validation and Alignment
        if "node" not in nodes_meta.columns:
            raise ValueError("nodes_meta.feather must contain a 'node' column.")
            
        meta_node_ids = nodes_meta["node"].astype(str).tolist()
        if meta_node_ids != node_ids_pt:
            raise ValueError("Mismatch between nodes_meta 'node' order and nodes_emb.pt 'node_ids'.")

        for col in ("source", "target", "key"):
            if col not in edges_meta.columns:
                raise ValueError(f"edges_meta.feather must contain a '{col}' column.")
                
        if not (edges_meta["source"].astype(str).tolist() == edge_src and 
                edges_meta["target"].astype(str).tolist() == edge_dst and 
                edges_meta["key"].astype(str).tolist() == edge_key):
            raise ValueError("Mismatch between edges_meta triplet layout and edges_emb.pt.")

        # 4. Bind embeddings back iteratively to dataframe format
        nodes_df = nodes_meta.copy()
        nodes_df["embedding"] = list(nodes_blob["emb"]) 

        edges_df = edges_meta.copy()
        edges_df["embedding"] = list(edges_blob["emb"]) 

        return nodes_df, edges_df, question_embedding

    def save_embeddings(self, graph_id: str, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, question_embedding: torch.Tensor = None) -> None:
        """
        Separately persists structural metadata and raw tensor representations locally.

        Parameters:
            - graph_id: Unique identifier for the graph, used in filenames.
            - nodes_df: DataFrame containing node metadata and embeddings.
            - edges_df: DataFrame containing edge metadata and embeddings.
            - question_embedding: Optional tensor containing the question embedding. If provided, it will be saved alongside the graph embeddings for reference.
        """
        meta_nodes_path = self.output_graph_path / f"nodes_meta.{graph_id}.feather"
        meta_edges_path = self.output_graph_path / f"edges_meta.{graph_id}.feather"

        meta_nodes = nodes_df.drop(columns=["embedding"])
        meta_edges = edges_df.drop(columns=["embedding"])
        
        meta_nodes.to_feather(meta_nodes_path)
        meta_edges.to_feather(meta_edges_path)

        # Save embeddings natively in PyTorch block archives 
        node_emb = torch.stack([TensorUtils.to_vec_32(x) for x in nodes_df["embedding"]])
        edge_emb = torch.stack([TensorUtils.to_vec_32(x) for x in edges_df["embedding"]])

        if question_embedding is not None:
            question_emb_path = self.output_graph_path / f"question_emb.{graph_id}.pt"
            torch.save({"embedding": TensorUtils.to_vec_32(question_embedding)}, question_emb_path)

        torch.save({
            "node_ids": meta_nodes["node"].tolist(), 
            "emb": node_emb.float()
        }, self.output_graph_path / f"nodes_emb.{graph_id}.pt")
        
        torch.save({
            "src": meta_edges["source"].tolist(),
            "dst": meta_edges["target"].tolist(),
            "key": meta_edges["key"].tolist(),
            "emb": edge_emb.float()
        }, self.output_graph_path / f"edges_emb.{graph_id}.pt")

        print(f"Saved embeddings for graph ID: {graph_id}")

    def reload_graphs_with_embeddings(self, graph_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reloads graph metadata and embeddings, ensuring they are correctly aligned and validated for downstream processing.

        Parameters:
            - graph_id: Unique identifier for the graph, used in filenames.
        """
        return self.load_split_graph(self.output_graph_path, graph_id)

    def extract_global_type_vocabulary(self, graph_ids: List[str]) -> Dict[str, int]:
        """
        Iterates over all available node parquet files to compute a consistent, 
        global vocabulary mapping of all unique types present across every graph.
        
        Returns:
            A dictionary mapping each unique type string to a globally consistent integer index.
        """
        print(f"Extracting global vocabulary across {len(graph_ids)} graphs...")

        def parse_types(t_val) -> List[str]:
            if isinstance(t_val, list):
                return [str(v).strip() for v in t_val]
            if pd.notna(t_val):
                return [v.strip().strip("'\"") for v in str(t_val).strip("[]").split(',') if v.strip()]
            return []

        all_types = set()
        
        for graph_id in graph_ids:
            path = self.input_graph_path / self.file_pattern_nodes.format(graph_id=graph_id)
            try:
                # Load only the 'types' column to save memory
                df = pd.read_parquet(path, engine="fastparquet", columns=["types"])
                for t_val in df["types"]:
                    all_types.update(parse_types(t_val))
            except Exception as e:
                print(f"Warning: Failed to extract types from {path}. Error: {e}")
                
        # Sort to insure consistent indexing across identical runs
        sorted_types = sorted(list(all_types))
        global_vocab = {t: i for i, t in enumerate(sorted_types)}
        
        print(f"Completed! Found {len(global_vocab)} unique semantic types globally.")
        return global_vocab

    def compute_embeddings(self, graphs_ids: List[str], save_intermediate: bool = False, global_vocab: Dict[str, int] = None, embedding_dim: int = 768) -> None:
        """
        Processes and translates textual relations and concepts dynamically over standard encoder layers.
        
        Parameters:
            - graphs_ids: List of graph identifiers to process.
            - save_intermediate: Whether to save intermediate DataFrames with embeddings before conversion to HeteroData.
        """

        for graph_id in graphs_ids:
            print(f"Processing graph ID: {graph_id}")
            
            nodes_path = self.input_graph_path / self.file_pattern_nodes.format(graph_id=graph_id)
            edges_path = self.input_graph_path / self.file_pattern_edges.format(graph_id=graph_id)
            graph_path = self.input_graph_path / f"full_graph.graph.{graph_id}.json"

            try:
                nodes_df = pd.read_parquet(nodes_path, engine="fastparquet")
                edges_df = pd.read_parquet(edges_path, engine="fastparquet")
                with open(graph_path, "r", encoding="utf-8") as f:
                    graph_info = json.load(f)
                question = graph_info.get("question", "")
                
            except Exception as e:
                print(f"Warning: Failed processing {graph_id} parquet blocks. {e}")
                continue

            if len(question) > 0:
                question_emb = self.text_to_encode.encode(question)
                question_emb = TensorUtils.to_vec_32(question_emb, embedding_dim)
                print(f"Encoded question for graph ID {graph_id}.")
            else:
                question_emb = None
                print(f"No question found for graph ID {graph_id}, skipping graph encoding.")
                continue
            # Batched encoding for nodes
            if not nodes_df.empty:
                if self.strategy == "simplename":
                    if nodes_df['name'].isnull().any():
                        nodes_df["name"] = nodes_df["name"].fillna(nodes_df["node"])
                    node_description = nodes_df["name"]
                else:
                    node_description = nodes_df["description"]
                node_descriptions = node_description.astype(str).tolist()
                nodes_encoded = self.text_to_encode.encode_batch(node_descriptions)
                nodes_df["embedding"] = [TensorUtils.to_vec_32(emb) for emb in nodes_encoded]
            else:
                nodes_df["embedding"] = []

            # Batched encoding for edges
            if not edges_df.empty:
                if self.strategy == "simplename":
                    edge_description = edges_df["key"]
                else:
                    edge_description = edges_df["description"]
                edge_descriptions = edge_description.astype(str).tolist()
                edges_encoded = self.text_to_encode.encode_batch_edges(edge_descriptions)
                edges_df["embedding"] = [TensorUtils.to_vec_32(emb) for emb in edges_encoded]
            else:
                edges_df["embedding"] = []

            nodes_df = nodes_df.sort_values("node").reset_index(drop=True)
            edges_df = edges_df.reset_index(drop=True)

            if save_intermediate:
                self.save_embeddings(graph_id, nodes_df, edges_df, question_embedding=question_emb)

            if self.strategy == "multihot":
                hetero_data, _ = self.convert_dfs_to_heterodata_multihot(
                    nodes_df,
                    edges_df,
                    question_embedding=question_emb,
                    graph_id=graph_id,
                    embedding_dim=embedding_dim,
                    sanitize_keys=True,
                    type_vocab=global_vocab
                )
            elif self.strategy == "classic" or self.strategy == "simplename":
                hetero_data, _ = self.convert_dfs_to_heterodata_classic(
                    nodes_df,
                    edges_df,
                    question_embedding=question_emb,
                    graph_id=graph_id,
                    embedding_dim=embedding_dim,
                    sanitize_keys=True
                )
            elif self.strategy == "singleton":
                hetero_data, _ = self.convert_dfs_to_heterodata_singleton(
                    nodes_df,
                    edges_df,
                    question_embedding=question_emb,
                    graph_id=graph_id,
                    embedding_dim=embedding_dim,
                    sanitize_keys=True
                )
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            sample_embedding_dim = hetero_data.node_stores[0].x.size(1)
            metadata_note = f"Generated with strategy: {self.strategy}"
            HeteroDataIO.save_heterodata(
                hetero_data,
                out_base=self.output_graph_path,
                graph_id=graph_id,
                save_node_ids=True,
                metadata={"embedding_dim": sample_embedding_dim, "note": metadata_note}
            )

            IOOperations.copy_and_rename(graph_path, self.output_graph_path, f"graph_meta.{graph_id}.json")
    
    def from_text_to_embeddings(self, save_intermediate: bool = False, compute_types_vocab: bool = False, embedding_dim: int = 768) -> None:
        """
        Gathers textual datasets and automatically maps them through LLM generation flows.
        """
        search_pattern = str(self.input_graph_path / self.file_pattern_nodes.replace("{graph_id}", "*"))
        all_nodes_paths = glob(search_pattern)
        
        # Deduplicate to strictly retrieve graph_ids
        all_graph_identifiers = sorted({Path(p).stem.split(".")[-1] for p in all_nodes_paths})
        print(f"Discovered {len(all_graph_identifiers)} total graphs.")
        
        if all_graph_identifiers:
            global_vocab = None
            if compute_types_vocab:
                global_vocab = self.extract_global_type_vocabulary(all_graph_identifiers)
                print(f"Global type vocabulary computed with {len(global_vocab)} unique types.")

            self.compute_embeddings(all_graph_identifiers, save_intermediate, global_vocab, embedding_dim)
    
