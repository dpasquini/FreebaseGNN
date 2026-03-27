import torch
import random
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple, Iterable
from torch_geometric.data import HeteroData

EdgeType = Tuple[str, str, str]  # (src_ntype, rel, dst_ntype)

class LinkPredictor(nn.Module):
    """
    Wraps a hetero GNN (e.g. HeteroGAT) and adds relation-specific decoders.

    Usage (as in your code):
      node_embeddings = model.encode(batch)
      scores = model.decode(node_embeddings, edge_label_index, edge_type)
    """

    def __init__(
        self,
        gnn: nn.Module,
        hidden_dim: int,
        edge_types: Iterable[EdgeType]
    ):
        super().__init__()
        self.gnn = gnn
        self.hidden_dim = hidden_dim

        # One small MLP per edge type
        self.rel_mlps = nn.ModuleDict()
        for et in edge_types:
            key = self._edge_key(et)
            self.rel_mlps[key] = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    @property
    def device(self) -> torch.device:
        """Utility property to get the device of the model parameters."""
        return next(self.parameters()).device

    @property
    def device_str(self) -> str:
        return str(self.device)

    # --- helper to build / fetch decoder for a given edge type ---
    def _edge_key(self, edge_type: EdgeType) -> str:
        """
        Builds a unique string key for an edge type tuple (src_ntype, rel, dst_ntype).
        This is used to store and retrieve the corresponding MLP decoder for that edge type.
        """
        # edge_type is ('ntype_entity', 'rel', 'ntype_entity') only in classic strategy, but we keep it general for any tuple of strings
        return "||".join(edge_type)

    def _get_rel_mlp(self, edge_type: EdgeType) -> nn.Module:
        """
        Retrieves the MLP decoder for a given edge type. The edge type is specified as a tuple (src_ntype, rel, dst_ntype).
        The method constructs a unique key from the edge type and looks it up in the rel_mlps ModuleDict.
        """
        key = self._edge_key(edge_type)
        if key not in self.rel_mlps:
            raise KeyError(f"No MLP localized for edge type: {edge_type}")
        return self.rel_mlps[key]

    def _collect_edge_inputs(self, data: HeteroData) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Moves graph attributes to the proper device and unpacks for GNN encoding.
        """
        dev = self.device
        x_dict = {nt: x.to(device=dev, dtype=torch.float32) for nt, x in data.x_dict.items()}
        edge_index_dict = {et: ei.to(dev) for et, ei in data.edge_index_dict.items()}
        edge_attr_dict = {}
        for et in data.edge_types:
            edge_attr_dict[et] = data[et].edge_attr.to(device=dev, dtype=torch.float32) \
                                 if hasattr(data[et], "edge_attr") else None
        return x_dict, edge_index_dict, edge_attr_dict

    def encode(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Converts a HeteroData batch into node embeddings using the GNN encoder. 
        The method first moves the data to the same device as the model, then collects the node features, edge indices, and edge attributes into dictionaries. 
        Finally, it calls the GNN encoder with these inputs and returns the resulting node embeddings.
        """
        data = data.to(self.device)
        x_dict, edge_index_dict, edge_attr_dict = self._collect_edge_inputs(data)
        return self.gnn(x_dict, edge_index_dict, edge_attr_dict)

    def decode(
        self,
        node_embeddings: Dict[str, torch.Tensor],
        edge_label_index: torch.Tensor,
        edge_type: EdgeType,
    ) -> torch.Tensor:
        """
        Decodes edge scores for a specific edge type using a relation-specific MLP.
        The method takes the node embeddings, edge label indices (which specify the source and destination nodes for the edges to score), 
        and the edge type (which determines which MLP to use).
        It retrieves the source and destination node types from the edge type, then gathers the corresponding node embeddings for the source and destination nodes.
        The source and destination embeddings are concatenated and passed through the MLP corresponding to the edge type to produce a score for each edge.

        Parameters:
        - node_embeddings: A dictionary mapping node types to their corresponding embeddings (shape: [num_nodes_ntype, hidden_dim]).
        - edge_label_index: A tensor of shape [2, num_edges] containing the source and destination node indices for the edges to score.
        - edge_type: A tuple (src_ntype, rel, dst_ntype) specifying the type of edge being scored, which determines which MLP decoder to use.

        Returns:
        - A tensor of shape [num_edges] containing the scores for each edge, as computed by the relation-specific MLP.
        """
        src_nt, _, dst_nt = edge_type

        # Make sure indices are on same device as node embeddings
        device = node_embeddings[src_nt].device
        edge_label_index = edge_label_index.to(device)

        src_idx, dst_idx = edge_label_index

        h_src = node_embeddings[src_nt][src_idx]
        h_dst = node_embeddings[dst_nt][dst_idx]

        h_cat = torch.cat([h_src, h_dst], dim=-1)
        mlp = self._get_rel_mlp(edge_type)

        logits = mlp(h_cat).squeeze(-1)
        return logits

    def forward(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        edge_type: EdgeType,
    ) -> torch.Tensor:
        """
        Convenience forward for code that does model(data, edge_label_index, edge_type)
        """
        node_embeddings = self.encode(data)
        logits = self.decode(node_embeddings, edge_label_index, edge_type)
        return logits

    def get_multi_seed_1hop_embeddings(self, data: HeteroData, seed_list=None, num_random_seeds=3) -> torch.Tensor:
        """
        Computes graph embeddings and extracts a deduplicated 1-hop subgraph.
        If seed_list is None, randomly selects `num_random_seeds` nodes from the graph.
        """
        self.eval()
        device = self.device
        data = data.to(device)
        
        # If no seeds provided, randomly select some nodes from the graph as seeds
        if seed_list is None:
            seed_list = []
            # Only consider node types that actually have nodes in this specific graph
            valid_node_types = [nt for nt in data.node_types if data[nt].num_nodes > 0]
            
            if valid_node_types:
                for _ in range(num_random_seeds):
                    # Pick a random node type
                    ntype = random.choice(valid_node_types)
                    
                    # PyG HeteroData stores this in num_nodes
                    num_nodes = data[ntype].num_nodes 
                    
                    # Randomly select a node ID from this node type
                    nid = torch.randint(0, num_nodes, (1,)).item()
                    seed_list.append((ntype, nid))
                    
                print(f"[*] No seeds provided. Randomly selected {num_random_seeds} seeds: {seed_list}")
            else:
                print("[Warning] Graph has no nodes at all! Cannot sample seeds.")
        
        # Compute all embeddings for the entire graph
        with torch.no_grad():
            x_dict = self.encode(data)
        
        # Setup a dictionary to collect unique node IDs per node type
        collected_nodes = {ntype: set() for ntype in data.node_types}
        seed_dict = {ntype: [] for ntype in data.node_types}
        
        for ntype, nid in seed_list:
            seed_dict[ntype].append(nid)
            collected_nodes[ntype].add(nid) # Add seeds themselves
            
        for ntype in seed_dict:
            seed_dict[ntype] = torch.tensor(seed_dict[ntype], dtype=torch.long, device=device)
        
        # Find all 1-hop neighbors across all relation types
        for src_t, rel, dst_t in data.edge_types:
            edge_index = data[src_t, rel, dst_t].edge_index
            
            # Outgoing edges
            if len(seed_dict[src_t]) > 0:
                mask = torch.isin(edge_index[0], seed_dict[src_t])
                neighbors = edge_index[1][mask]
                collected_nodes[dst_t].update(neighbors.tolist())
                
            # Incoming edges
            if len(seed_dict[dst_t]) > 0:
                mask = torch.isin(edge_index[1], seed_dict[dst_t])
                neighbors = edge_index[0][mask]
                collected_nodes[src_t].update(neighbors.tolist())
                
        # 4. Extract and concatenate the actual embedding vectors
        extracted_embeddings = []
        
        for ntype, unique_nids in collected_nodes.items():
            if len(unique_nids) > 0:
                nid_tensor = torch.tensor(list(unique_nids), dtype=torch.long, device=device)
                extracted_embeddings.append(x_dict[ntype][nid_tensor])
                
        if len(extracted_embeddings) > 0:
            gnn_node_embeddings = torch.cat(extracted_embeddings, dim=0)
        else:
            # Failsafe for an empty graph
            first_ntype = data.node_types[0]
            gnn_node_embeddings = torch.empty((0, x_dict[first_ntype].shape[-1]), device=device)
            
        return gnn_node_embeddings

    def get_multi_seed_nhop_embeddings(self, data: HeteroData, seed_list=None, num_random_seeds=3, n_hop=1) -> torch.Tensor:
        """
        Computes graph embeddings and extracts a deduplicated n-hop subgraph.
        If seed_list is None, randomly selects `num_random_seeds` nodes from the graph.
        """
        self.eval()
        device = self.device
        data = data.to(device)
        
        # if no seeds provided, randomly select some nodes from the graph as seeds
        if seed_list is None:
            seed_list = []
            valid_node_types = [nt for nt in data.node_types if data[nt].num_nodes > 0]
            
            if valid_node_types:
                for _ in range(num_random_seeds):
                    ntype = random.choice(valid_node_types)
                    num_nodes = data[ntype].num_nodes 
                    nid = torch.randint(0, num_nodes, (1,)).item()
                    seed_list.append((ntype, nid))
                    
                print(f"No seeds provided. Randomly selected {num_random_seeds} seeds: {seed_list}")
            else:
                print("[Warning] Graph has no nodes at all! Cannot sample seeds.")
                
        # Compute all embeddings for the entire graph
        with torch.no_grad():
            x_dict = self.encode(data)
        
        # Initialize BFS structures
        # visited_nodes: keeps track of all unique nodes we've seen so far across all hops
        visited_nodes = {ntype: set() for ntype in data.node_types}
        
        # current_frontier: the nodes we just discovered in the previous hop (starting with our seeds)
        current_frontier = {ntype: [] for ntype in data.node_types}
        
        for ntype, nid in seed_list:
            current_frontier[ntype].append(nid)
            visited_nodes[ntype].add(nid)
            
        for ntype in current_frontier:
            current_frontier[ntype] = torch.tensor(current_frontier[ntype], dtype=torch.long, device=device)
        
        # Perform n-hop expansion
        for hop in range(n_hop):
            # The new nodes we will discover in this specific hop
            next_frontier = {ntype: set() for ntype in data.node_types}
            
            for src_t, rel, dst_t in data.edge_types:
                # Safely skip edge types that might not exist in this specific batch
                if (src_t, rel, dst_t) not in data.edge_types:
                    continue
                    
                edge_index = data[src_t, rel, dst_t].edge_index
                if edge_index is None or edge_index.numel() == 0:
                    continue
                
                # Outgoing edges: frontier nodes act as the source
                if len(current_frontier[src_t]) > 0:
                    mask = torch.isin(edge_index[0], current_frontier[src_t])
                    neighbors = edge_index[1][mask]
                    next_frontier[dst_t].update(neighbors.tolist())
                    
                # Incoming edges: frontier nodes act as the destination
                if len(current_frontier[dst_t]) > 0:
                    mask = torch.isin(edge_index[1], current_frontier[dst_t])
                    neighbors = edge_index[0][mask]
                    next_frontier[src_t].update(neighbors.tolist())
                    
            # Prepare the frontier for the next hop
            for ntype in data.node_types:
                # Keep only the purely new nodes to avoid redundant expansions
                new_nodes = next_frontier[ntype] - visited_nodes[ntype]
                
                # Update our master visited list
                visited_nodes[ntype].update(new_nodes)
                
                # Set the new frontier as tensors for the next iteration
                current_frontier[ntype] = torch.tensor(list(new_nodes), dtype=torch.long, device=device)

        # 5. Extract and concatenate the actual embedding vectors
        extracted_embeddings = []
        
        for ntype, unique_nids in visited_nodes.items():
            if len(unique_nids) > 0:
                nid_tensor = torch.tensor(list(unique_nids), dtype=torch.long, device=device)
                extracted_embeddings.append(x_dict[ntype][nid_tensor])
                
        if len(extracted_embeddings) > 0:
            gnn_node_embeddings = torch.cat(extracted_embeddings, dim=0)
        else:
            # Failsafe for an empty graph
            first_ntype = data.node_types[0]
            gnn_node_embeddings = torch.empty((0, x_dict[first_ntype].shape[-1]), device=device)
            
        return gnn_node_embeddings
    

    def get_ranked_nhop_embeddings(self, data: HeteroData, question_embedding: torch.Tensor = None, seed_list: list = None, num_random_seeds: int = 1, k_max: int = 50, n_hop: int = 1) -> torch.Tensor:
        """
        Expands n-hop from seeds, ranking new nodes at each hop by cosine similarity 
        to the question embedding. Truncates to k_max nodes and returns their GNN embeddings.
        """
        self.eval()
        device = self.device
        data = data.to(device)

        # We check both data.question_embedding and the function argument question_embedding 
        # to ensure we have the question embedding available on the correct device 
        # for similarity calculations during ranking.
        if data.question_embedding is not None:
            question_embedding = data.question_embedding.to(device)
        elif question_embedding is not None:
            question_embedding = question_embedding.to(device)
        
        # if no seeds provided, randomly select some nodes from the graph as seeds
        """
        if seed_list is None:
            seed_list = []
            valid_node_types = [nt for nt in data.node_types if data[nt].num_nodes > 0]
            
            if valid_node_types:
                for _ in range(num_random_seeds):
                    ntype = random.choice(valid_node_types)
                    num_nodes = data[ntype].num_nodes 
                    nid = torch.randint(0, num_nodes, (1,)).item()
                    seed_list.append((ntype, nid))
                    
                print(f"No seeds provided. Randomly selected {num_random_seeds} seeds: {seed_list}")
            else:
                print("[Warning] Graph has no nodes at all! Cannot sample seeds.")
        """
        if not seed_list:  # Catches both None and empty lists []
            seed_list = []
            all_candidates = []
            
            # Ensure question embedding is 2D for broadcasting
            if question_embedding.dim() == 1:
                q_emb = question_embedding.unsqueeze(0)
            else:
                q_emb = question_embedding
                
            # 1. Score every node in the graph against the question
            for ntype in data.node_types:
                if data[ntype].num_nodes > 0 and hasattr(data[ntype], 'x') and data[ntype].x is not None:
                    initial_embs = data[ntype].x
                    
                    # Compute cosine similarity for all nodes of this type
                    scores = F.cosine_similarity(initial_embs, q_emb, dim=-1)
                    
                    # Only keep the top K from this specific node type 
                    # to prevent building massive lists in memory
                    k_for_type = min(num_random_seeds, data[ntype].num_nodes)
                    top_scores, top_indices = torch.topk(scores, k_for_type)
                    
                    for score, nid in zip(top_scores.tolist(), top_indices.tolist()):
                        all_candidates.append({
                            'ntype': ntype, 
                            'nid': nid, 
                            'score': score
                        })
                        
            # 2. Sort all candidates globally across all node types
            all_candidates.sort(key=lambda x: x['score'], reverse=True)
            
            # 3. Take the absolute best `num_random_seeds` nodes
            best_candidates = all_candidates[:num_random_seeds]
            
            for cand in best_candidates:
                seed_list.append((cand['ntype'], cand['nid']))
                
            print(f"Auto-Seeded top {len(seed_list)} semantic matches: "
                  f"{[(c['ntype'], c['nid'], round(c['score'], 4)) for c in best_candidates]}")
            
            if not seed_list:
                print("[Warning] Graph has no valid features! Cannot auto-seed.")
        
        # Compute final GNN embeddings (what we actually feed the LLM later)
        with torch.no_grad():
            gnn_x_dict = self.encode(data)
            
        # Tracking structures
        visited_nodes = {ntype: set() for ntype in data.node_types}
        current_frontier = {ntype: [] for ntype in data.node_types}
        
        # This list maintains our strict ordering: Seed -> Hop 1 (sorted) -> Hop 2 (sorted)
        ordered_node_sequence = []
            
        for ntype, nid in seed_list:
            if nid not in visited_nodes[ntype]:
                current_frontier[ntype].append(nid)
                visited_nodes[ntype].add(nid)
                ordered_node_sequence.append((ntype, nid))
                
        # Convert frontier to tensors for fast edge masking
        for ntype in current_frontier:
            current_frontier[ntype] = torch.tensor(current_frontier[ntype], dtype=torch.long, device=device)
            
        # Hop-by-Hop Expansion and Ranking
        for hop in range(n_hop):
            # Stop early if we already have enough nodes to fill the k_max quota
            if len(ordered_node_sequence) >= k_max:
                break
                
            next_frontier = {ntype: set() for ntype in data.node_types}
            
            # Find neighbors for the current frontier
            for src_t, rel, dst_t in data.edge_types:
                if (src_t, rel, dst_t) not in data.edge_types: 
                    continue
                edge_index = data[src_t, rel, dst_t].edge_index
                if edge_index is None or edge_index.numel() == 0: 
                    continue
                
                # Outgoing edges
                if len(current_frontier[src_t]) > 0:
                    mask = torch.isin(edge_index[0], current_frontier[src_t])
                    next_frontier[dst_t].update(edge_index[1][mask].tolist())
                    
                # Incoming edges
                if len(current_frontier[dst_t]) > 0:
                    mask = torch.isin(edge_index[1], current_frontier[dst_t])
                    next_frontier[src_t].update(edge_index[0][mask].tolist())
            
            # Rank the newly discovered nodes in THIS specific hop
            hop_new_nodes = [] 
            
            for ntype in data.node_types:
                # Filter out nodes we've already seen in previous hops
                new_nids = list(next_frontier[ntype] - visited_nodes[ntype])
                if len(new_nids) == 0:
                    continue
                    
                visited_nodes[ntype].update(new_nids)
                new_nids_tensor = torch.tensor(new_nids, dtype=torch.long, device=device)
                
                # Ranking Step: We use the INITIAL (pre-GNN) embeddings to calculate similarity to the question
                initial_embs = data[ntype].x[new_nids_tensor] 
                
                # Compute Cosine Similarity against the question
                # Ensure question_embedding is 2D for broadcasting: (1, dim)
                if question_embedding.dim() == 1:
                    question_embedding = question_embedding.unsqueeze(0)
                    
                sim_scores = F.cosine_similarity(initial_embs, question_embedding, dim=-1)
                
                # Store the node and its score
                for i, nid in enumerate(new_nids):
                    hop_new_nodes.append({
                        'ntype': ntype,
                        'nid': nid,
                        'score': sim_scores[i].item()
                    })
                    
                # Set up the frontier for the next hop
                current_frontier[ntype] = new_nids_tensor
                
            # Sort all new nodes discovered in this hop by their similarity score (Descending)
            hop_new_nodes.sort(key=lambda x: x['score'], reverse=True)
            
            # Append the sorted block of nodes to our master sequence
            for node_info in hop_new_nodes:
                ordered_node_sequence.append((node_info['ntype'], node_info['nid']))
                
        # Truncate strictly to k_max
        ordered_node_sequence = ordered_node_sequence[:k_max]
        
        # Extract the final GNN embeddings in the exact sorted order
        extracted_embeddings = []
        for ntype, nid in ordered_node_sequence:
            # We use unsqueeze(0) to keep the batch dimension [1, gnn_dim]
            emb = gnn_x_dict[ntype][nid].unsqueeze(0) 
            extracted_embeddings.append(emb)
            
        if len(extracted_embeddings) > 0:
            # Stack into final tensor: Shape (K, gnn_dim)
            final_ordered_tensor = torch.cat(extracted_embeddings, dim=0) 
        else:
            first_ntype = data.node_types[0]
            final_ordered_tensor = torch.empty((0, gnn_x_dict[first_ntype].shape[-1]), device=device)
            
        return final_ordered_tensor

