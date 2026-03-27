import torch
from collections import defaultdict
import tqdm
from pathlib import Path
from torch_geometric.utils import coalesce
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import HeteroData
import sys
import os
from utils.HeteroDataIO import HeteroDataIO

class GNNDataProcessor:
    def __init__(self, input_dir: str, file_pattern: str, random_initialization: bool = False):
        self.input_dir = Path(input_dir)
        self.file_pattern = file_pattern
        self.random_initialization = random_initialization

    def prepare_data(self, top_k_relations=150, edge_existence: bool = False, homogenization: bool = False):
        print("Starting data preparation pipeline...")
        train_data, val_data, test_data, new_edge_types, node_types = self.load_and_preprocess_graphs(
            top_k_relations=top_k_relations,
            edge_existence=edge_existence,
            homogenization=homogenization
        )
        
        print("Making train packs...")
        train_packs = self.make_packs(train_data)
        print("Making validation packs...")
        val_packs = self.make_packs(val_data)
        print("Making test packs...")
        test_packs = self.make_packs(test_data)
        
        return train_packs, val_packs, test_packs, new_edge_types, node_types

    @staticmethod
    def _strip_ntype_prefix(ntype: str) -> str:
        ntype = str(ntype)
        return ntype[len("ntype_"):] if ntype.startswith("ntype_") else ntype

    def flatten_edge_types_for_edge_existence(self, graph):
        """
        Rewrites all relation names to generic node-type based relations, e.g.
        (ntype_entity, instanceof, ntype_type) -> (ntype_entity, generic_entity_to_type, ntype_type).
        """
        new_g = graph.clone()
        groups = defaultdict(list)

        for et in graph.edge_types:
            src_t, _, dst_t = et
            rel = f"generic_{self._strip_ntype_prefix(src_t)}_to_{self._strip_ntype_prefix(dst_t)}"
            new_key = (src_t, rel, dst_t)
            groups[new_key].append(graph[et].edge_index)

        for et in list(new_g.edge_types):
            del new_g[et]

        for new_et, indices_list in groups.items():
            if len(indices_list) == 0:
                continue
            merged_index = torch.cat(indices_list, dim=1)
            src_t, _, dst_t = new_et
            num_src = graph[src_t].num_nodes
            num_dst = graph[dst_t].num_nodes
            merged_index, _ = coalesce(
                merged_index,
                None,
                num_nodes=max(num_src, num_dst)
            )
            new_g[new_et].edge_index = merged_index

        return new_g

    def homogenize_graph_for_edge_existence(self, graph):
        """
        Collapses all node types into a single generic node type and all edge
        types into one generic relation between generic nodes.
        """
        generic_ntype = "ntype_generic"
        generic_rel = "generic_generic_to_generic"

        new_g = HeteroData()

        if hasattr(graph, "graph_id"):
            new_g.graph_id = graph.graph_id
        if hasattr(graph, "gmeta"):
            new_g.gmeta = graph.gmeta
        if hasattr(graph, "question_embedding"):
            new_g.question_embedding = graph.question_embedding

        # Build node offsets per original node type so we can remap edge indices.
        offsets = {}
        x_parts = []
        nid_all = []
        total_nodes = 0

        for ntype in graph.node_types:
            num_nodes = int(graph[ntype].num_nodes)
            if num_nodes <= 0:
                continue

            offsets[ntype] = total_nodes
            total_nodes += num_nodes

            if hasattr(graph[ntype], "x") and graph[ntype].x is not None:
                x_parts.append(graph[ntype].x)

            nids = getattr(graph[ntype], "nid", None)
            if nids is not None:
                nid_all.extend([str(v) for v in nids])
            else:
                nid_all.extend([f"{ntype}:{i}" for i in range(num_nodes)])

        if x_parts:
            new_g[generic_ntype].x = torch.cat(x_parts, dim=0)
        else:
            new_g[generic_ntype].x = torch.empty((0, 0), dtype=torch.float32)
        new_g[generic_ntype].nid = nid_all

        edge_index_parts = []
        edge_attr_parts = []
        has_edge_attr = True

        for et in graph.edge_types:
            src_t, _, dst_t = et
            if src_t not in offsets or dst_t not in offsets:
                continue

            ei = graph[et].edge_index
            if ei is None or ei.numel() == 0:
                continue

            remapped_src = ei[0] + offsets[src_t]
            remapped_dst = ei[1] + offsets[dst_t]
            edge_index_parts.append(torch.stack([remapped_src, remapped_dst], dim=0))

            if hasattr(graph[et], "edge_attr") and graph[et].edge_attr is not None:
                edge_attr_parts.append(graph[et].edge_attr)
            else:
                has_edge_attr = False

        if edge_index_parts:
            merged_edge_index = torch.cat(edge_index_parts, dim=1)
            if has_edge_attr and edge_attr_parts:
                merged_edge_attr = torch.cat(edge_attr_parts, dim=0)
                # Keep edge_index and edge_attr aligned after duplicate removal.
                merged_edge_index, merged_edge_attr = coalesce(
                    merged_edge_index,
                    merged_edge_attr,
                    num_nodes=total_nodes
                )
                new_g[(generic_ntype, generic_rel, generic_ntype)].edge_index = merged_edge_index
                new_g[(generic_ntype, generic_rel, generic_ntype)].edge_attr = merged_edge_attr
            else:
                merged_edge_index, _ = coalesce(
                    merged_edge_index,
                    None,
                    num_nodes=total_nodes
                )
                new_g[(generic_ntype, generic_rel, generic_ntype)].edge_index = merged_edge_index

        return new_g

    def create_relation_mapping(self, train_data, top_k=50):
        """
        Creates a mapping of edge types to a reduced set of relation buckets based on the top_k most frequent relations in the training data.
        This helps to reduce the number of unique edge types the model has to handle, which can improve generalization and reduce overfitting on rare relations.
        
        The method counts the occurrences of each edge type in the training data, identifies the top_k most frequent relations, and then creates a mapping for all edge types.
        Edge types that are not in the top_k are bucketed based on their domain (extracted from the relation name) to create a new key for the mapping. 
        The method returns the mapping and a sorted list of the new unique keys.

        Parameters:
        - train_data: The training dataset containing the graph data with edge types.
        - top_k: The number of top relations to keep as is, while the rest will
            be bucketed (default is 50).
        Returns:
        - mapping: A dictionary mapping original edge types to their corresponding new edge types (either the original if in top_k or a bucketed version).
        - new_unique_keys: A sorted list of the new unique edge types after mapping, which includes the top_k relations and the new bucketed relations.     
        """
        edge_counts = defaultdict(int)
        for g in train_data:
            for et in g.edge_types:
                count = g[et].edge_index.size(1)
                edge_counts[et] += count

        sorted_rels = sorted(edge_counts.keys(), key=lambda k: edge_counts[k], reverse=True)
        top_relations = set(sorted_rels[:top_k])
        
        print(f"Total unique relations: {len(sorted_rels)}")
        print(f"Top {top_k} relations cover {sum(edge_counts[et] for et in top_relations)} edges out of {sum(edge_counts.values())} total edges.")

        mapping = {} 
        new_unique_keys = set()
        
        all_known_types = sorted_rels 
        
        # We iterate over all known edge types and check if they are in the top relations. 
        # If they are, we keep them as is. If not, we bucket them based on their domain (extracted from the relation name) 
        # and create a new key for the mapping.
        for et in all_known_types:
            if et in top_relations:
                mapping[et] = et
                new_unique_keys.add(et)
            else:
                src_t, rel_name, dst_t = et
                if "_DOT_" in rel_name:
                    domain = rel_name.split('_DOT_')[0]
                elif "." in rel_name:
                    domain = rel_name.split('.')[0]
                else:
                    domain = rel_name.split('_')[0] 
                    
                bucket_rel = f"BUCKET_{domain}"
                new_key = (src_t, bucket_rel, dst_t)
                mapping[et] = new_key
                new_unique_keys.add(new_key)

        print(f"Mapping complete. Reduced {len(all_known_types)} into {len(new_unique_keys)} types.")
        return mapping, sorted(list(new_unique_keys))

    def apply_relation_mapping(self, graph, mapping):
        """
        Applies the relation mapping to a given graph, transforming its edge types according to the provided mapping.
        The method creates a new graph where the edge types are replaced based on the mapping. 
        For edge types that are not in the mapping (e.g., new relations in the test set), it attempts to bucket them based on their domain. 
        If the predicted bucket is not in the known target types, it drops those edge types to avoid crashes during training or inference.
        """
        new_g = graph.clone()
        groups = defaultdict(list)
        
        # Precompute the set of known target types for quick lookup during dynamic bucketing
        known_target_types = set(mapping.values())

        for et in graph.edge_types:
            # Case 1: Known relation type (seen in training) -> Direct Mapping
            if et in mapping:
                target_key = mapping[et]
                groups[target_key].append(graph[et].edge_index)
            
            # Case 2: New relation (e.g., in the Test Set) -> Dynamic Bucketing
            else:
                src_t, rel_name, dst_t = et
                
                # Extract domain from the relation name using the same logic as in the mapping creation
                if "_DOT_" in rel_name: domain = rel_name.split('_DOT_')[0]
                elif "." in rel_name:   domain = rel_name.split('.')[0]
                else:                   domain = rel_name.split('_')[0]
                
                # Construct the predicted bucket key
                predicted_bucket = (src_t, f"BUCKET_{domain}", dst_t)
                
                # if the predicted bucket is in the known target types, we use it; 
                # otherwise, we drop this relation type to avoid crashes (better to lose an edge than to crash)
                if predicted_bucket in known_target_types:
                    groups[predicted_bucket].append(graph[et].edge_index)
                else:
                    # CASO C: Relazione aliena intraducibile. 
                    # Ignoriamola per sicurezza (meglio perdere un arco che crashare)
                    # print(f"[Warning] Dropping unknown relation type: {et}")
                    pass
                
        # We remove all original edge types from the new graph, as we will reconstruct them based on the mapping.
        for et in list(new_g.edge_types):
            del new_g[et]
            
        # Now we create new edge types in the new graph based on the grouped edge indices. 
        # For each new edge type (which could be an original top relation or a bucketed relation), 
        # we concatenate all the corresponding edge indices and assign them to the new graph. 
        # We also apply coalescing to remove duplicates and sort the edges, 
        # which is important for efficient processing in GNNs.
        for new_et, indices_list in groups.items():
            if len(indices_list) > 0:
                merged_index = torch.cat(indices_list, dim=1)
                
                src_t, _, dst_t = new_et
                num_src = graph[src_t].num_nodes
                num_dst = graph[dst_t].num_nodes
                
                # Coalesce to remove duplicates and sort. This is crucial for GNN processing, 
                # as it ensures that the edge indices are in a canonical form.
                merged_index, _ = coalesce(
                    merged_index,
                    None,
                    num_nodes=max(num_src, num_dst)
                )
                
                new_g[new_et].edge_index = merged_index
                
        return new_g

    def split(self, data, train_ratio=0.7, test_ratio=0.15, val_ratio=0.15):
        """
        Splits the data into train, test, and validation sets based on the provided ratios.

        Parameters:
        - data: The dataset to be split (e.g., a list of samples).
        - train_ratio: The proportion of the dataset to be used for training (default is 0.7).
        - test_ratio: The proportion of the dataset to be used for testing (default is 0.15).
        - val_ratio: The proportion of the dataset to be used for validation (default is 0.15).
        """
        total = len(data)
        train_end = int(total * train_ratio)
        test_end = train_end + int(total * test_ratio)

        train_data = data[:train_end]
        test_data = data[train_end:test_end]
        val_data = data[test_end:]

        print(f"Data split into: {len(train_data)} train, {len(test_data)} test, {len(val_data)} validation samples.")

        return train_data, test_data, val_data
    
    def randomize_features(self, graph):
        """
        Randomizes the node features in the given graph data. 
        This can be used as a control experiment to test 
        whether the model is learning meaningful patterns from the features or just relying on the graph structure.

        Parameters:
        - graph: The graph data (e.g., a HeteroData object) whose node features are to be randomized.
        """
        for node_type in graph.node_types:
            num_nodes = graph[node_type].num_nodes
            feature_dim = graph[node_type].x.size(1)
            rand_x = torch.empty((num_nodes, feature_dim))
            # We use Xavier uniform initialization to fill the random features, 
            # which is a common practice for initializing neural network weights.
            torch.nn.init.xavier_uniform_(rand_x)
            graph[node_type].x = rand_x

    def make_packs(self, graphs):
        """
        Prepares the graph data for training by applying a RandomLinkSplit to create train, validation, and test splits for link prediction.
        The method iterates over the list of graphs, applies the RandomLinkSplit transformation to each graph, 
        and collects the resulting train, validation, and test splits along with the full positive edge indices
        for each graph. It also handles cases where a graph might be empty or problematic (e.g., no edges) by skipping them and counting how many were skipped.
        The resulting packs are returned as a list of tuples, where each tuple contains the train, validation, and test splits along with the full positive edge indices for a graph. 

        Parameters:
        - graphs: A list of graph data objects (e.g., HeteroData) to be processed and split into train, validation, and test sets for link prediction.
        Returns:
        - packs: A list of tuples, where each tuple contains:
            the train, validation, and test splits along with the full positive edge indices for a graph. 
            Each tuple is of the form (train_graph, val_graph, test_graph, full_positive_edges).
        """
        packs = []
        skipped_count = 0
        for g in graphs:
            if not g.edge_types: 
                skipped_count += 1
                continue
            full_pos = {et: g[et].edge_index.clone() for et in g.edge_types}
            splitter = RandomLinkSplit(num_val=0.1, num_test=0.1, edge_types=list(g.edge_types), add_negative_train_samples=False, disjoint_train_ratio=0.2, is_undirected=False)
            try:
                tr, va, te = splitter(g)
                packs.append((tr, va, te, full_pos))
            except Exception as e:
                print(f"Skipping graph due to split error: {e}")
                skipped_count += 1
        if skipped_count > 0:
            print(f"Warning: Skipped {skipped_count} empty/problematic graphs.")
        return packs

    def load_and_preprocess_graphs(self, top_k_relations=150, read_graph_metadata=False, edge_existence: bool = False, homogenization: bool = False):
        """
        Preprocesses the raw graph data by loading it, splitting it into train/validation/test sets, 
        applying relation mapping to reduce the number of edge types, 
        and optionally randomizing node features. 
        This method prepares the data for training the GNN model by ensuring that the graphs are in the correct format and that the edge types are manageable for the model. 
        """
        search_pattern = self.file_pattern.replace("{graph_id}", "*")
        if not search_pattern.endswith('.pt'):
            search_pattern += ".pt"
        
        all_graphs = list(self.input_dir.glob(search_pattern))
        print(f"Found {len(all_graphs)} graph files matching the pattern.")

        graphs = []
        for graph_path in tqdm.tqdm(all_graphs, desc="Loading graphs"):
            try:
                # Extract graph_id by removing the pattern prefix from the stem
                # This assumes the pattern is roughly "prefix.{graph_id}.suffix" and file is "prefix.ID.suffix"
                prefix = self.file_pattern.split("{graph_id}")[0]
                graph_id = graph_path.stem.replace(prefix, "", 1)
                
                graph = HeteroDataIO().load_heterodata(self.input_dir, graph_id=graph_id, file_pattern=self.file_pattern)
                if read_graph_metadata:
                    metadata_path = self.input_dir / f"graph_meta.{graph_id}.json"
                    if metadata_path.exists():
                        metadata = HeteroDataIO().load_metadata(metadata_path)
                        graph.gmeta = metadata
                graphs.append((graph_id, graph))
            except Exception as e:
                print(f"Error loading {graph_path}: {e}")
        
        raw_graphs = [g for _, g in graphs]
        print(f"Successfully loaded {len(raw_graphs)} graphs.")

        if homogenization:
            print("Homogenization mode active: collapsing all node types/edge types into generic types...")
            raw_graphs = [self.homogenize_graph_for_edge_existence(g) for g in tqdm.tqdm(raw_graphs, desc="Homogenizing graphs")]
        elif edge_existence:
            print("Edge existence mode active: flattening relation names to generic node-type relations...")
            raw_graphs = [self.flatten_edge_types_for_edge_existence(g) for g in tqdm.tqdm(raw_graphs, desc="Flattening edge types")]

        train_data, test_data, val_data = self.split(raw_graphs)

        if homogenization:
            new_edge_types = set(et for g in train_data for et in g.edge_types)
            print(f"Homogenization mode: using {len(new_edge_types)} generic edge types from training data.")
        elif edge_existence:
            new_edge_types = set(et for g in train_data for et in g.edge_types)
            print(f"Edge existence mode: using {len(new_edge_types)} generic edge types from training data.")
        elif top_k_relations is not None:
            relation_mapping, new_edge_types = self.create_relation_mapping(train_data, top_k=top_k_relations)
            print(f"Applying relation mapping to training data...")
            train_data = [self.apply_relation_mapping(g, relation_mapping) for g in tqdm.tqdm(train_data, desc="Mapping train graphs")]
            print(f"Applying relation mapping to validation data...")
            val_data = [self.apply_relation_mapping(g, relation_mapping) for g in tqdm.tqdm(val_data, desc="Mapping val graphs")]
            print(f"Applying relation mapping to test data...")
            test_data = [self.apply_relation_mapping(g, relation_mapping) for g in tqdm.tqdm(test_data, desc="Mapping test graphs")]
        else:
            new_edge_types = set(et for g in train_data for et in g.edge_types)
            print(f"No relation mapping applied. Using all {len(new_edge_types)} unique edge types from training data.")
        
        node_types = set(nt for g in train_data for nt in g.node_types)
        print(f"Unique node types in training data: {node_types}")
        print(f"Unique edge types after mapping: {new_edge_types}")

        if self.random_initialization:
            print("Randomizing node features for all graphs...")
            for g in tqdm.tqdm(train_data + val_data + test_data, desc="Randomizing features"):
                self.randomize_features(g)
        
        return train_data, val_data, test_data, new_edge_types, node_types


