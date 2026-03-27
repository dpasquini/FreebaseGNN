import json
import multiprocessing
import random
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import constants
from utils.DBManager import DBManager
from utils.FreebaseIndex import FreebaseIndex
from utils.GraphUtils import GraphUtils
from utils.IOOperations import IOOperations


class GraphEnhance:

    def __init__(self, param):
        self.parameters = param
        self.entities_idx = FreebaseIndex(self.parameters.entities_db_path)
        self.literals_idx = FreebaseIndex(self.parameters.literals_db_path)
        self.types_idx = FreebaseIndex(self.parameters.types_db_path)
        self.strategy = getattr(self.parameters, "type_strategy", "classic")

    def pick_types_from_list(self, candidates, type_sources_dict):
        """
        Select types for a given node based on frequency thresholds.

        Parameters:
            - candidates: list of (pred_t, t) tuples for a node
            - type_sources_dict[t]  = collection of sources for that type
        Returns: list of (pred_t, t) tuples that meet the highest frequency threshold, or empty list if none meet any threshold
        """
        for thr in constants.THRESHOLDS:
            level = [
                (pred_t, t)
                for (pred_t, t) in candidates
                if len(type_sources_dict[t]) >= thr
            ]
            if level:
                return level
        return []
    
    def pick_max_size_type_for_node(self, candidates, type_sources_dict):
        """
        Select types for a given node based on frequency thresholds, 
        but if multiple types meet the same threshold, pick the one with the largest number of sources.

        @parameters:
            - candidates: list of (pred_t, t) tuples for that node
            - type_sources_dict[t]  = collection of sources for that type

        Returns: (pred_t, t) for the selected type where pred_t is the predicate and t is the type, or None if no types meet any threshold
        """
        type_with_max_sources = None
        for thr in constants.THRESHOLDS:
            for pred_t, t in candidates:
                if len(type_sources_dict[t]) >= thr:
                    if type_with_max_sources is None:
                        type_with_max_sources = (pred_t, t)
                    else:    
                        if len(type_sources_dict[t]) > len(type_sources_dict[type_with_max_sources[1]]):
                            type_with_max_sources = (pred_t, t)
        
        return type_with_max_sources

    def enhance_graph_with_neighs(self, graph_id):
        """
        Enhance the graph by adding new neighbors for existing nodes, 
        along with type edges for the newly added neighbors based on the heuristic strategy.
        If the input graph is already typed, we do not attempt to re-type existing nodes or add new types to existing nodes,
        but we do add new structural edges and type edges for newly discovered neighbors.

        Parameters:
            - graph_id: identifier for the graph to enhance (used to locate the input graph files and save the enhanced graph)
        Returns:
            - enhanced_nodes: DataFrame of nodes in the enhanced graph
            - enhanced_edges: DataFrame of edges in the enhanced graph
            - graph_attrs: dictionary of graph attributes (same as input graph, can be modified if needed)
        """
        # We assume the input graph is already typed, so we only add new structural edges and type edges for newly discovered neighbors, 
        # but we do not attempt to re-type existing nodes or add new types to existing nodes. 
        # This way we keep the enhancement focused on expanding the graph with new entities and their most relevant types, 
        # without altering the original typing of existing nodes.

        IS_ENHANCED = 1 # flag to indicate new edges/nodes added by this enhancement step

        # Set up paths for the input typed graph
        typed_graphs_dir = Path(self.parameters.typed_graphs_dir)

        edges_path = typed_graphs_dir.joinpath(f"typed_graph.edges.{graph_id}.parquet")
        nodes_path = typed_graphs_dir.joinpath(f"typed_graph.nodes.{graph_id}.parquet")
        graph_path = typed_graphs_dir.joinpath(f"typed_graph.graph.{graph_id}.json")

        edges = pd.read_parquet(edges_path, engine="fastparquet")
        nodes = pd.read_parquet(nodes_path, engine="fastparquet")
        with graph_path.open("r", encoding="utf-8") as gf:
            graph_attrs = json.load(gf)

        def norm_key(k: str) -> str:
            """
            Normalize predicate keys by sanitizing and applying any additional rules.
            This should be consistent with how predicates are normalized when building the index and when saving graphs.
            """
            return IOOperations.sanitize(k)

        # Exclude certain predicates from consideration for enhancement (e.g., literals, types, meta-preds)
        excluded_predicates = constants.LITERAL_PREDICATES_SAN.union({constants.TYPE_PREDICATE_SAN})

        # Build a fast lookup of existing normalized edges to prevent re-adding them
        existing_edges = set(zip(
            edges["source"].astype(str),
            edges["target"].astype(str),
            edges["key"].astype(str).map(norm_key),
        ))

        # Prepare DB connection for neighbor retrieval
        entities_db_path = Path(self.parameters.entities_db_path)
        conn_db = DBManager.get_db(entities_db_path)
        cursor = conn_db.cursor()

        # Consider only m./g. nodes already present
        mask = (nodes["node"].astype(str).str.startswith("m.") | nodes["node"].astype(str).str.startswith("g."))
        nodes_to_query = set(nodes.loc[mask, "node"].astype(str))

        query_cache = {}
        nodes_to_fetch = [n for n in nodes_to_query if n not in query_cache]
        if nodes_to_fetch:
            batch_size = 900
            for i in range(0, len(nodes_to_fetch), batch_size):
                batch = nodes_to_fetch[i:i + batch_size]
                placeholders = ",".join(["?"] * len(batch)) # for SQL IN clause
                cursor.execute(f"SELECT key, value FROM freebase_index WHERE key IN ({placeholders})", batch) # fetch neighbors for the batch
                for key, value_json in cursor.fetchall():
                    query_cache[key] = json.loads(value_json)

                # chunked extensions
                for key in batch:
                    query_cache.setdefault(key, []) # ensure key exists even if no neighbors
                    chunk_index = 0
                    while True:
                        chunk_key = f"{key}_chunk_{chunk_index}" # consistent with how neighbors are stored in the index
                        cursor.execute("SELECT value FROM freebase_index WHERE key = ?", (chunk_key,)) # fetch chunk
                        r = cursor.fetchone()
                        if not r:
                            break
                        query_cache[key].extend(json.loads(r[0])) # append chunk neighbors to main list
                        chunk_index += 1

        # We will collect new edges in a set to ensure no duplicates, and only convert to DataFrame at the end. Same for new nodes.
        all_discovered_neighbors = set()  # nodes to add
        new_struct_edges = set()  # (source, target, key) structural edges
        
        # We'll build type edges later after we know the neighbors' types
        random_seed = getattr(self.parameters, "random_seed", None)
        if random_seed is not None:
            random.seed(random_seed)

        for node in nodes_to_query:
            conns = query_cache.get(node) or []
            if not conns:
                continue

            # Split incoming/outgoing, sanitize predicates, filter excluded/prefixes
            incoming, outgoing = [], []
            for c in conns:
                p_norm = norm_key(c.get("p", "")) # normalize predicate key for consistent filtering and edge representation
                if (not p_norm) or (p_norm in excluded_predicates) or any(
                        p_norm.startswith(pref) for pref in constants.EXCLUDED_PREDICATE_PREFIXES
                ): # filter out edges with excluded predicates or prefixes (e.g., literals, types, meta-preds)
                    continue
                
                # store with normalized predicate and consistent source/target keys for later edge construction
                if c.get("dir") == "out": # outgoing edge: node -> c.get("o")
                    outgoing.append({"p": p_norm, "s": node, "o": c.get("o")}) 
                else:
                    incoming.append({"p": p_norm, "s": c.get("s"), "o": node})

            # Sample neighbors if there are more than the threshold, ensuring we keep the same number of incoming/outgoing if possible
            k_out = min(constants.SAMPLE_SIZE_OUT, len(outgoing))
            k_in = min(constants.SAMPLE_SIZE_IN, len(incoming))
            sampled = (random.sample(outgoing, k_out) if k_out > 0 else []) + \
                      (random.sample(incoming, k_in) if k_in > 0 else [])

            if not sampled:
                continue

            # Deduplicate the sampled connections by (src, dst, key)
            # Note: we use the normalized predicate key for deduplication to ensure consistency with existing edges, but we keep the original keys in the edge data for later construction (we'll normalize again when building the final edge set)
            dedup_sampled = {
                (str(c["s"]), str(c["o"]), c["p"]) for c in sampled
                if isinstance(c["s"], str) and isinstance(c["o"], str) and isinstance(c["p"], str)
            }

            # Keep only truly new edges (not in existing graph, not already added in this run)
            for src, dst, key in dedup_sampled:
                e = (src, dst, key)
                if e not in existing_edges and e not in new_struct_edges:
                    new_struct_edges.add(e)
                
                # Determine the neighbor w.r.t. the current center `node`
                if dst == node:
                    neigh = src         # incoming edge: neighbor is the source
                elif src == node:
                    neigh = dst         # outgoing edge: neighbor is the target
                else:
                    # Shouldn't happen for properly built samples, but skip if it does
                    continue

                # track only MID/GUID neighbors to add as nodes
                if (neigh.startswith("m.") or neigh.startswith("g.")):
                    all_discovered_neighbors.add(neigh)

        # If no edges discovered, return original graph
        if not new_struct_edges and not all_discovered_neighbors:
            return nodes, edges, graph_attrs

        # we will add type edges for all discovered neighbors that have types in the index, 
        # but we need to pick which types to add based on the heuristic 
        # (e.g., if multiple types, prefer those with more sources, but allow single-source types if no multi-source ones). 
        # To do this, we first gather all candidate types for all discovered neighbors, along with their source counts, 
        # and then apply the heuristic to pick which ones to actually add as edges. 
        # This way we ensure that we only add a manageable number of type edges per neighbor, 
        # while still capturing the most relevant types according to our criteria.
        
        all_nonmeta_by_node = defaultdict(list)  # node -> [(pred_t_norm, type_id)]
        all_meta_by_node = defaultdict(list)
        nonmeta_type_sources = defaultdict(set)  # type_id -> {node,...}
        meta_type_sources = defaultdict(set)
        name_by_node = {}

        for n in all_discovered_neighbors:
            # types
            for pred_t, t in self.types_idx.get_types_for_node(n):
                if not isinstance(t, str) or not t:
                    continue
                pred_t_norm = norm_key(pred_t)
                if GraphUtils.is_meta_type(t):
                    all_meta_by_node[n].append((pred_t_norm, t))
                    meta_type_sources[t].add(n)
                else:
                    all_nonmeta_by_node[n].append((pred_t_norm, t))
                    nonmeta_type_sources[t].add(n)
            # literal name
            lit = self.literals_idx.get_literals_for_node(n)
            if lit is not None:
                name_by_node[n] = str(lit[1])

        # choose types per neighbor
        type_edges_set = set()
        type_nodes = set()
        new_nodes_records = []

        for n in all_discovered_neighbors:
            nonmeta_list = all_nonmeta_by_node.get(n, [])
            meta_list = all_meta_by_node.get(n, [])
            
            if self.strategy == "multihot" or self.strategy == "classic":
                chosen = self.pick_types_from_list(nonmeta_list, nonmeta_type_sources)
                if not chosen:
                    # if no non-meta types meet any threshold, consider meta types as backup options (but only if they have at least <MAX_THRESHOLD> sources to avoid very noisy single-source meta types)
                    chosen = self.pick_types_from_list(meta_list, meta_type_sources)

                for pred_t_norm, t in chosen:
                    e = (str(n), str(t), pred_t_norm)
                    if e not in existing_edges and e not in type_edges_set:
                        type_edges_set.add(e)
                    type_nodes.add(t)

                types_for_node = [t for (_, t) in chosen] or None
                new_nodes_records.append({
                    "node": n,
                    "is_original": False,
                    "is_type": False,
                    "types": types_for_node,
                    "name": name_by_node.get(n),
                    "enhanced": IS_ENHANCED,
                })
            elif self.strategy == "singleton":
                chosen = self.pick_max_size_type_for_node(nonmeta_list, nonmeta_type_sources)
                if not chosen:
                    # if no non-meta types meet any threshold, consider meta types as backup options (but only if they have at least <MAX_THRESHOLD> sources to avoid very noisy single-source meta types)
                    chosen = self.pick_max_size_type_for_node(meta_list, meta_type_sources)
                    if chosen is None:
                        # we can use a placeholder type to indicate that we attempted to type the node but found no suitable types
                        # which is different from not having attempted to type it at all (e.g., if it had no types in the index). 
                        # This way we can still add a type edge to 'unknown_type' for these nodes, which might be useful for downstream models to learn that these nodes are of unknown type rather than just untyped.
                        chosen = 'unknown_type'

                new_nodes_records.append({
                    "node": n,
                    "is_original": False,
                    "is_type": False,
                    "types": [chosen[1]], # wrap in list for consistency with multihot format
                    "name": name_by_node.get(n),
                    "enhanced": IS_ENHANCED,
                })

        new_nodes_df = pd.DataFrame(new_nodes_records).drop_duplicates(subset=["node"])
        new_edges_df = pd.DataFrame(list(new_struct_edges), columns=["source", "target", "key"])
        if not new_edges_df.empty:
            new_edges_df["enhanced"] = IS_ENHANCED
    
        # existing nodes + new neighbor nodes (without types yet)
        enhanced_nodes = pd.concat([nodes, new_nodes_df], ignore_index=True)  
        # existing edges + new structural edges (without type edges yet)
        enhanced_edges = pd.concat([edges, new_edges_df], ignore_index=True)
        
        # for classic strategy, we add type nodes as regular nodes in the graph (with is_type=True), and connect them with type edges.
        if self.strategy == "classic":
            type_nodes_records = [{
                "node": t,
                "is_original": False,
                "is_type": True,
                "types": None,
                "name": t,
                "enhanced": IS_ENHANCED,
            } for t in type_nodes]
            
            type_nodes_df = pd.DataFrame(type_nodes_records).drop_duplicates(subset=["node"])
            type_edges_df = pd.DataFrame(list(type_edges_set), columns=["source", "target", "key"])
            if not type_edges_df.empty:
                type_edges_df["enhanced"] = IS_ENHANCED
            # add type nodes to the graph
            enhanced_nodes = pd.concat([enhanced_nodes, type_nodes_df], ignore_index=True)
            # add type edges to the graph
            enhanced_edges = pd.concat([enhanced_edges, type_edges_df], ignore_index=True)

        # final deduplication of nodes and edges to ensure no duplicates, keeping the first occurrence (which will be the original graph's node/edge if it exists, otherwise the newly added one)
        enhanced_nodes = enhanced_nodes.drop_duplicates(subset=["node"], keep="first")
        enhanced_edges = enhanced_edges.drop_duplicates(subset=["source", "target", "key"], keep="first")

        return enhanced_nodes, enhanced_edges, graph_attrs

    def infer_cvt_type_candidates_from_preds(self, preds):
        """
        Given all predicates incident to a node, infer CVT-type prefixes:
        any prefix before last '.' that appears with >= <MAX_THRESHOLDS> distinct predicates.
        """
        by_prefix = defaultdict(set)
        for p in preds:
            if not isinstance(p, str):
                continue
            if "." not in p:
                continue
            prefix = p.rsplit(".", 1)[0]
            by_prefix[prefix].add(p)

        # Try prefixes with >= <MAX_THRESHOLDS> distinct preds (non-meta only)
        for thr in constants.THRESHOLDS:
            candidates = {
                prefix: sorted(ps)
                for prefix, ps in by_prefix.items()
                if len(ps) >= thr and not GraphUtils.is_meta_type(prefix)
            }
            if candidates:
                return candidates
        return {}

    def choose_bridge_name(self, details: dict) -> str:
        """
        details: { T: { "cvt_preds": [...], "incoming_like_preds": [...], "inferred": bool } }
        Choose a canonical type name for the bridge node.
        """
        items = list(details.items())
        # Prefer non-inferred types
        explicit = [(t, info) for t, info in items if not info.get("inferred", False)]
        candidates = explicit if explicit else items
        # Pick the one with most cvt_preds, tie-breaker: lexicographically smallest T
        candidates.sort(key=lambda x: (-len(x[1]["cvt_preds"]), x[0]))
        return candidates[0][0]

    def build_node_row_for_id(self, node_id, nodes_columns):
        """
        Create a node row for a non-bridge node (entity or type),
        respecting: node, is_original, name, types, is_type.
        """
        base = {c: pd.NA for c in nodes_columns}
        base["node"] = node_id

        # Heuristic: if not a MID, treat as schema/type node
        if not GraphUtils.is_mid(node_id):
            base["is_type"] = True
            base["name"] = node_id
            base["types"] = []
        else:
            # Entity-like
            base["is_type"] = False
            # Literal as name if exists
            lit = self.literals_idx.get_literals_for_node(node_id)
            if lit:
                # lit is (pred, value)
                _, val = lit
                base["name"] = val
            else:
                base["name"] = node_id

            # Types: non-meta only
            type_pairs = self.types_idx.get_types_for_node(node_id)
            ts = [IOOperations.sanitize(t) for _, t in type_pairs if not GraphUtils.is_meta_type(t)]
            base["types"] = ts

        # Newly created nodes from this function are not original graph nodes
        base["is_original"] = False
        base["enhanced"] = 1

        return base

    def enhance_graph_with_bridge(self, graph_id):

        def pred_matches_type_prefix(t_raw: str, p_raw: str) -> bool:
            """
            Heuristic to check if a predicate matches the expected pattern for a type prefix,
            which is that the predicate starts with the type prefix followed by a dot (e.g., for type "people.person", predicates like "people.person.name" would match).
            """
            return isinstance(t_raw, str) and isinstance(p_raw, str) and p_raw.startswith(t_raw + ".")

        def norm_key(k: str) -> str:
            """
            Normalize predicate keys by sanitizing and applying any additional rules.
            This should be consistent with how predicates are normalized when building the index and when saving graphs.
            """
            return IOOperations.sanitize(k)

        def get_type_label(type_id: str):
            """
            Get a human-readable label for a type ID, preferring the literal label if available, otherwise returning the type ID itself.
            """
            lit = self.literals_idx.get_literals_for_node(type_id)
            return str(lit[1]) if lit else None

        enhanced_graphs_dir = Path(self.parameters.enhanced_graphs_dir)

        edges_path = enhanced_graphs_dir.joinpath(f"enhanced_graph.edges.{graph_id}.parquet")
        nodes_path = enhanced_graphs_dir.joinpath(f"enhanced_graph.nodes.{graph_id}.parquet")
        graph_path = enhanced_graphs_dir.joinpath(f"enhanced_graph.graph.{graph_id}.json")

        edges = pd.read_parquet(edges_path, engine="fastparquet")
        nodes = pd.read_parquet(nodes_path, engine="fastparquet")
        with graph_path.open("r", encoding="utf-8") as gf:
            graph_attrs = json.load(gf)

        # Ensure 'is_bridge' column exists to mark bridge nodes, defaulting to False for existing nodes
        if "is_bridge" not in nodes.columns:
            nodes["is_bridge"] = pd.Series([False] * len(nodes), dtype="boolean")

        node_cols = list(nodes.columns)
        nodes_out = nodes.copy() # we will modify this in-place for existing nodes and also append new nodes to it, to keep all nodes together for easier deduplication and final output construction

        existing_nodes = set(nodes_out["node"].astype(str))
        existing_edges = set(zip(
            edges["source"].astype(str),
            edges["target"].astype(str),
            edges["key"].astype(str).map(norm_key),
        ))

        new_nodes_rows = []
        new_edges_rows = []

        bridge_map = {}

        # Detect potential CVT/bridge nodes among the existing nodes in the graph based on their neighbors and types in the index, 
        # and build a mapping of candidate bridge nodes to their potential CVT types and associated predicates.
        for _, row in nodes_out.iterrows():
            mid = row.get("node")
            if not isinstance(mid, str) or (not mid.startswith("m.") and not mid.startswith("g.")):
                continue

            # Likely not a CVT bridge if it already has a literal label
            if self.literals_idx.get_literals_for_node(mid) is not None:
                continue

            # get all neighbors and predicates for this node from the index (not just those in the graph, to have a more complete picture for type inference)
            neighbors = self.entities_idx.get_bridge_neighs(mid)
            if not neighbors:
                continue

            all_preds_raw = [p for (p, neigh) in neighbors if isinstance(p, str)]
            if not all_preds_raw:
                continue

            # get all explicit types for this node from the index, and filter out meta-types. We will use these to check for type-predicate patterns that are
            # indicative of CVT/bridge nodes 
            # (e.g., if we see a type "people.deceased_person" and predicates like "people.deceased_person.date_of_death", 
            # that would be a strong signal that this node is a CVT/bridge of type "people.deceased_person").
            type_pairs = self.types_idx.get_types_for_node(mid)
            explicit_types_raw = [t for _, t in type_pairs if isinstance(t, str) and not GraphUtils.is_meta_type(t)]

            cvt_type_to_preds = {}
            for t_raw in explicit_types_raw:
                t_style_preds = {
                    p_raw for p_raw in all_preds_raw
                    if pred_matches_type_prefix(t_raw, p_raw) and not GraphUtils.is_meta_pred(p_raw)
                }
                if t_style_preds:
                    cvt_type_to_preds[t_raw] = {"cvt_preds": sorted(t_style_preds), "inferred": False}

            if not cvt_type_to_preds:
                inferred = self.infer_cvt_type_candidates_from_preds(all_preds_raw)
                for t_raw, t_preds in inferred.items():
                    cvt_type_to_preds[t_raw] = {"cvt_preds": t_preds, "inferred": True}

            if cvt_type_to_preds:
                bridge_map[mid] = cvt_type_to_preds

        # For each candidate bridge node, 
        # we will add it to the graph if not already present, 
        # and connect it with type edges to its candidate CVT types (choosing which ones to connect based on the heuristic strategy),
        # and also connect it with structural edges to its neighbors that match the CVT-type predicate patterns.
        for mid, details in bridge_map.items():
            # todo enhance 
            neighbors = self.entities_idx.get_bridge_neighs(mid) or []
            if not neighbors:
                continue

            neighbors = list({(p, neigh) for (p, neigh) in neighbors
                              if isinstance(p, str) and isinstance(neigh, str)})

            chosen_type_id = self.choose_bridge_name(details)  # canonical type id
            chosen_label = get_type_label(chosen_type_id)  # prefer literal label if present
            chosen_name = chosen_label if chosen_label else chosen_type_id
            chosen_types = list(details.keys())
            # for the bridge node itself, we will assign the chosen canonical type name as its name,
            # and we will assign all candidate CVT types as its types (even if we only connect it with type edges to a subset of them based on the heuristic),
            # no heuristic for types asssigned to the node itself
            for p_raw, neigh in neighbors:
                if neigh not in existing_nodes and not GraphUtils.is_mid(neigh):
                    chosen_types.append(neigh)
            

            if mid in existing_nodes:
                # EXISTING node: update in place 
                # (set is_bridge=True, fill name if missing/blank/equals id, 
                # fill types if empty/NA but only if we have candidate CVT types to assign,
                # modify enhanced flag only if it is a new (bridge) node without any existing info, 
                # otherwise keep original enhanced flag to preserve info about whether it was added in a previous enhancement step or is an original node)
                idx = nodes_out.index[nodes_out["node"] == mid]
                if len(idx) > 0:
                    i = idx[0]
                    # name: set if missing/blank/equals id
                    cur_name = nodes_out.at[i, "name"] if "name" in nodes_out.columns else None
                    if pd.isna(cur_name) or str(cur_name).strip() in ("", mid):
                        nodes_out.at[i, "name"] = chosen_name
                    # types: fill only if empty/NA
                    cur_types = nodes_out.at[i, "types"] if "types" in nodes_out.columns else None
                    if (isinstance(cur_types, float) and pd.isna(cur_types)) or cur_types in (None, [], ()):
                        nodes_out.at[i, "types"] = list(set(chosen_types))
                    nodes_out.at[i, "is_bridge"] = True
            else:
                # NEW (bridge) node: add to graph with all available info (name, types) and is_bridge=True, enhanced=1
                existing_nodes.add(mid)
                base = {c: pd.NA for c in node_cols}
                base["node"] = mid
                base["is_type"] = False
                base["types"] = list(set(chosen_types))
                base["name"] = chosen_name
                base["is_original"] = False
                base["enhanced"] = 1
                base["is_bridge"] = True
                new_nodes_rows.append(base)

            # per-MID edge dedupe
            added_here = set()

            for t_raw, info in details.items():
                cvt_pred_set_raw = set(info["cvt_preds"])

                # Ensure type node (use label if available)
                # only add type nodes and type edges for classic strategy, 
                # for other strategies we will just add the type into the types column of the bridge node, 
                # without adding separate type nodes. 
                if self.strategy == "classic":
                    if t_raw not in existing_nodes:
                        existing_nodes.add(t_raw)
                        base_t = {c: pd.NA for c in node_cols}
                        base_t["node"] = t_raw
                        base_t["is_type"] = True
                        type_label = get_type_label(t_raw)
                        base_t["name"] = type_label if type_label else t_raw
                        base_t["types"] = []
                        base_t["is_original"] = False
                        base_t["enhanced"] = 1
                        base_t["is_bridge"] = False
                        new_nodes_rows.append(base_t)

                    # type edge from bridge node to type node
                    type_key_norm = norm_key("type.object.type")
                    e = (str(mid), str(t_raw), type_key_norm)
                    if e not in existing_edges and e not in added_here:
                        existing_edges.add(e);
                        added_here.add(e)
                        new_edges_rows.append({"source": mid, "target": t_raw, "key": type_key_norm, "enhanced": 1})

                # structural edges to neighbors that match the CVT-type predicate patterns
                for p_raw, neigh in neighbors:
                    if neigh not in existing_nodes:
                        if not GraphUtils.is_mid(neigh) and self.strategy != "classic":
                            continue
                        existing_nodes.add(neigh)
                        base_nei = {c: pd.NA for c in node_cols}
                        base_nei["node"] = neigh
                        base_nei["is_type"] = False if GraphUtils.is_mid(neigh) else True
                        # optional: set neighbor label if present (not required by your ask)
                        lit_nei = self.literals_idx.get_literals_for_node(neigh)
                        base_nei["name"] = str(lit_nei[1]) if lit_nei else neigh
                        base_nei["types"] = []
                        base_nei["is_original"] = False
                        base_nei["enhanced"] = 1
                        base_nei["is_bridge"] = False
                        new_nodes_rows.append(base_nei)

                    if p_raw in cvt_pred_set_raw:
                        key_norm = norm_key(p_raw)
                        e2 = (str(mid), str(neigh), key_norm)
                        if e2 not in existing_edges and e2 not in added_here:
                            existing_edges.add(e2);
                            added_here.add(e2)
                            new_edges_rows.append({"source": mid, "target": neigh, "key": key_norm, "enhanced": 1})

        # After processing all candidate bridge nodes, we will have collected new nodes and edges to add to the graph.
        # We will then construct the final enhanced graph by combining the original graph with the new nodes and edges, 
        # ensuring to deduplicate nodes and edges properly (e.g., if a bridge node was 
        # already present in the original graph, we update it in place rather than adding a duplicate, and we ensure no duplicate edges are added).
        nodes_enhanced = (
            pd.concat([nodes_out, pd.DataFrame(new_nodes_rows)], ignore_index=True)
            if new_nodes_rows else nodes_out
        )
        nodes_enhanced["is_bridge"] = nodes_enhanced["is_bridge"].astype("boolean") # Ensure dtype is nullable boolean (concat can upcast)
        nodes_enhanced = nodes_enhanced.drop_duplicates(subset=["node"], keep="first")
        nodes_enhanced["is_bridge"] = nodes_enhanced["is_bridge"].astype("boolean").fillna(False)

        edges_enhanced = (
            pd.concat([edges, pd.DataFrame(new_edges_rows)], ignore_index=True)
            if new_edges_rows else edges
        )
        edges_enhanced["key"] = edges_enhanced["key"].astype(str).map(norm_key)
        edges_enhanced = edges_enhanced.drop_duplicates(subset=["source", "target", "key"], keep="first")

        return edges_enhanced, nodes_enhanced, graph_attrs

    def enhance_graphs_w_neighbors(self, files_to_process, output_dir):
        """
        Parallel enhancement of graphs with neighbors, controlled by a provided list of graph identifiers to process.

        Parameters:
            - files_to_process: List of graph identifiers (e.g., ['12345', '67890']) that still need enhancement.
            - output_dir: Path object pointing to the directory where enhanced graphs should be saved.
        """
        GraphUtils.log_with_time(f"Starting parallel enhancement for {len(files_to_process)} remaining graphs.")

        # This line is now controlled by the new argument, not os.listdir
        graph_files = files_to_process

        num_processes = 6
        worker_func = partial(self.enhance_graph_with_neighs)

        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=2) as pool:
            results_iterator = pool.imap_unordered(worker_func, graph_files)
            for nodes, edges, graph_attrs in tqdm(results_iterator, total=len(graph_files),
                                                  desc="Enhancing Remaining Graphs"):
                IOOperations.save_graph(edges, nodes, graph_attrs, output_dir, prefix="enhanced_graph")

        GraphUtils.log_with_time("Parallel enhancement phase complete.")

    def retrieve_bridges(self, files_to_process, output_dir):
        """Main function to orchestrate parallel data retrieval."""

        num_processes = self.parameters.num_processes

        worker_func = partial(self.enhance_graph_with_bridge)
        print(f"Starting parallel processing on {len(files_to_process)} graphs using {num_processes} cores...")

        with multiprocessing.Pool(
                processes=num_processes, maxtasksperchild=5
        ) as pool:
            results_iterator = pool.imap_unordered(worker_func, files_to_process)
            for edges, nodes, graph_attrs in tqdm(results_iterator, total=len(files_to_process),
                                                  desc="Processing graphs"):
                IOOperations.save_graph(edges, nodes, graph_attrs, output_dir, "bridge_enhanced_graph")

        print("Parallel processing complete. Merging results...")

    def enhance_graphs(self):
        """
        Main function to orchestrate the graph enhancement process, including both neighbor enhancement and bridge resolution steps.
        """
        # enhance graphs with freebase neighbors 5 + 5 (original version)
        enhanced_graphs_dir = self.parameters.enhanced_graphs_dir
        typed_graphs_dir = self.parameters.typed_graphs_dir
        output_dir = self.parameters.output_dir

        enhanced_graphs_dir_path = Path(enhanced_graphs_dir)
        typed_graphs_dir_path = Path(typed_graphs_dir)

        if not typed_graphs_dir_path.exists():
            print(f"Typed graphs directory {typed_graphs_dir} does not exist. Please run the typing step first.")
            return

        output_dir_path = Path(output_dir)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True)

        # Identify what's already done
        all_original_files = {
            path.name.split(".")[2]  # enhanced_graph.edges.<ID>.parquet
            for path in typed_graphs_dir_path.glob("typed_graph.edges.*.parquet")
            if len(path.name.split(".")) >= 4  # safety check; optional
        }

        # Check completed enhanced files
        completed_original_files = {
            path.name.split(".")[2]  # enhanced_graph.edges.<ID>.parquet
            for path in enhanced_graphs_dir_path.glob("enhanced_graph.edges.*.parquet")
            if len(path.name.split(".")) >= 4  # safety check; optional
        }

        GraphUtils.log_with_time(
            f"Found {len(completed_original_files)} already enhanced graphs out of {len(all_original_files)} total.")

        # Identify and process remaining files
        files_to_process = list(all_original_files - completed_original_files)

        if not files_to_process:
            GraphUtils.log_with_time("All graphs have already been enhanced. Nothing to do.")
        else:
            GraphUtils.log_with_time(f"Found {len(files_to_process)} graphs remaining to be enhanced.")

            # Run the parallel enhancer ONLY on the remaining files
            self.enhance_graphs_w_neighbors(files_to_process, output_dir_path)

        GraphUtils.log_with_time("Process complete.")
        DBManager.close_all()

    def resolve_bridges(self):
        enhanced_graphs_dir = self.parameters.enhanced_graphs_dir
        enhanced_graphs_dir_path = Path(enhanced_graphs_dir)
        if not enhanced_graphs_dir_path.exists():
            print(f"Enhanced graphs directory {enhanced_graphs_dir} does not exist. Please run the neighbor enhancement step first.")
            return

        output_dir = self.parameters.output_dir
        output_dir_path = Path(output_dir)

        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True)

        all_graph_identifiers = {
            path.name.split(".")[2]  # enhanced_graph.edges.<ID>.parquet
            for path in enhanced_graphs_dir_path.glob("enhanced_graph.edges.*.parquet")
            if len(path.name.split(".")) >= 4  # safety check; optional
        }
        print(f"Discovered {len(all_graph_identifiers)} total possible graphs.")

        files_to_process = [
            identifier for identifier in all_graph_identifiers
        ]

        self.retrieve_bridges(files_to_process, output_dir_path)
        DBManager.close_all()
