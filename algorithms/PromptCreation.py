import json
import os
import pickle
import random
import time
from collections import defaultdict
from functools import partial
from itertools import chain  # For more efficient iteration
from multiprocessing import Pool, cpu_count
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.FreebaseIndex import FreebaseIndex
from utils.IOOperations import IOOperations


IGNORE_TYPES = {'common.topic', 'freebase', 'ontology', 'common.notable_for', 'measurement_unit',
                'common.webpage'}  # Add other generic types here

TYPE_PROMPT_TEMPLATE = (
    "Below is the name of a conceptual type from a knowledge graph. "
    "Write a compact definition focused on discriminative meaning.\n\n"
    "Here are the details:\n\n"
    "- Concept Name: {type_name}\n\n"
    "Rules:\n"
    "- Use exactly one sentence.\n"
    "- Avoid generic fillers (e.g., important, notable, common).\n"
    "- Avoid repeating the concept name verbatim if unnecessary.\n"
    "- Keep only task-useful meaning.\n"
    "Return only the definition sentence."
)

PROMPT_TEMPLATE = """
You are an expert in knowledge graphs.

Write a compact functional description of the node from type, label, and connections.

Rules:
- Output exactly 2 bullet points.
- Each bullet must contain one non-redundant, discriminative fact.
- Do not repeat the label unless needed for disambiguation.
- Prefer relation-bearing facts (role, domain, strong links).
- Avoid filler and vague adjectives.
- Max 14 words per bullet.
- Output only the 2 bullets.

Type: {node_type}
Label: {label}
Connections:
{connections_sample}

Description:
"""

class PromptCreation:

    def __init__(self, param):
        self.parameters = param
        self.bridge_types_idx = FreebaseIndex(self.parameters.bridge_types_db_path)
        self.entities_idx = FreebaseIndex(self.parameters.entities_db_path)

    def build_edge_prompt(self, rel, edge_set):
        rel_parts = rel.split('_DOT_')
        type_edge = f"{rel_parts[0]}.{rel_parts[1]}"
        prop = f"{rel_parts[-1]}"
        formatted_lines = [
            f"\t - {source_name} ({source_type}) → {target_name} ({target_type})"
            for source_type, target_type, _, _, source_name, target_name in edge_set
        ]
        formatted_lines = "\n".join(formatted_lines)

        prompt = (
            f"Write as if for Wikipedia Simple English edition. Below is a set of raw knowledge-graph data about a relationship type. Each line shows a source entity connected to a target entity, with their respective types provided in parentheses.\n"
            f"Think for a moment about what the samples and their types imply, then give a precise and general definition of this relationship, suitable for a general audience (at least 60 words; never exceed 80 words, aim for ~70).\n"
            f"Use your best judgment to express the meaning in natural English. Use the provided types to understand the core pattern, but describe that pattern using general, descriptive terms (like 'an organization' or 'a creative work') rather than the raw technical labels.\n"
            f"Avoid simply echoing the labels, listing examples or name any source/target. You may allude to the distinct roles of the connected entities if doing so is essential to explaining the relationship's core meaning.\n"
            f"Here are the details:\n"
            f"– Relationship label:\n"
            f"\t type:{type_edge}, property: {prop}\n"
            f"– Sample connections (Source (Source Type) → Target (Target Type)):\n")
        prompt += formatted_lines
        prompt += "\n"
        prompt += f"Do not mention any of the sample entities by name.\n"
        return prompt

    def build_prompt(self, node_name, neighbors, variant='B'):
        conn_list_incoming = []
        conn_list_outgoing = []
        for rel, obj, literal, edge_type in neighbors:
            rel_parts = rel.split('_DOT_')
            if edge_type == "i":
                if variant == 'B':
                    if len(rel_parts) == 3:
                        conn_list_incoming.append(
                            f'domain: {rel_parts[0]}, type: {rel_parts[1]}, property: {rel_parts[-1]}, object: {literal}')
                    elif len(rel_parts) == 4:
                        conn_list_incoming.append(
                            f'domain: {rel_parts[0]}, subject type: {rel_parts[1]}, mediator property: {rel_parts[2]}, mediator\'s property; {rel_parts[-1]}, object: {literal}')
                else:
                    conn_list_incoming.append(f'{rel_parts[-1]}: {literal}')
            elif edge_type == "o":
                if variant == 'B':
                    if len(rel_parts) == 3:
                        conn_list_outgoing.append(
                            f'domain: {rel_parts[0]}, type: {rel_parts[1]}, property: {rel_parts[-1]}, object: {literal}')
                    elif len(rel_parts) == 4:
                        conn_list_outgoing.append(
                            f'domain: {rel_parts[0]}, subject type: {rel_parts[1]}, mediator property: {rel_parts[2]}, mediator\'s property: {rel_parts[-1]}, object: {literal}')
                else:
                    conn_list_outgoing.append(f'{rel_parts[-1]}: {literal}')
        conn_text_incoming = "\n\t- ".join(conn_list_incoming)
        conn_text_outgoing = "\n\t- ".join(conn_list_outgoing)

        if variant == 'A':
            prompt = (
                f"Below is a set of raw knowledge graph data about an entity. Each fact is represented as a property and value, using technical labels (such as birth_date, place_of_birth, president_of, spouse, etc.). Some connections refer to relationships with other entities.\n"
                f"Your task is to produce compact, task-useful facts about the main entity. Use your best judgment to interpret the property labels and present information in natural English. "
                f"Keep only discriminative information and drop repetitions.\n\n"
                f"Here are the details:\n\n"
                f"- Name: {node_name}\n")
            if conn_list_outgoing:
                prompt += (f"- Outgoing properties:\n"
                           f"\t- {conn_text_outgoing}\n")
            if conn_list_incoming:
                prompt += (f"- Incoming properties:\n"
                           f"\t- {conn_text_incoming}\n\n")
            prompt += (
                "Return exactly 3 bullet points with unique facts. "
                "Each bullet max 16 words. Prefer role/type/strong relations over generic biography.")
        else:  # Variant B code, kept for completeness
            prompt = (
                f"Below is a set of raw knowledge graph data about an entity. Each fact is represented as a property and value, using technical labels (such as birth_date, place_of_birth, president_of, spouse, etc.). Some connections refer to relationships with other entities.\n"
                f"Your task is to produce compact, task-useful facts about the main entity. Use your best judgment to interpret the property labels and present information in natural English. "
                f"Keep only discriminative information and drop repetitions.\n\n"
                f"Here are the details:\n\n"
                f"- Name: {node_name}\n")
            if conn_list_outgoing:
                prompt += (f"- Outgoing properties:\n"
                           f"\t- {conn_text_outgoing}\n")
            if conn_list_incoming:
                prompt += (f"- Incoming properties:\n"
                           f"\t- {conn_text_incoming}\n\n")
            prompt += (
                "Return exactly 3 bullet points with unique facts. "
                "Each bullet max 16 words. Prefer role/type/strong relations over generic biography.")
        return prompt

    def build_intermediary_prompt(self, node_type, label, connections):
        """
        Formats the prompt template with specific data for an intermediary node.

        Parameters:
            - node_type (str): The technical type of the node.
            - label (str): The human-readable label for the node type.
            - connections (list): A list of (predicate, name) tuples.

        Returns:
            - str: The fully formatted prompt ready to be sent to the LLM.
        """
        # Format the list of connections into a clean, multi-line string
        connections_sample = "\n".join([f"- {pred}: '{name}'" for pred, name in connections])

        return PROMPT_TEMPLATE.format(
            node_type=node_type,
            label=label,
            connections_sample=connections_sample
        )

    def normalize_type(self, t, max_types=3):
        """
        Normalize a type or list of types into a concise, human-readable string. 
        This is a heuristic that can be customized based on the specific type naming conventions.
        For example, if t is a list of type IDs, we can:
        - Format the type IDs into a more readable form (e.g., converting 'people.person' to 'Person').
        
        Parameters:
            - t (str or list): A single type ID or a list of type IDs.
            - max_types (int): Maximum number of types to include in the output string.
        Returns:
            - str: A normalized, human-readable string representing the type(s).
        """
        if isinstance(t, list):
            # ensure consistent ordering
            t = sorted(set(t))
            t = ', '.join(t[:max_types])
        return str(t)

    def format_type_name(self, node_id):
        """
        Convert a raw type ID into a more human-readable format. This is a simple heuristic that can be improved with more complex logic if needed.
        For example, '/people/person' would become 'People Person', and '/location/city' would become 'Location City'.
        
        Parameters:
            - node_id (str): The raw type ID from the graph.
        Returns:
            - str: A human-readable version of the type name.
        """
        if not isinstance(node_id, str):
            return "Unknown Type"
        # Cleans up the ID and makes it title case
        return node_id.strip('/').replace('/', ' ').replace('.', ' ').replace('_', ' ').title()


    def new_process_single_graph_df_fast(self, identifier, take_type=True, prefix_graph_file_name="bridge_enhanced"):
        """
        Columns available:
          nodes: node, is_original, name, types, is_type, enhanced, is_bridge
          edges: source, target, key, enhanced

        Behavior:
          - PASS 1 (bridges): determine canonical type for each bridge, then call
            self.get_sample_connections_for_intermediary(canonical_type) to fetch
            (label, sampled_connections) from the DB, and build the intermediary prompt.
            (No graph-based sampling for bridges.)
          - PASS 2 (other originals): sample from the graph to build prompts.
          - local_edges_to_node_type: populated for all processed centers using current graph edges.
          - PRUNE: remove all enhanced nodes, except those that are type nodes AND are
            the target of at least one outgoing edge from an original node.
          - Save prompted_graph.* files.
        """

        t0 = time.perf_counter()
        graphs_dir = Path(self.parameters.graphs_to_prompt_dir)
        sample_sizes = getattr(self.parameters, "sample_sizes", {"o": 5, "i": 5})

        # Load graph dataframes and attributes
        edges_path = graphs_dir / f"{prefix_graph_file_name}.edges.{identifier}.parquet"
        nodes_path = graphs_dir / f"{prefix_graph_file_name}.nodes.{identifier}.parquet"
        graph_json_path = graphs_dir / f"{prefix_graph_file_name}.graph.{identifier}.json"

        edge_df = pd.read_parquet(edges_path, engine="fastparquet").copy()
        node_df = pd.read_parquet(nodes_path, engine="fastparquet").copy()
        with open(graph_json_path, "r") as f:
            graph_attrs = json.load(f)

        def norm_key(k: str) -> str:
            # Normalize keys for consistent matching, e.g., by lowercasing and replacing certain characters. 
            # This should match the normalization used when building the index.
            return IOOperations.sanitize(k)

        TYPE_PREDICATE = norm_key("type.object.type")

        # Ensure boolean columns are properly typed and missing values are handled
        for col in ("is_original", "is_type", "is_bridge"):
            if col in node_df.columns:
                node_df[col] = node_df[col].astype("boolean").fillna(False)
            else:
                node_df[col] = pd.Series(False, index=node_df.index, dtype="boolean")

        node_df["enhanced"] = node_df.get("enhanced", 0)
        node_df["enhanced"] = node_df["enhanced"].fillna(0).astype(int)
        edge_df["enhanced"] = edge_df.get("enhanced", 0)
        edge_df["enhanced"] = edge_df["enhanced"].fillna(0).astype(int)

        if "prompt" not in node_df.columns:
            node_df["prompt"] = pd.NA

        # make node id index
        node_df = node_df.set_index("node", drop=False)

        # precompute source/target indices for efficient access
        edge_sources = edge_df["source"].astype(str).to_numpy()
        edge_targets = edge_df["target"].astype(str).to_numpy()
        edge_keys = edge_df["key"].astype(str).to_numpy()

        # build quick lookup for edges by source and target
        src_to_idx, tgt_to_idx = defaultdict(list), defaultdict(list)
        for i, s in enumerate(edge_sources):
            src_to_idx[s].append(i)
        for i, t in enumerate(edge_targets):
            tgt_to_idx[t].append(i)

        def get_node_label(nid: str) -> str:
            """
            Get the human-readable label for a node, preferring the 'name' column if available and non-empty, otherwise falling back to the node ID itself.

            Parameters:
                - nid (str): The node ID for which to retrieve the label.
            Returns:
                - str: The label to use for this node in prompts.
            """
            if nid in node_df.index:
                val = node_df.at[nid, "name"]
                if isinstance(val, str) and val.strip():
                    return val
            return nid

        local_edges_to_node_type = defaultdict(set)

        # helper to choose canonical type for a bridge:
        def pick_canonical_type(mid: str):
            """
            Prefer an explicit type edge (type.object.type) from mid; 
            else first element in `types`; 
            else None."""
            # Retrieve type from explicit type edge if available
            for i in src_to_idx.get(mid, []):
                if edge_keys[i] == TYPE_PREDICATE:
                    return edge_targets[i]
            # Fallback to first type in 'types' column if available
            if mid in node_df.index:
                tv = node_df.at[mid, "types"]
                if isinstance(tv, (list, tuple)) and len(tv) > 0:
                    return tv[0]
            return None

        def _normalize_type_list(type_list):
            """
            Turn a list/tuple of type IDs into the normalized text we use downstream.
            """
            if hasattr(self, "normalize_type") and callable(self.normalize_type):
                return self.normalize_type(type_list)
            # minimal fallback: stable, human-ish single string
            if not type_list:
                return "entity"
            # ensure list of strings, unique, sorted
            ts = sorted({str(t) for t in (type_list if isinstance(type_list, (list, tuple)) else [type_list])})
            return "|".join(ts)

        def type_text_for_node(nid: str) -> str:
            """
            Return the *actual* type text for nid.
            - type nodes: the node id itself (or formatted), because their 'type' is the type
            - bridges: the canonical type id (selected the same way as for prompts)
            - entities: the 'types' list from nodes parquet
            """
            if nid not in node_df.index:
                return "entity"

            row = node_df.loc[nid]
            # 1) plain type nodes → their own id as the type text
            if bool(row.get("is_type", False)):
                # if you prefer a pretty label: return self.format_type_name(nid)
                return _normalize_type_list([self.format_type_name(nid)])

            # 2) bridges → canonical type id
            if bool(row.get("is_bridge", False)):
                ctype = pick_canonical_type(nid)
                if ctype:
                    return _normalize_type_list([ctype])

            # 3) regular entities → use the 'types' column
            ts = row.get("types", None)
            if isinstance(ts, (list, tuple)) and len(ts) > 0:
                return _normalize_type_list(ts)

            # 4) fallback
            return "entity"

        if take_type:
            type_nodes = node_df.index[node_df["is_type"] == True].tolist()
            for tid in type_nodes:
                # ensure a readable type name
                tname = node_df.at[tid, "name"]
                if not isinstance(tname, str) or not tname.strip():
                    tname = self.format_type_name(tid)
                    node_df.at[tid, "name"] = tname
                else:
                    tname = self.format_type_name(tname)

                # simple prompt for type nodes
                node_df.at[tid, "prompt"] = TYPE_PROMPT_TEMPLATE.format(type_name=tname)

                # fill local_edges_to_node_type for this type center (both directions; include all keys)
                for i in src_to_idx.get(tid, []):
                    k = edge_keys[i]
                    o = edge_targets[i]
                    local_edges_to_node_type[k].add((
                        type_text_for_node(tid), type_text_for_node(o),
                        tid, o,
                        get_node_label(tid), get_node_label(o)
                    ))
                for i in tgt_to_idx.get(tid, []):
                    k = edge_keys[i]
                    s = edge_sources[i]
                    local_edges_to_node_type[k].add((
                        type_text_for_node(tid), type_text_for_node(s),
                        tid, s,
                        get_node_label(tid), get_node_label(s)
                    ))

        # Pass 1: Handle bridges first, without graph-based sampling, using DB lookups for prompts. 
        # This ensures we have the most accurate type info for bridges when we populate local_edges_to_node_type for the next pass.
        bridges = node_df.index[node_df["is_bridge"] == True].tolist()
        for nid in bridges:
            canonical_type = pick_canonical_type(nid)
            # local_edges_to_node_type is always filled from the current graph
            # (both directions, excluding type edges)
            for i in src_to_idx.get(nid, []):
                k = edge_keys[i]
                if k == TYPE_PREDICATE:
                    continue
                o = edge_targets[i]
                local_edges_to_node_type[k].add((
                    type_text_for_node(nid), type_text_for_node(o),
                    nid, o,
                    get_node_label(nid), get_node_label(o)
                ))
            for i in tgt_to_idx.get(nid, []):
                k = edge_keys[i]
                if k == TYPE_PREDICATE:
                    continue
                s = edge_sources[i]
                local_edges_to_node_type[k].add((
                    type_text_for_node(nid), type_text_for_node(s),
                    nid, s,
                    get_node_label(nid), get_node_label(s)
                ))

            # Build prompt using the DB (the method returns (label, sampled_connections) or None)
            # sampled_connections is expected as list of (predicate, object)
            label, sampled_connections = None, []
            if canonical_type:
                try:
                    db_res = self.bridge_types_idx.get_neighs_for_type(canonical_type, 15)
                except Exception:
                    db_res = None
                if db_res:
                    label, sampled_connections = db_res

            # fallback if DB misses: use canonical type as label (after formatting), and no connections
            if not label:
                # final fallback: the bridge node's own 'name'
                br_name = node_df.at[nid, "name"] if nid in node_df.index else None
                label = br_name if isinstance(br_name, str) and br_name.strip() else canonical_type or nid
            if not label:
                # prefer type label from nodes if present (e.g., type nodes may carry names)
                if canonical_type and canonical_type in node_df.index:
                    cand = node_df.at[canonical_type, "name"]
                    if isinstance(cand, str) and cand.strip():
                        label = cand

            # NOTE: sampled_connections come from DB (already diversified). Do NOT resample here.
            # If DB returns raw predicates, pass them as-is to the prompt builder.
            node_df.at[nid, "prompt"] = self.build_intermediary_prompt(
                canonical_type or nid,
                label,
                sampled_connections or []
            )

        # Pass 2: Now handle the remaining original nodes, using graph-based sampling. 
        # By this point, local_edges_to_node_type is fully populated with all edges from the graph, 
        # including those connected to bridges with their canonical types.
        originals = node_df.index[(node_df["is_original"] == True) & (node_df["is_bridge"] != True)].tolist()

        empty_obj = np.empty(0, dtype=object)
        empty_dir = np.empty(0, dtype="U1")

        for nid in originals:
            center_label = get_node_label(nid)

            out_idx = src_to_idx.get(nid)
            in_idx = tgt_to_idx.get(nid)

            if out_idx:
                out_idx_arr = np.fromiter(out_idx, dtype=np.int64)
                out_keys = edge_keys[out_idx_arr]
                out_neighbors = edge_targets[out_idx_arr]
                dirs_out = np.full(out_keys.shape[0], "o", dtype="U1")
            else:
                out_keys, out_neighbors, dirs_out = empty_obj, empty_obj, empty_dir

            if in_idx:
                in_idx_arr = np.fromiter(in_idx, dtype=np.int64)
                in_keys = edge_keys[in_idx_arr]
                in_neighbors = edge_sources[in_idx_arr]
                dirs_in = np.full(in_keys.shape[0], "i", dtype="U1")
            else:
                in_keys, in_neighbors, dirs_in = empty_obj, empty_obj, empty_dir

            if out_keys.size == 0 and in_keys.size == 0:
                node_df.at[nid, "prompt"] = self.build_prompt(center_label, [], "A")
                continue

            keys = np.concatenate((out_keys, in_keys))
            neighs = np.concatenate((out_neighbors, in_neighbors))
            dirs = np.concatenate((dirs_out, dirs_in))

            grouped_i, grouped_o = [], []
            for k, d, neigh in zip(keys, dirs, neighs):
                # record in local_edges_to_node_type
                local_edges_to_node_type[k].add((
                    type_text_for_node(nid), type_text_for_node(neigh),
                    nid, neigh,
                    get_node_label(nid), get_node_label(neigh)
                ))
                tup = (k, neigh, get_node_label(neigh), d)
                if d == "i":
                    grouped_i.append(tup)
                else:
                    grouped_o.append(tup)

            # per-direction sampling from the graph
            sample_neighborhood = []
            for direction, bucket in (("o", grouped_o), ("i", grouped_i)):
                max_size = int(sample_sizes.get(direction, 0))
                if max_size <= 0 or not bucket:
                    continue
                random.shuffle(bucket)
                sample_neighborhood.extend(bucket[:max_size])

            node_df.at[nid, "prompt"] = self.build_prompt(center_label, sample_neighborhood, "A")

        # Pass 3: PRUNE. Now that all prompts are built and local_edges_to_node_type is fully populated, 
        # we can prune the graph by removing all enhanced nodes except those that are type nodes and are the target of at least one outgoing edge from an original node. 
        # This keeps the graph focused on the original nodes and their most relevant enhanced type nodes, 
        # while removing other enhanced nodes that may have been added for context but are not central to the original nodes' prompts.
        original_nodes = set(node_df.loc[node_df["is_original"] == True, "node"].astype(str))
        enhanced_type_nodes = set(
            node_df.loc[(node_df["enhanced"] != 0) & (node_df["is_type"] == True), "node"].astype(str))

        edge_sources_s = edge_df["source"].astype(str)
        edge_targets_s = edge_df["target"].astype(str)
        survive_targets = set(edge_targets_s[edge_sources_s.isin(original_nodes)])

        surviving_enhanced_types = enhanced_type_nodes.intersection(survive_targets)
        all_enhanced_nodes = set(node_df.loc[node_df["enhanced"] != 0, "node"].astype(str))
        to_remove = all_enhanced_nodes.difference(surviving_enhanced_types)

        if to_remove:
            node_df = node_df[~node_df["node"].astype(str).isin(to_remove)]
            edge_df = edge_df[
                ~edge_df["source"].astype(str).isin(to_remove) &
                ~edge_df["target"].astype(str).isin(to_remove)
                ]

        # optional tidy: drop isolated non-originals
        remaining_nodes = set(edge_df["source"].astype(str)).union(edge_df["target"].astype(str))
        node_df = node_df[
            node_df["is_original"] | node_df["node"].astype(str).isin(remaining_nodes)
            ]

        node_df = node_df.copy()
        node_df.index.name = None         # avoid name collision if some writer resets index
        node_df = node_df.reset_index(drop=True)  # ensure a plain RangeIndex with no extra column

        # Pass 4: Save the prompted graph and the local_edges_to_node_type for this graph.
        out_dir = Path(self.parameters.output_dir)

        IOOperations.save_graph(edge_df, node_df, graph_attrs, out_dir, "prompted_graph")

        print(f"[{identifier}] TOTAL took {time.perf_counter() - t0:.3f}s")
        return identifier, local_edges_to_node_type

    # Worker function for parallelizing edge prompt generation (unchanged)
    def generate_single_edge_prompt(self, item):
        k, v, sample_size = item
        s = min(sample_size, len(v))
        sample_nodes = random.sample(list(v), s)
        p = self.build_edge_prompt(k, sample_nodes)
        return k, p

    def create_prompt(self):
        """
        Main function to create prompts for all graphs in the input directory, with checkpointing and parallel processing.

        Checkpointing strategy:
        - After processing every N graphs (e.g., 30), save a checkpoint of the accumulated edges_to_node_type dictionary for that batch, and also save a cumulative list of all processed graph identifiers so far.
        - On startup, load the latest checkpoint (if any) to resume from the last saved state, ensuring that already processed graphs are not reprocessed.
        - This allows for efficient memory usage and the ability to resume after interruptions without losing progress.
        """
        graph_prefix_filename = self.parameters.graph_prefix_filename
        num_processes = self.parameters.num_processes
        graphs_to_prompt_dir_path = Path(self.parameters.graphs_to_prompt_dir)
        if not graphs_to_prompt_dir_path.exists():
            print(f"Error: Input directory not found at '{graphs_to_prompt_dir_path}'. Exiting.")
            return

        prompted_dir_path = Path(self.parameters.output_dir)
        if not prompted_dir_path.exists():
            prompted_dir_path.mkdir(parents=True)

        checkpoint_dir = Path(self.parameters.checkpoint_dir)
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True)

        # load existing checkpoint state
        checkpoint_files = sorted(checkpoint_dir.glob("edges_to_node_type_checkpoint*.pkl"))
        # cumulative list of processed graphs
        checkpoint_progress = sorted(checkpoint_dir.glob("edges_to_node_type_checkpoint*_processed_graphs.txt"))

        all_processed_identifiers = []

        # Load the latest checkpoint progress if available, to resume from there. 
        # This is a cumulative list of graph identifiers that have been processed in all previous batches.
        if checkpoint_progress:
            last_txt = checkpoint_progress[-1]
            print(f"Found latest processed-graphs checkpoint: {last_txt}")
            with open(last_txt, "r") as f:
                all_processed_identifiers = [
                    line.strip() for line in f
                    if line.strip() and not line.startswith("#")
                ]
            print(f"Loaded {len(all_processed_identifiers)} processed graph IDs from checkpoint.")
        else:
            print("No processed-graphs checkpoint found -> starting from scratch.")

        processed_set = set(all_processed_identifiers)

        # Discover all graph identifiers in the input directory based on the naming pattern.
        all_graph_identifiers = [
            path.name.split('.')[2]
            for path in graphs_to_prompt_dir_path.glob(f"{graph_prefix_filename}.edges.*.parquet")
        ]

        all_graph_identifiers = sorted(set(all_graph_identifiers))
        print(f"Discovered {len(all_graph_identifiers)} total possible graphs.")

        # Determine which graphs still need to be processed by comparing against the checkpointed list.
        files_to_process = [
            g for g in all_graph_identifiers
            if g not in processed_set
        ]

        if not files_to_process:
            print("All discovered graphs have been processed according to the latest checkpoint.")
            return

        #files_to_process = files_to_process[:80]
        print(f"Resuming. Will process the remaining {len(files_to_process)} graphs.")

        # Determine the starting batch index based on existing checkpoint files. 
        # This ensures we continue with a new batch number for the next checkpoint.
        if checkpoint_files:
            # Extract batch indices from existing checkpoint files to find the max index.
            def _batch_idx_from_path(p):
                name = p.name
                # edges_to_node_type_checkpoint{N}.pkl
                mid = name.split("edges_to_node_type_checkpoint", 1)[1]
                idx = mid.split(".pkl", 1)[0]
                return int(idx)

            batch_index = max(_batch_idx_from_path(p) for p in checkpoint_files)
        else:
            batch_index = 0

        # Current batch dict: only temporary accumulation (then flush and reset)
        final_edges_to_node_type = defaultdict(set)

        # Set a checkpoint frequency (e.g., every 30 graphs) to balance between memory usage and checkpoint granularity.
        checkpoint_every = 30

        # Process graphs in parallel using a worker function that returns the local_edges_to_node_type for each graph.
        worker_func = partial(
            self.new_process_single_graph_df_fast,
            take_type=True,
            prefix_graph_file_name = graph_prefix_filename
        )

        print(f"Starting parallel processing on {len(files_to_process)} graphs "
              f"using {num_processes} cores...")

        with Pool(processes=num_processes, maxtasksperchild=2) as pool:
            results_iterator = pool.imap_unordered(worker_func, files_to_process)

            for identifier, result_dict in tqdm(results_iterator,
                                                total=len(files_to_process),
                                                desc="Processing graphs"):
                if result_dict is None:
                    continue

                all_processed_identifiers.append(identifier)

                # Merge the local_edges_to_node_type from this graph into the current batch dictionary.
                for key, value_set in result_dict.items():
                    if value_set:
                        final_edges_to_node_type[key].update(value_set)

                # Every N graphs > flush batch to disk + reset RAM
                if len(all_processed_identifiers) % checkpoint_every == 0:
                    batch_index += 1

                    base_path = checkpoint_dir / f"edges_to_node_type_checkpoint{batch_index}"

                    # Paths for pkl
                    pickle_path_tmp = base_path.with_suffix(".pkl.tmp")
                    pickle_path = base_path.with_suffix(".pkl")

                    # Paths for txt cumulative list
                    txt_base_name = base_path.name + "_processed_graphs"
                    txt_path_tmp = base_path.with_name(txt_base_name + ".txt.tmp")
                    txt_path = base_path.with_name(txt_base_name + ".txt")

                    # Save ONLY the current batch dictionary
                    with pickle_path_tmp.open("wb") as f:
                        pickle.dump(final_edges_to_node_type, f)
                    pickle_path_tmp.replace(pickle_path)

                    # Save CUMULATIVE list of processed graphs so far
                    with txt_path_tmp.open("w") as f:
                        f.write(f"# List of {len(all_processed_identifiers)} graphs processed.\n")
                        for gid in sorted(all_processed_identifiers):
                            f.write(gid + "\n")
                    txt_path_tmp.replace(txt_path)

                    # Remove ONLY the previous batch txt (not the .pkl, to keep all batch dicts on disk until final aggregation)
                    if batch_index > 1:
                        prev_txt = checkpoint_dir / f"edges_to_node_type_checkpoint{batch_index - 1}_processed_graphs.txt"
                        if prev_txt.exists():
                            try:
                                prev_txt.unlink()
                            except OSError as e:
                                print(f"[Checkpoint] Warning: could not remove previous txt: {e}")

                    print(f"\n[Checkpoint] Saved batch {batch_index}: "
                          f"{len(final_edges_to_node_type)} keys -> {pickle_path}")

                    # Reset the batch dictionary for the next round of accumulation.
                    final_edges_to_node_type = defaultdict(set)

        # After processing all graphs, if there are any remaining entries in the batch dictionary that haven't been checkpointed yet, save them as a final batch.
        if final_edges_to_node_type:
            batch_index += 1
            base_path = checkpoint_dir / f"edges_to_node_type_checkpoint{batch_index}"

            # pkl paths
            pickle_path_tmp = base_path.with_suffix(".pkl.tmp")
            pickle_path = base_path.with_suffix(".pkl")

            # txt paths for cumulative list
            txt_base_name = base_path.name + "_processed_graphs"
            txt_path_tmp = base_path.with_name(txt_base_name + ".txt.tmp")
            txt_path = base_path.with_name(txt_base_name + ".txt")

            with pickle_path_tmp.open("wb") as f:
                pickle.dump(final_edges_to_node_type, f)
            pickle_path_tmp.replace(pickle_path)

            with txt_path_tmp.open("w") as f:
                f.write(f"# List of {len(all_processed_identifiers)} graphs processed.\n")
                for gid in sorted(all_processed_identifiers):
                    f.write(gid + "\n")
            txt_path_tmp.replace(txt_path)

            if batch_index > 1:
                prev_txt = checkpoint_dir / f"edges_to_node_type_checkpoint{batch_index - 1}_processed_graphs.txt"
                if prev_txt.exists():
                    try:
                        prev_txt.unlink()
                    except OSError as e:
                        print(f"[Checkpoint] Warning: could not remove previous txt: {e}")

            print(
                f"\n[Checkpoint] Saved final batch {batch_index}: "
                f"{len(final_edges_to_node_type)} keys -> {pickle_path}"
            )

        print("Parallel processing for node prompts finished.")

        print("\nAggregating all checkpoint batches into final_edges_to_node_type...")
        batch_pkls = sorted(checkpoint_dir.glob("edges_to_node_type_checkpoint*.pkl"))

        final_edges_to_node_type = defaultdict(set)

        for pkl_path in tqdm(batch_pkls, desc="Merging checkpoint batches"):
            with pkl_path.open("rb") as f:
                part = pickle.load(f)
            for k, vs in part.items():
                if vs:
                    final_edges_to_node_type[k].update(vs)

        edge_to_node_dict_path = checkpoint_dir / "full_edges_to_node_type.pkl"
        with edge_to_node_dict_path.open("wb") as f:
            pickle.dump(final_edges_to_node_type, f)

        total_unique_edges = len(final_edges_to_node_type)
        print(f"Final aggregated edge type dictionary saved at {edge_to_node_dict_path}.")
        print(f"Total unique edge keys: {total_unique_edges}")

        
        # Generate edge prompts in parallel using the final aggregated dictionary.
        # Each key is an edge type, and the value is a set of node type pairs that can be used as examples for prompt generation.
        print(f"\nStarting parallel generation of edge prompts for {total_unique_edges} unique edge types...")

        final_prompts = {}

        def edge_prompt_iter():
            for k, v in final_edges_to_node_type.items():
                yield (k, v, 10)

        with Pool(processes=num_processes) as pool:
            for k, p in tqdm(
                    pool.imap_unordered(self.generate_single_edge_prompt, edge_prompt_iter()),
                    total=total_unique_edges,
                    desc="Generating Edge Prompts"
            ):
                final_prompts[k] = p

        print("Parallel edge prompt generation finished.")

        # Write the final edge prompts in each graph's edge parquet file, matching on the 'key' column. 
        # This is done sequentially to avoid write conflicts, but could be parallelized if needed.
        print("\nAdding edge prompts to individual graph edge files...")

        if not all_processed_identifiers:
            print("Warning: No processed graph identifiers found. Skipping edge prompt addition.")
        else:

            for identifier in tqdm(sorted(all_processed_identifiers), desc="Updating graph edge files"):
                edge_path = prompted_dir_path / f"prompted_graph.edges.{identifier}.parquet"

                if not edge_path.exists():
                    print(f"Warning: Edge file not found for identifier '{identifier}' at {edge_path}. Skipping.")
                    continue

                try:
                    edge_df = pd.read_parquet(edge_path)  # pandas accepts Path objects
                    edge_df["prompt"] = edge_df["key"].map(final_prompts)
                    edge_df.to_parquet(edge_path, index=False)
                except Exception as e:
                    print(f"Error processing edge file for identifier '{identifier}': {e}. Skipping.")
                    continue
