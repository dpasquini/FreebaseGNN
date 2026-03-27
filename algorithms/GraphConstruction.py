#!/usr/bin/env python3
import itertools
import json
import multiprocessing
import os
import pickle
import random
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils import constants
from utils.FreebaseIndex import FreebaseIndex
from utils.GraphUtils import GraphUtils
from utils.IOOperations import IOOperations


class GraphConstruction:

    TYPE_STRATEGY = ('classic', 'singleton', 'multihot') # classic: the original heuristic with thresholds; singleton: pick only the single type with the most sources (even if <MAX THRESHOLD>); multihot: keep all types that meet the thresholds (potentially multiple per node)

    def __init__(self, param):
        self.parameters = param
        if param.literals_db_path and self.parameters.types_db_path:
            self.literals_idx = FreebaseIndex(self.parameters.literals_db_path)
            self.types_idx = FreebaseIndex(self.parameters.types_db_path)
        else: 
            self.literals_idx = None
            self.types_idx = None
        
        self.strategy = self.parameters.type_strategy
        self.n_graphs = self.parameters.n_graphs_for_test
        self.num_processes = self.parameters.num_processes

    def pick_types_for_node(self, node_id, by_node_dict, type_sources_dict):
        """
        Select types for a given node based on frequency thresholds.

        by_node_dict[node_id] = list[(pred_t, t)]
        type_sources_dict[t]  = collection of sources for that type
        """
        candidates = by_node_dict.get(node_id, [])
        for thr in constants.THRESHOLDS:
            level = [
                (pred_t, t)
                for (pred_t, t) in candidates
                if len(type_sources_dict[t]) >= thr
            ]
            if level:
                return level
        return []  # nothing found

    def pick_max_size_type_for_node(self, node_id, by_node_dict, type_sources_dict):
        """
        Select types for a given node based on frequency thresholds, 
        but if multiple types meet the same threshold, pick the one with the largest number of sources.

        @parameters:
            - node_id: the node for which we want to pick types
            - by_node_dict[node_id] = list[(pred_t, t)]
            - type_sources_dict[t]  = collection of sources for that type

        Returns: (pred_t, t) for the selected type where pred_t is the predicate and t is the type, or None if no types meet any threshold
        """
        candidates = by_node_dict.get(node_id, [])
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

    def read_types_literals(self, path, to_sanitize=True):
        """
        Reads a TSV file of types/literals into a dictionary mapping node_id -> set of (predicate, object) pairs.

        Parameters:
            - path: the path to the TSV file
            - to_sanitize: whether to sanitize the predicate and object values

        Returns:
            - types_literals: a dictionary mapping node_id -> set of (predicate, object) pairs
        """
        types_literals = defaultdict(set)
        i = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split('\t')
                if len(parts) != 3:
                    continue
                subj, predicate, obj = parts
                if to_sanitize:
                    predicate = IOOperations.sanitize(predicate)
                    if predicate == constants.TYPE_PREDICATE:
                        obj = IOOperations.sanitize(obj)
                types_literals[subj].add((predicate, obj))
                i += 1
                if i % 5000 == 0:
                    GraphUtils.log_with_time(f"Read {i:,} lines...")
                    print(f"Current line, subject: {subj}, predicate: {predicate}, object: {obj}")
        return types_literals

    @staticmethod
    def read_in_chunks_from_handle(handle, chunk_size):
        """
        Generator that yields lists of lines from a file handle in chunks of the specified size.
        
        Parameters:
            - handle: an open file handle to read from
            - chunk_size: the number of lines to include in each chunk
        Yields:
            - list of lines (strings) of length up to chunk_size
        """
        while True:
            lines = list(itertools.islice(handle, chunk_size))
            if not lines:
                break
            yield lines

    @staticmethod
    def init_worker(shared_entities, out_dir):
        """
        Called once per worker process. Stores shared state in globals.
        """
        global target_entities, output_dir, worker_file
        target_entities = shared_entities
        output_dir = out_dir

        # each worker writes to its own file based on PID
        os.makedirs(output_dir, exist_ok=True)
        worker_file = os.path.join(output_dir, f"worker_{os.getpid()}.tsv")

    def process_line_chunk(self, lines):
        """
        Worker: process a chunk of lines and append results to the worker's shard file.
        Returns the number of matches written (int).
        """
        results = []
        for line in lines:
            lm = constants.LITERAL_PATTERN.match(line)
            if lm:
                mid, predicate, literal_val, lang_code = lm.groups()
                if predicate in constants.LITERAL_PREDICATES:
                    lang = GraphUtils.normalize_lang(lang_code)
                    if lang == "en" or lang is None:
                        results.append(f"{mid}\t{predicate}\t{literal_val}\n")
                continue

            tm = constants.TYPE_PATTERN.match(line)
            if tm:
                mid, predicate, obj_type = tm.groups()
                if mid in target_entities and predicate == constants.TYPE_PREDICATE:
                    if not any(obj_type.startswith(sub) for sub in constants.NOT_COMMONS):
                        results.append(f"{mid}\t{predicate}\t{obj_type}\n")

        if results:
            with open(worker_file, "a", encoding="utf-8") as wf:
                wf.writelines(results)

        return len(results)

    def parallel_retrieve_types(self, entities, output_dir, chunk_size=10_000, log_interval_chunks=10):
        """
        Parses Freebase RDF (from stdin) in parallel.
        Each worker appends to its own shard file. Parent only tracks counts.
        """
        GraphUtils.log_with_time("Script execution started.")
        progress_file = os.path.join(output_dir, "parser.progress")
        shards_dir = os.path.join(output_dir, "worker_shards")
        os.makedirs(shards_dir, exist_ok=True)

        # resume progress
        start_line = 0
        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as pf:
                    start_line = int(pf.read().strip())
                    GraphUtils.log_with_time(f"Resuming from line {start_line + 1}")
            except ValueError:
                GraphUtils.log_with_time("Progress file corrupted. Starting fresh.")
                start_line = 0

        # skip lines in stdin to resume
        in_f = sys.stdin
        if start_line > 0:
            GraphUtils.log_with_time(f"Skipping {start_line:,} lines...")
            for _ in range(start_line):
                try:
                    next(in_f)
                except StopIteration:
                    break
            GraphUtils.log_with_time("Finished skipping.")

        num_processes = min(12, multiprocessing.cpu_count())
        GraphUtils.log_with_time(f"Using {num_processes} workers, chunk size {chunk_size} lines.")

        entities_set = set(entities)

        processed_lines_count = start_line
        total_matches = 0
        last_log_time = datetime.now()
        chunk_count_since_log = 0

        with multiprocessing.Pool(
                processes=num_processes,
                initializer=self.init_worker,
                initargs=(entities_set, shards_dir),
        ) as pool:
            for matches in tqdm(
                    pool.imap_unordered(self.process_line_chunk,
                                        self.read_in_chunks_from_handle(in_f, chunk_size)),
                    desc="Processing", unit="chunks"
            ):
                processed_lines_count += chunk_size  # approx
                total_matches += matches
                chunk_count_since_log += 1

                # save progress + log every few chunks
                if chunk_count_since_log >= log_interval_chunks:
                    with open(progress_file, "w") as pf:
                        pf.write(str(processed_lines_count))
                    elapsed = (datetime.now() - last_log_time).total_seconds()
                    if elapsed > 0:
                        speed = (chunk_count_since_log * chunk_size) / elapsed
                        GraphUtils.log_with_time(f"Processed ~{processed_lines_count:,} lines "
                                           f"({speed:,.0f} lines/sec).")
                    last_log_time = datetime.now()
                    chunk_count_since_log = 0

        # final save
        with open(progress_file, "w") as pf:
            pf.write(str(processed_lines_count))

        GraphUtils.log_with_time(f"Complete. Matches: {total_matches:,}")
        GraphUtils.log_with_time(f"Shards in: {shards_dir}")
        final_out = os.path.join(output_dir, "final_unique_entities.tsv")
        print(f"cat {shards_dir}/worker_*.tsv | sort -u > {final_out}")
        return shards_dir

    def create_graph_from_edge_list_w_index(self, g, resolved_predicates, resolved_entities):
        """
        Similar to create_graph_from_edge_list but retrieves types/literals on the fly from the index.
        Parameters:
            - g: graph dict from input JSON
            - resolved_predicates: dict raw_pred -> resolved_pred
            - resolved_entities: dict raw_entity -> resolved_entity

        Returns: nodes_df, edges_df, graph_attrs
        """
        
        question = g['question']
        answers = g['answers']
        edge_list = g['subgraph']['tuples']
        gid = g['id']

        answers_text = [ans['text'] for ans in answers]
        answers_concatenated = ", ".join(str(a) for a in answers_text if a is not None)

        graph_attrs = {
            'question': question,
            'answers': answers,
            'answers_text': answers_text,
            'answers_concatenated': answers_concatenated,
            'id': gid,
        }

        edges = set()
        nodes = set()

        ORIGINAL = True

        # Create base graph from edge list and classify nodes as original vs new, type vs non-type.
        # We also skip literal edges here (we don't want literal nodes) but we keep track of literals for name assignment later.
        for s_raw, p_raw, t_raw in edge_list:
            source = resolved_entities[s_raw]
            predicate = resolved_predicates[p_raw]
            target = resolved_entities[t_raw]

            # Skip explicit literal edges: we don't want literal nodes
            if predicate in constants.LITERAL_PREDICATES:
                # Ensure source exists as an original node
                nodes.add((source, ORIGINAL, None, None, False))
                continue

            # Normal edge
            edges.add((source, target, predicate))

            # Target node classification
            if predicate == constants.TYPE_PREDICATE:
                # target is a type node from the original KG
                nodes.add((target, ORIGINAL, None, None, True))
            else:
                nodes.add((target, ORIGINAL, None, None, False))

            # Source is always an original non-type node (for our purposes here)
            nodes.add((source, ORIGINAL, None, None, False))

        # Determine original (non-type) nodes from what we have so far
        original_nodes = {
            n for (n, is_orig, name, types, is_type) in nodes
            if is_orig and not is_type
        }

        # Collect all types from index for each node + build frequency maps
        all_nonmeta_by_node = defaultdict(list)   # node -> list[(pred, type)]
        all_meta_by_node = defaultdict(list)      # node -> list[(pred, type)]
        nonmeta_type_sources = defaultdict(set)   # type -> set(nodes)
        meta_type_sources = defaultdict(set)      # type -> set(nodes)

        # Name map from literals (applied later; no literal nodes)
        name_by_node = {}

        for node_id in original_nodes:
            # Types from index
            for pred_t, target_type in self.types_idx.get_types_for_node(node_id):
                if GraphUtils.is_meta_type(target_type):
                    all_meta_by_node[node_id].append((IOOperations.sanitize(pred_t), target_type))
                    meta_type_sources[target_type].add(node_id)
                else:
                    all_nonmeta_by_node[node_id].append((IOOperations.sanitize(pred_t), target_type))
                    nonmeta_type_sources[target_type].add(node_id)

            # Literals: only store as "name", no edges, no literal nodes
            literals_result = self.literals_idx.get_literals_for_node(node_id)
            if literals_result is not None:
                _, literal_value = literals_result
                name_by_node[node_id] = str(literal_value)

        # For each node, select types according to the priority:
        #    1) non-meta types with >= 2 nodes
        #    2) if none, non-meta types with 1 node
        #    3) if none, meta   types with >= 2 nodes
        #    4) if none, meta   types with 1 node

        for node_id in original_nodes:
            chosen = self.pick_types_for_node(node_id, all_nonmeta_by_node, nonmeta_type_sources)
            if not chosen:
                # If no non-meta, try meta with thresholds 4,3,2,1
                chosen = self.pick_types_for_node(node_id, all_meta_by_node, meta_type_sources)

            # Add chosen type edges + mark type nodes
            for pred_t, target_type in chosen:
                edges.add((node_id, target_type, pred_t))
                nodes.add((target_type, False, None, None, True))

        # Build final node info (now apply literal/name rules)
        node_info = {}
        for n_id, is_orig, name, types, is_type in nodes:
            info = node_info.get(
                n_id,
                {'is_original': False, 'is_type': False}
            )
            info['is_original'] = info['is_original'] or is_orig
            info['is_type'] = info['is_type'] or is_type
            node_info[n_id] = info

        # Only keep nodes that actually appear in edges
        present_nodes = set()
        for s, t, _ in edges:
            present_nodes.add(s)
            present_nodes.add(t)

        final_nodes_rows = []
        for n_id in present_nodes:
            info = node_info.get(n_id, {'is_original': False, 'is_type': False})

            # Name logic:
            # - if type node -> name = node_id (string)
            # - else if literal exists -> that literal
            # - else -> None
            if info['is_type']:
                name = n_id
            else:
                name = name_by_node.get(n_id, None)

            final_nodes_rows.append(
                (n_id, info['is_original'], name, None, info['is_type'])
            )

        edges_df = pd.DataFrame(list(edges), columns=['source', 'target', 'key'])
        nodes_df = pd.DataFrame(
            final_nodes_rows,
            columns=['node', 'is_original', 'name', 'types', 'is_type']
        )

        # Fill 'types' column with list of type nodes for each non-type node
        type_map = defaultdict(list)
        for s, t, p in edges:
            if t in node_info and node_info[t]['is_type']:
                type_map[s].append(t)

        nodes_df['types'] = nodes_df.apply(
            lambda row: type_map.get(row['node'], []) if not row['is_type'] else None,
            axis=1
        )

        nodes_df['enhanced'] = 0
        edges_df['enhanced'] = 0

        return nodes_df, edges_df, graph_attrs

    def create_graph_from_edge_list(self, g, resolved_predicates, resolved_entities, retrieved_types_literals):
        """
        Graph construction method that doesn't retrieve from index on the fly, but relies on the retrieved_types_literals dict passed as an argument.
        Parameters:
            - g: graph dict from input JSON
            - resolved_predicates: dict raw_pred -> resolved_pred
            - resolved_entities: dict raw_entity -> resolved_entity
            - retrieved_types_literals: dict node_id -> set of (predicate, object) pairs retrieved for that node, used for type selection and name assignment
        Returns: nodes_df, edges_df, graph_attrs
        """
        question = g['question']
        answers = g['answers']
        edge_list = g['subgraph']['tuples']
        gid = g['id']

        answers_text = [ans['text'] for ans in answers]
        answers_concatenated = ", ".join(str(a) for a in answers_text if a is not None)

        graph_attrs = {
            'question': question,
            'answers': answers,
            'answers_text': answers_text,
            'answers_concatenated': answers_concatenated,
            'id': gid,
        }

        edges = set()
        nodes = set()
        ORIGINAL = True

        # Create base graph from edge list and classify nodes as original vs new, type vs non-type.
        for s_raw, p_raw, t_raw in edge_list:
            source = resolved_entities[s_raw]
            predicate = resolved_predicates[p_raw]
            target = resolved_entities[t_raw]

            # Skip explicit literal edges: we don't want literal nodes
            if predicate in constants.LITERAL_PREDICATES:
                # Ensure source exists as an original node
                nodes.add((source, ORIGINAL, None, None, False))
                continue

            # Normal edge
            edges.add((source, target, predicate))

            # Target node classification
            if predicate == constants.TYPE_PREDICATE:
                nodes.add((target, ORIGINAL, None, None, True))
            else:
                nodes.add((target, ORIGINAL, None, None, False))

            # Source is always an original non-type node
            nodes.add((source, ORIGINAL, None, None, False))

        original_nodes = {
            n for (n, is_orig, name, types, is_type) in nodes
            if is_orig and not is_type
        }

        # Collect all types from dictionary for each node + build frequency maps
        all_nonmeta_by_node = defaultdict(list)
        all_meta_by_node = defaultdict(list)
        nonmeta_type_sources = defaultdict(set)
        meta_type_sources = defaultdict(set)
        name_by_node = {}

        for node_id in original_nodes:
            # retrieved_types_literals: node_id -> set[(predicate, obj)]
            for pred, obj in retrieved_types_literals.get(node_id, set()):
                if pred == constants.TYPE_PREDICATE:
                    if GraphUtils.is_meta_type(obj):
                        all_meta_by_node[node_id].append((pred, obj))
                        meta_type_sources[obj].add(node_id)
                    else:
                        all_nonmeta_by_node[node_id].append((pred, obj))
                        nonmeta_type_sources[obj].add(node_id)
                elif pred in constants.LITERAL_PREDICATES:
                    if node_id not in name_by_node:
                        name_by_node[node_id] = str(obj)

        # Select types using the heuristic prioritization
        for node_id in original_nodes:
            chosen = self.pick_types_for_node(node_id, all_nonmeta_by_node, nonmeta_type_sources)
            if not chosen:
                chosen = self.pick_types_for_node(node_id, all_meta_by_node, meta_type_sources)

            for pred_t, target_type in chosen:
                edges.add((node_id, target_type, pred_t))
                nodes.add((target_type, False, None, None, True))

        # Build final node info
        node_info = {}
        for n_id, is_orig, name, types, is_type in nodes:
            info = node_info.get(
                n_id,
                {'is_original': False, 'is_type': False}
            )
            info['is_original'] = info['is_original'] or is_orig
            info['is_type'] = info['is_type'] or is_type
            node_info[n_id] = info

        present_nodes = set()
        for s, t, _ in edges:
            present_nodes.add(s)
            present_nodes.add(t)

        final_nodes_rows = []
        for n_id in present_nodes:
            info = node_info.get(n_id, {'is_original': False, 'is_type': False})

            if info['is_type']:
                name = n_id
            else:
                name = name_by_node.get(n_id, None)

            final_nodes_rows.append(
                (n_id, info['is_original'], name, None, info['is_type'])
            )

        edges_df = pd.DataFrame(list(edges), columns=['source', 'target', 'key'])
        nodes_df = pd.DataFrame(
            final_nodes_rows,
            columns=['node', 'is_original', 'name', 'types', 'is_type']
        )

        # Fill 'types' column
        type_map = defaultdict(list)
        for s, t, p in edges:
            if t in node_info and node_info[t]['is_type']:
                type_map[s].append(t)

        nodes_df['types'] = nodes_df.apply(
            lambda row: type_map.get(row['node'], []) if not row['is_type'] else None,
            axis=1
        )

        nodes_df['enhanced'] = 0
        edges_df['enhanced'] = 0

        return nodes_df, edges_df, graph_attrs


    def create_graph_from_edge_list_w_index_majority_type(self, g, resolved_predicates, resolved_entities):
        """
        retrieves types/literals on the fly from the index, but only for the original nodes in the graph (i.e. those that appear in the edge list), 
        and applies the type selection heuristic to assign a single type per node. We also apply the name logic (using literals) but we don't create literal nodes.

        @parameters:
            - g: graph dict from input JSON
            - resolved_predicates: dict raw_pred -> resolved_pred
            - resolved_entities: dict raw_entity -> resolved_entity
        @returns: nodes_df, edges_df, graph_attrs
        """
        question = g['question']
        answers = g['answers']
        edge_list = g['subgraph']['tuples']
        gid = g['id']

        answers_text = [ans['text'] for ans in answers]
        answers_concatenated = ", ".join(str(a) for a in answers_text if a is not None)

        graph_attrs = {
            'question': question,
            'answers': answers,
            'answers_text': answers_text,
            'answers_concatenated': answers_concatenated,
            'id': gid,
        }

        edges = set()
        nodes = set()
        ORIGINAL = True

        for s_raw, p_raw, t_raw in edge_list:
            source = resolved_entities[s_raw]
            predicate = resolved_predicates[p_raw]
            target = resolved_entities[t_raw]

            # Skip explicit literal edges: we don't want literal nodes
            if predicate in constants.LITERAL_PREDICATES:
                nodes.add((source, ORIGINAL, None, None, False))
                continue

            # Normal edge
            edges.add((source, target, predicate))

            # Target node classification
            if predicate == constants.TYPE_PREDICATE:
                # target is a type node from the original KG
                nodes.add((target, ORIGINAL, None, None, True))
            else:
                # target is a non-type node from the original KG
                nodes.add((target, ORIGINAL, None, None, False))

            # Source is always an original non-type node
            nodes.add((source, ORIGINAL, None, None, False))

        original_nodes = {n for (n, is_orig, name, types, is_type) in nodes if is_orig and not is_type}

        all_nonmeta_by_node = defaultdict(list)
        all_meta_by_node = defaultdict(list)
        nonmeta_type_sources = defaultdict(set)
        meta_type_sources = defaultdict(set)
        name_by_node = {}

        for node_id in original_nodes:
            # for each node, get types/literals from the index and separate meta vs non-meta types
            for pred_t, target_type in self.types_idx.get_types_for_node(node_id):
                if GraphUtils.is_meta_type(target_type):
                    all_meta_by_node[node_id].append((IOOperations.sanitize(pred_t), target_type))
                    meta_type_sources[target_type].add(node_id)
                else:
                    all_nonmeta_by_node[node_id].append((IOOperations.sanitize(pred_t), target_type))
                    nonmeta_type_sources[target_type].add(node_id)

            literals_result = self.literals_idx.get_literals_for_node(node_id)
            if literals_result is not None:
                _, literal_value = literals_result
                name_by_node[node_id] = str(literal_value)

        assigned_types_for_node = {}
        # for each node, pick types according to the priority: non-meta with >=2 nodes, then non-meta with 1 node, then meta with >=2 nodes, then meta with 1 node
        for node_id in original_nodes:
            chosen = self.pick_max_size_type_for_node(node_id, all_nonmeta_by_node, nonmeta_type_sources)
            if chosen is None:
                chosen = self.pick_max_size_type_for_node(node_id, all_meta_by_node, meta_type_sources)

            if chosen is not None:
                assigned_types_for_node[node_id] = chosen[1]  # chosen is (pred, type)
            else:
                assigned_types_for_node[node_id] = "unknown_type"

        final_nodes_rows = []
        # build final node rows, applying the name logic: if type node -> name = node_id (string), else if literal exists -> that literal, else None
        for n_id in original_nodes:
            name = name_by_node.get(n_id, None)
            node_type = [assigned_types_for_node.get(n_id, "unknown_type")]
            final_nodes_rows.append((n_id, ORIGINAL, name, node_type, False))

        valid_edges = [(s, t, p) for s, t, p in edges if s in original_nodes and t in original_nodes]
        
        edges_df = pd.DataFrame(valid_edges, columns=['source', 'target', 'key'])
        nodes_df = pd.DataFrame(final_nodes_rows, columns=['node', 'is_original', 'name', 'types', 'is_type'])

        nodes_df['enhanced'] = 0
        edges_df['enhanced'] = 0

        return nodes_df, edges_df, graph_attrs

    def create_graph_from_edge_list_w_index_multi_hot(self, g, resolved_predicates, resolved_entities):
        """
        We keep all types and insert them in the 'types' column.
        This is a simpler approach that doesn't require picking a single type per node.

        Parameters:
            - g: graph dict from input JSON
            - resolved_predicates: dict raw_pred -> resolved_pred
            - resolved_entities: dict raw_entity -> resolved_entity
        Returns: nodes_df, edges_df, graph_attrs
        """
        question = g['question']
        answers = g['answers']
        edge_list = g['subgraph']['tuples']
        gid = g['id']

        answers_text = [ans['text'] for ans in answers]
        answers_concatenated = ", ".join(str(a) for a in answers_text if a is not None)

        graph_attrs = {
            'question': question,
            'answers': answers,
            'answers_text': answers_text,
            'answers_concatenated': answers_concatenated,
            'id': gid,
        }

        edges = set()
        nodes = set()
        ORIGINAL = True

        for s_raw, p_raw, t_raw in edge_list:
            source = resolved_entities[s_raw]
            predicate = resolved_predicates[p_raw]
            target = resolved_entities[t_raw]

            if predicate in constants.LITERAL_PREDICATES:
                nodes.add((source, ORIGINAL, None, None, False))
                continue

            edges.add((source, target, predicate))

            if predicate == constants.TYPE_PREDICATE:
                nodes.add((target, ORIGINAL, None, None, True))
            else:
                nodes.add((target, ORIGINAL, None, None, False))

            nodes.add((source, ORIGINAL, None, None, False))

        original_nodes = {n for (n, is_orig, name, types, is_type) in nodes if is_orig and not is_type}

        all_nonmeta_by_node = defaultdict(list)
        all_meta_by_node = defaultdict(list)
        nonmeta_type_sources = defaultdict(set)
        meta_type_sources = defaultdict(set)
        name_by_node = {}

        for node_id in original_nodes:
            for pred_t, target_type in self.types_idx.get_types_for_node(node_id):
                if GraphUtils.is_meta_type(target_type):
                    all_meta_by_node[node_id].append((IOOperations.sanitize(pred_t), target_type))
                    meta_type_sources[target_type].add(node_id)
                else:
                    all_nonmeta_by_node[node_id].append((IOOperations.sanitize(pred_t), target_type))
                    nonmeta_type_sources[target_type].add(node_id)

            literals_result = self.literals_idx.get_literals_for_node(node_id)
            if literals_result is not None:
                _, literal_value = literals_result
                name_by_node[node_id] = str(literal_value)

        assigned_types_for_node = {}
        for node_id in original_nodes:
            chosen = self.pick_types_for_node(node_id, all_nonmeta_by_node, nonmeta_type_sources)
            if not chosen:
                chosen = self.pick_types_for_node(node_id, all_meta_by_node, meta_type_sources)

            types_list = [target_type for pred_t, target_type in chosen]
            assigned_types_for_node[node_id] = types_list

        final_nodes_rows = []
        for n_id in original_nodes:
            name = name_by_node.get(n_id, None)
            node_type = assigned_types_for_node.get(n_id, [])
            final_nodes_rows.append((n_id, ORIGINAL, name, node_type, False))

        valid_edges = [(s, t, p) for s, t, p in edges if s in original_nodes and t in original_nodes]
        
        edges_df = pd.DataFrame(valid_edges, columns=['source', 'target', 'key'])
        nodes_df = pd.DataFrame(final_nodes_rows, columns=['node', 'is_original', 'name', 'types', 'is_type'])

        nodes_df['enhanced'] = 0
        edges_df['enhanced'] = 0

        return nodes_df, edges_df, graph_attrs

    def create_graph_from_edge_list_majority_type(self, g, resolved_predicates, resolved_entities, retrieved_types_literals):
        """
        We retrieve types and literals for original nodes, but instead of keeping all types, we apply the heuristic to pick a single type per node. We also apply the name logic but we don't create literal nodes. 
        we use the retrieved_types_literals dict that is passed as an argument. 
        
        Parameters:
            - g: graph dict from input JSON
            - resolved_predicates: dict raw_pred -> resolved_pred
            - resolved_entities: dict raw_entity -> resolved_entity
            - retrieved_types_literals: dict node_id -> set[(predicate, obj)] containing all retrieved
                types/literals for all nodes (not just original nodes, but we will only use the relevant ones)

        Returns: nodes_df, edges_df, graph_attrs
        """
        question = g['question']
        answers = g['answers']
        edge_list = g['subgraph']['tuples']
        gid = g['id']

        answers_text = [ans['text'] for ans in answers]
        answers_concatenated = ", ".join(str(a) for a in answers_text if a is not None)

        graph_attrs = {
            'question': question,
            'answers': answers,
            'answers_text': answers_text,
            'answers_concatenated': answers_concatenated,
            'id': gid,
        }

        edges = set()
        nodes = set()
        ORIGINAL = True

        # Base edges + original nodes (NO literal nodes)
        for s_raw, p_raw, t_raw in edge_list:
            source = resolved_entities[s_raw]
            predicate = resolved_predicates[p_raw]
            target = resolved_entities[t_raw]

            # Skip explicit literal edges: we don't want literal nodes
            if predicate in constants.LITERAL_PREDICATES:
                nodes.add((source, ORIGINAL, None, None, False))
                continue

            edges.add((source, target, predicate))

            # Target node classification
            if predicate == constants.TYPE_PREDICATE:
                nodes.add((target, ORIGINAL, None, None, True))
            else:
                nodes.add((target, ORIGINAL, None, None, False))

            nodes.add((source, ORIGINAL, None, None, False))

        original_nodes = {n for (n, is_orig, name, types, is_type) in nodes if is_orig and not is_type}

        all_nonmeta_by_node = defaultdict(list)
        all_meta_by_node = defaultdict(list)
        nonmeta_type_sources = defaultdict(set)
        meta_type_sources = defaultdict(set)
        name_by_node = {}

        # For each original node, get types/literals from the retrieved_types_literals dict and 
        # separate meta vs non-meta types, also apply name logic for literals
        for node_id in original_nodes:
            for pred, obj in retrieved_types_literals.get(node_id, set()):
                if pred == constants.TYPE_PREDICATE:
                    if GraphUtils.is_meta_type(obj):
                        all_meta_by_node[node_id].append((pred, obj))
                        meta_type_sources[obj].add(node_id)
                    else:
                        all_nonmeta_by_node[node_id].append((pred, obj))
                        nonmeta_type_sources[obj].add(node_id)
                elif pred in constants.LITERAL_PREDICATES:
                    if node_id not in name_by_node:
                        name_by_node[node_id] = str(obj)

        assigned_types_for_node = {}
        for node_id in original_nodes:
            chosen = self.pick_max_size_type_for_node(node_id, all_nonmeta_by_node, nonmeta_type_sources)
            if chosen is None:
                chosen = self.pick_max_size_type_for_node(node_id, all_meta_by_node, meta_type_sources)

            if chosen is not None:
                assigned_types_for_node[node_id] = chosen[1]  # chosen is (pred, type)
            else:
                assigned_types_for_node[node_id] = "unknown_type"

        final_nodes_rows = []
        for n_id in original_nodes:
            name = name_by_node.get(n_id, None)
            node_type = [assigned_types_for_node.get(n_id, "unknown_type")]
            final_nodes_rows.append((n_id, ORIGINAL, name, node_type, False))

        valid_edges = [(s, t, p) for s, t, p in edges if s in original_nodes and t in original_nodes]
        
        edges_df = pd.DataFrame(valid_edges, columns=['source', 'target', 'key'])
        nodes_df = pd.DataFrame(final_nodes_rows, columns=['node', 'is_original', 'name', 'types', 'is_type'])

        nodes_df['enhanced'] = 0
        edges_df['enhanced'] = 0

        return nodes_df, edges_df, graph_attrs

    def create_graph_from_edge_list_multi_hot(self, g, resolved_predicates, resolved_entities, retrieved_types_literals):
        """
        Create a graph from a list of edges, where each node can have multiple types.
        This is a simpler approach that doesn't require picking a single type per node, 
        but instead keeps all types in a list in the 'types' column.

        Parameters:
            - g: graph dict from input JSON
            - resolved_predicates: dict raw_pred -> resolved_pred
            - resolved_entities: dict raw_entity -> resolved_entity
            - retrieved_types_literals: dict node_id -> set[(predicate, obj)] containing all retrieved
                types/literals for all nodes (not just original nodes, but we will only use the
                relevant ones)
        
        Returns: nodes_df, edges_df, graph_attrs
        """

        question = g['question']
        answers = g['answers']
        edge_list = g['subgraph']['tuples']
        gid = g['id']

        answers_text = [ans['text'] for ans in answers]
        answers_concatenated = ", ".join(str(a) for a in answers_text if a is not None)

        graph_attrs = {
            'question': question,
            'answers': answers,
            'answers_text': answers_text,
            'answers_concatenated': answers_concatenated,
            'id': gid,
        }

        edges = set()
        nodes = set()
        ORIGINAL = True

        for s_raw, p_raw, t_raw in edge_list:
            source = resolved_entities[s_raw]
            predicate = resolved_predicates[p_raw]
            target = resolved_entities[t_raw]

            if predicate in constants.LITERAL_PREDICATES:
                nodes.add((source, ORIGINAL, None, None, False))
                continue

            edges.add((source, target, predicate))

            if predicate == constants.TYPE_PREDICATE:
                nodes.add((target, ORIGINAL, None, None, True))
            else:
                nodes.add((target, ORIGINAL, None, None, False))

            nodes.add((source, ORIGINAL, None, None, False))

        original_nodes = {n for (n, is_orig, name, types, is_type) in nodes if is_orig and not is_type}

        all_nonmeta_by_node = defaultdict(list)
        all_meta_by_node = defaultdict(list)
        nonmeta_type_sources = defaultdict(set)
        meta_type_sources = defaultdict(set)
        name_by_node = {}

        for node_id in original_nodes:
            for pred, obj in retrieved_types_literals.get(node_id, set()):
                if pred == constants.TYPE_PREDICATE:
                    if GraphUtils.is_meta_type(obj):
                        all_meta_by_node[node_id].append((pred, obj))
                        meta_type_sources[obj].add(node_id)
                    else:
                        all_nonmeta_by_node[node_id].append((pred, obj))
                        nonmeta_type_sources[obj].add(node_id)
                elif pred in constants.LITERAL_PREDICATES:
                    if node_id not in name_by_node:
                        name_by_node[node_id] = str(obj)

        assigned_types_for_node = {}
        for node_id in original_nodes:
            chosen = self.pick_types_for_node(node_id, all_nonmeta_by_node, nonmeta_type_sources)
            if not chosen:
                chosen = self.pick_types_for_node(node_id, all_meta_by_node, meta_type_sources)

            types_list = [target_type for pred_t, target_type in chosen]
            assigned_types_for_node[node_id] = types_list

        final_nodes_rows = []
        for n_id in original_nodes:
            name = name_by_node.get(n_id, None)
            node_type = assigned_types_for_node.get(n_id, [])
            final_nodes_rows.append((n_id, ORIGINAL, name, node_type, False))

        valid_edges = [(s, t, p) for s, t, p in edges if s in original_nodes and t in original_nodes]
        
        edges_df = pd.DataFrame(valid_edges, columns=['source', 'target', 'key'])
        nodes_df = pd.DataFrame(final_nodes_rows, columns=['node', 'is_original', 'name', 'types', 'is_type'])

        nodes_df['enhanced'] = 0
        edges_df['enhanced'] = 0

        return nodes_df, edges_df, graph_attrs

    def merge_sort_results(self, input_dir, output_dir, file_name="types_literals.tsv"):
        input_dir_path = Path(input_dir)
        output_dir_path = Path(output_dir)
        output_dir_file = output_dir_path.joinpath(output_dir_path, file_name)

        files = sorted(input_dir_path.glob("worker_*.tsv"))
        if not files:
            raise FileNotFoundError("No worker_*.tsv files found")

        with output_dir_file.open("w", encoding="utf-8") as out:
            subprocess.run(
                ["sort", "-u", *[str(p) for p in files]],
                check=True,
                stdout=out,
            )
        return output_dir_file

    def retrieve_types_literals(self):
        p = Path(self.parameters.db_path)

        if not p.is_file():  # retrieve types and literals using freebase
            entities_path = self.parameters.resolved_nodes

            output_dir = self.parameters.output_dir

            resolved_entities = IOOperations.read_csv_file(entities_path, to_sanitize=False)

            # run parallel reader; it reads from stdin
            # Example run:
            # pigz -dc /path/to/freebase.gz | python3 this_script.py
            # retrieve types/literals for all entities in the dataset
            shards_dir = self.parallel_retrieve_types(resolved_entities, output_dir, chunk_size=10_000)
            merge_sort_file_path = self.merge_sort_results(shards_dir, output_dir)

            return merge_sort_file_path

    def generate_typed_graphs(self):
        graphs_path = self.parameters.graphs_input_file
        predicates_path = self.parameters.resolved_predicates
        entities_path = self.parameters.resolved_nodes

        output_dir = self.parameters.output_dir
        types_literals_file = self.parameters.types_literals_file

        resolved_predicates = IOOperations.read_csv_file(predicates_path, to_sanitize=True)
        resolved_entities = IOOperations.read_csv_file(entities_path, to_sanitize=False)
        graphs = IOOperations.read_json(graphs_path)

        # limit number of graphs for testing
        if self.n_graphs > 0:
            graphs = graphs[:self.n_graphs]

        # if types_literals_file is provided, read the retrieved types/literals and use them to build the graphs
        # we are not using the index here, but directly the retrieved types/literals from the file
        if types_literals_file is not None and self.literals_idx is None and self.types_idx is None:
            types_literals_path = Path(types_literals_file)
            if not types_literals_path.exists():
                print("Types and literals not found, please generate them manually...")
                sys.exit(0)
            else:
                print("Reading types/literals...")
                types_literal = self.read_types_literals(types_literals_path, to_sanitize=True)
                for g in tqdm(graphs, desc="Building graphs"):
                    nodes, edges, graph_attrs = self.create_graph_from_edge_list(g, resolved_predicates,
                                                                                 resolved_entities,
                                                                                 types_literal)
                    IOOperations.save_graph(nodes, edges, graph_attrs, output_dir)

        # if types_literals_file is not provided, we retrieve types/literals on the fly from the index for each graph and build the graphs using that information
        if types_literals_file is None and self.literals_idx is not None and self.types_idx is not None:
            if self.strategy == "multihot":
                w_func = self.create_graph_from_edge_list_w_index_multi_hot
            elif self.strategy == "classic":
                w_func = self.create_graph_from_edge_list_w_index
            elif self.strategy == "singleton":
                w_func = self.create_graph_from_edge_list_w_index_majority_type
            
            worker_func =   partial(w_func, 
                                    resolved_predicates=resolved_predicates,
                                    resolved_entities=resolved_entities)
            with multiprocessing.Pool(processes=self.num_processes, maxtasksperchild=5) as pool:
                results_iterator = pool.imap_unordered(worker_func, graphs)
                for nodes, edges, graph_attrs in tqdm(results_iterator, total=len(graphs),
                                                    desc="Building Graphs"):
                    IOOperations.save_graph(edges, nodes, graph_attrs, output_dir, prefix="typed_graph")

