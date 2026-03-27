import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

class Statistics:

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir

    def read_graph(self, graph_id, graph_prefix_filename="bridge_enhanced_graph"):

        input_dir = Path(self.input_dir)
        edges_path = input_dir / f"{graph_prefix_filename}.edges.{graph_id}.parquet"
        nodes_path = input_dir / f"{graph_prefix_filename}.nodes.{graph_id}.parquet"
        graph_path = input_dir / f"{graph_prefix_filename}.graph.{graph_id}.json"

        edges = pd.read_parquet(edges_path, engine="fastparquet")
        nodes = pd.read_parquet(nodes_path, engine="fastparquet")

        with graph_path.open("r") as f:
            graph_attrs = json.load(f)

        return edges, nodes, graph_attrs

    def compute_total_nodes(self, nodes):
        return nodes.shape[0]

    def compute_total_type_nodes(self, nodes):
        type_nodes = nodes[nodes['is_type'] == True]
        return type_nodes.shape[0]

    def compute_total_bridge_nodes(self, nodes):
        bridge_nodes = nodes[nodes['is_bridge'] == True]
        return bridge_nodes.shape[0]

    def compute_total_edges(self, edges):
        return edges.shape[0]

    def compute_total_original_nodes(self, nodes):
        original_nodes = nodes[nodes['is_original'] == True]
        return original_nodes.shape[0]

    def compute_total_not_original_nodes(self, nodes):
        not_original_nodes = nodes[nodes['is_original'] == False]
        return not_original_nodes.shape[0]

    def compute_total_original_type_nodes(self, nodes):
        original_type_nodes = nodes[(nodes['is_type'] == True) & (nodes['is_original'] == True)]
        return original_type_nodes.shape[0]

    def compute_total_not_original_type_nodes(self, nodes):
        not_original_type_nodes = nodes[(nodes['is_type'] == True) & (nodes['is_original'] == False)]
        return not_original_type_nodes.shape[0]

    def compute_total_original_bridge_nodes(self, nodes):
        original_bridge_nodes = nodes[(nodes['is_bridge'] == True) & (nodes['is_original'] == True)]
        return original_bridge_nodes.shape[0]

    def compute_total_not_original_bridge_nodes(self, nodes):
        not_original_bridge_nodes = nodes[(nodes['is_bridge'] == True) & (nodes['is_original'] == False)]
        return not_original_bridge_nodes.shape[0]

    def compute_total_original_edges(self, edges):
        original_edges = edges[edges['enhanced'] == 1]
        return original_edges.shape[0]

    def compute_total_not_original_edges(self, edges):
        not_original_edges = edges[edges['enhanced'] == 0]
        return not_original_edges.shape[0]

    def compute_average(self, elements: np.array):
        avg = np.mean(elements)
        return avg

    def compute_median(self, elements: np.array):
        median = np.median(elements)
        return median

    def compute_standard_deviation(self, elements: np.array):
        sd = np.std(elements)
        return sd

    def compute_variance(self, elements: np.array):
        variance = np.var(elements)
        return variance

    def compute_statistics(self, graph_prefix_filename="bridge_enhanced_graph"):
        dir = Path(self.input_dir)

        all_graph_identifiers_by_edge = [
            path.name.split('.')[2]
            for path in dir.glob(f"{graph_prefix_filename}.edges.*.parquet")
        ]


        all_graph_identifiers_by_node = [
            path.name.split('.')[2]
            for path in dir.glob(f"{graph_prefix_filename}.nodes.*.parquet")
        ]

        all_graph_identifiers_by_gattrs = [
            path.name.split('.')[2]
            for path in dir.glob(f"{graph_prefix_filename}.graph.*.json")
        ]

        print(f"Found {len(all_graph_identifiers_by_edge)} graphs by edges, {len(all_graph_identifiers_by_node)} by nodes, {len(all_graph_identifiers_by_gattrs)} by gattrs")

        if not (len(all_graph_identifiers_by_edge) == len(all_graph_identifiers_by_node) == len(all_graph_identifiers_by_gattrs)):
            print(f"Graph identifiers do not match")
            return

        ids = all_graph_identifiers_by_edge
        stats = {
            "total_nodes": [],
            "total_edges": [],
            "total_original_nodes": [],
            "total_not_original_nodes": [],
            "total_type_nodes": [],
            "total_bridge_nodes": [],
            "total_original_type_nodes": [],
            "total_not_original_type_nodes": [],
            "total_original_bridge_nodes": [],
            "total_not_original_bridge_nodes": [],
            "total_original_edges": [],
            "total_not_original_edges": []
        }

        for graph_id in tqdm(ids, desc="Computing statistics for graphs"):
            edges, nodes, attrs = self.read_graph(graph_id, graph_prefix_filename)

            stats["total_nodes"].append(self.compute_total_nodes(nodes))
            stats["total_edges"].append(self.compute_total_edges(edges))
            stats["total_original_nodes"].append(self.compute_total_original_nodes(nodes))
            stats["total_not_original_nodes"].append(self.compute_total_not_original_nodes(nodes))
            stats["total_type_nodes"].append(self.compute_total_type_nodes(nodes))
            stats["total_bridge_nodes"].append(self.compute_total_bridge_nodes(nodes))
            stats["total_original_type_nodes"].append(self.compute_total_original_type_nodes(nodes))
            stats["total_not_original_type_nodes"].append(self.compute_total_not_original_type_nodes(nodes))
            stats["total_original_bridge_nodes"].append(self.compute_total_original_bridge_nodes(nodes))
            stats["total_not_original_bridge_nodes"].append(self.compute_total_not_original_bridge_nodes(nodes))
            stats["total_original_edges"].append(self.compute_total_original_edges(edges))
            stats["total_not_original_edges"].append(self.compute_total_not_original_edges(edges))

        stats["total_nodes"] = np.array(stats["total_nodes"])
        stats["total_edges"] = np.array(stats["total_edges"])
        stats["total_original_nodes"]= np.array(stats["total_original_nodes"])
        stats["total_not_original_nodes"] = np.array(stats["total_not_original_nodes"])
        stats["total_type_nodes"] = np.array(stats["total_type_nodes"])
        stats["total_bridge_nodes"] = np.array(stats["total_bridge_nodes"])
        stats["total_original_type_nodes"] = np.array(stats["total_original_type_nodes"])
        stats["total_not_original_type_nodes"] = np.array(stats["total_not_original_type_nodes"])
        stats["total_original_bridge_nodes"] = np.array(stats["total_original_bridge_nodes"])
        stats["total_not_original_bridge_nodes"] = np.array(stats["total_not_original_bridge_nodes"])
        stats["total_original_edges"] = np.array(stats["total_original_edges"])
        stats["total_not_original_edges"] = np.array(stats["total_not_original_edges"])
        
        c_stats = {}
        for k, v in stats.items():
            c_stats[k+"_avg"] = self.compute_average(v)
            c_stats[k+"_median"] = self.compute_median(v)
            c_stats[k+"_std"] = self.compute_standard_deviation(v)
            c_stats[k+"_var"] = self.compute_variance(v)

        stats.update(c_stats)
        return stats
