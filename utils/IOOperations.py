import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import networkx as nx
import pandas as pd
import shutil


class IOOperations:
    """
    Utility class for generic I/O operations and graph data transformations.
    """

    @staticmethod
    def copy_and_rename(src_path, dest_path, new_name):
        """
        Copies a file from src_path to dest_path and renames it to new_name.

        Parameters:
        - src_path: The source file path to copy from.
        - dest_path: The destination directory path to copy to.
        - new_name: The new name for the copied file.
        """
        # Copy the file
        shutil.copy(src_path, dest_path)

        # Rename the copied file
        file_name = Path(src_path).name
        new_path = f"{dest_path}/{new_name}"
        shutil.move(f"{dest_path}/{file_name}", new_path)

    @staticmethod
    def sanitize(name: str) -> str:
        """
        Replace '.', ' ', '/' and '#' to make the name safe for use as keys or module names.
        """
        if not isinstance(name, str):
            return str(name)
            
        return (
            name.replace('.', '_DOT_')
            .replace(' ', '_SPACE_')
            .replace('/', "_SLASH_")
            .replace('#', "_DOT_")
        )


    @staticmethod
    def read_csv_file(path: Union[str, Path], to_sanitize: bool = False) -> List[str]:
        """
        Reads a CSV file and returns a list of values from the first column.
        If to_sanitize is True, the values will be sanitized using the sanitize method.
        """
        resolved: List[str] = []
        file_path = Path(path)
        
        with file_path.open("r", newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                val = row[0]
                if to_sanitize:
                    val = IOOperations.sanitize(val)
                resolved.append(val)
                
        return resolved

    @staticmethod
    def read_json(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Reads a JSON Lines file (.jsonl) where each line is a separate JSON object.
        """
        data: List[Dict[str, Any]] = []
        path = Path(file_path)
        
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
                    
        return data

    @staticmethod
    def read_edge_list_as_graph(
        edge_list_path: Union[str, Path], 
        node_list_path: Union[str, Path], 
        graph_attrs_path: Union[str, Path]
    ) -> nx.MultiDiGraph:
        """
        Rebuilds the complete NetworkX graph structure from persisted Parquet edge/node lists.
        """
        # Load Edges
        edge_df_loaded = pd.read_parquet(edge_list_path, engine='fastparquet')
        if 'key' in edge_df_loaded.columns:
            edge_df_loaded['key'] = edge_df_loaded['key'].astype(str).str.replace('#', '_DOT_', regex=False)
            
        g = nx.from_pandas_edgelist(
            edge_df_loaded,
            source='source',
            target='target',
            edge_key='key',
            edge_attr=True,
            create_using=nx.MultiDiGraph
        )

        # Load Node Attributes
        node_df_loaded = pd.read_parquet(node_list_path, engine='fastparquet')
        # Ensuring the DataFrame has an index that NetworkX can use as node identifiers
        nx.set_node_attributes(g, node_df_loaded.to_dict('index'))

        # Load and Set Graph Attributes
        with Path(graph_attrs_path).open('r', encoding='utf-8') as f:
            g.graph.update(json.load(f))

        return g

    @staticmethod
    def save_graph(
        edges: pd.DataFrame, 
        nodes: pd.DataFrame, 
        graph_attrs: Dict[str, Any], 
        output_dir: Union[str, Path], 
        prefix: str = "graph"
    ) -> None:
        """
        Persists graph components into Parquet and JSON files to disk.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)  # Avoids race conditions

        graph_id = graph_attrs.get('id', 'unknown')
        
        output_edges_path = output_dir_path / f"{prefix}.edges.{graph_id}.parquet"
        output_nodes_path = output_dir_path / f"{prefix}.nodes.{graph_id}.parquet"
        output_graph_path = output_dir_path / f"{prefix}.graph.{graph_id}.json"

        edges.to_parquet(output_edges_path, engine='fastparquet')
        nodes.to_parquet(output_nodes_path, engine='fastparquet')
        
        with output_graph_path.open("w", encoding="utf-8") as f:
            json.dump(graph_attrs, f, indent=4)  # Added indent for better readability

    @staticmethod
    def transform_and_save_graph(
        graph: nx.Graph, 
        output_dir: Union[str, Path], 
        prefix: str = "graph"
    ) -> None:
        """
        Extracts DataFrames from a NetworkX graph and defers to save_graph.
        """
        edge_df = nx.to_pandas_edgelist(graph, source='source', target='target', edge_key='key')
        
        node_data = {node: data for node, data in graph.nodes(data=True)}
        node_df = pd.DataFrame.from_dict(node_data, orient='index').reset_index(drop=True)
        
        IOOperations.save_graph(edge_df, node_df, graph.graph, output_dir, prefix=prefix)

