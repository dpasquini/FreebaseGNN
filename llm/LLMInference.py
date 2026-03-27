import json
import os
import shutil
import pandas as pd
from tqdm import tqdm
import glob
from pathlib import Path
from typing import List

from llm.LLMService import LLMService
from utils.TextUtils import TextUtils
from utils.IOOperations import IOOperations

class LLMInference:
    """
    This class orchestrates performing LLM inference using vLLM to generate 
    descriptions for nodes and edges in prompted graph data.
    """
    def __init__(self, model_name: str, input_dir: str, output_dir: str, file_pattern_nodes: str, file_pattern_edges: str, file_pattern_graphs: str, huggingface_token: str = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.file_pattern_nodes = file_pattern_nodes
        self.file_pattern_edges = file_pattern_edges
        self.file_pattern_graphs = file_pattern_graphs
        self.llm_service = LLMService(model_name=model_name, huggingface_token=huggingface_token)
    
    def _get_graphs_ids(self, file_pattern: str) -> List[str]:
        search_pattern = self.input_dir / file_pattern.replace("{graph_id}", "*")
        all_files = glob.glob(str(search_pattern))
        # Extracts graph id assuming format: prefix.nodes|edges.{graph_id}.parquet
        return sorted(list(set([os.path.basename(path).split('.')[2] for path in all_files])))

    def query_questions(self, file_pattern: str, graphs_ids: List[str]):
        """
        For each graph, reads the question and other information from the graph info files, constructs a prompt, queries the LLM for an answer, and saves the results in a structured JSON format in the output directory.
        The method operates in three main stages:
        1. Collecting all questions: It iterates through the specified graph info files, extracts the questions (and other information), and organizes them into a list of conversations formatted for LLM input.
        2. Running vLLM batch inference: It sends the collected conversations to the LLM service in batches and retrieves the raw outputs.
        3. Processing and saving results: It cleans the raw LLM outputs, merges them with the original question and answer data, and saves the final results in JSON files in the output directory, one file per graph.

        Parameters:
        - file_pattern: The pattern to identify the graph info files containing the questions and answers.
        - graphs_ids: A list of graph identifiers to process, which will be used to locate the corresponding graph info files.
        """
        print("Stage 1: Collecting all questions from all graphs...")
        all_conversations = []
        all_metadata = []

        # Defines a clear role, explicit constraints, and uses "Answer:" grounding to prevent repetitive logic headers.
        prompt_pattern = (
            #"Answer using your general knowledge. Output only the final answer, with no explanation, no full sentence, and no extra text.\n"
            "Question: {question}\n"
            "Answer:"
        )

        for graph_id in tqdm(graphs_ids, desc="Loading graphs"):
            graph_path = self.input_dir / file_pattern.replace("{graph_id}", graph_id)
            if not graph_path.exists():
                continue
                
            try:
                with open(graph_path, "r", encoding="utf-8") as f:
                    graph_info = json.load(f)
                
                answers = graph_info.get("answers", [])
                answers_text = graph_info.get("answers_text", [])
                answers_concatenated = graph_info.get("answers_concatenated", "")
                question = graph_info.get("question", "")

                prompt_text = prompt_pattern.format(question=question)
                all_conversations.append([{"role": "user", "content": prompt_text}])
                all_metadata.append((graph_id, question, answers, answers_text, answers_concatenated))
            except Exception as e:                
                print(f"Warning: Could not process {graph_path}. Error: {e}")

        print(f"Collected {len(all_conversations)} total prompts to process.")
        if not all_conversations:
            print("No prompts found. Exiting.")
            return

        print("Stage 2: Running vLLM batch inference...")
        raw_outputs = self.llm_service.generate_batch(all_conversations)
        
        print("Stage 3: Processing and saving results...")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for i, raw_result in enumerate(tqdm(raw_outputs, desc="Post-processing")):
            clean_result = TextUtils.extract_model_response(raw_result) 
            clean_result = TextUtils.clean_gemma_output(clean_result)
            
            g_index, question, answers, answers_text, answers_concatenated = all_metadata[i]
            
            result = {
                "graph_id": g_index,
                "question": question,
                "answers": answers,
                "answers_text": answers_text,
                "answers_concatenated": answers_concatenated,
                "model_response": clean_result.strip()
            }

            json_output_path = self.output_dir / f"qa_results.{g_index}.json"
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

    def query_nodes(self, file_pattern: str, graphs_ids: List[str]):
        """
        Processes node prompts using vLLM in batches with deduping.
        """
        print("Stage 1: Collecting unique prompts form all graphs...")
        unique_nodes_to_query = {} # Map node_id -> prompt

        for graph_id in tqdm(graphs_ids, desc="Loading graphs"):
            nodes_path = self.input_dir / file_pattern.replace("{graph_id}", graph_id)
            if not nodes_path.exists():
                continue
                
            try:
                nodes_df = pd.read_parquet(nodes_path, engine="pyarrow")
                
                # Iterate rows to find unique nodes
                for row in nodes_df.itertuples():
                    node_index = row.node
                    # Only add if we haven't seen this node ID yet
                    if node_index not in unique_nodes_to_query:
                        prompt_text = getattr(row, 'prompt', None)
                        if isinstance(prompt_text, str) and prompt_text:
                            unique_nodes_to_query[node_index] = prompt_text
                            
            except Exception as e:
                print(f"Warning: Could not process {nodes_path}. Error: {e}")

        print(f"Found {len(unique_nodes_to_query)} unique nodes to query.")
        if not unique_nodes_to_query:
            print("No prompts found. Exiting.")
            return

        # Prepare lists for batch inference
        node_ids_list = list(unique_nodes_to_query.keys())
        all_conversations = [[{"role": "user", "content": unique_nodes_to_query[nid]}] for nid in node_ids_list]

        print("Stage 2: Running vLLM batch inference...")
        raw_outputs = self.llm_service.generate_batch(all_conversations)
        
        print("Stage 3: Building cache and saving results...")
        
        # Build a lookup cache: node_id -> generated description
        node_desc_cache = {}
        for i, raw_result in enumerate(tqdm(raw_outputs, desc="Post-processing")):
            clean_result = TextUtils.extract_model_response(raw_result) 
            clean_result = TextUtils.clean_gemma_output(clean_result)
            clean_result = TextUtils.compress_description(clean_result, max_facts=3, max_words_per_fact=16)
            node_desc_cache[node_ids_list[i]] = clean_result.strip()

        # Save results back to files
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for graph_id in tqdm(graphs_ids, desc="Saving files"):
            nodes_path = self.input_dir / file_pattern.replace("{graph_id}", graph_id)
            if not nodes_path.exists():
                continue

            try:
                nodes_df = pd.read_parquet(nodes_path, engine="pyarrow")
                # Map the descriptions using the node ID
                # Note: We use .map() so every row simply looks up its description
                nodes_df['description'] = nodes_df['node'].map(node_desc_cache)
                
                # Filter out entries that didn't get a description (if any) and save
                to_save = nodes_df[["node", "name", "types", 'is_type', "description"]].dropna(subset=["description"])
                
                output_file = self.output_dir / f"full_graph.nodes.{graph_id}.parquet"
                to_save.to_parquet(output_file, engine="pyarrow", index=False)
            except Exception as e:
                print(f"Warning: Could not process {nodes_path}. Error: {e}")

        print(f"Finished and saved all results in {self.output_dir}")

    def query_nodes_standard(self, file_pattern: str, graphs_ids: List[str]):
        """
        Processes node prompts using vLLM in batches and saves the updated graphs.
        This function reads all the prompted graph node files, extracts the prompts, and organizes them into a batch format for vLLM. 
        After receiving the raw outputs from the model, it cleans and processes the results, 
        merges them back with the original node data, and saves the final output in a specified directory.
        """
        print("Stage 1: Collecting all prompts from all graphs...")
        all_conversations = []
        all_metadata = []
        original_nodes_dfs = {} 

        for graph_id in tqdm(graphs_ids, desc="Loading graphs"):
            nodes_path = self.input_dir / file_pattern.replace("{graph_id}", graph_id)
            if not nodes_path.exists():
                continue
                
            try:
                nodes_df = pd.read_parquet(nodes_path, engine="pyarrow")
                original_nodes_dfs[graph_id] = nodes_df 
                
                for row in nodes_df.itertuples():
                    prompt_text = getattr(row, 'prompt', None)
                    node_index = row.node
                    if isinstance(prompt_text, str) and prompt_text:
                        all_conversations.append([{"role": "user", "content": prompt_text}])
                        all_metadata.append((graph_id, node_index))
            except Exception as e:
                print(f"Warning: Could not process {nodes_path}. Error: {e}")

        print(f"Collected {len(all_conversations)} total prompts to process.")
        if not all_conversations:
            print("No prompts found. Exiting.")
            return

        print("Stage 2: Running vLLM batch inference...")
        raw_outputs = self.llm_service.generate_batch(all_conversations)
        
        print("Stage 3: Processing and saving results...")
        all_results_data = []

        for i, raw_result in enumerate(tqdm(raw_outputs, desc="Post-processing")):
            clean_result = TextUtils.extract_model_response(raw_result) 
            clean_result = TextUtils.clean_gemma_output(clean_result)
            clean_result = TextUtils.compress_description(clean_result, max_facts=3, max_words_per_fact=16)
            
            g_index, n_index = all_metadata[i]
            all_results_data.append((g_index, n_index, clean_result))

        results_df = pd.DataFrame(all_results_data, columns=["graph_id", "node", "description"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for g_index, group_df in tqdm(results_df.groupby('graph_id'), desc="Saving files"):
            if g_index not in original_nodes_dfs:
                continue
                
            nodes_df = original_nodes_dfs[g_index]
            merged = nodes_df.merge(group_df, on="node")
            to_save = merged[["node", "name", "types", 'is_type', "description"]]
            
            output_file = self.output_dir / f"full_graph.nodes.{g_index}.parquet"
            to_save.to_parquet(output_file, engine="pyarrow", index=False)

        print(f"Finished and saved all results in {self.output_dir}")

    def query_edges(self, file_pattern: str, graphs_ids: List[str]):
        """
        Processes edge prompts using caching to avoid redundant queries.
        This function identifies unique relation keys across all prompted graph edge files, queries the LLM for descriptions of these unique relations, 
        and then merges the results back into the full graph format. It operates in three main steps:
        
        1. Triage - Find all unique relations: It scans through all the prompted graph edge files to identify all unique relation keys and their corresponding prompts. This creates a mapping of relation keys to their prompts, which will be used for querying the LLM.
        2. Query LLM for unique relations: It constructs a batch of conversations for the unique relation prompts and queries the LLM to get descriptions for each unique relation key.
        3. Assemble final results using the cache: It merges the LLM-generated descriptions back into the original edge data for each graph, ensuring that the same relation key gets the same description across all graphs. Finally, it saves the updated edge data in the specified output directory.
        """
        print("Step 1: Finding unique relations to query...")
        unique_rels_to_query = {}

        for graph_id in tqdm(graphs_ids, desc="Scanning edge files"):
            edges_path = self.input_dir / file_pattern.replace("{graph_id}", graph_id)
            if not edges_path.exists():
                continue
                
            try:
                edge_df = pd.read_parquet(edges_path, engine="pyarrow")
                for row in edge_df.itertuples():
                    rel = row.key
                    prompt_text = getattr(row, 'prompt', None)
                    if rel not in unique_rels_to_query and isinstance(prompt_text, str) and prompt_text:
                        unique_rels_to_query[rel] = prompt_text
            except Exception as e:
                print(f"Warning: Could not read {edges_path}. Error: {e}")
                
        print(f"Found {len(unique_rels_to_query)} unique relations to query.")
        if not unique_rels_to_query:
            return

        relations_list = list(unique_rels_to_query.keys())
        prompts_for_unique_rels = [unique_rels_to_query[rel] for rel in relations_list]

        print("Step 2: Querying LLM for unique relations with vLLM...")
        all_conversations = [[{"role": "user", "content": prompt}] for prompt in prompts_for_unique_rels]
        raw_outputs = self.llm_service.generate_batch(all_conversations)

        print("Inference complete. Building cache...")
        rel_cache = {}
        for i, raw_result in enumerate(tqdm(raw_outputs, desc="Processing results")):
            clean_result = TextUtils.extract_model_response(raw_result)
            clean_result = TextUtils.clean_gemma_output(clean_result)
            clean_result = TextUtils.compress_description(clean_result, max_facts=2, max_words_per_fact=18)
            rel_cache[relations_list[i]] = clean_result

        rel_cache_df = pd.DataFrame(list(rel_cache.items()), columns=["key", "description"])
        
        print("Step 3: Assembling final results using the cache...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for graph_id in tqdm(graphs_ids, desc="Assembling final graphs"):
            edges_path = self.input_dir / file_pattern.replace("{graph_id}", graph_id)
            if not edges_path.exists():
                continue

            try:
                edge_df = pd.read_parquet(edges_path, engine="pyarrow")
                merged = edge_df.merge(rel_cache_df, on="key", how="left")
                to_save = merged[["source", "target", "key", "description"]]
                output_file = self.output_dir / f"full_graph.edges.{graph_id}.parquet"
                to_save.to_parquet(output_file, engine="pyarrow", index=False)
            except Exception as e:
                print(f"Warning: Could not process {edges_path}. Error: {e}")

        print(f"Finished and saved all edge results in {self.output_dir}")

    def answer_questions(self):
        """
        Main method for answering questions about the graphs.
        """
        if not self.input_dir.exists():
            print(f"Input directory {self.input_dir} does not exist. Exiting.")
            return
        
        graphs_ids = self._get_graphs_ids(self.file_pattern_nodes)
        print(f"Discovered {len(graphs_ids)} total graphs.")
        if graphs_ids:
            self.query_questions(
                file_pattern=self.file_pattern_graphs,
                graphs_ids=graphs_ids,
            )

    def generate_nodes_descriptions(self):
        """Main method for generating node descriptions."""
        if not self.input_dir.exists():
            print(f"Input directory {self.input_dir} does not exist. Exiting.")
            return
        
        graphs_ids = self._get_graphs_ids(self.file_pattern_nodes)
        print(f"Discovered {len(graphs_ids)} total node graphs.")
        if graphs_ids:
            self.query_nodes(
                file_pattern=self.file_pattern_nodes,
                graphs_ids=graphs_ids,
            )
    
    def generate_edges_descriptions(self):
        """
        Main method for generating edge descriptions.
        """
        if not self.input_dir.exists():
            print(f"Input directory {self.input_dir} does not exist. Exiting.")
            return
        
        graphs_ids = self._get_graphs_ids(self.file_pattern_edges)
        print(f"Discovered {len(graphs_ids)} total edge graphs.")
        if graphs_ids:
            self.query_edges(
                file_pattern=self.file_pattern_edges,
                graphs_ids=graphs_ids,
            )
    
    def get_graphs_info(self):
        """
        Main method for retrieving graph information.
        Copies the full graph info files from the input directory to the output directory for all discovered graph ids.
        """
        if not self.input_dir.exists():
            print(f"Input directory {self.input_dir} does not exist. Exiting.")
            return
        
        graphs_ids = self._get_graphs_ids(self.file_pattern_graphs)
        print(f"Discovered {len(graphs_ids)} total graphs.")

        if graphs_ids:
            for graph_id in tqdm(graphs_ids, desc="Loading graph info"):
                graph_path = self.input_dir / self.file_pattern_graphs.replace("{graph_id}", graph_id)
                if not graph_path.exists():
                    continue
                try:
                    IOOperations.copy_and_rename(graph_path, self.output_dir, f"full_graph.graph.{graph_id}.json")
                except Exception as e:
                    print(f"Warning: Could not read {graph_path}. Error: {e}")

    @staticmethod
    def adjust_prompted_nodes(graphs_dir_prompted: str, graphs_dir_full: str, file_pattern: str = "prompted_graph.nodes.{graph_id}.parquet"):
        """
        Merges 'is_type' information back into the full graph node files.
        """
        graphs_path_prompted = Path(graphs_dir_prompted)
        graphs_path_full = Path(graphs_dir_full)
        
        search_pattern = graphs_path_prompted / file_pattern.replace("{graph_id}", "*")
        all_nodes_paths = glob.glob(str(search_pattern))
        all_graph_identifiers = sorted(list(set([os.path.basename(path).split('.')[2] for path in all_nodes_paths])))
        print(f"Discovered {len(all_graph_identifiers)} total graphs to adjust.")

        for graph_id in tqdm(all_graph_identifiers, desc="Adjusting graphs"):
            nodes_path = graphs_path_prompted / file_pattern.replace("{graph_id}", graph_id)
            full_nodes_path = graphs_path_full / f"full_graph.nodes.{graph_id}.parquet"
            
            if not nodes_path.exists() or not full_nodes_path.exists():
                continue

            try:
                nodes_df_prompted = pd.read_parquet(nodes_path, engine="pyarrow")
                nodes_df_full = pd.read_parquet(full_nodes_path, engine="pyarrow")
                
                merged = nodes_df_full.merge(nodes_df_prompted[["node", "is_type"]], on='node', how="inner")
                to_save = merged[["node", "name", "types", 'is_type', "description"]]
                to_save.to_parquet(full_nodes_path, engine="pyarrow", index=False)
            except Exception as e:
                print(f"Warning: Could not adjust {nodes_path}. Error: {e}")



