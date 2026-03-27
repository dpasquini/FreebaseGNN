import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import multiprocessing as mp

# Dataclasses definitions
@dataclass
class TypedGraphsParameters:
    type_strategy: str
    graphs_input_file: str
    output_dir: str
    resolved_predicates: str
    resolved_nodes: str
    types_literals_file: str
    literals_db_path: str
    types_db_path: str
    n_graphs_for_test: int
    num_processes: int

@dataclass
class RetrieveParameters:
    graphs_input_file: str
    output_dir: str
    resolved_nodes: str
    freebase_file: str
    db_path: str

@dataclass
class EnhanceParameters:
    type_strategy: str
    output_dir: str
    enhanced_graphs_dir: str
    typed_graphs_dir: str
    entities_db_path: str
    literals_db_path: str
    types_db_path: str
    num_processes: int

@dataclass
class PromptParameters:
    output_dir: str
    num_processes: int
    checkpoint_dir: str
    graphs_to_prompt_dir: str
    sample_sizes: list
    entities_db_path: str
    bridge_types_db_path: str
    graph_prefix_filename: str

@dataclass
class LLMInferenceParameters:
    input_dir_nodes: str
    input_dir_edges: str
    output_dir_nodes: str
    output_dir_edges: str
    file_pattern_nodes: str
    file_pattern_edges: str
    file_pattern_graphs: str

@dataclass
class LLMSetup:
    model_name: str
    huggingface_token: str

@dataclass
class EmbeddingsGenerationParameters:
    output_dir: str
    input_dir: str
    file_pattern_nodes: str
    file_pattern_edges: str
    embedding_dim: int
    save_embeddings: bool
    compute_types_vocab: bool

@dataclass
class GNNTrainingParameters:
    input_dir: str
    output_dir: str
    file_pattern: str
    num_epochs: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    num_layers: int
    num_heads: int
    embedding_dim: int
    early_stopping_patience: int
    gnn_model: str
    predictor_model: str
    random_initialization: bool
    max_positive_edge: int
    stochastic_sampling: bool
    max_rel_per_graph: int
    train_negative_ratio: int
    edge_existence: bool
    homogenization: bool



def load_properties(file_path: str = "properties/prop.json") -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="FreebaseQA Graph Processing Pipeline")
    parser.add_argument('--properties', type=str, default="properties/prop.json", help='Properties file path')
    args = parser.parse_args()

    prop = load_properties(args.properties)
    
    # Global Parameters
    type_strategy = prop.get("type_strategy", "classic")
    tasks = prop.get("to_do", {})

    print("Pipeline Execution Plan:")
    print(f" - Retrieve Types/Literals: {tasks.get('retrieve_types_literals')}")
    print(f" - Typed Graphs:            {tasks.get('typed_graphs')}")
    print(f" - Enhance Graphs:          {tasks.get('enhance_graphs')}")
    print(f" - Resolve Bridges:         {tasks.get('enhance_with_bridges')}")
    print(f" - Create Type Index:       {tasks.get('create_bridge_types_index')}")
    print(f" - Create Prompts:          {tasks.get('create_prompts')}")
    print(f" - Compute Stats:           {tasks.get('compute_stats')}")
    print(f" - LLM Inference:            {tasks.get('llm_inference')}")
    print(f" - Embeddings Generation:   {tasks.get('embeddings_generation')}")
    print(f" - GNN Training:             {tasks.get('gnn_training')}")
    print()

    # Shared instances (initialized to None, imported locally)
    graph_construction = None
    graph_enhance = None

    if tasks.get("retrieve_types_literals"):
        from algorithms.GraphConstruction import GraphConstruction
        cfg = prop["graph_construction"]
        p = RetrieveParameters(
            graphs_input_file=cfg["graphs_input_file"],
            output_dir=cfg["output_dir"],
            resolved_nodes=cfg["resolved_nodes"],
            freebase_file=cfg["freebase"],
            db_path=cfg["db_path"]
        )
        graph_construction = GraphConstruction(p)

    if tasks.get("typed_graphs"):
        if graph_construction is None: # Need to import if not already done
             from algorithms.GraphConstruction import GraphConstruction
        
        cfg = prop["typed_graphs"]
        p = TypedGraphsParameters(
            type_strategy=type_strategy,
            graphs_input_file=cfg["graphs_input_file"],
            output_dir=cfg["output_dir"],
            resolved_predicates=cfg["resolved_predicates"],
            resolved_nodes=cfg["resolved_nodes"],
            types_literals_file=cfg["types_literals_file"],
            literals_db_path=cfg["literals_db_path"],
            types_db_path=cfg["types_db_path"],
            n_graphs_for_test=cfg.get("test_graphs", 0),
            num_processes=cfg.get("num_processes", 12)
        )
        if not graph_construction:
            graph_construction = GraphConstruction(p)
        graph_construction.generate_typed_graphs()

    if tasks.get("enhance_graphs") or tasks.get("enhance_with_bridges"):
        from algorithms.GraphEnhance import GraphEnhance
        cfg = prop["enhance_graphs"]
        p = EnhanceParameters(
            type_strategy=type_strategy,
            output_dir=cfg["output_dir"],
            enhanced_graphs_dir=cfg["enhanced_graphs_dir"],
            typed_graphs_dir=cfg["typed_graphs_dir"],
            entities_db_path=cfg["entities_db_path"],
            literals_db_path=cfg["literals_db_path"],
            types_db_path=cfg["types_db_path"],
            num_processes=cfg.get("num_processes", 12)
        )
        if not graph_enhance:
            graph_enhance = GraphEnhance(p)
        
        if tasks.get("enhance_graphs"):
            graph_enhance.enhance_graphs()
        
        if tasks.get("enhance_with_bridges"):
            graph_enhance.resolve_bridges()

    if tasks.get("create_prompts"):
        from algorithms.PromptCreation import PromptCreation
        cfg = prop["prompted_graph"]
        p = PromptParameters(
            output_dir=cfg["output_dir"],
            num_processes=cfg.get("num_processes", 12),
            checkpoint_dir=cfg["checkpoint_dir"],
            graphs_to_prompt_dir=cfg["graphs_to_prompt_dir"],
            sample_sizes=cfg["sample_sizes"],
            entities_db_path=cfg["entities_db_path"],
            bridge_types_db_path=cfg["bridge_types_db_path"],
            graph_prefix_filename=cfg["graph_prefix_filename"]
        )
        cp = PromptCreation(p)
        cp.create_prompt()

    if tasks.get("create_bridge_types_index"):
        from algorithms.CreateIndex import CreateIndex
        cfg = prop["type_index"]
        ci = CreateIndex(
            index_path=cfg["index_path"],
            literals_db_path=cfg["literals_db_path"],
            input_graphs_path=cfg["input_graphs_path"],
            graph_prefix_filename=cfg["graph_prefix_filename"]
        )
        ci.build_bridge_type_outgoing_index()

    if tasks.get("compute_stats"):
        from stats.Statistics import Statistics
        cfg = prop["stats"]
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        s = Statistics(input_dir=cfg["input_dir"], output_dir=str(output_dir))
        stats = s.compute_statistics()

        stats_file = output_dir / "stats.json"
        
        # Guard against stats objects that might not have a .tolist() method
        serializable = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in stats.items()}
        
        with open(stats_file, "w") as f:
            json.dump(serializable, f, indent=4)

    if tasks.get("llm_inference"):
        from llm.LLMInference import LLMInference
        cfg = prop["llm_inference"]
        cfg_llm_setup = prop["llm"]
        llm_setup = LLMSetup(
            model_name=cfg_llm_setup["model"],
            huggingface_token=cfg_llm_setup["huggingface_token"]
        )
        p = LLMInferenceParameters(
            input_dir_nodes=cfg["input_dir_nodes"],
            input_dir_edges=cfg["input_dir_edges"],
            output_dir_nodes=cfg["output_dir_nodes"],
            output_dir_edges=cfg["output_dir_edges"],
            file_pattern_nodes=cfg["file_pattern_nodes"],
            file_pattern_edges=cfg["file_pattern_edges"],
            file_pattern_graphs=cfg["file_pattern_graphs"]
        )
        llm_inf = LLMInference(model_name=llm_setup.model_name, 
                               input_dir=p.input_dir_nodes,
                               output_dir=p.output_dir_nodes,
                               file_pattern_nodes=p.file_pattern_nodes,
                               file_pattern_edges=p.file_pattern_edges,
                               file_pattern_graphs=p.file_pattern_graphs,
                               huggingface_token=llm_setup.huggingface_token)
        llm_inf.get_graphs_info()
        llm_inf.answer_questions()
        llm_inf.generate_nodes_descriptions()
        llm_inf.generate_edges_descriptions()
    
    if tasks.get("embeddings_generation"):
        from gnn.EmbeddingGraphs import EmbeddingGraphs
        cfg = prop["embeddings_generation"]
        p = EmbeddingsGenerationParameters(
            output_dir=cfg["output_dir"],
            input_dir=cfg["input_dir"],
            file_pattern_nodes=cfg["file_pattern_nodes"],
            file_pattern_edges=cfg["file_pattern_edges"],
            embedding_dim=cfg["embedding_dim"],
            save_embeddings=cfg.get("save_embeddings", True),
            compute_types_vocab=cfg.get("compute_types_vocab", False)
        )
        eg = EmbeddingGraphs(input_graph_path=p.input_dir, 
                             output_graph_path=p.output_dir, 
                             file_pattern_nodes=p.file_pattern_nodes, 
                             file_pattern_edges=p.file_pattern_edges, 
                             strategy=type_strategy)
        eg.from_text_to_embeddings(save_intermediate=p.save_embeddings, compute_types_vocab=p.compute_types_vocab, embedding_dim = p.embedding_dim)

    if tasks.get("gnn_training"):
        from gnn.preprocess.GNNDataProcessor import GNNDataProcessor
        from gnn.train.GNNTraining import GNNTraining
        from gnn.evaluation.GNNEvaluator import GNNEvaluator
        from gnn.evaluation.Evaluator import Evaluator
        from gnn.evaluation.GNNScorer import GNNScorer
        from gnn.train.BaselineTrainer import BaselineTrainer
        from gnn.model.RawMLPConcatPredictor import RawMLPConcatPredictor
        from gnn.evaluation.CosineBaselineScorer import CosineBaselineScorer
        from gnn.evaluation.MLPBaselineScorer import MLPBaselineScorer
        
        cfg = prop["gnn_training"]

        p = GNNTrainingParameters(
            input_dir=cfg["input_dir"],
            output_dir=cfg["output_model_dir"],
            file_pattern=cfg["file_pattern"],
            num_epochs=cfg["num_epochs"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            embedding_dim=cfg["embedding_dim"],
            early_stopping_patience=cfg["early_stopping_patience"],
            gnn_model=cfg["gnn_model"],
            predictor_model=cfg["predictor_model"],
            random_initialization=cfg.get("random_initialization", False),
            max_positive_edge=cfg.get("max_positive_edge", 256),
            stochastic_sampling=cfg.get("stochastic_sampling", True),
            max_rel_per_graph=cfg.get("max_rel_per_graph", 8),
            train_negative_ratio=cfg.get("train_negative_ratio", 19),
            edge_existence=cfg.get("edge_existence", False),
            homogenization=cfg.get("homogenization", False)
        )

        # Prepare PyTorch Geometric Data once
        processor = GNNDataProcessor(
            input_dir=p.input_dir, 
            file_pattern=p.file_pattern, 
            random_initialization=p.random_initialization
        )
        train_packs, val_packs, test_packs, new_edge_types, node_types = processor.prepare_data(
            top_k_relations=150,
            edge_existence=p.edge_existence,
            homogenization=p.homogenization
        )

        # Auto-detect embedding dimension from data to fix mismatch with multi-hot encoding
        if type_strategy == "multihot":
            # Check the first graph in the training set for node features to determine the actual embedding dimension
            if len(train_packs) > 0:
                sample_graph = train_packs[0][0] # (tr, va, te, full_pos) -> tr
                # Find the first node type that has features
                for ntype in sample_graph.node_types:
                    if sample_graph[ntype].num_nodes > 0 and sample_graph[ntype].x is not None:
                        actual_dim = sample_graph[ntype].x.shape[1]
                        if actual_dim != p.embedding_dim:
                            print(f"Overriding embedding_dim: Configured={p.embedding_dim}, Detected={actual_dim}")
                            p.embedding_dim = actual_dim
                        break

        mlp_model = RawMLPConcatPredictor(in_dim=p.embedding_dim, hidden=p.hidden_dim, dropout=p.dropout)
        mlp_trainer = BaselineTrainer(mlp_model, edge_types=new_edge_types, lr=1e-3)
        mlp_trainer.train(train_packs=train_packs, epochs=p.num_epochs, neg_ratio=p.train_negative_ratio)
        shared_evaluator = Evaluator()

        # Setup the Training Loop (the clean refactored trainer)
        trainer = GNNTraining(
            output_dir=p.output_dir,
            gnn_model=p.gnn_model,
            predictor_model=p.predictor_model,
            embedding_dim=p.embedding_dim,
            num_epochs=p.num_epochs,
            hidden_dim=p.hidden_dim,
            num_layers=p.num_layers,
            num_heads=p.num_heads,
            dropout=p.dropout,
            learning_rate=p.learning_rate,
            weight_decay=p.weight_decay,
            early_stopping_patience=p.early_stopping_patience
        )

        # Run the training
        trained_model = trainer.train(train_packs, val_packs, new_edge_types, node_types, max_positive_edge=p.max_positive_edge, stochastic_sampling=p.stochastic_sampling, max_rel_per_graph=p.max_rel_per_graph, train_negative_ratio=p.train_negative_ratio)
        ratio = [1, 9, 100]
        for scorer_name, scorer in [
            ("Cosine", CosineBaselineScorer()),
            ("MLP", MLPBaselineScorer(mlp_model)),
            ("GNN", GNNScorer(trained_model)) 
        ]:
            for r in ratio:
                bt_r = shared_evaluator.find_best_threshold(val_packs, new_edge_types, r, scorer)
                
                # Unified evaluation master call loops cleanly over them without redefining loops natively!
                shared_evaluator.evaluate_metrics_master(
                    name=scorer_name,
                    packs=test_packs,
                    edge_types=new_edge_types,
                    neg_ratio=r,
                    scorer=scorer,
                    split="test",
                    threshold=bt_r
                )

        # Comprehensive Evaluation
        evaluator = GNNEvaluator(model=trained_model)
        evaluator.evaluate(val_packs, test_packs, edge_types=new_edge_types, ratio=ratio, train_negative_ratio=p.train_negative_ratio)
    

if __name__ == '__main__':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()


