# Linguistic Initialization for Inductive Reasoning in Heterogeneous Knowledge Graphs

Codebase for the paper accepted at the LREC 2026 Workshop on Knowledge Graphs and Large Language Models (KG-LLM).

This repository implements a configurable pipeline over Freebase-derived graphs, combining graph processing, LLM-based enrichment, embedding generation, and GNN training/evaluation.

## What This Repository Currently Runs

The execution flow is controlled by `to_do` flags in `properties/prop.json` and orchestrated by `Main.py`.

Implemented and wired tasks in `Main.py`:

1. `retrieve_types_literals`
2. `typed_graphs`
3. `enhance_graphs`
4. `enhance_with_bridges`
5. `create_prompts`
6. `create_bridge_types_index`
7. `compute_stats`
8. `llm_inference`
9. `embeddings_generation`
10. `gnn_training`

## Project Layout

```
FreebaseGNN/
|-- algorithms/
|   |-- GraphConstruction.py
|   |-- GraphEnhance.py
|   |-- PromptCreation.py
|   `-- CreateIndex.py
|-- gnn/
|   |-- EmbeddingGraphs.py
|   |-- preprocess/
|   |-- model/
|   |-- train/
|   `-- evaluation/
|-- llm/
|   |-- LLMInference.py
|   `-- LLMService.py
|-- stats/
|-- utils/
|-- properties/
|   `-- prop.json
`-- Main.py
```

## Requirements

Recommended environment:

1. Python 3.9+
2. PyTorch + PyTorch Geometric
3. transformers
4. vllm (if running LLM inference through vLLM)
5. pandas
6. pyarrow and/or fastparquet
7. tqdm

Install with your preferred environment manager (venv/conda).

## Configuration

All pipeline behavior is driven by `properties/prop.json`.

### Task Switches (`to_do`)

Example:

```json
"to_do": {
  "retrieve_types_literals": false,
  "typed_graphs": false,
  "enhance_graphs": false,
  "enhance_with_bridges": false,
  "create_prompts": false,
  "create_bridge_types_index": false,
  "compute_stats": false,
  "llm_inference": false,
  "embeddings_generation": false,
  "gnn_training": false,
  "qa": false,
  "generate_structural_questions": false
}
```

### Config Blocks Used by the Pipeline

Depending on enabled tasks, `Main.py` reads:

1. `typed_graphs`
2. `enhance_graphs`
3. `prompted_graph`
4. `type_index`
5. `stats`
6. `llm`
7. `llm_inference`
8. `embeddings_generation`
9. `gnn_training`

Note:

1. Keep secrets out of the repository. Do not store tokens directly in `prop.json`.
2. Pass sensitive values through environment variables or a local ignored config.

## Run

From repository root:

```bash
python Main.py --properties properties/prop.json
```

To run a single stage, set only the corresponding `to_do` flag to `true` and keep others `false`.

## GNN Training and Evaluation

The `gnn_training` stage includes:

1. Data preparation via `GNNDataProcessor`
2. Baseline MLP training
3. GNN training through `GNNTraining`
4. Evaluation with multiple scorers:
   - Cosine baseline
   - MLP baseline
   - GNN scorer

## Authors

Daniele Pasquini, Danilo Croce, Roberto Basili
