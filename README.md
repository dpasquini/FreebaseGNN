# FreebaseQA Graph Processing Pipeline

FreebaseQA is a comprehensive framework designed for performing Question Answering (QA) tasks over Freebase knowledge graphs. The project implements a multi-stage pipeline that ingests raw Freebase data, constructs and enhances graph structures, leverages Large Language Models (LLMs) for semantic enrichment, and trains Graph Neural Networks (GNNs) for advanced link prediction and reasoning.

## Features

The pipeline consists of several modular components controlled via a configuration file:

1.  **Graph Construction & Typing**: Extracts subgraphs from Freebase and resolves entity types and literals.
2.  **Graph Enhancement**: Enriches graphs with additional structural information and resolves "bridge" nodes to connect distant entities.
3.  **Prompt Engineering**: Automatically generates prompts for nodes and edges to elicit semantic descriptions.
4.  **LLM Inference**: Integrates with **vLLM** and HuggingFace models (e.g., Gemma) to generate high-quality text descriptions for graph elements.
5.  **Embedding Generation**: Converts textual descriptions and graph structures into vector embeddings (Multihot, etc.).
6.  **GNN Training**: Trains Heterogeneous Graph Attention Networks (**HeteroGAT**) for link prediction tasks.
7.  **Evaluation**: Includes a robust evaluation suite compatible with PyTorch Geometric.

## Project Structure

```
FreebaseQA/
├── algorithms/       # Core graph processing algorithms (Construction, Enhancement, Indexing)
├── gnn/              # Graph Neural Network modules
│   ├── model/        # GNN architectures (HeteroGAT, LinkPredictor)
│   ├── train/        # Training loops and logic
│   ├── preprocess/   # Data loading and transformation (PyG HeteroData)
│   └── evaluation/   # Scorers and Evaluators
├── llm/              # LLM integration components
│   ├── LLMInference.py # Inference orchestration
│   └── LLMService.py   # vLLM service wrapper
├── properties/       # Configuration files
│   └── prop.json     # Main pipeline configuration
├── stats/            # Statistical analysis tools
├── utils/            # Helper utilities (IO, Text processing, Tensors)
└── Main.py           # Application Entry Point
```

## 🛠️ Installation & Requirements

The project relies on Python (>3.8) and several key libraries for deep learning and graph processing.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/FreebaseQA.git
    cd FreebaseQA
    ```

2.  **Install Dependencies:**
    It is recommended to use a Conda environment. Key dependencies include:
    *   `torch`
    *   `torch_geometric` (PyG)
    *   `vllm`
    *   `transformers`
    *   `pandas`
    *   `tqdm`
    *   `fastparquet` / `pyarrow`

    *(Note: Ensure your CUDA version matches the Torch installation for GPU support).*

## Configuration

The pipeline is entirely driven by the `properties/prop.json` file. This JSON file controls which steps of the pipeline are executed and provides the necessary paths and parameters for each step.

### The `to_do` Block
The `to_do` section in `prop.json` acts as a switchboard:

```json
"to_do": {
  "retrieve_types_literals": false,  // Step 1: Raw data extraction
  "typed_graphs": false,             // Step 2: Build typed graphs
  "enhance_graphs": false,           // Step 3: Add external knowledge
  "create_prompts": false,           // Step 4: Generate LLM prompts
  "llm_inference": true,             // Step 5: Run LLM on prompts
  "embeddings_generation": true,     // Step 6: Create embeddings from text
  "gnn_training": true               // Step 7: Train GNN model
}
```

### Module Configuration
Each step has a corresponding configuration block in `prop.json` (e.g., `"gnn_training"`, `"llm_inference"`) where you specify input/output directories and model hyperparameters (learning rate, epochs, embedding dimensions, etc.).

## ▶️ How to Run

To run the pipeline, simply execute the `Main.py` script. By default, it looks for `properties/prop.json`.

```bash
python Main.py --properties properties/prop.json
```

### Running Specific Modules
To run only specific parts of the pipeline (e.g., only **GNN Training**), modify the `properties/prop.json` file:

1.  Open `properties/prop.json`.
2.  Set `"gnn_training": true` inside the `"to_do"` block.
3.  Set all other steps to `false`.
4.  Run `python Main.py`.

## GNN Model Details

The project implements a **Heterogeneous Graph Attention Network (HeteroGAT)**.
*   **Encoders**: Handles diverse node types (Entities, Types, etc.) with specific embedding strategies (Multihot, Textual).
*   **Training**: Supports stochastic sampling for large graphs and negative sampling for link prediction.
*   **Predictor**: Uses a Link Predictor or MLP to score edge existence.

## 🤖 LLM Integration

The `LLMInference` module leverages **vLLM** for high-throughput inference. It supports batch processing of graph nodes/edges to generate descriptions, which are then used to enrich the initial node embeddings for the GNN.

---
*Author: Daniele Pasquini, Danilo Croce, Roberto Basili*
