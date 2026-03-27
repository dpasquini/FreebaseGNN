

import torch
import torch.nn.functional as F
import tqdm
from pathlib import Path
from gnn.model.ModelFactory import ModelFactory
import random
import numpy as np

from utils.GraphUtils import GraphUtils

class GNNTraining:
    def __init__(self,  output_dir: str, 
                        gnn_model: str,
                        predictor_model: str,
                        embedding_dim: int,
                        hidden_dim: int, 
                        num_layers: int, 
                        learning_rate: float, 
                        num_epochs: int, 
                        weight_decay: float, 
                        num_heads: int, 
                        dropout: float, 
                        early_stopping_patience: int):
        self.output_dir = Path(output_dir)
        self.gnn_model = gnn_model
        self.predictor_model = predictor_model
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.num_heads = num_heads
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience

    def train(self, train_packs, val_packs, new_edge_types, node_types, max_positive_edge=256, stochastic_sampling=True, max_rel_per_graph=8, train_negative_ratio=19):
        """
        Main training loop for the GNN model. This method will handle the entire training process, including data loading, preprocessing, model initialization, training iterations, and evaluation on validation and test sets. 
        It will also implement early stopping based on the validation performance to prevent overfitting.
        """

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gnn_metadata = (node_types, new_edge_types)
        
        gnn_encoder = ModelFactory.create_gnn(
            model_name=self.gnn_model,
            gnn_metadata=gnn_metadata,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            n_heads=self.num_heads,
            dropout_rate=self.dropout,
            input_dim=self.embedding_dim
        )
        
        model = ModelFactory.create_predictor(
            predictor_name=self.predictor_model,
            gnn_model=gnn_encoder,
            hidden_dim=self.hidden_dim,
            edge_types=new_edge_types
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

        best_val_loss = float('inf')

        val_negative_ratio = train_negative_ratio

        model_path = self.output_dir / "best_model.pt"

        for epoch in tqdm.tqdm(range(1, self.num_epochs + 1), desc="Training epochs"):
            
            model.train()
            total_loss, used = 0.0, 0
            random.shuffle(train_packs)

            # We iterate over the training packs, which contain the train, validation, and test splits 
            # along with the full positive edge indices for each graph.
            for tr_data, _, _, full_pos in train_packs:
                tr_data = tr_data.to(device)
                optimizer.zero_grad()
                emb = model.encode(tr_data)
                
                loss_sum, n_sum = 0.0, 0

                # Only consider types that actually exist in this graph partition and have supervision edges
                available_types = []
                for et in new_edge_types:
                    if et in tr_data.edge_types and et in full_pos:
                        # Quick check if it has edges in the training split
                        if hasattr(tr_data[et], "edge_label_index") and tr_data[et].edge_label_index.numel() > 0:
                            available_types.append(et)

                # Stochastic Sampling of Relation Types (Memory Cap)
                # If there are more available relation types than the specified max_rel_per_graph, 
                # we randomly sample a subset of them for training in this epoch.
                # This helps to reduce memory usage and can also act as a form of regularization by 
                # not always training on all relation types at once, especially for large graphs with many relations.
                if stochastic_sampling and len(available_types) > max_rel_per_graph:
                    target_types = random.sample(available_types, max_rel_per_graph)
                else:
                    target_types = available_types

                for et in target_types:
                    all_pos = GraphUtils.get_pos_edges_from_split(tr_data, et)
                    if all_pos is None: continue
                    all_pos = all_pos.to(device)
                    
                    num_pos = all_pos.size(1)
                    
                    # Cap the number of positive edges to max_positive_edge to control memory usage.
                    if num_pos > max_positive_edge:
                        perm = torch.randperm(num_pos, device=device)[:max_positive_edge]
                        pos_batch = all_pos[:, perm]
                    else:
                        pos_batch = all_pos
                    
                    # Compute the number of negative samples to generate based on 
                    # the number of positive samples and the structure of the graph.
                    num_src = tr_data[et[0]].num_nodes
                    num_dst = tr_data[et[2]].num_nodes
                    # The number of forbidden negative edges is equal to 
                    # the total number of positive edges for this edge type, 
                    # as we want to exclude them from the negative sampling.
                    n_forbidden = full_pos[et].size(1) 
                    
                    # The maximum number of negative samples is the total possible edges 
                    # (num_src * num_dst) minus the forbidden edges (the positive ones).
                    max_neg = max(0, num_src * num_dst - n_forbidden)
                    
                    # We calculate the target number of negative samples based on 
                    # the specified train_negative_ratio and the number of positive samples in the batch.
                    target_neg = int(pos_batch.size(1) * train_negative_ratio)
                    
                    # We cap the target number of negative samples to 
                    # the maximum possible to avoid trying to sample more negatives than exist,
                    # which would cause an error.
                    target_neg = min(target_neg, max_neg)

                    if target_neg <= 0: 
                        continue # Skip if we cannot sample any negatives, which can happen if the graph is very small or has many positives.

                    # We sample negative edges while ensuring that they do not overlap with the positive edges.
                    neg = GraphUtils.sample_negatives_excluding_positives(
                        full_pos[et].to(device), 
                        num_src, 
                        num_dst, 
                        target_neg,
                        (et[0]==et[2]), 
                        device, 
                        "sparse"
                    )
                    # If no negative edges could be sampled (e.g., if the graph is very small or has many positives), we skip this edge type for this batch.
                    if neg.size(1)==0: 
                        continue
                    
                    # Create labels for the positive and negative edges (1 for positives, 0 for negatives) and concatenate them.
                    lbl = torch.cat([torch.ones(pos_batch.size(1), device=device), torch.zeros(neg.size(1), device=device)])
                    sc = model.decode(emb, torch.cat([pos_batch, neg], dim=1), et).view(-1)
                    
                    loss_sum += criterion(sc, lbl)
                    n_sum += 1

                if n_sum > 0:
                    (loss_sum/n_sum).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    total_loss += (loss_sum/n_sum).item()
                    used += 1

            # After each epoch of training, we evaluate the model on 
            # the validation set to monitor its performance and implement checkpointing.
            model.eval()
            val_loss, v_used = 0.0, 0
            with torch.no_grad():
                for _, va_data, _, full_pos in val_packs:
                    va_data = va_data.to(device); emb = model.encode(va_data)
                    l_sum, n_sum = 0.0, 0
                    for et in new_edge_types:
                        if et not in va_data.edge_types: continue
                        pos = GraphUtils.get_pos_edges_from_split(va_data, et)
                        if pos is None: continue
                        pos = pos.to(device)
                        
                        # For val loss, also cap negatives to be fast
                        num_src = va_data[et[0]].num_nodes
                        num_dst = va_data[et[2]].num_nodes
                        max_neg = max(0, num_src * num_dst - full_pos[et].size(1))
                        
                        target_neg = int(pos.size(1) * val_negative_ratio)
                        target_neg = min(target_neg, max_neg)
                        
                        if target_neg <= 0: 
                            continue

                        neg = GraphUtils.sample_negatives_excluding_positives(
                            full_pos[et].to(device), 
                            num_src, 
                            num_dst, 
                            target_neg, 
                            (et[0]==et[2]), 
                            device, 
                            "sparse"
                        )

                        if neg.size(1)==0: 
                            continue

                        sc = model.decode(emb, torch.cat([pos, neg], dim=1), et).view(-1)
                        lbl = torch.cat([torch.ones(pos.size(1), device=device), torch.zeros(neg.size(1), device=device)])
                        l_sum += criterion(sc, lbl).item(); n_sum += 1
                    if n_sum > 0: 
                        val_loss += l_sum/n_sum; v_used += 1
            
            avg_vl = val_loss/max(1, v_used)
            
            print(f"Ep {epoch:03d} | Tr: {total_loss/max(1,used):.4f} | Val: {avg_vl:.4f}")
            
            if avg_vl < best_val_loss:
                best_val_loss = avg_vl
                torch.save(model.state_dict(), model_path)

        return model
