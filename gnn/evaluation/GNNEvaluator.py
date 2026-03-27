import torch
import numpy as np
from collections import defaultdict
import random
from sklearn.metrics import f1_score, precision_score, recall_score
from utils.GraphUtils import GraphUtils
import torch.nn.functional as F

from gnn.evaluation.Evaluator import Evaluator
from gnn.evaluation.GNNScorer import GNNScorer

class GNNEvaluator(Evaluator):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.scorer = GNNScorer(model)

    def create_hard_test_set(self, test_packs, max_degree=1):
        """
        This method creates a "hard" test set by filtering the original test edges to include only those that are truly challenging for the model.
        The filtering criteria are based on the degree of the nodes involved in the edges, 
        specifically targeting edges where at least one of the nodes has a degree less than or equal to 
        a specified threshold (max_degree) in the training graph.
        The rationale behind this approach is that edges involving low-degree nodes are often more difficult for models to predict accurately, 
        as they have less contextual information available during training. 
        By creating a test set that focuses on these "hard" edges, we can gain deeper insights into the model's performance in scenarios that are more representative of real-world challenges, 
        such as cold-start situations or cases where the model has limited information about certain nodes.

        
        Parameters:
        - test_packs: A list of tuples, where each tuple contains (tr_data, va_data, te_data, full_pos) for a graph. 
                      tr_data is the training graph, va_data is the validation graph, te_data is the original test graph, and full_pos contains all positive edges across splits.
        - max_degree: An integer threshold for node degree. Edges will be included in the hard test set if at least one of the nodes has a degree <= max_degree in the training graph.

        """
        print(f"\n--- CREATING HARD TEST SET (Max Context Degree <= {max_degree}) ---")
        
        hard_packs = []
        total_edges = 0
        hard_edges = 0
        
        for (tr_data, va_data, te_data, full_pos) in test_packs:
            # tr_data contains the full graph with all edges (train + val + test) and is what the model sees during training.
            # te_data contains only the test edges (positives) and is what we evaluate on.
            # full_pos contains the positive edges for all splits and is used to avoid sampling negatives that are actually positives.
            
            # Compute the degree of each node in the training graph (tr_data) to identify "hard" nodes.
            node_degrees = defaultdict(lambda: torch.zeros(0))
            
            # Initialize degree counts for all node types
            for nt in tr_data.node_types:
                node_degrees[nt] = torch.zeros(tr_data[nt].num_nodes, device='cpu')
                
            for et in tr_data.edge_types:
                src_t, _, dst_t = et
                row, col = tr_data[et].edge_index
                
                # We move to CPU for degree calculation to avoid GPU memory issues, 
                # as we only need the degree counts and not the full graph on GPU for this step.
                row, col = row.cpu(), col.cpu()
                
                node_degrees[src_t].index_add_(0, row, torch.ones_like(row, dtype=torch.float))
                node_degrees[dst_t].index_add_(0, col, torch.ones_like(col, dtype=torch.float))
                
            # Filter the test edges in te_data to keep only those where 
            # at least one of the nodes has degree <= max_degree in the training graph.
            new_full_pos = {}
            has_edges = False
            
            for et in te_data.edge_types:
                src_t, _, dst_t = et
                
                # Retrieve the positive edges for this edge type from the test data
                pos_edge_index = GraphUtils.get_pos_edges_from_split(te_data, et)
                if pos_edge_index is None: 
                    continue
                
                pos_edge_index = pos_edge_index.cpu()
                total_edges += pos_edge_index.size(1)
                
                src_degs = node_degrees[src_t][pos_edge_index[0]]
                dst_degs = node_degrees[dst_t][pos_edge_index[1]]
                
                # Hard condition: Source OR Dest degree <= max_degree
                mask = (src_degs <= max_degree) | (dst_degs <= max_degree)
                
                hard_pos_edges = pos_edge_index[:, mask]
                
                if hard_pos_edges.size(1) > 0:
                    # Store the hard positive edges for this edge type. 
                    # This will be used to create the new test set and also to 
                    # guide negative sampling (to avoid sampling these as negatives).
                    if et not in new_full_pos:
                        new_full_pos[et] = hard_pos_edges # Store for neg sampling later
                    
                    # Count hard edges for reporting
                    hard_edges += hard_pos_edges.size(1)
                    has_edges = True
                    
            if has_edges:
                # We keep the same training and validation data (tr_data, va_data) since we are only modifying the test set.
                # we create a new test data (te_data_hard) that contains only the hard positive edges.
                # we pass full_pos unfiltered to maintain the same negative sampling constraints 
                # (avoid sampling any positive edge as negative), 
                # but the evaluation will only consider the hard edges in te_data_hard.
                
                te_data_hard = te_data.clone()
                
                #  Overwrite test edges with hard positives only
                for et in new_full_pos:
                    store = te_data_hard[et]
                    # Assuming te_data[et] has edge_index and optionally edge_attr, we replace edge_index with the hard positives.
                    store.edge_label_index = new_full_pos[et]
                    store.edge_label = torch.ones(new_full_pos[et].size(1))
                
                # Remove edge types from te_data_hard that do not have any hard positives, to avoid evaluating on them.
                for et in te_data.edge_types:
                    if et not in new_full_pos:
                        if hasattr(te_data_hard[et], 'edge_label_index'):
                            del te_data_hard[et].edge_label_index
                            del te_data_hard[et].edge_label

                # Append the modified pack with the hard test set
                # We keep the original full_pos for negative sampling to ensure that 
                # we do not accidentally sample any of the hard positives as negatives 
                # during evaluation, which would invalidate the evaluation results.
                hard_packs.append((tr_data, va_data, te_data_hard, full_pos))

        print(f"Original Test Edges: {total_edges}")
        print(f"Hard Test Edges (Cold Start): {hard_edges} ({hard_edges/total_edges:.1%})")
        print(f"Hard Graphs: {len(hard_packs)}")
        
        return hard_packs

    def evaluate_real_hard_set_entity_to_entity(self, model, packs, edge_types, neg_ratio=100, threshold=0.5):
        """
        Evaluates on the 'Real Hard' subset:
        1. Source Node Degree <= 1 (Cold Start / Inductive)
        2. Destination is NOT a 'Type' node (Removes Hub/Schema bias)
        3. Relation is NOT a 'BUCKET' (Removes broad grouping bias)
        """
        device = next(model.parameters()).device
        model.eval()
        
        y_true_all = []
        y_score_all = []
        
        print(f"\n" + "█"*70)
        print(f"   REAL HARD SET: Entity-to-Entity Cold Start (1:{neg_ratio})")
        print(f"   (Filters: SrcDeg<=1, Dst!=Type, Rel!=Bucket)")
        print("█"*70)

        with torch.no_grad():
            for _, _, data, full_pos in packs:
                data = data.to(device)
                x_dict = model.encode(data)
                
                node_degrees = {}
                for et in data.edge_types:
                    src_t, _, _ = et
                    s, _ = data[et].edge_index
                    if src_t not in node_degrees: 
                        node_degrees[src_t] = torch.zeros(data[src_t].num_nodes, device=device)
                    node_degrees[src_t].index_add_(0, s, torch.ones_like(s, dtype=torch.float))

                for et in edge_types:
                    src_t, rel_name, dst_t = et
                    
                    if "BUCKET" in rel_name: 
                        continue
                        
                    if dst_t == "ntype_type": 
                        continue

                    if et not in data.edge_types or et not in full_pos: 
                        continue
                    
                    pos_edge_index = GraphUtils.get_pos_edges_from_split(data, et)
                    if pos_edge_index is None: 
                        continue
                    pos_edge_index = pos_edge_index.to(device)
                    
                    if src_t not in node_degrees: 
                        continue
                    
                    src_ids = pos_edge_index[0]
                    degrees = node_degrees[src_t][src_ids]
                    
                    hard_mask = (degrees <= 1)
                    
                    if hard_mask.sum() == 0: 
                        continue
                    
                    pos_hard = pos_edge_index[:, hard_mask]
                    
                    num_pos = pos_hard.size(1)
                    num_src = data[src_t].num_nodes
                    num_dst = data[dst_t].num_nodes
                    
                    max_neg = max(0, num_src * num_dst - full_pos[et].size(1))
                    target_neg = min(int(num_pos * neg_ratio), max_neg)
                    
                    if target_neg <= 0: continue
                    
                    neg = GraphUtils.sample_negatives_excluding_positives(
                        full_pos[et].to(device), 
                        num_src, 
                        num_dst, 
                        target_neg, 
                        (src_t == dst_t), 
                        device, 
                        "sparse"
                    )
                    
                    if neg.size(1) == 0: 
                        continue
                    
                    pos_scores = torch.sigmoid(model.decode(x_dict, pos_hard, et).view(-1))
                    neg_scores = torch.sigmoid(model.decode(x_dict, neg, et).view(-1))
                    
                    scores = torch.cat([pos_scores, neg_scores])
                    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
                    
                    y_true_all.append(labels.cpu().numpy())
                    y_score_all.append(scores.cpu().numpy())

        if not y_true_all:
            print("[Warning] No edges met the 'Real Hard' criteria.")
            return

        y_true = np.concatenate(y_true_all)
        y_score = np.concatenate(y_score_all)
        
        acc, pr, re, f1 = self.metrics_from_scores(y_true, y_score, threshold)
        
        print(f"Total Edges Evaluated: {len(y_true)}")
        print(f"Positives (Hard): {int(y_true.sum())}")
        print(f"Negatives: {len(y_true) - int(y_true.sum())}")
        print("-" * 40)
        print(f"MICRO METRICS (Thr={threshold:.4f}):")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  Precision: {pr:.4f}")
        print(f"  Recall   : {re:.4f}")
        print(f"  Accuracy : {acc:.4f}")
        print("="*70 + "\n")
    
    def run_comprehensive_error_analysis(self, model, test_packs, train_negative_ratio, threshold=0.5, k_examples=5):
        """
        This method performs a comprehensive error analysis on the model's predictions, focusing on the "Real Hard" subset of the test set.
        The analysis includes:
        1. Error Categorization: Classifying errors into True Positives (TP), False Positives (FP), and False Negatives (FN) based on the model's predictions compared to the true labels.
        2. Degree-Based Analysis: Evaluating the distribution of errors based on the degree of the source nodes, categorizing them into "low", "medium", and "high" degree buckets to understand how node connectivity affects model performance.
        3. Semantic Similarity Analysis: For False Positives, computing the cosine similarity between the source and destination node embeddings to 
            identify cases where the model may have made a "plausible" prediction that is semantically close to the true relationship,
            even if it is not correct according to the test labels. This can help identify potential missing links in the graph.
        4. Case Studies: Collecting specific examples of errors for qualitative analysis, including:
            - Rare True Positives: Correct predictions on rare edge types or low-degree nodes.
            - Plausible False Positives: Incorrect predictions that have high semantic similarity, suggesting they may be reasonable predictions that are not labeled as such in the test set.
            - Hard False Negatives: Missed predictions that have high semantic similarity, indicating the model missed an obvious semantic link.
            - Stupid False Negatives: Missed predictions with low semantic similarity, suggesting a potential structural failure in the model's understanding of the graph.
        The insights gained from this analysis can inform future model improvements and help understand the limitations of the current model in handling the most challenging cases in the graph.

        """
        device = next(model.parameters()).device
        model.eval()
        
        print("\n" + "="*60)
        print("   COMPREHENSIVE ERROR ANALYSIS & CASE STUDIES")
        print("="*60)

        stats_tail = {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0}
        stats_head = {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0}
        
        degree_bins = {'low': [], 'med': [], 'high': []} # List of (pred == target) boolean
        
        # For semantic analysis of FPs
        fp_similarities = [] 
        
        # Case studies storage
        cases = {
            'rare_tp': [],      # Bucket relations predicted correctly
            'plausible_fp': [], # FP with high cosine similarity (Potential Missing Link)
            'hard_fn': [],      # FN with high cosine similarity (Model missed obvious semantic link)
            'stupid_fn': []     # FN with low cosine similarity (Structural failure)
        }

        with torch.no_grad():
            for _, _, data, full_pos in test_packs:
                data = data.to(device)
                
                # Compute node degrees for all node types in the graph. 
                # This will be used to identify "hard" nodes based on their degree (e.g., nodes with degree <= 2 are "low", 3-10 "med", >10 "high").
                node_degrees = {}
                for et in data.edge_types:
                    src, _, dst = et
                    s, d = data[et].edge_index
                    if src not in node_degrees: 
                        node_degrees[src] = torch.zeros(data[src].num_nodes, device=device)
                    if dst not in node_degrees: 
                        node_degrees[dst] = torch.zeros(data[dst].num_nodes, device=device)
                    # In-degree + Out-degree sum
                    node_degrees[src].index_add_(0, s, torch.ones_like(s, dtype=torch.float))
                    node_degrees[dst].index_add_(0, d, torch.ones_like(d, dtype=torch.float))

                x_dict = model.encode(data)

                for et in data.edge_types:
                    # Identify if this edge type is a "bucket" relation (rare) or a "head" relation (frequent) based on its name.
                    is_bucket = "BUCKET" in et[1]
                    
                    # --- Preparazione Dati ---
                    pos_edge_index = GraphUtils.get_pos_edges_from_split(data, et)
                    if pos_edge_index is None: 
                        continue
                    pos_edge_index = pos_edge_index.to(device)
                    
                    if et not in full_pos: 
                        continue
                    
                    # We use the train_negative_ratio to determine how many negative samples to generate for this analysis,
                    # as we want to maintain the same evaluation conditions as during training, 
                    # ensuring that the error analysis reflects the model's performance under realistic negative sampling scenarios.
                    neg_ratio_analysis = train_negative_ratio 
                    
                    # Calculate how many negative samples to generate based on
                    # the number of positive edges and the specified negative ratio, 
                    # while ensuring we do not exceed the maximum possible negatives 
                    # given the graph size. 
                    # This is crucial for maintaining a realistic evaluation scenario that 
                    # reflects the conditions under which the model was trained, 
                    # allowing us to analyze errors in a context that is consistent with the model's learning environment. 
                    # The generated negative samples will be used alongside the positive samples to 
                    # compute predictions and evaluate metrics, 
                    # as well as to perform semantic analysis and case studies on the errors made by the model.
                    num_src = data[et[0]].num_nodes
                    num_dst = data[et[2]].num_nodes
                    max_neg = max(0, num_src * num_dst - full_pos[et].size(1))
                    
                    # Qui ratio 19 o 100
                    target_neg = int(pos_edge_index.size(1) * neg_ratio_analysis)
                    target_neg = min(target_neg, max_neg)
                    
                    if target_neg <= 0: 
                        continue
                    
                    neg_edge_index = GraphUtils.sample_negatives_excluding_positives(
                        full_pos[et].to(device), 
                        num_src, 
                        num_dst,
                        target_neg, 
                        (et[0]==et[2]), 
                        device, 
                        "sparse"
                    )
                    
                    if neg_edge_index.size(1) == 0: 
                        continue

                    # Merge pos and neg for unified processing
                    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
                    y_true = torch.cat([
                        torch.ones(pos_edge_index.size(1), device=device),
                        torch.zeros(neg_edge_index.size(1), device=device)
                    ])
                    
                    # Compute predictions for all edges (pos + neg) in this edge type using the model's decode function,
                    logits = model.decode(x_dict, edge_label_index, et).view(-1)
                    probs = torch.sigmoid(logits)
                    preds = (probs >= threshold).float()
                    
                    # Compute cosine similarity between source and destination node embeddings for all edges (pos + neg) in this edge type.
                    src_emb = x_dict[et[0]][edge_label_index[0]]
                    dst_emb = x_dict[et[2]][edge_label_index[1]]
                    # Cosine similarity will be used in the semantic analysis of errors,
                    # as it provides insight into how semantically related the source and destination nodes are,
                    # which can help us understand whether certain errors (e.g., false positives) 
                    # are due to the model confusing semantically similar pairs (potentially plausible errors) or 
                    # if they are random mistakes (structural errors).
                    semantic_sim = F.cosine_similarity(src_emb, dst_emb)

                    # Masks for TP, FP, FN based on true labels and predictions
                    is_pos = (y_true == 1)
                    is_neg = (y_true == 0)
                    is_pred_pos = (preds == 1)
                    is_pred_neg = (preds == 0)
                    
                    # TP / FP / FN masks
                    tp_mask = is_pos & is_pred_pos
                    fp_mask = is_neg & is_pred_pos
                    fn_mask = is_pos & is_pred_neg
                    
                    # Step 1: Long Tail Analysis (Bucket vs Head)
                    target_stats = stats_tail if is_bucket else stats_head
                    target_stats['tp'] += tp_mask.sum().item()
                    target_stats['fp'] += fp_mask.sum().item()
                    target_stats['fn'] += fn_mask.sum().item()
                    target_stats['count'] += len(y_true)
                    
                    # Step 2: Topological Analysis (Only on Positives)
                    if is_pos.sum() > 0:
                        pos_indices = torch.where(is_pos)[0]
                        src_nodes_pos = edge_label_index[0][pos_indices]
                        # Get degrees for the source nodes of the positive edges and bin them into low/med/high degree categories,
                        degrees = node_degrees[et[0]][src_nodes_pos]
                        # We will analyze the recall (TP rate) for positives in each degree bin 
                        # to see if the model struggles more with low-degree nodes (potentially due to less training signal) 
                        # compared to high-degree nodes (which might be easier to learn due to more connectivity).
                        corrects = preds[pos_indices] 
                        
                        for deg, corr in zip(degrees, corrects):
                            d_val = deg.item()
                            if d_val <= 2: degree_bins['low'].append(corr.item())
                            elif d_val <= 10: degree_bins['med'].append(corr.item())
                            else: degree_bins['high'].append(corr.item())

                    # Step 3: Semantic Analysis of Errors (Focus on FPs, but also look at FNs)
                    
                    # FP Analysis (Global stats + Case Studies)
                    if fp_mask.sum() > 0:
                        idxs = torch.where(fp_mask)[0]
                        # We collect the cosine similarities of the false positives to analyze their distribution,
                        # as this can indicate whether the model is making "plausible" errors 
                        # (high similarity, potentially confusing semantically related pairs)
                        fp_similarities.extend(semantic_sim[idxs].detach().cpu().tolist())
                        
                        for i in idxs:
                            sim_val = semantic_sim[i].item()
                            # Plausible FP (High Sim)
                            if sim_val > 0.7: 
                                cases['plausible_fp'].append({
                                    'src': edge_label_index[0][i].item(), 'dst': edge_label_index[1][i].item(),
                                    'rel': et[1], 'score': probs[i].item(), 'sim': sim_val
                                })

                    # TP Analysis (Rare only)
                    if is_bucket and tp_mask.sum() > 0:
                        idxs = torch.where(tp_mask)[0]
                        # We collect some of the true positives for rare (bucket) relations to analyze successful cases in the long tail,
                        # as this can provide insights into what the model is getting right in the more challenging, less frequent relations,
                        # potentially revealing patterns or characteristics of the cases that the model can successfully learn in the long tail,
                        # which can inform future improvements in model design or training strategies to further enhance performance on these difficult cases.
                        sel_idx = idxs[:5] 
                        for i in sel_idx:
                            cases['rare_tp'].append({
                                'src': edge_label_index[0][i].item(), 'dst': edge_label_index[1][i].item(),
                                'rel': et[1], 'score': probs[i].item(), 'sim': semantic_sim[i].item()
                            })
                    
                    # FN Analysis
                    if fn_mask.sum() > 0:
                        idxs = torch.where(fn_mask)[0]
                        for i in idxs:
                            sim_val = semantic_sim[i].item()
                            # Hard FN (Should have been easy)
                            if sim_val > 0.6: 
                                cases['hard_fn'].append({
                                    'src': edge_label_index[0][i].item(), 'dst': edge_label_index[1][i].item(),
                                    'rel': et[1], 'score': probs[i].item(), 'sim': sim_val
                                })
                            # Structural FN (Nodes far apart)
                            elif sim_val < 0.2:
                                cases['stupid_fn'].append({
                                    'src': edge_label_index[0][i].item(), 'dst': edge_label_index[1][i].item(),
                                    'rel': et[1], 'score': probs[i].item(), 'sim': sim_val
                                })

        
        # Step 4: Print Comprehensive Report
        # 1. LONG TAIL (Head vs Bucket)
        print("\n### 1. LONG TAIL STRESS TEST (Generalization)")
        print(f"{'Category':<10} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Count'}")
        print("-" * 55)
        for name, s in [('HEAD (Freq)', stats_head), ('TAIL (Rare)', stats_tail)]:
            p = s['tp'] / (s['tp'] + s['fp'] + 1e-9)
            r = s['tp'] / (s['tp'] + s['fn'] + 1e-9)
            f1 = 2*p*r/(p+r+1e-9)
            print(f"\n{name:<10} | {p:.4f}   | {r:.4f}   | {f1:.4f}   | {s['count']}")

        # 2. TOPOLOGICAL BIAS (Recall by Node Degree)
        print("\n### 2. TOPOLOGICAL BIAS (Recall by Node Degree)")
        for cat in ['low', 'med', 'high']:
            vals = degree_bins[cat]
            if vals:
                rec = sum(vals)/len(vals)
                print(f"\n  - {cat.upper():<4} Degree: Recall = {rec:.4f} (Samples: {len(vals)})")
            else:
                print(f"\n  - {cat.upper():<4} Degree: No samples")

        # 3. SEMANTIC ANALYSIS OF ERRORS (Focus on FPs)
        print("\n### 3. SEMANTIC ANALYSIS OF ERRORS")
        if fp_similarities:
            avg_fp_sim = np.mean(fp_similarities)
            print(f"\n  - Average Cosine Similarity of False Positives: {avg_fp_sim:.4f}")
            if avg_fp_sim > 0.5:
                print("    -> HIGH. Model hallucinates semantically related pairs (Plausible/Missing Links).")
            else:
                print("    -> LOW. Model makes random errors (Structural Hallucinations).")
        else:
            print("  - No False Positives found.")

        # 4. QUALITATIVE CASE STUDIES
        print(f"\n### 4. CASE STUDIES (Top {k_examples} examples)")
        
        def print_cases(title, case_list, sort_key, reverse=True):
            print(f"\n\n[ {title} ]")
            sorted_cases = sorted(case_list, key=lambda x: x[sort_key], reverse=reverse)[:k_examples]
            if not sorted_cases:
                print("  (None found)")
                return
            print(f"\n  {'Rel':<25} | {'Score':<6} | {'Sim':<6}")
            for c in sorted_cases:
                print(f"\n  {c['rel'][:23]:<25} | {c['score']:.4f} | {c['sim']:.4f}")

        print_cases("Rare TPs (Bucket Success)", cases['rare_tp'], 'score', True)
        print_cases("Plausible FPs (Missing Links?)", cases['plausible_fp'], 'sim', True)
        print_cases("Hard FNs (Missed Easy)", cases['hard_fn'], 'sim', True)
        print_cases("Structural FNs (Cold Start)", cases['stupid_fn'], 'sim', False)
        
        print("="*60 + "\n")

    def run_advanced_case_studies(self, model, test_packs, top_k=10):
        """
        Advanced Diagnostic for Heterogeneous Graphs.
        Identifies 'Cold Start' vs 'Hub' nodes based on their TOTAL degree across all relations
        in the training set (visible context), then checks performance on hidden test edges.
        """
        device = next(model.parameters()).device
        model.eval()
        
        print("\n" + "█"*70)
        print("   ADVANCED DIAGNOSTIC: COLD START vs HUBS (Heterogeneous)")
        print("█"*70)
        
        # Prepare Data from the first test batch
        _, _, te_data, full_pos = test_packs[0]
        te_data = te_data.to(device)
        
        # Step 1: Compute node degrees across ALL edge types in the training graph (visible context) to identify cold start vs hub nodes.
        # This is crucial for understanding how the model performs on nodes with varying levels of connectivity,
        # which can reveal whether the model struggles with "cold start" nodes that have very few connections 
        # (potentially due to lack of training signal) compared to "hub" nodes that have many connections 
        # (which might be easier to learn due to more training signal).
        # The analysis will focus on the hidden test edges originating from these nodes to evaluate the model's ability to generalize to unseen connections based on the node's degree in the training graph.
        # We need the degree of each node summing up ALL edge types in the input graph
        node_degrees = {} # {node_type: Tensor(num_nodes)}
        
        # Initialize zeros
        for nt in te_data.node_types:
            node_degrees[nt] = torch.zeros(te_data[nt].num_nodes, device=device)
            
        # Sum degrees from all edge types present in te_data (the context)
        for et in te_data.edge_types:
            src_t, _, dst_t = et
            s, d = te_data[et].edge_index
            
            # Add Out-Degree to Source
            node_degrees[src_t].index_add_(0, s, torch.ones_like(s, dtype=torch.float))
            # Add In-Degree to Destination (undirected logic for connectivity strength)
            node_degrees[dst_t].index_add_(0, d, torch.ones_like(d, dtype=torch.float))

        # Now node_degrees contains the total degree for each node across all relations in the training graph, 
        # which we will use to categorize nodes as "cold start" (low degree) or "hubs" (high degree) 
        # for our analysis of hidden test edges.
        # We need to know which nodes have hidden test edges to predict
        test_targets_map = defaultdict(lambda: defaultdict(list)) # {src_type: {src_id: [(dst_type, dst_id, rel_name)]}}
        
        for et in te_data.edge_types:
            pos_test = GraphUtils.get_pos_edges_from_split(te_data, et)
            if pos_test is None: continue
            
            src_t, rel, dst_t = et
            s_list = pos_test[0].tolist()
            d_list = pos_test[1].tolist()
            
            for s, d in zip(s_list, d_list):
                test_targets_map[src_t][s].append( (dst_t, d, rel) )

        # We will analyze a few examples from the "cold start" category (nodes with degree <= 2) 
        # and the "hub" category (nodes with degree >= 20) to see how the model performs on 
        # hidden test edges originating from these nodes.
        candidates_cold = [] # (node_type, node_id) with degree <= 2
        candidates_hubs = [] # (node_type, node_id) with degree >= 20
        
        for nt, id_map in test_targets_map.items():
            for nid in id_map.keys():
                deg = node_degrees[nt][nid].item()
                if deg <= 2:
                    candidates_cold.append((nt, nid))
                elif deg >= 20:
                    candidates_hubs.append((nt, nid))
        
        # Sample 3 of each
        sel_cold = random.sample(candidates_cold, min(3, len(candidates_cold)))
        sel_hubs = random.sample(candidates_hubs, min(3, len(candidates_hubs)))
        
        # For each selected node, we will analyze:
        # 1. The number of hidden test edges (targets) it has (context stats)
        # 2. The specific target edges (relations and destination nodes) it has in the test set (targets)
        # 3. The model's predictions for all candidate destination nodes for those relations (prediction), 
        #   along with the cosine similarity of embeddings to understand if errors are due to semantic
        def analyze_node(nt, nid, tag):
            print(f"\n\n{tag} | Node: {nt} (ID: {nid})")
            
            # 1. Context Stats
            deg = node_degrees[nt][nid].item()
            print(f"\n   Context (Train Edges): {int(deg)} total neighbors (across all types)")
            
            # 2. Targets
            targets = test_targets_map[nt][nid] # List of (dst_type, dst_id, rel)
            print(f"\n   Targets (Hidden): {len(targets)} edges")
            for t in targets:
                print(f"\n    -> via '{t[2]}' to {t[0]} ID {t[1]}")
                
            # 3. Prediction (Complexity: We must check ALL relations originating from this node)
            # To keep it readable, we only check the relations relevant to the Targets
            # (Prediction across all types/nodes is too expensive computationally)
            
            with torch.no_grad():
                x_dict = model.encode(te_data)
                
                # Group targets by relation type to batch predictions
                target_rels = defaultdict(list)
                for t in targets:
                    target_rels[(nt, t[2], t[0])].append(t[1])
                    
                # For each relevant relation, predict top-k against ALL candidate destination nodes
                for et, target_dst_ids in target_rels.items():
                    src_t, rel, dst_t = et
                    print(f"\n   [Predicting for relation: {rel}]")
                    
                    num_dst = te_data[dst_t].num_nodes
                    
                    # Batch: (nid) -> All nodes of type dst_t
                    src_vec = torch.full((num_dst,), nid, device=device, dtype=torch.long)
                    dst_vec = torch.arange(num_dst, device=device, dtype=torch.long)
                    edge_index_all = torch.stack([src_vec, dst_vec], dim=0)
                    
                    logits = model.decode(x_dict, edge_index_all, et).view(-1)
                    probs = torch.sigmoid(logits)
                    
                    # Top-K
                    k_local = top_k
                    top_v, top_i = torch.topk(probs, k=k_local)
                    
                    # Check hits
                    target_set = set(target_dst_ids)
                    
                    # Embedding Sim
                    src_emb = x_dict[src_t][nid]
                    dst_embs = x_dict[dst_t]
                    
                    for p, idx in zip(top_v.tolist(), top_i.tolist()):
                        idx = int(idx)
                        
                        # Sim
                        sim = F.cosine_similarity(src_emb.unsqueeze(0), dst_embs[idx].unsqueeze(0)).item()
                        
                        marker = ""
                        if idx in target_set:
                            marker = "HIT (TEST)"
                        else:
                            marker = ".." # Just noise or training edge
                            
                        print(f"\n     DstID {idx:<5} | P: {p:.4f} | Sim: {sim:.4f} | {marker}")

        # --- EXECUTE ---
        if sel_cold:
            print("\n" + "="*60)
            print("   CATEGORY: COLD START (Total Degree <= 2)")
            print("   (Success here implies Semantic Generalization)")
            print("="*60)
            for nt, nid in sel_cold: analyze_node(nt, nid, "[COLD]")
        else:
            print("\n(No Cold Start candidates found)")
            
        if sel_hubs:
            print("\n" + "="*60)
            print("   CATEGORY: HUBS (Total Degree >= 20)")
            print("   (Success here implies Structural Learning)")
            print("="*60)
            for nt, nid in sel_hubs: analyze_node(nt, nid, "[HUB]")
            
        print("\n" + "█"*70 + "\n")

    def analyze_hard_set_granularity(self, model, packs, hard_max_degree=1, threshold=0.5, edge_types=None):
        device = next(model.parameters()).device
        model.eval()
        
        # Categories
        results = {
            'HEAD': {'tp':0, 'fp':0, 'fn':0},
            'BUCKET': {'tp':0, 'fp':0, 'fn':0}
        }

        print("Analyzing hard set granularity...")
        
        with torch.no_grad():
            for _, _, data, full_pos in packs:
                data = data.to(device)
                
                # Compute node degrees for all node types in the graph. 
                # This will be used to identify "hard" nodes based on their degree (e.g., nodes with degree <= hard_max_degree).
                node_degrees = {}
                # Initialize zeros for all node types present
                for nt in data.node_types:
                    node_degrees[nt] = torch.zeros(data[nt].num_nodes, device=device)

                # Sum degrees from all edge types
                for et in data.edge_types:
                    src_t, _, dst_t = et
                    s, d = data[et].edge_index
                    
                    # Add Out-Degree to Source
                    node_degrees[src_t].index_add_(0, s, torch.ones_like(s, dtype=torch.float))
                    # Add In-Degree to Destination (optional, but good for total connectivity)
                    if src_t != dst_t: # Avoid double counting self-loops if treated undirected
                        node_degrees[dst_t].index_add_(0, d, torch.ones_like(d, dtype=torch.float))

                x_dict = model.encode(data)
                
                for et in edge_types:
                    if et not in full_pos or et not in data.edge_types: 
                        continue
                    
                    # We focus on the positive edges for this analysis, 
                    # as we want to see how well the model predicts the "hard" positives (those involving low-degree nodes).
                    rel_cat = 'BUCKET' if 'BUCKET' in et[1] else 'HEAD'
                    
                    pos = GraphUtils.get_pos_edges_from_split(data, et)
                    if pos is None: 
                        continue
                    pos = pos.to(device)
                    
                    # Identify "hard" positives based on the degree of the source nodes.
                    src_type = et[0]
                    # Get degrees for the source nodes involved in these positive edges
                    src_ids = pos[0]
                    current_degrees = node_degrees[src_type][src_ids]
                    
                    hard_mask = (current_degrees <= hard_max_degree)
                    
                    if hard_mask.sum() == 0: 
                        continue
                    
                    pos_hard = pos[:, hard_mask]
                    
                    # Sample negatives (1:1 ratio for this specific analysis)
                    neg = GraphUtils.sample_negatives_excluding_positives(
                        full_pos[et].to(device), data[et[0]].num_nodes, data[et[2]].num_nodes, 
                        pos_hard.size(1), (et[0]==et[2]), device, "sparse"
                    )
                    
                    if neg.size(1) == 0: 
                        continue

                    # Compute scores for the hard positives and 
                    # the sampled negatives using the model's decode function.
                    pos_scores = torch.sigmoid(model.decode(x_dict, pos_hard, et).view(-1))
                    neg_scores = torch.sigmoid(model.decode(x_dict, neg, et).view(-1))
                    
                    # Metrics
                    # TP: Positives predicted as Pos
                    results[rel_cat]['tp'] += (pos_scores >= threshold).sum().item()
                    # FN: Positives predicted as Neg
                    results[rel_cat]['fn'] += (pos_scores < threshold).sum().item()
                    # FP: Negatives predicted as Pos
                    results[rel_cat]['fp'] += (neg_scores >= threshold).sum().item()

        # Print Report
        print(f"{'Category':<10} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Count'}")
        print("-" * 55)
        for cat in ['HEAD', 'BUCKET']:
            s = results[cat]
            count = s['tp'] + s['fn'] # Total positives
            if count == 0: 
                print(f"\n{cat:<10} | N/A        | N/A        | N/A        | 0")
                continue
                
            p = s['tp'] / (s['tp'] + s['fp'] + 1e-9)
            r = s['tp'] / (s['tp'] + s['fn'] + 1e-9)
            f1 = 2*p*r / (p+r+1e-9)
            print(f"\n{cat:<10} | {p:.4f}   | {r:.4f}   | {f1:.4f}   | {int(count)}")


    def evaluate(self, val_packs, test_packs, edge_types, ratio=[1, 9, 100], train_negative_ratio=19):
        device = next(self.model.parameters()).device
        bt = self.find_best_threshold(val_packs, edge_types, train_negative_ratio, self.scorer)
        self.run_comprehensive_error_analysis(self.model, test_packs, train_negative_ratio, threshold=bt, k_examples=5)
        self.run_advanced_case_studies(self.model, test_packs, top_k=20)
        hard_test_packs = self.create_hard_test_set(test_packs, max_degree=1)

        for r in ratio:
            print(f"\nEvaluating with Negative Ratio: {r}")
            self.evaluate_real_hard_set_entity_to_entity(self.model, test_packs, edge_types, neg_ratio=r, threshold=bt)
            
            bt_r = self.find_best_threshold(val_packs, edge_types, r, self.scorer)
            self.evaluate_metrics_master("GNN - Val Thr", test_packs, edge_types, r, self.scorer, threshold=bt_r, split="test")
            
            if hard_test_packs:
                print(f"\nEvaluating on Hard Test Set with Negative Ratio: {r}")
                bt_hard = self.find_best_threshold(val_packs, edge_types, r, self.scorer)
                self.evaluate_metrics_master("GNN Hard Set", hard_test_packs, edge_types, r, self.scorer, threshold=bt_hard, split="test")
