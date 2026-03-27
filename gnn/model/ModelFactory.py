from gnn.model.HeteroGAT import HeteroGAT
from gnn.model.LinkPredictor import LinkPredictor
from gnn.model.TypePairPredictor import TypePairPredictor

class ModelFactory:
    @staticmethod
    def create_gnn(model_name: str, gnn_metadata, hidden_dim, num_layers, n_heads, dropout_rate, input_dim, edge_feature_dim=None):
        """
        Factory method to create a GNN model based on the specified type.
        """
        if edge_feature_dim is None:
            edge_feature_dim = input_dim
            
        if model_name.lower() == "heterogat":
            return HeteroGAT(
                metadata=gnn_metadata, 
                hidden_dim=hidden_dim, 
                num_layers=num_layers, 
                n_heads=n_heads, 
                dropout_rate=dropout_rate, 
                input_dim=input_dim, 
                edge_feature_dim=edge_feature_dim
            )
        # other models with elif can be added here in the future
        else:
            raise ValueError(f"Unknown GNN model type: {model_name}")

    @staticmethod
    def create_predictor(predictor_name: str, gnn_model, hidden_dim, edge_types):
        if predictor_name.lower() == "linkpredictor":
            return LinkPredictor(
                gnn=gnn_model, 
                hidden_dim=hidden_dim, 
                edge_types=edge_types
            )
        elif predictor_name.lower() == "typepairpredictor":
            return TypePairPredictor(
                gnn=gnn_model,
                hidden_dim=hidden_dim,
                edge_types=edge_types
            )
        # future predictor models here (e.g., NodePredictor)
        else:
            raise ValueError(f"Unknown Predictor type: {predictor_name}")
