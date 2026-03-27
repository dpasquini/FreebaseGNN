import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import tqdm

class TextToEncode:
    def __init__(self, model_name="answerdotai/ModernBERT-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Step 1: Define the wrappers processing pooled vs cls outputs
        class TransformerWithPooling(nn.Module):
            """
            A wrapper around a transformer model that applies mean pooling to the token embeddings.
            This allows us to get a fixed-size vector representation for each input text, regardless of its length.
            """
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, input_ids, attention_mask):
                output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = output.last_hidden_state
                
                # Mean Pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
                sum_mask = input_mask_expanded.sum(dim=1)
                return sum_embeddings / sum_mask

        class TransformerWithCLS(nn.Module):
            """
            A wrapper around a transformer model that extracts the CLS token representation.
            Used typically for edge summarization/specialized classification token usages.
            """
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, input_ids, attention_mask):
                output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                # The CLS token is always at index 0 of the sequence length dimension
                cls_embedding = output.last_hidden_state[:, 0, :]
                return cls_embedding

        # We need a robust mock that handles @torch.compile(dynamic=True) AND torch.compile(model)
        def robust_compile_mock(*args, **kwargs):
            # Case 1: Called as a decorator factory with args (e.g. @torch.compile(dynamic=True))
            # We return a dummy decorator that just returns the function unchanged.
            if len(args) == 0 and len(kwargs) > 0:
                return lambda func: func
            
            # Case 2: Called directly on a function/model (e.g. torch.compile(model))
            # We just return the model unchanged.
            if len(args) > 0:
                return args[0]
            
            # Fallback
            return lambda func: func

        # Apply the mock
        original_compile = torch.compile
        try:
            torch.compile = robust_compile_mock
            
            # Load the models (now it ignores the @torch.compile(dynamic=True) inside ModernBERT)
            base_model_pooling = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            base_model_cls = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        finally:
            # Restore original just in case
            torch.compile = original_compile
        
        # Step 2: Wrap and Parallelize
        self.model_pooling = TransformerWithPooling(base_model_pooling)
        self.model_cls = TransformerWithCLS(base_model_cls)
        
        if torch.cuda.device_count() > 1:
            print(f"TextToEncode: Active on {torch.cuda.device_count()} GPUs (Eager Mode).")
            self.model_pooling = nn.DataParallel(self.model_pooling)
            self.model_cls = nn.DataParallel(self.model_cls)
        
        self.model_pooling.to(self.device)
        self.model_pooling.eval()

        self.model_cls.to(self.device)
        self.model_cls.eval()

    def encode_batch(self, texts: list, batch_size=None):
        """
        Encode a batch of texts using Mean Pooling.
        If batch_size is not provided, it defaults to 128 per GPU.
        """
        return self._run_inference_batch(texts, self.model_pooling, batch_size, desc="Encoding nodes (pooling)")

    def encode_batch_edges(self, texts: list, batch_size=None):
        """
        Encode a batch of texts using the CLS token representation, intended for edges.
        If batch_size is not provided, it defaults to 128 per GPU.
        """
        return self._run_inference_batch(texts, self.model_cls, batch_size, desc="Encoding edges (CLS)")

    def _run_inference_batch(self, texts: list, model: nn.Module, batch_size: int = None, desc: str = "Encoding text batches"):
        """
        Shared internal loop processing batches dynamically on GPUs
        """
        if batch_size is None:
            # Safe default: 128 per GPU
            batch_size = 128 * max(1, torch.cuda.device_count())

        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, len(texts), batch_size), desc=desc):
                batch_texts = texts[i : i + batch_size]
                
                encoded_input = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                # Forward pass
                batch_emb = model(
                    input_ids=encoded_input['input_ids'], 
                    attention_mask=encoded_input['attention_mask']
                )
                
                # Move to CPU immediately
                all_embeddings.append(batch_emb.cpu().float())
                
        if not all_embeddings:
            return torch.empty(0, 768)

        return torch.cat(all_embeddings, dim=0)

    def encode(self, text):
        if isinstance(text, str):
            text = [text]
        return self.encode_batch(text, batch_size=1)
        
    def encode_edge(self, text):
        if isinstance(text, str):
            text = [text]
        return self.encode_batch_edges(text, batch_size=1)