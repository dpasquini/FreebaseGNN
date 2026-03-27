import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams
from typing import List, Dict, Tuple, Optional
from huggingface_hub import login

class LLMService:
    """
    Service class encapsulating the vLLM setup and inference generation.
    This class initializes the vLLM model with specified parameters and provides a method to generate responses for a batch of conversations. 
    It abstracts away the details of interacting with the vLLM library, 
    allowing other parts of the code to simply call the generate_batch method with the appropriate input format.
    """
    
    def __init__(self, model_name: str, max_tokens: int = 128, temperature: float = 0.0, huggingface_token: Optional[str] = None):
        self.model_name = model_name
        self.huggingface_token = huggingface_token
        self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        if self.huggingface_token:
            login(token=self.huggingface_token)
        self.llm = self._setup_llm()


    def _setup_llm(self) -> LLM:
        """
        Initializes the vLLM instance with specified model and quantization settings.
        """
        return LLM(
            model=self.model_name, 
            quantization="bitsandbytes",
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.9
        )

    def generate_batch(self, conversations: List[List[Dict[str, str]]]) -> List[str]:
        """
        Generates responses for a batch of conversations using vLLM.
        """
        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=self.sampling_params
        )
        return [output.outputs[0].text for output in outputs]
