"""HuggingFace model loading for activation access (steering).

vLLM handles bulk generation, but the residual stream is only reachable through
transformers + forward hooks, so steering loads the model this way.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_hf(model_name, dtype="bfloat16"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
    model.to("cuda")
    model.eval()
    return model, tokenizer
