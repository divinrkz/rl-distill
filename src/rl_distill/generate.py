"""vLLM generation. One model resident at a time; free_llm releases the GPU."""

import gc


def load_llm(model_name, vllm_config):
    from vllm import LLM

    return LLM(model=model_name, **vllm_config)


def free_llm(llm):
    import torch

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sampling_params(gen_config):
    from vllm import SamplingParams

    return SamplingParams(
        n=gen_config.get("n", 1),
        temperature=gen_config["temperature"],
        top_p=gen_config["top_p"],
        max_tokens=gen_config["max_tokens"],
        seed=gen_config.get("seed"),
    )


def generate(llm, prompts, gen_config):
    """Return list (per prompt) of lists of completion strings."""
    outputs = llm.generate(prompts, sampling_params(gen_config))
    return [[c.text for c in out.outputs] for out in outputs]
