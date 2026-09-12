"""Generate with a steering vector added to the residual stream."""

import contextlib

import torch

from .hooks import layers_of, steer


def resolve_layers(spec, n_layers):
    if spec == "all":
        return list(range(n_layers))
    return list(spec)


def layer_vectors(vector, layers, device):
    """vector: tensor[n_layers, hidden] -> {layer_idx: tensor[hidden]} on device."""
    return {i: vector[i].to(device) for i in layers}


def generate_steered(model, tokenizer, prompt, gen_kwargs, vec_by_layer=None, coeff=0.0):
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    ctx = (
        steer(model, vec_by_layer, coeff)
        if vec_by_layer and coeff != 0.0
        else contextlib.nullcontext()
    )
    with torch.no_grad(), ctx:
        out = model.generate(ids, **gen_kwargs)
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=False)
