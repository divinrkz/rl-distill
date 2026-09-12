"""Forward hooks on decoder layers for activation capture and steering.

Both models are Qwen2.5-based: `model.model.layers[i]` is a decoder layer whose
forward returns a tuple with hidden states (the residual stream) at index 0.
Capture reads output[0]; steering adds coeff*vector to output[0].
"""

import contextlib


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_hidden(out, hs):
    return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs


def layers_of(model):
    return model.model.layers


@contextlib.contextmanager
def capture(model):
    """Capture per-layer residual output for one forward pass.

    Yields a dict {layer_idx: tensor[batch, seq, hidden]} filled after the pass.
    """
    store = {}
    handles = []

    def make_hook(i):
        def hook(module, inp, out):
            store[i] = _hidden(out).detach()

        return hook

    for i, layer in enumerate(layers_of(model)):
        handles.append(layer.register_forward_hook(make_hook(i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def steer(model, vector_by_layer, coeff):
    """Add coeff*vector to the residual at the given layers during generation.

    vector_by_layer: {layer_idx: tensor[hidden]} on the model's device/dtype.
    """
    handles = []

    def make_hook(vec):
        def hook(module, inp, out):
            hs = _hidden(out)
            hs = hs + coeff * vec.to(device=hs.device, dtype=hs.dtype)
            return _replace_hidden(out, hs)

        return hook

    layers = layers_of(model)
    for i, vec in vector_by_layer.items():
        handles.append(layers[i].register_forward_hook(make_hook(vec)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
