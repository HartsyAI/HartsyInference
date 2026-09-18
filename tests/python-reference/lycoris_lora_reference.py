"""Reference deltas for LoraDeltaMathTests (LoHa / LoKr / conv LoRA / DoRA).

Transcribes ComfyUI's comfy/weight_adapter/{loha,lokr,lora}.py and base.py::weight_decompose
on tiny fixed tensors and prints the expected values the C# test asserts against. Values are
small exact-in-binary32 integers so the only tolerance the test needs is for DoRA's sqrt.

Usage (any environment with torch):
    python3 tests/python-reference/lycoris_lora_reference.py
"""

import torch


def show(name, tensor):
    flat = tensor.reshape(-1).tolist()
    print(f"{name} shape={tuple(tensor.shape)}")
    print("  " + ", ".join(f"{v:.8g}f" for v in flat))


def seq(*shape, start=1.0, step=1.0):
    """Deterministic ramp; reproduced verbatim in the C# fixtures."""
    count = 1
    for dim in shape:
        count *= dim
    return torch.arange(count, dtype=torch.float32).mul_(step).add_(start).reshape(shape)


def loha_plain():
    # out=4, in=6, rank=2 — ComfyUI: m1 = mm(w1a, w1b), m2 = mm(w2a, w2b), diff = m1 * m2.
    w1a = seq(4, 2)
    w1b = seq(2, 6, start=-3.0)
    w2a = seq(4, 2, start=2.0)
    w2b = seq(2, 6, start=1.0, step=-1.0)
    m1 = torch.mm(w1a, w1b)
    m2 = torch.mm(w2a, w2b)
    show("loha_plain", m1 * m2)


def loha_tucker():
    # Conv LoHa: einsum("i j k l, j r, i p -> p r k l", t, wb, wa), flattened from dim 1.
    t1 = seq(2, 2, 2, 2)
    w1a = seq(2, 3)
    w1b = seq(2, 2, start=-1.0)
    t2 = seq(2, 2, 2, 2, start=2.0)
    w2a = seq(2, 3, start=-2.0)
    w2b = seq(2, 2, start=1.0, step=-1.0)
    m1 = torch.einsum("i j k l, j r, i p -> p r k l", t1, w1b, w1a)
    m2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)
    show("loha_tucker", (m1 * m2).flatten(start_dim=1))


def lokr_whole():
    # Neither factor low-rank: ComfyUI's dim stays None, so the file's alpha is ignored (scale 1.0).
    w1 = seq(2, 2)
    w2 = seq(2, 3, start=-2.0)
    show("lokr_whole", torch.kron(w1, w2))


def lokr_factored():
    # Both factors low-rank; alpha divides by w2_b.shape[0] (the LAST factored side).
    w1a = seq(2, 1)
    w1b = seq(1, 2, start=3.0)
    w2a = seq(2, 1, start=-1.0, step=3.0)
    w2b = seq(1, 3, start=2.0)
    show("lokr_factored", torch.kron(torch.mm(w1a, w1b), torch.mm(w2a, w2b)))


def lokr_conv():
    # Rank-4 right factor: ComfyUI unsqueezes w1 to [a,b,1,1] before kron, which equals the 2-D
    # kron of w1 with w2 flattened from dim 1.
    w1 = seq(2, 2)
    w2 = seq(2, 1, 2, 2, start=-1.0)
    delta = torch.kron(w1.unsqueeze(2).unsqueeze(2), w2)
    show("lokr_conv", delta.flatten(start_dim=1))


def conv_lora():
    # Standard conv adapter: mm(up.flatten(1), down.flatten(1)) -> [out, in*kh*kw].
    up = seq(4, 2, 1, 1)
    down = seq(2, 3, 3, 3, start=-5.0)
    show("conv_lora", torch.mm(up.flatten(start_dim=1), down.flatten(start_dim=1)))


def weight_decompose(dora_scale, weight, lora_diff, alpha, strength):
    """comfy/weight_adapter/base.py::weight_decompose, float32, no function hook."""
    weight = weight.clone()
    lora_diff = lora_diff * alpha
    weight_calc = weight + lora_diff
    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0]
    if wd_on_output_axis:
        weight_norm = weight.reshape(weight.shape[0], -1).norm(dim=1, keepdim=True)
    else:
        weight_norm = (
            weight_calc.transpose(0, 1)
            .reshape(weight_calc.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .transpose(0, 1)
        )
    weight_norm = weight_norm + torch.finfo(torch.float32).eps
    weight_calc = weight_calc * (dora_scale / weight_norm)
    if strength != 1.0:
        weight_calc = weight_calc - weight
        weight = weight + strength * weight_calc
    else:
        weight = weight_calc
    return weight


def dora():
    weight = seq(3, 4, start=-4.0, step=0.5)
    delta = seq(3, 4, start=1.0, step=-0.25)
    on_output = torch.tensor([[2.0], [3.0], [0.5]])
    on_input = torch.tensor([[1.5, 2.0, 0.25, 3.0]])
    show("dora_output_axis_strength1", weight_decompose(on_output, weight, delta, 0.5, 1.0))
    show("dora_output_axis_strength_half", weight_decompose(on_output, weight, delta, 0.5, 0.5))
    show("dora_input_axis_strength1", weight_decompose(on_input, weight, delta, 0.5, 1.0))


if __name__ == "__main__":
    loha_plain()
    loha_tucker()
    lokr_whole()
    lokr_factored()
    lokr_conv()
    conv_lora()
    dora()
