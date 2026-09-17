"""Dumps the released MERT2 audio encoder's activations, for the C# port to be checked against (gates S2/S3).

Unlike dump_sheetsage2_reference.py — which only needs the symbolic layer and stubs torch — this runs the real
network, so it needs a torch that can load the checkpoint. Use the ComfyUI venv:

    "/path/to/ComfyUI/venv/bin/python3" dump_mert2_reference.py \
        --comfyui "/path/to/ComfyUI" \
        --checkpoint /path/to/sheetsage2_bf16.safetensors

The forward runs on the CPU in float32 (CUDA is hidden from torch before it is imported) so a dump can never
collide with a GPU test or with the live ComfyUI service. The checkout is only read, never started.

What is dumped, and why each stage earns a gate:
  * mel         — the frontend is the only stage with no learned weights, so a wrong window/filterbank/log law
                  shows up here and nowhere else.
  * subsampled  — the ConvNeXt stack, whose GlobalResponseNorm reduces over the WHOLE time axis; an implementation
                  that normalizes per-frame or per-chunk still produces plausible numbers, just not these.
  * mixed       — 24 Conformer layers plus the learned 25-way layer mix (RoPE, the depthwise conv module).
  * projected   — encoder_projection, 1024 -> 512: exactly the tensor the decoder cross-attends over.

The waveform is generated here rather than committed: a fixed sum of sines plus an LCG noise floor, so the dump
is reproducible from this file alone. It is written into the output directory so the C# side reads bytes instead
of re-deriving the generator.

Writes mert2_reference/ (gitignored): tensors.safetensors + meta.json.
"""
import os

# Before torch: a hidden GPU is the only way to guarantee this never competes with a GPU test suite.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 24_000
WINDOW_SECONDS = 300.0

# The released encoder's own constants, restated so a silent upstream change fails the dump instead of the gate.
EXPECTED_MEL_BINS = 128
EXPECTED_DIM = 1024
EXPECTED_LAYERS = 24


def synthetic_waveform(samples: int, seed: int = 1234) -> np.ndarray:
    """Deterministic test signal: three inharmonic partials over an LCG noise floor.

    Sines alone leave most mel bins at the log floor, where any implementation agrees; the noise floor is what
    makes every bin carry a real number. The LCG is the standard PCG multiplier/increment pair, reproduced
    exactly so this file is the whole specification of the input.
    """
    index = np.arange(samples, dtype=np.float64)
    tone = (0.50 * np.sin(2.0 * np.pi * 220.0 * index / SAMPLE_RATE)
            + 0.30 * np.sin(2.0 * np.pi * 439.7 * index / SAMPLE_RATE + 0.4)
            + 0.18 * np.sin(2.0 * np.pi * 1970.3 * index / SAMPLE_RATE + 1.1))
    state = np.uint64(seed)
    multiplier = np.uint64(6364136223846793005)
    increment = np.uint64(1442695040888963407)
    noise = np.empty(samples, dtype=np.float64)
    with np.errstate(over="ignore"):
        for i in range(samples):
            state = multiplier * state + increment
            noise[i] = float(state >> np.uint64(40)) / float(1 << 24) * 2.0 - 1.0
    return (tone + 0.05 * noise).astype(np.float32)


def load_released_model(comfyui: Path, checkpoint: Path):
    """Imports the released MERT2 out of a ComfyUI checkout and fills it from the checkpoint, in float32."""
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(comfyui))
    from comfy.cli_args import args as comfy_args

    # comfy.model_management probes the torch device at import time; --cpu is what keeps that probe off CUDA.
    comfy_args.cpu = True
    import comfy.ops
    from comfy.audio_encoders.mert2 import MERT2
    from safetensors.torch import load_file

    state = load_file(str(checkpoint))
    model = MERT2(device="cpu", dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    encoder_state = {key[len("encoder."):]: value.float() for key, value in state.items() if key.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"MERT2 state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval()

    layer_weight = state["layer_weight"].float()
    projection_weight = state["encoder_projection.weight"].float()
    projection_bias = state["encoder_projection.bias"].float()
    if layer_weight.shape[0] != len(model.layers) + 1:
        raise SystemExit(f"layer_weight is {tuple(layer_weight.shape)}, expected {len(model.layers) + 1}")
    return model, layer_weight, projection_weight, projection_bias


class AttentionProbe:
    """Wraps the attention callable to record what an F16 attention path would have to survive.

    The engine runs this encoder's attention through cuDNN with F16 I/O, which is only safe while the scores and
    the values stay inside F16's range. The score bound is Cauchy-Schwarz on the per-row norms (cheap and
    rigorous) rather than the materialized 7500x7500 product.
    """

    def __init__(self, inner):
        self.inner = inner
        self.max_score_bound = 0.0
        self.max_value = 0.0

    def __call__(self, q, k, v, heads, **kwargs):
        scale = 1.0 / (q.shape[-1] ** 0.5)
        bound = float(q.norm(dim=-1).max()) * float(k.norm(dim=-1).max()) * scale
        self.max_score_bound = max(self.max_score_bound, bound)
        self.max_value = max(self.max_value, float(v.abs().max()))
        return self.inner(q, k, v, heads, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", required=True, type=Path, help="ComfyUI checkout (read only; never start it)")
    parser.add_argument("--checkpoint", required=True, type=Path, help="sheetsage2_bf16.safetensors")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "mert2_reference")
    parser.add_argument("--seconds", type=float, default=12.0, help="clip length before the 300 s window pad")
    parser.add_argument("--seed", type=int, default=1234)
    options = parser.parse_args()

    torch.manual_seed(options.seed)
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count() or 8)

    model, layer_weight, projection_weight, projection_bias = load_released_model(options.comfyui, options.checkpoint)
    import comfy.audio_encoders.mert2 as mert2_module

    clip = synthetic_waveform(int(round(options.seconds * SAMPLE_RATE)), options.seed)
    waveform = torch.from_numpy(clip)[None]
    padded = torch.nn.functional.pad(waveform, (0, max(0, int(WINDOW_SECONDS * SAMPLE_RATE) - waveform.shape[-1])))

    started = time.time()
    mel = model.feature_extractor(padded.float())
    print(f"mel {tuple(mel.shape)} in {time.time() - started:.1f}s")
    if mel.shape[-1] != EXPECTED_MEL_BINS:
        raise SystemExit(f"mel has {mel.shape[-1]} bins, expected {EXPECTED_MEL_BINS}")

    subsampled = {}
    handle = model.subsampling_module.register_forward_hook(
        lambda module, inputs, output: subsampled.__setitem__("value", output.clone()))
    probe = AttentionProbe(mert2_module.optimized_attention_for_device(torch.device("cpu")))
    original = mert2_module.optimized_attention_for_device
    mert2_module.optimized_attention_for_device = lambda *a, **k: probe
    try:
        started = time.time()
        mixed, _ = model(mel, layer_weight)
        print(f"encoder {tuple(mixed.shape)} in {time.time() - started:.1f}s")
    finally:
        mert2_module.optimized_attention_for_device = original
        handle.remove()

    # RoPE tables get their own gate: the released angles are a FLOAT32 product of the position and the inverse
    # frequency, and at token 7499 a double-precision angle drifts from that by ~5e-4 rad. Dumping cos/sin turns
    # "did the port reproduce the rounding" into a measurement instead of an argument.
    embeddings = model.position_embeddings(subsampled["value"])
    rope_cos = embeddings[0, 0, :, :, 0, 0].contiguous()
    rope_sin = embeddings[0, 0, :, :, 1, 0].contiguous()

    projected = torch.nn.functional.linear(mixed, projection_weight, projection_bias)
    if mixed.shape[-1] != EXPECTED_DIM or len(model.layers) != EXPECTED_LAYERS:
        raise SystemExit("MERT2 geometry changed; re-read the reference before trusting this dump")

    options.out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    tensors = {
        "waveform": waveform.contiguous(),
        "mel": mel.contiguous(),
        "subsampled": subsampled["value"].contiguous(),
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "mixed": mixed.contiguous(),
        "projected": projected.contiguous(),
        "layer_weights": layer_weight.softmax(dim=0).contiguous(),
    }
    save_file(tensors, str(options.out / "tensors.safetensors"))

    meta = {
        "sampleRate": SAMPLE_RATE,
        "windowSeconds": WINDOW_SECONDS,
        "clipSeconds": options.seconds,
        "seed": options.seed,
        "checkpoint": options.checkpoint.name,
        "shapes": {name: list(value.shape) for name, value in tensors.items()},
        "peaks": {name: float(value.abs().max()) for name, value in tensors.items()},
        "attentionScoreBound": probe.max_score_bound,
        "attentionMaxValue": probe.max_value,
    }
    (options.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
