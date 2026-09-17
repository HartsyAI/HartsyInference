"""Dumps SheetSage2's decode side from the released implementation, for the C# port to be checked against.

Covers gates S4 (the grammar mask applied inside the decode loop) and S5 (greedy argmax decoding). Everything
here needs the real checkpoint, so unlike `dump_sheetsage2_reference.py` it imports the released module out of a
ComfyUI checkout instead of exec-ing source blocks out of it: the decoder's own classes have to run, not a
re-reading of them. `comfy.model_prefetch` is stubbed because its accelerator half is not importable outside a
running ComfyUI, and every entry point it offers is a no-op on the eager CPU path this dump runs.

    python3 dump_sheetsage2_decoder_reference.py \
        --comfy /path/to/ComfyUI \
        --checkpoint /path/to/sheetsage2_bf16.safetensors \
        --audio /path/to/clip.wav --steps 256

Writes sheetsage2_decoder_reference_tensors/: protocol.json, memory.bin and logits.bin. The reference
runs on CPU in float32 over float32-upcast BF16 weights, which is the dtype the C# decoder runs in — a BF16
reference would put a dtype gap inside a token-exact gate and hide what the gate is supposed to measure.

Never point --comfy at a running ComfyUI install's live service; a checkout is fine, it is only read.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np


def install_prefetch_stub():
    """Stubs comfy.model_prefetch, whose accelerator module is absent outside a running ComfyUI."""
    stub = types.ModuleType("comfy.model_prefetch")
    stub.malloc_graph_enabled = lambda device: False
    stub.malloc_graph_begin = lambda device: None
    stub.malloc_graph_end = lambda: None
    stub.make_prefetch_queue = lambda layers, device, options=None: None
    stub.cleanup_prefetch_queues = lambda: None

    def prefetch_queue_pop(queue, device, layer, dtype=None, core=None, enable_graph=False, malloc_scope=None):
        if core is not None:
            core()

    stub.prefetch_queue_pop = prefetch_queue_pop
    sys.modules["comfy.model_prefetch"] = stub
    return stub


def load_module(comfy_root: Path):
    sys.path.insert(0, str(comfy_root))
    stub = install_prefetch_stub()
    import comfy  # noqa: E402

    comfy.model_prefetch = stub
    from comfy.audio_encoders import sheetsage2  # noqa: E402

    return sheetsage2


def load_waveform(path: Path, sample_rate: int):
    """Reads a WAV clip to mono float32 at the model's rate, mixing channels as the released forward() does."""
    import torch
    import torchaudio
    from scipy.io import wavfile

    rate, samples = wavfile.read(str(path))
    full_scale = {np.dtype("int16"): 32768.0, np.dtype("int32"): 2147483648.0, np.dtype("uint8"): 128.0}
    samples = samples.astype(np.float32) / full_scale.get(samples.dtype, 1.0)
    waveform = torch.from_numpy(np.ascontiguousarray(samples))
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    waveform = waveform.mean(dim=1)[None]
    if rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, rate, sample_rate)
    return torch.clamp(waveform, -1.0, 1.0)


def build_model(sheetsage2, checkpoint: Path, with_encoder: bool):
    """Builds the released model in float32 on CPU and loads the checkpoint into it.

    Without --audio only the decode side is built, so the dump does not pay for MERT2's 24 layers to produce a
    memory it is about to overwrite with a synthetic one.
    """
    import comfy.ops
    import torch
    from torch import nn
    from safetensors.torch import load_file

    state = load_file(str(checkpoint))
    state = {key: value.float() for key, value in state.items()}
    operations = comfy.ops.disable_weight_init
    device, dtype = torch.device("cpu"), torch.float32

    if with_encoder:
        model = sheetsage2.SheetSage2(device=device, dtype=dtype, operations=operations)
        model.load_state_dict(state, strict=True)
    else:
        model = sheetsage2.SheetSage2.__new__(sheetsage2.SheetSage2)
        nn.Module.__init__(model)
        model.dtype = dtype
        model.max_tokens = 5120
        model.tokenizer = sheetsage2.ScoreTokenizer()
        model.token_embedding = operations.Embedding(model.tokenizer.n_tokens, 512, device=device, dtype=dtype)
        model.decoder = sheetsage2.Decoder(512, 2048, 8, 6, 5120, device=device, dtype=dtype, operations=operations)
        model.output_projection = operations.Linear(512, model.tokenizer.n_tokens, bias=False, device=device, dtype=dtype)
        decode_side = {key: value for key, value in state.items()
                       if key.startswith(("decoder.", "token_embedding.", "output_projection."))}
        model.load_state_dict(decode_side, strict=True)

    model.eval()
    return model


def synthetic_memory(tokens: int, dim: int, seed: int):
    """A deterministic stand-in for the encoder's output, for a dump run without a clip."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, tokens, dim, generator=generator, dtype=torch.float32)


def record_decode(sheetsage2, model, memory, steps: int, stop_seconds: float):
    """Runs the released generate_tokens, capturing the logits every step produced.

    decode() is shadowed on the instance rather than reimplemented, so the loop, its grammar mask and its stop
    rules are the released ones; only the logits leaving decode() are observed.
    """
    import torch

    prefix = model.tokenizer.prompt_prefix()
    model.max_tokens = len(prefix) + steps
    captured = []
    released_decode = type(model).decode

    def capture(ids, positions, cache, decode_buffer=None):
        logits = released_decode(model, ids, positions, cache, decode_buffer=decode_buffer)
        captured.append(logits[0, -1].float().clone())
        return logits

    model.decode = capture
    with torch.no_grad():
        tokens = model.generate_tokens(memory, stop_seconds)
    del model.decode
    return prefix, tokens, torch.stack(captured)


def grammar_trace(sheetsage2, model, prefix, tokens, logits):
    """Replays the grammar over the produced sequence to recover each step's mask and its top-2 margin.

    The replay is exact — generate_tokens is greedy, so the mask at step i is a function of the tokens before it.
    """
    import torch

    tokenizer = model.tokenizer
    state = sheetsage2.PromptGrammarState(tokenizer)
    for token in prefix[prefix.index(tokenizer.out_token) + 1:]:
        state.update(token)

    trace = []
    device = torch.device("cpu")
    for step in range(logits.shape[0]):
        allowed = state.allowed(device)
        scores = logits[step].masked_fill(~allowed, -torch.inf)
        top = torch.topk(scores, 2)
        chosen = int(top.indices[0])
        expected = tokens[len(prefix) + step]
        trace.append({
            "step": step,
            "token": chosen,
            "allowedCount": int(allowed.sum()),
            "top1": float(top.values[0]),
            "top2": float(top.values[1]),
            "margin": float(top.values[0] - top.values[1]),
            "unmaskedArgmax": int(logits[step].argmax()),
        })
        if chosen != expected:
            raise RuntimeError(f"grammar replay diverged from the recorded decode at step {step}")
        if state.update(chosen):
            break
    return trace


def dump_case(sheetsage2, model, out: Path, name: str, memory, memory_source: str, steps: int, logit_steps: int):
    """Decodes one memory and writes its case directory: protocol.json, memory.bin, logits.bin."""
    memory = memory.contiguous()
    prefix, tokens, logits = record_decode(sheetsage2, model, memory, steps, stop_seconds=300.0)
    trace = grammar_trace(sheetsage2, model, prefix, tokens, logits)

    case = out / name
    case.mkdir(parents=True, exist_ok=True)
    memory.numpy().astype(np.float32).tofile(case / "memory.bin")
    rows = min(logit_steps, logits.shape[0])
    logits[:rows].numpy().astype(np.float32).tofile(case / "logits.bin")

    margins = [entry["margin"] for entry in trace]
    payload = {
        "dim": 512,
        "intermediate": 2048,
        "heads": 8,
        "layers": 6,
        "maxTokens": model.max_tokens,
        "positionOffset": 2,
        "embedPositionRows": model.decoder.embed_positions.weight.shape[0],
        "nTokens": model.tokenizer.n_tokens,
        "memoryTokens": int(memory.shape[1]),
        "memorySource": memory_source,
        "dtype": "float32",
        "promptPrefix": list(prefix),
        "tokens": [int(t) for t in tokens],
        "steps": int(logits.shape[0]),
        "logitRows": rows,
        "trace": trace,
        "marginMin": min(margins),
        "marginMedian": float(np.median(margins)),
        "maskChangedTheArgmax": sum(1 for e in trace if e["token"] != e["unmaskedArgmax"]),
    }
    (case / "protocol.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {case} ({payload['steps']} steps, {len(payload['tokens'])} tokens, "
          f"memory={memory_source}, min margin {payload['marginMin']:.4g}, "
          f"mask changed the argmax at {payload['maskChangedTheArgmax']} steps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfy", type=Path, required=True, help="ComfyUI checkout root (read only)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="sheetsage2_bf16.safetensors")
    parser.add_argument("--audio", type=Path, default=None, help="clip to encode into the 'audio' case")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "sheetsage2_decoder_reference_tensors")
    parser.add_argument("--steps", type=int, default=256, help="decode steps to run and pin")
    parser.add_argument("--logit-steps", type=int, default=16, help="steps whose full logit row is dumped")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    sheetsage2 = load_module(args.comfy)
    import torch

    torch.manual_seed(args.seed)
    model = build_model(sheetsage2, args.checkpoint, with_encoder=args.audio is not None)

    # Both cases earn their place. A real clip is what the decoder meets in production, but the checkpoint is
    # good enough there that its unmasked argmax already obeys the grammar, so that case cannot tell a masked
    # loop from an unmasked one. A synthetic memory puts the model off its manifold, where the mask decides
    # several tokens — which is the only way gate S4 can fail for its own reason.
    if args.audio is not None:
        waveform = load_waveform(args.audio, 24000)
        with torch.no_grad():
            memory, _ = model.encode(waveform)
        dump_case(sheetsage2, model, args.out, "audio", memory, args.audio.name, args.steps, args.logit_steps)

    dump_case(sheetsage2, model, args.out, "synthetic", synthetic_memory(7500, 512, args.seed),
              f"synthetic(seed={args.seed})", args.steps, args.logit_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
