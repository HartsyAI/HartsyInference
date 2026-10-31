"""Dumps ComfyUI's Anima LLM adapter (the text conditioning path) on fixed inputs, F32 on CPU, for the engine's
text-adapter parity test. The adapter weights are the checkpoint's net.llm_adapter.* tensors. Every key must map
(strict load), so a mismatch fails here instead of comparing the wrong weights.

Outputs (safetensors): in.qwen [1, S, 1024] source hidden states, in.t5ids [1, T] int64 target ids,
out.refined [1, T, 1024].
Usage:
  tests/python-reference/.venv/bin/python tests/python-reference/dump_anima_text_adapter.py \\
    --comfy <ComfyUI checkout> --checkpoint <anima-base-v1.0.safetensors> \\
    --out tests/python-reference/anima_text_adapter.safetensors
"""
import argparse
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-len", type=int, default=20)
    parser.add_argument("--target-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    sys.path.insert(0, args.comfy)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.model_management  # noqa: F401  (device bookkeeping must be initialized first)
    import comfy.ops
    from comfy.ldm.anima.model import LLMAdapter

    torch.set_grad_enabled(False)
    prefix = "net.llm_adapter."
    state = {}
    with safe_open(args.checkpoint, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = f.get_tensor(key).to(torch.float32)
    print("adapter tensors:", len(state))

    adapter = LLMAdapter(device="cpu", dtype=torch.float32, operations=comfy.ops.disable_weight_init).eval()
    result = adapter.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(f"key mismatch: missing {result.missing_keys[:5]} unexpected {result.unexpected_keys[:5]}")

    generator = torch.Generator().manual_seed(args.seed)
    qwen = torch.randn(1, args.source_len, 1024, generator=generator)
    vocab = adapter.embed.weight.shape[0]
    ids = torch.randint(0, vocab, (1, args.target_len), generator=generator)
    out = adapter(qwen, ids)
    tensors = {
        "in.qwen": qwen.contiguous(),
        "in.t5ids": ids.to(torch.float32).contiguous(),
        "out.refined": out.contiguous(),
    }
    metadata = {"source": "ComfyUI LLMAdapter (net.llm_adapter.*), F32 CPU", "seed": str(args.seed)}
    save_file(tensors, args.out, metadata=metadata)
    print("wrote", args.out, "out", tuple(out.shape), "vocab", vocab)


if __name__ == "__main__":
    main()
