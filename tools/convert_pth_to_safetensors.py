#!/usr/bin/env python3
"""
Convert PyTorch .pt / .pth checkpoints to .safetensors for use with HartsyInference.

Why this exists
---------------
Several audio models in our research scope ship only .pth pickles:

  - Kokoro 82M  (kokoro-v1_0.pth — dict of state_dicts)
  - StyleTTS 2  (epochs_2nd_00020.pth — dict of state_dicts)
  - ChatTTS     (gpt.pt, dvae.pt, decoder.pt — flat state_dicts)
  - Moonshine   (also has safetensors on HF, but the original .pt was the canonical drop)
  - RNNoise     (rnnoise10Ga_*.pth — training checkpoint wrapping state_dict alongside
                 model_kwargs/loss/epoch. Prefer this over the shipped rnnoise_data.c /
                 weights_blob.bin, which carry int8-quantized copies of some layers;
                 the .pth is the unquantized float source.)

HartsyInference reads only safetensors at runtime (project rule: pure C#, no python
pickle parsing). This is the offline one-shot tool that produces the artifact our
ModelHandler.SafeTensors loader consumes.

Usage
-----
  python tools/convert_pth_to_safetensors.py kokoro-v1_0.pth   -o kokoro.safetensors
  python tools/convert_pth_to_safetensors.py styletts2.pth     -o styletts2.safetensors
  python tools/convert_pth_to_safetensors.py chat_gpt.pt       -o chat_gpt.safetensors
  python tools/convert_pth_to_safetensors.py rnnoise10Ga_12.pth --list-only

The script:

  1. Loads the .pth via torch.load(map_location='cpu', weights_only=False).
     We pass weights_only=False because some checkpoints contain training state
     (config, optimizer, dvae)  in addition to weights. The script extracts only
     the tensor leaves.
  2. Recursively walks the loaded object — handles plain Tensor, dict of Tensor,
     and dict of {section_name: state_dict} (the StyleTTS2 / Kokoro pattern).
  3. Unwraps a training-checkpoint envelope (a 'state_dict' / 'model' / 'net' /
     'module' key) so keys come out as "conv1.weight", not "state_dict.conv1.weight".
  4. Nested keys are always flattened to a dotted path, so the final safetensors
     file has flat keys like "bert.embeddings.word_embeddings.weight" rather than
     nested groupings. (There is no --flatten switch; earlier revisions of this
     docstring advertised one that was never implemented.)
  5. Writes to safetensors using safetensors.torch.save_file().

Requires
--------
  pip install torch safetensors

The script is intentionally NOT invoked at HartsyInference build/runtime. It runs
once per checkpoint, and the resulting .safetensors gets cached under
~/.cache/hartsyinference/models/ — or uploaded to a HuggingFace mirror we control.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

try:
    import torch
    from safetensors.torch import save_file
except ImportError as e:
    print(f"missing dependency: {e}. Install with: pip install torch safetensors", file=sys.stderr)
    sys.exit(1)


def collect_tensors(obj: Any, prefix: str = "", out: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """Walk `obj` and collect every tensor leaf into `out`, keyed by dotted path.

    Handles three common .pth shapes:
      * a raw state_dict  ({name -> Tensor})
      * a dict of state_dicts  ({section -> {name -> Tensor}})  — StyleTTS2 / Kokoro
      * a dict containing 'state_dict' / 'model' / 'net' subkey wrapping (1) or (2).
    """
    if out is None:
        out = {}

    if isinstance(obj, torch.Tensor):
        out[prefix.rstrip(".")] = obj.contiguous().detach().cpu()
        return out

    if isinstance(obj, dict):
        # Unwrap a training-checkpoint envelope before recursing. The test is "no sibling
        # holds a tensor", not a key-count limit: a real state_dict has tensors at the top
        # level, whereas an envelope's siblings are scalars/config (RNNoise ships
        # model_args/model_kwargs/state_dict/loss/epoch — five keys, so the old `len <= 3`
        # guard silently left a "state_dict." prefix on every tensor name).
        for wrapper in ("state_dict", "model", "net", "module"):
            if wrapper in obj and isinstance(obj[wrapper], (dict, torch.Tensor)):
                siblings_are_metadata = not any(
                    isinstance(v, torch.Tensor) for k, v in obj.items() if k != wrapper
                )
                if siblings_are_metadata:
                    return collect_tensors(obj[wrapper], prefix, out)

        for k, v in obj.items():
            # Strip DataParallel "module." prefixes that some training pipelines bake in.
            key = k[len("module."):] if k.startswith("module.") else k
            new_prefix = f"{prefix}{key}." if isinstance(v, dict) else f"{prefix}{key}"
            if isinstance(v, dict):
                collect_tensors(v, new_prefix, out)
            elif isinstance(v, torch.Tensor):
                out[new_prefix] = v.contiguous().detach().cpu()
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    collect_tensors(item, f"{prefix}{key}.{i}.", out)
            # silently ignore non-tensor leaves (config dicts, ints, strings, etc.)
        return out

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            collect_tensors(item, f"{prefix}{i}.", out)
        return out

    # Anything else — config object, optimizer state — is skipped.
    return out


def cast_dtype(tensors: dict[str, torch.Tensor], target: str | None) -> dict[str, torch.Tensor]:
    if target is None:
        return tensors
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if target not in dtype_map:
        raise ValueError(f"--dtype must be one of fp32/fp16/bf16, got '{target}'")
    target_dtype = dtype_map[target]
    out = {}
    for k, v in tensors.items():
        # Don't downcast integer tensors (token-id buffers, positional tables) —
        # only weight matrices in floating-point dtypes.
        if v.dtype.is_floating_point:
            out[k] = v.to(target_dtype)
        else:
            out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .pth / .pt to .safetensors for HartsyInference.")
    parser.add_argument("input", help="Path to the .pth / .pt file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .safetensors path. Required unless --list-only.")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None,
                        help="Optional dtype cast for all floating-point tensors.")
    parser.add_argument("--list-only", action="store_true",
                        help="Print the discovered tensor names + shapes, don't write output.")
    parser.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE",
                        help="Add a metadata entry to the safetensors header. Repeatable.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1

    if args.output is None and not args.list_only:
        print("-o/--output is required unless --list-only is given.", file=sys.stderr)
        return 1

    print(f"loading {args.input} ...")
    obj = torch.load(args.input, map_location="cpu", weights_only=False)

    tensors = collect_tensors(obj)
    print(f"found {len(tensors)} tensors")

    if args.list_only:
        for k, v in sorted(tensors.items()):
            print(f"  {k:60s}  {tuple(v.shape)}  {v.dtype}")
        return 0

    tensors = cast_dtype(tensors, args.dtype)

    metadata = {}
    for entry in args.metadata:
        if "=" not in entry:
            print(f"invalid --metadata entry (need KEY=VALUE): {entry}", file=sys.stderr)
            return 1
        k, v = entry.split("=", 1)
        metadata[k] = v

    metadata.setdefault("converted_by", "hartsyinference convert_pth_to_safetensors.py")
    metadata.setdefault("source", os.path.basename(args.input))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    save_file(tensors, args.output, metadata=metadata)
    print(f"wrote {args.output} ({len(tensors)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
