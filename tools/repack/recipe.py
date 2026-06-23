#!/usr/bin/env python3
"""Core repack framework: Recipe / Component model + merge / sha256 / manifest helpers.

Why this exists
---------------
Audio models in HartsyInference arrive in wildly different shapes: single .pth pickles
(Kokoro, StyleTTS2), 4-6 separate component safetensors (CosyVoice, FishSpeech, Zonos),
multi-shard safetensors (VibeVoice). Every pipeline's LoadAsync hardcodes its own filenames
and does ad-hoc key surgery at runtime (Kokoro's `module.` strip, Chatterbox's SubDictionary).

A *Recipe* declares, once and offline, how to turn a model's official checkpoint(s) into ONE
`.safetensors` with a flat, documented key layout. Multi-component models merge their
components under per-component prefixes (`llm.* / flow.* / vocoder.*`), mirroring Chatterbox's
existing prefix-routed layout, so the engine can split a merged file with the same
`SubDictionary` pattern it already uses.

The engine stays pure C#; this is an offline dev tool, same as convert_pth_to_safetensors.py.

Requires: pip install torch safetensors huggingface_hub
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

try:
    import torch
    from safetensors.torch import load_file as st_load_file
    from safetensors.torch import save_file as st_save_file
except ImportError as e:  # pragma: no cover - dependency guard
    print(f"missing dependency: {e}. Install with: pip install torch safetensors huggingface_hub",
          file=sys.stderr)
    raise

# Reuse the tensor-walking logic from the existing single-file converter rather than
# duplicating it. convert_pth_to_safetensors.py lives one directory up (tools/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convert_pth_to_safetensors import collect_tensors  # noqa: E402

TOOL_VERSION = "1"

# ---------------------------------------------------------------------------
# Key transforms. A transform maps an original key -> new key, or returns None
# to drop the key. They are named so the manifest can record exactly what ran.
# ---------------------------------------------------------------------------

def _strip_inner_module(key: str) -> str:
    """Drop a `module.` segment that sits immediately after the first dotted section.

    The hexgrad Kokoro checkpoint was saved under nn.DataParallel, so each top-level
    module's inner state-dict is wrapped: `bert.module.embeddings.weight`. This strips
    that one wrapper -> `bert.embeddings.weight`, matching what KokoroPlBert.LoadWeights
    expects. Equivalent to the runtime strip in KokoroPipeline.LoadAsync.
    """
    dot = key.find(".")
    if dot < 0:
        return key
    rest = key[dot + 1:]
    if rest.startswith("module."):
        return key[: dot + 1] + rest[len("module."):]
    return key


def _strip_leading_module(key: str) -> str:
    """Drop a leading `module.` (plain DataParallel wrap at the root)."""
    return key[len("module."):] if key.startswith("module.") else key


# Registry of built-in transforms, referenced by name from a Component so the
# manifest stays declarative and auditable.
TRANSFORMS: dict[str, Callable[[str], str | None]] = {
    "strip_inner_module": _strip_inner_module,
    "strip_leading_module": _strip_leading_module,
}


@dataclass
class Component:
    """One source weight file that contributes a prefixed slice of the merged output.

    source_file : filename within the model's HF repo (e.g. "llm.pt", "model.safetensors").
    prefix      : prepended to every key from this file. "" for single-component models;
                  "flow." / "llm." etc. for multi-component models so the engine can split.
    flatten     : for .pth/.pt dict-of-state-dicts, recursively flatten to dotted keys.
    transforms  : ordered list of transform names (see TRANSFORMS) applied to each key.
    """

    source_file: str
    prefix: str = ""
    flatten: bool = False
    transforms: Sequence[str] = field(default_factory=tuple)


@dataclass
class Recipe:
    """Declarative repack spec for one model. Subclass and fill the fields; override
    build() only for models whose merge can't be expressed as prefixed components."""

    name: str                       # output basename, e.g. "kokoro-82m"
    source_repo: str                # HF repo of the official weights
    license: str                    # SPDX-ish id, e.g. "apache-2.0", "cc-by-nc-4.0"
    target_repo: str                # where the repack is hosted, e.g. "Hartsy/kokoro-82m-safetensors"
    components: Sequence[Component]
    redistribute: bool = False      # gate for upload; flip True only after license sign-off
    revision: str = "main"
    extra_files: Sequence[str] = field(default_factory=tuple)  # copied alongside (config.json, vocab.txt)
    dtype: str | None = None        # optional cast: "fp32" | "fp16" | "bf16"

    # -- overridable build ---------------------------------------------------

    def build(self, fetch: Callable[[str], str]) -> dict[str, "torch.Tensor"]:
        """Download every component via `fetch`, load, transform, prefix, and merge into
        one flat tensor dict. `fetch(filename) -> local_path` resolves a repo file."""
        merged: dict[str, torch.Tensor] = {}
        for comp in self.components:
            path = fetch(comp.source_file)
            tensors = load_component(path, flatten=comp.flatten)
            fns = [_resolve_transform(t) for t in comp.transforms]
            for key, tensor in tensors.items():
                new_key = key
                dropped = False
                for fn in fns:
                    out = fn(new_key)
                    if out is None:
                        dropped = True
                        break
                    new_key = out
                if dropped:
                    continue
                final_key = comp.prefix + new_key
                if final_key in merged:
                    raise ValueError(
                        f"key collision while merging '{comp.source_file}': '{final_key}' "
                        f"already present. Fix the component prefix to disambiguate.")
                merged[final_key] = tensor
        return cast_dtype(merged, self.dtype)


def _resolve_transform(name: str) -> Callable[[str], str | None]:
    if name not in TRANSFORMS:
        raise KeyError(f"unknown transform '{name}'. Known: {sorted(TRANSFORMS)}")
    return TRANSFORMS[name]


# ---------------------------------------------------------------------------
# Loading / dtype / hashing helpers
# ---------------------------------------------------------------------------

def load_component(path: str, flatten: bool = False) -> dict[str, "torch.Tensor"]:
    """Load a single source weight file into a flat {key -> Tensor} dict.

    Dispatches by extension: .safetensors via safetensors, everything else
    (.pth/.pt/.bin/.th) via the pickle path reusing collect_tensors."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        return st_load_file(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return collect_tensors(obj) if flatten else _flat_state_dict(obj)


def _flat_state_dict(obj: Any) -> dict[str, "torch.Tensor"]:
    """Extract a flat state_dict for the common non-nested pickle case, unwrapping the
    usual training-checkpoint wrapper keys. Falls back to collect_tensors for anything
    deeper."""
    if isinstance(obj, dict):
        for wrapper in ("state_dict", "model", "net", "module"):
            if wrapper in obj and isinstance(obj[wrapper], dict) and len(obj) <= 3:
                obj = obj[wrapper]
                break
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return {(_strip_leading_module(k)): v.contiguous().detach().cpu()
                    for k, v in obj.items()}
    return collect_tensors(obj)


def cast_dtype(tensors: dict[str, "torch.Tensor"], target: str | None) -> dict[str, "torch.Tensor"]:
    if target is None:
        return tensors
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if target not in dtype_map:
        raise ValueError(f"dtype must be one of fp32/fp16/bf16, got '{target}'")
    td = dtype_map[target]
    return {k: (v.to(td) if v.dtype.is_floating_point else v) for k, v in tensors.items()}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def manifest_path(output_path: str) -> str:
    base = output_path[:-len(".safetensors")] if output_path.endswith(".safetensors") else output_path
    return base + ".manifest.json"


def write_manifest(recipe: Recipe, output_path: str, source_hashes: dict[str, str],
                   tensor_count: int) -> str:
    """Write the provenance/verification manifest next to the output file. Returns its path."""
    data = {
        "tool_version": TOOL_VERSION,
        "name": recipe.name,
        "source_repo": recipe.source_repo,
        "revision": recipe.revision,
        "license": recipe.license,
        "redistribute": recipe.redistribute,
        "target_repo": recipe.target_repo,
        "output_file": os.path.basename(output_path),
        "output_sha256": sha256_file(output_path),
        "tensor_count": tensor_count,
        "dtype_cast": recipe.dtype,
        "sources": [{"file": f, "sha256": h} for f, h in sorted(source_hashes.items())],
        "components": [
            {
                "source_file": c.source_file,
                "prefix": c.prefix,
                "flatten": c.flatten,
                "transforms": list(c.transforms),
            }
            for c in recipe.components
        ],
        "extra_files": list(recipe.extra_files),
    }
    mpath = manifest_path(output_path)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return mpath


def save_safetensors(tensors: dict[str, "torch.Tensor"], output_path: str, recipe: Recipe) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    metadata = {
        "converted_by": "hartsyinference tools/repack",
        "source_repo": recipe.source_repo,
        "name": recipe.name,
    }
    # safetensors requires contiguous tensors with no shared storage.
    clean = {k: v.contiguous().detach().cpu() for k, v in tensors.items()}
    st_save_file(clean, output_path, metadata=metadata)
