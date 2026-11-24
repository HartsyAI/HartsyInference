"""Tiny Sage-SciPerception projectors through the upstream `AtomProjector`, saved for the C# projector test.

Offline only. Fetches upstream's `modeling_projector.py` (12 KB of source, no weights) at the pinned SciPerception revision, builds small random
projectors with the option sets the released projectors use, and writes for each case: config.json (sage-projector-v1), projector.safetensors,
and the inputs and output of one forward pass as .npy files.

    python tools/parity/sciperception_reference.py --out tests/HartsyInference.LLM.Tests/Fixtures/Glm5Next/sciperception

Cases (the option sets of the released projectors): `plain` (cif/ecg/eeg/psr/dna: position embedding + MLP), `discrete` (spectra/qm9s: a
category embedding column), `side` (rna/ncrna: side-measurement MLP), `readout` (protein-seq/mpbind: frozen per-row classifier bridge, a side
MLP that the readout does not read, and the unused task_head).
"""
import argparse
import importlib.util
import json
import os
import urllib.request

import numpy as np
import torch
from safetensors.torch import save_file

REVISION = "7c228fc03394027fed19bda1ff26388394b0031a"
URL = f"https://huggingface.co/internlm/Sage-SciPerception/resolve/{REVISION}/modeling_projector.py"
OUT_DIM = 24

CASES = {
    "plain": dict(kind="atom", feature_dim=10, index_dim=6, max_atoms=16, hidden_dim=32),
    "discrete": dict(kind="atom", feature_dim=9, index_dim=6, max_atoms=16, hidden_dim=32, discrete_dim=5, discrete_sizes=[3, 4]),
    "side": dict(kind="atom", feature_dim=8, index_dim=6, max_atoms=16, hidden_dim=32, side_dim=7, side_hidden_dim=20),
    "readout": dict(kind="atom", feature_dim=12, index_dim=6, max_atoms=16, hidden_dim=32, task_head_outputs=1, readout_classes=1,
                    readout_uses_side=False, side_dim=2, side_hidden_dim=16),
}
ROWS = 11


def load_upstream(directory):
    path = os.path.join(directory, "modeling_projector.py")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    spec = importlib.util.spec_from_file_location("modeling_projector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def randomize(model, gen):
    """Upstream zero-initializes exits so deep stacks start as identity; every tensor must matter here."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * (0.4 if p.ndim > 1 else 0.2))
        for name, b in model.named_buffers():
            if name.endswith("feature_std"):
                b.copy_(0.5 + torch.rand(b.shape, generator=gen))
            else:
                b.copy_(torch.randn(b.shape, generator=gen) * 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    up = load_upstream("/tmp")
    gen = torch.Generator().manual_seed(args.seed)
    for name, spec in CASES.items():
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        model = up.build(spec, OUT_DIM).float().eval()
        randomize(model, gen)
        values = torch.randn(ROWS, spec["feature_dim"], generator=gen)
        positions = torch.randint(0, spec["max_atoms"], (ROWS,), generator=gen)
        sizes = spec.get("discrete_sizes") or []
        discrete = torch.stack([torch.randint(0, n, (ROWS,), generator=gen) for n in sizes], dim=1) if sizes else None
        side = torch.randn(ROWS, spec["side_dim"], generator=gen) if spec.get("side_dim") else None
        with torch.no_grad():
            out = model(values, positions, discrete, side=side)
        save_file({k: v.detach().float().contiguous() for k, v in model.state_dict().items()}, os.path.join(d, "projector.safetensors"))
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"format_version": "sage-projector-v1", "component_type": "scientific_projector", "task": f"tiny/{name}",
                       "weights": "projector.safetensors", "projector": spec, "output_dim": OUT_DIM}, f, indent=1)
        np.save(os.path.join(d, "features.npy"), values.numpy())
        np.save(os.path.join(d, "positions.npy"), positions.numpy().astype(np.int32))
        if discrete is not None:
            np.save(os.path.join(d, "discrete.npy"), discrete.numpy().astype(np.int32))
        if side is not None:
            np.save(os.path.join(d, "side.npy"), side.numpy())
        np.save(os.path.join(d, "output.npy"), out.numpy())
        print(name, tuple(out.shape), float(out.abs().mean()))


if __name__ == "__main__":
    main()
