"""Upstream MeanFlow head at several step counts and guidance weights on the tiny Haru checkpoint.

    python head_steps_reference.py <path to the upstream repo's haru/ package dir> <fixture dir>

Reads haru_tiny.safetensors, haru_tiny_expected.safetensors and config.json from <fixture dir> (written by haru_reference.py) and
writes haru_head_steps_expected.safetensors beside them: normalized latents of `head.sample` for the same hidden states and noise,
one tensor per (steps, guidance) pair, named `ref.steps{S}.guidance{G}`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

CASES = [(1, 1.0), (1, 2.0), (2, 1.5), (3, 1.25)]


@torch.no_grad()
def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    package, fixture = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, package)
    from haru.config import HaruConfig
    from haru.models.haru import HaruModel

    torch.set_num_threads(1)
    cfg = HaruConfig.from_dict(json.loads((fixture / "config.json").read_text()))
    model = HaruModel(cfg).eval()
    model.load_state_dict(load_file(str(fixture / "haru_tiny.safetensors")))
    model.eval().requires_grad_(False)
    expected = load_file(str(fixture / "haru_tiny_expected.safetensors"))
    hidden, noise = expected["ref.hidden"], expected["in.noise"]
    out = {f"ref.steps{steps}.guidance{guidance}": model.head.sample(hidden, noise, guidance, steps).float() for steps, guidance in CASES}
    save_file({k: v.contiguous() for k, v in out.items()}, str(fixture / "haru_head_steps_expected.safetensors"), metadata={"format": "pt"})
    print("wrote", list(out))


if __name__ == "__main__":
    main()
