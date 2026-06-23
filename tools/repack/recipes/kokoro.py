"""Repack recipe for Kokoro 82M (hexgrad/Kokoro-82M, Apache-2.0).

The official release ships a single PyTorch checkpoint, kokoro-v1_0.pth, whose top-level
keys are the module names (bert, bert_encoder, predictor, text_encoder, decoder) and whose
inner state-dicts are wrapped in a `module.` segment (saved under nn.DataParallel).

This recipe bakes the runtime surgery currently done in KokoroPipeline.LoadAsync into one
offline step:
  * recursive flatten of the dict-of-state-dicts  ->  dotted keys
  * strip the inner `module.` wrapper             ->  bert.embeddings.weight, ...
producing kokoro-82m.safetensors with exactly the flat keys the submodule LoadWeights
methods already expect. Single-component, so prefix is "".
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recipe import Component, Recipe  # noqa: E402

RECIPE = Recipe(
    name="kokoro-82m",
    source_repo="hexgrad/Kokoro-82M",
    license="apache-2.0",
    target_repo="Hartsy/kokoro-82m-safetensors",
    redistribute=False,  # flip True after confirming Apache-2.0 redistribution sign-off
    extra_files=("config.json",),
    components=(
        Component(
            source_file="kokoro-v1_0.pth",
            prefix="",
            flatten=True,
            transforms=("strip_inner_module",),
        ),
    ),
)
