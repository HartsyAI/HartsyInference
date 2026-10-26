"""Repack recipe for Hurston-1 (destined/hurston-1, Apache-2.0), a LoRA fine-tune of Maya1.

The official release ships two bf16 safetensors shards plus model.safetensors.index.json. This
recipe merges both shards into one flat, bf16 .safetensors with the source key layout unchanged.
No transforms: the keys are already the plain Llama-3B names Orpheus and Maya1 use.

The source repo is gated on Hugging Face. Fetching it needs an authenticated token from an
account that accepted the model's terms (HF_TOKEN or `huggingface-cli login`).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recipe import Component, Recipe  # noqa: E402

RECIPE = Recipe(
    name="hurston-1",
    source_repo="destined/hurston-1",
    license="apache-2.0",
    target_repo="Hartsy/hurston-1-safetensors",
    redistribute=False,  # flip True only after explicit upload approval from the user
    extra_files=(
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "hurston.py",
    ),
    dtype="bf16",
    # HurstonTtsModel loads through AudioCheckpoints, which reads model.safetensors.
    output_name="model.safetensors",
    components=(
        Component(source_file="model-00001-of-00002.safetensors", prefix=""),
        Component(source_file="model-00002-of-00002.safetensors", prefix=""),
    ),
)
