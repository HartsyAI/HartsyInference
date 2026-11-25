"""Writes the YaRN inverse frequencies and attention scaling transformers computes for SenseNova-Loop-9B's real rope parameters
(head_dim 256, partial rotary 0.25 so 64 rotary channels, theta 1e7, factor 2, original 262144), for SenseNovaLoopYarnTests.

Usage: python yarn_reference.py <SenseNova-Loop-9B config.json> <output json>   (needs transformers, torch; run in the CPU container)
"""
import json, sys
import torch
from transformers import Qwen3_5TextConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

cfg = json.load(open(sys.argv[1]))["text_config"]
config = Qwen3_5TextConfig(**cfg)
inv_freq, scaling = ROPE_INIT_FUNCTIONS["yarn"](config, "cpu")
json.dump({"dim": int(inv_freq.numel()) * 2, "inv_freq": [float(x) for x in inv_freq.double()], "attention_scaling": float(scaling),
           "transformers": __import__("transformers").__version__, "torch": torch.__version__}, open(sys.argv[2], "w"))
print(inv_freq.shape, scaling)
