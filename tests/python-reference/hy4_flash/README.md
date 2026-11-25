# Hy4 Flash tiny-config reference

`dump_tiny_reference.py` runs the real Hugging Face classes from transformers PR 50903 (pin `eb1d5db464b0df81793f914849142e9b82e056c7`, unmerged) on a
3-layer CPU config and saves the weights (checkpoint key names), the inputs and every stage the C# model is compared with. The script has **not been run**: it
needs torch, which does not import on the Windows Python here.

Run it on WSL or a pod the user starts (CPU is enough, a few GB of RAM, under a minute):

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch numpy safetensors "transformers[torch,vision] @ git+https://github.com/huggingface/transformers.git@eb1d5db464b0df81793f914849142e9b82e056c7"
python tests/python-reference/hy4_flash/dump_tiny_reference.py --out /tmp/hy4_flash_tiny
```

In the `hartsy-ref:cpu` image (run.sh in the main checkout under `.claude/run-outputs/refenv/`) the pinned transformers needs `pip install --no-deps https://github.com/huggingface/transformers/archive/eb1d5db464b0df81793f914849142e9b82e056c7.tar.gz` first. The committed `data/` folder was produced that way on 2026-11-24 (manifest records torch and the pin).

The script refuses a transformers install that is not that exact checkout (`--allow-unpinned` overrides, and the dump then is not the reference). Commit the
resulting folder's small files (the weights are about 1 MB) under `tests/python-reference/hy4_flash/data/` with the manifest; the C# stage tests read it. The dump covers a text case (full forward plus a 37-token prefill and three cached steps) and an image case (a tiny tower and an image span in the middle of the sequence); `tiny_config.json` is read by both the script and a C# test.

`data/engram_fp8/` holds the same tiny model with `engram_fp8_embedding` true (`dump_tiny_reference.py --engram-fp8`), keeping only the config, the inputs, layer 0's output and the Engram layer 1 output: the in-layer fp8 round trip is checked against it, with the weights of `data/`.
