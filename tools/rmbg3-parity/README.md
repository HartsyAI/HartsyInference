# RMBG 3.0 parity harness

Offline scripts for the GPU server. Nothing here runs in the engine, and nothing here has been run on real weights.

## Pinned references

- diffusers `98dfc1aa4146e0d8bcbdb2d00ff68e698ce3304c` (`pipeline_bria_fibo2.py`, `transformer_bria_fibo2.py`, `image_processor.py`), diffusers >= 0.42.
- Comfy repack `briaai/RMBG-3.0-Comfy` @ `63880aa074782b09e7ae17184960dfd6de70ab85`.

## Files and sizes the server may need

| File | Bytes | sha256 |
|---|---|---|
| `diffusion_models/rmbg3_bf16.safetensors` | 16,452,934,194 | `a5df048f799025510f29a743c0dca0d4f1362478ec00f6f7c86ee66d57b4e5d9` |
| `diffusion_models/rmbg3_int8_convrot.safetensors` | 8,408,562,474 | `bc29b9a87ed378596c5ca5bc0c5bdfaa8aabc8415d33f3c767f9a3cbd9fd566d` |
| `qwen3vl_4b_int8_convrot.safetensors` (encoder, int8 path) | 4,858,559,240 | `32de113c…` (full hash in the architecture doc) |
| Fibo 2.0 encoder (bf16) and FLUX.2 VAE | see the architecture doc | shared with Fibo 2.0 |

The diffusers reference needs a diffusers-layout copy of the pipeline. The Comfy repack above is a different layout, so the reference script cannot read it. If only the repack is on the server, stop there and report.

## Input images

Fixed set: the first 4 files by sorted name in `Datasets/coco2017/images` (the same folder on the dev machine). Copy them to the server unchanged and record their sha256 in `manifest.json`.

## (1) diffusers reference on the same input images

```bash
python diffusers_reference.py --model-dir /path/to/diffusers-layout-rmbg3 --images /path/to/coco-subset --out /path/to/ref --count 4
```

Writes `<stem>.alpha.png`, `<stem>.npz` and `<stem>.noise.npy` per image, plus `manifest.json`.

**Gap:** the engine's vision path has no way to load a saved noise tensor. Its seed does not match torch's generator, so the engine and diffusers do not start from the same noise yet. Check (1) on saved noise needs an engine hook that reads `<stem>.noise.npy`. That hook is not written.

## (2) Engine CPU vs CUDA, same input

```bash
hartsy vision /path/to/coco-subset/<image> -m rmbg3 -b cpu  --mode matting -o out/<stem>.cpu.png
hartsy vision /path/to/coco-subset/<image> -m rmbg3 -b cuda --mode matting -o out/<stem>.cuda.png
python compare_alpha.py out/<stem>.cpu.png out/<stem>.cuda.png
```

Assumes `rmbg3` resolves its weights from the catalog and `--mode matting` is the mode for these ids. Neither was run.

## (3) 12 GB int8 check

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1 > vram_int8.csv &
hartsy vision /path/to/coco-subset/<image> -m rmbg3-int8 -b cuda --mode matting -o out/<stem>.int8.png
kill %1
python compare_alpha.py out/<stem>.bf16.png out/<stem>.int8.png
```

Run this on a 12 GB card, or cap the device to 12 GB. The RTX 4090 has 24 GB and does not show the 12 GB fit. Peak `memory.used` comes from `vram_int8.csv`.

## Pass lines (proposed, not validated)

- (1) diffusers vs engine: pearson >= 0.999, mean_abs <= 1.0 level, max_abs <= 8 levels.
- (2) CPU vs CUDA: pearson >= 0.999, mean_abs <= 1.0 level, max_abs <= 8 levels.
- (3) int8: completes with no OOM on 12 GB, peak `memory.used` <= 12288 MiB. The int8 vs bf16 alpha metrics are reported, not gated.

The parity owner sets the final numbers before the first run.

## Not yet done

- The `diffusers_reference.py` pipeline call and signature are unverified against the pinned commit. The script refuses to run if the signature differs.
- The engine noise hook for check (1) is not written.
- No command in this README has been run.
