# Python Reference Scripts

Scripts that generate reference tensors and statistics from the Python (diffusers/PyTorch) SD1.5 pipeline. These are used by the C# cross-runtime validation tests to verify numerical correctness.

## Setup

```bash
python -m venv tests/python-reference/.venv
tests/python-reference/.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
tests/python-reference/.venv/Scripts/pip install diffusers transformers safetensors
```

## Scripts

| Script | What it does | Output |
|---|---|---|
| `dump_reference_stats.py` | Runs full pipeline (noise, CLIP, 20 UNet steps, VAE), saves key tensors | `reference_tensors/*.bin` |
| `dump_layer_outputs.py` | Hooks every UNet layer, saves per-layer output tensors | `reference_tensors/layers/*.bin` + `index.json` |
| `dump_attn_sublayers.py` | Manually steps through first CrossAttentionBlock sub-operations | `reference_tensors/attn0_sublayers/*.bin` |

## Running

```bash
"tests/python-reference/.venv/Scripts/python" "tests/python-reference/<script>.py"
```

## Output Structure

```
reference_tensors/
├── initial_noise.bin                    # Gaussian noise [1,4,32,32] float32
├── text_embeddings.bin                  # CLIP output [2,77,768] float32
├── unet_step0_input.bin                 # Scaled input to first UNet call
├── unet_step0_output_uncond.bin         # UNet unconditional output
├── unet_step0_text_emb.bin             # Text embeddings for first call
├── step0_scaled_input.bin              # Pipeline-level step 0 input
├── step0_noise_pred_cond.bin           # Conditional noise prediction
├── step0_noise_pred_uncond.bin         # Unconditional noise prediction
├── final_latents.bin                    # Final denoised latent
├── layers/                              # Per-layer UNet outputs
│   ├── index.json                       # Layer name → file mapping
│   ├── conv_in.bin
│   ├── down_blocks_0_resnets_0.bin
│   └── ...
└── attn0_sublayers/                     # CrossAttentionBlock breakdown
    ├── 01_groupnorm.bin
    ├── 03_proj_in.bin
    ├── 05_self_q.bin
    └── ...
```

All binary files are raw float32 tensors in C-contiguous (row-major) layout.
