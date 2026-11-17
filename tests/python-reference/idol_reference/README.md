# Idol, Idol-turbo and Budget3D reference dumps, on the CPU

Upstream code is imported from clones, never copied: idol-sdk at `956439f5e664a7e8c5db25a8055e72d6365aafa5` (the commit
Budget3D pins; later ones need diffusers 0.40) and Budget3D at `95bbeab46063375d560f9b20fa7b9d9fedb078f7`. Everything runs in
float32 on the CPU with PyTorch SDPA. Outputs are safetensors files written outside the repository.

## Setup

A separate virtual environment with Budget3D's pinned versions (the CPU wheels are enough):

```
python -m venv venv
venv/Scripts/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python -m pip install transformers==4.57.6 diffusers==0.36.0 huggingface_hub==0.35.3 safetensors==0.8.0 \
    tensordict==0.10.0 accelerate==1.14.0 numpy==2.2.6 scipy==1.17.0 pillow==12.1.1 einops==0.8.2 tqdm plyfile==1.1.5 pyyaml
```

Environment: `IDOL_ROOT` (idol-sdk clone), `BUDGET3D_ROOT` (Budget3D clone), `IDOL_MODELS` (the engine's `Models/3D`).
The weights are the engine's own files: `idol/` (Comfy-Org/Idol repack plus `speridlabs/config.json` and the pinned
`pos_pe_f32_1x8192x3.bin`), `idol-turbo/`, `budget3d-idol/`, `budget3d-idol-turbo/`.

## Order

Each script reads the previous one's output folder. Run them with `python -I` from this folder.

| Script | Reads | Writes |
|---|---|---|
| `dump_idol_conditioning.py <image> <cond>` | an image (RGB is keyed to RGBA and saved as `input.png`) | prepared pixels, DINOv3 ViT-H+ tokens before and after its final norm, `feature1`, VAE mean, `feature2` |
| `dump_idol_denoiser.py <cond> <den>` | conditioning | seeded latent and camera; one dense forward per model with stage taps (Idol, Budget3D-Idol), both turbo phases |
| `dump_budget3d.py <den> <b3d>` | denoiser inputs | octree tables, layouts at each budget (from the latent's salience and from seeded scores), sparse forwards |
| `dump_idol_sampler.py <den> <samp>` | denoiser inputs | short sampler runs: Idol 4 steps with guidance, the official turbo loop, Budget3D at a budget |
| `dump_idol_decoders.py <samp> <dec>` | the turbo run's latent | octree logits per level on the recorded parents, Gaussian decoder features, activated Gaussians, PLY values |

The anchor sampler draws from torch's RNG, so only its per-level logits are compared, not the anchors it picks.
