# Oasis-500m — Research Notes

> Status: Complete (full inference code + all configs captured from `etched-ai/open-oasis`; safetensors key dump still required) | Last Updated: 2026-05-24 | Needed Before: HartsyInference.World (`OasisPipeline`, Phase 10 — world models)
> Source of truth: [etched-ai/open-oasis (GitHub)](https://github.com/etched-ai/open-oasis), [HF `Etched/oasis-500m`](https://huggingface.co/Etched/oasis-500m), [Oasis blog](https://oasis-model.github.io/), [Decart publication](https://decart.ai/publications/oasis-interactive-ai-video-game-model)
> License: **MIT** (both code and weights)
> Related: [`DIFFUSION_SCHEDULERS.md`](DIFFUSION_SCHEDULERS.md), [`LANCE_ARCHITECTURE.md`](LANCE_ARCHITECTURE.md) (DiT lineage), [`FLOW_MATCHING_AUDIO.md`](FLOW_MATCHING_AUDIO.md), [`CONV2D_CUDA.md`](CONV2D_CUDA.md). Net-new module-class introduced by this doc: **continuous ViT-VAE** (not VQ — see § 6).

> **Stub.** The narrative walkthrough, restated pseudocode and resolved open questions were
> removed on 2026-08-06 — this model is built and verified, so the C# is the source of truth for
> *how it works*. What remains is what the code cannot tell you: upstream provenance, reference
> constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> Full history is in git. Parity evidence: `docs/Checklists/PARITY_VERIFICATION.md`.

## Summary

Oasis is an **interactive, action-conditioned, autoregressive video world model** trained on Minecraft gameplay, released by Decart and Etched on 2024-10-31 under MIT. The publicly released checkpoint is a **down-scaled 500M-parameter "DiT-S/2"** transformer; the production demo at oasis.us.decart.ai is a larger model (referred to in press as 7B) that is **not** released. For HartsyInference, "Oasis-500m" means **only the 500M open checkpoint** — the cheap, pedagogical, real-time world model that serves as our CI smoke test and reference port for future world-model pipelines.

The architecture has two parts, both pure Transformers:

1. A **continuous Gaussian ViT-VAE** (`vit-l-20-shallow-encoder`, ~917 MB), patch-size 20, encoding 360 × 640 RGB frames into 18 × 32 = 576 latent tokens with 16 channels each. This is **NOT** a discrete VQ-VAE/VQ-GAN — it is a vanilla KL-regularised continuous autoencoder with a 16-dim Gaussian latent. (Section 6 explains why this matters for HartsyInference: no new VQ codebook infrastructure is required.)
2. A **spatio-temporal DiT** (`DiT-S/2`, 16 layers / hidden 1024 / 16 heads, ~2.43 GB FP16) operating in this 16-channel latent space. Each transformer block alternates a **spatial axial attention** (over the 18 × 32 = 576 patch tokens per frame, bidirectional, with 2-D rotary position embeddings) and a **temporal axial attention** (over the time axis at each spatial location, **causal** with 1-D rotary). Per-block adaptive layer-norm (adaLN-zero, DiT-style) is conditioned on a sum of (a) sinusoidal diffusion-timestep embedding and (b) a 25-dim Minecraft action vector projected to hidden through `nn.Linear(25, 1024)`.

Generation is **autoregressive in time, diffusion in space**: starting from N prompt frames already encoded to clean latents, the loop appends a Gaussian-noise latent for the next frame and denoises **only that frame** using 10 DDIM steps, while the prompt frames are held at a near-clean "stabilisation" noise level of 15/1000. This is a degenerate form of [**Diffusion Forcing**](https://www.boyuan.space/diffusion-forcing/) (Chen et al., NeurIPS 2024) where each frame can carry an independent noise level — at inference Oasis exploits this by giving the context near-zero noise and the target frame full noise. After 10 DDIM steps the new frame's clean latent is committed and the loop advances. The noise schedule is the **sigmoid β-schedule** with `T=1000`. A sliding window caps the visible context to `max_frames = 32` latent frames.

Action conditioning is a 25-dim per-frame vector covering the Minecraft VPT action space: 23 binary keys (`inventory`, `ESC`, `hotbar.1..9`, `forward/back/left/right`, `jump/sneak/sprint/swapHands/attack/use/pickItem/drop`) and 2 normalised camera floats (`cameraX`, `cameraY` ∈ [−1, 1]). The model emits frames at **20 FPS at 360 × 640** on H100; on Etched's Sohu it claims real-time at up to 4K. Inference cost is **47 ms / frame** on H100 (10 DDIM steps × the DiT-S/2 forward pass).

For HartsyInference, Oasis-500m is the **Phase 10 reference world model**: tiny (~3.4 GB total FP16), fully MIT, end-to-end in pure C# without any new heavy primitives, and trivially validated frame-by-frame against the upstream Python (because both action stream and prompt frames are deterministic given a fixed seed).

## Key Numbers / Constants

| Constant | Value | Where used |
|---|---|---|
| Image resolution | **360 × 640** | VAE input/output |
| VAE patch size | **20** | square |
| VAE latent grid | **18 × 32 = 576 tokens** | per frame |
| VAE latent channels | **16** | `latent_dim` |
| VAE encoder dim | **1024** | `enc_dim` |
| VAE encoder depth | **6** | `enc_depth` (shallow) |
| VAE encoder heads | **16** | head_dim = 64 |
| VAE decoder dim | **1024** | `dec_dim` |
| VAE decoder depth | **12** | `dec_depth` |
| VAE decoder heads | **16** | head_dim = 64 |
| VAE MLP ratio | **4.0** | both halves |
| VAE LayerNorm eps | **1e-6** | both halves |
| VAE `use_variational` | **True** | Gaussian KL |
| VAE `logvar` clamp | **[-30, 20]** | bottleneck |
| VAE `qkv_bias` | **True** | inside `AttentionBlock` |
| VAE inner-attn rotary `dim` | **16** | = head_dim / 4 (NOT /2) |
| VAE inner-attn rotary `max_freq` | **576** | = H × W |
| VAE patch_dim | **1200** | = 3 × 20² |
| Latent scaling factor | **0.07843137255** | `generate.py` line ≈ 49 |
| DiT input H | **18** | matches VAE grid |
| DiT input W | **32** | matches VAE grid |
| DiT in_channels | **16** | matches VAE latent_dim |
| DiT patch size | **2** | latent-internal patchify → 9 × 16 = **144 tokens/frame** |
| DiT hidden_size | **1024** | model dim |
| DiT depth | **16** | layers |
| DiT num_heads | **16** | head_dim = **64** |
| DiT mlp_ratio | **4.0** | MLP hidden = 4096 |
| DiT timestep freq dim | **256** | `frequency_embedding_size` |
| DiT external_cond_dim | **25** | action vector dim |
| DiT max_frames | **32** | sliding window |
| Spatial RoPE dim | **32** | = head_dim / 2 |
| Spatial RoPE `freqs_for` | `"pixel"` | 2-D axial |
| Spatial RoPE max_freq | **256** | — |
| Temporal RoPE dim | **64** | = head_dim |
| Temporal attention causal | **True** | left-to-right time |
| Spatial attention causal | **False** | bidirectional in space |
| AdaLN modulation per block | `Linear(1024, 6 × 1024) × 2` | spatial + temporal halves |
| MLP activation | `GELU(approximate="tanh")` | DiT MLPs |
| LayerNorm eps (DiT) | **1e-6** | `elementwise_affine=False` |
| Action keys count | **25** | exact list above |
| Action camera normalization | `max_val=20, bin_size=0.5` | VPT-style |
| Action prepended zero frame | **1** | aligns action[i] ↔ frame[i] |
| Diffusion `T` (`max_noise_level`) | **1000** | sigmoid β schedule |
| β schedule `start, end, tau` | **-3, 3, 1** | sigmoid |
| DDIM steps | **10** | `--ddim-steps` |
| Initial-noise clamp | **±20** | `noise_abs_max` |
| Stabilization level | **15** | context-frame noise idx |
| Default `num_frames` | **32** | matches `max_frames` |
| Default `n_prompt_frames` | **1** | single-image prompt |
| Output FPS | **20** | quoted real-time on H100 |
| Per-frame latency (H100) | **47 ms** | Decart blog |
| Per-iter training cost | **150 ms** | Decart blog |
| Storage (DiT) | **2.43 GB** | safetensors FP32 |
| Storage (VAE) | **917 MB** | safetensors FP32 |
| Storage (total FP16) | **≈ 1.7 GB** | full pipeline cast to FP16 |
| License | **MIT** | code and weights |

## Data Layouts / Formats

### 9.1 HF repo `Etched/oasis-500m` file tree

```
Etched/oasis-500m/
├── .gitattributes                  115 B
├── LICENSE                        1.07 kB    MIT
├── README.md                       893 B
├── oasis500m.safetensors          2.43 GB    DiT-S/2 backbone, FP32
├── oasis500m.pt                   2.43 GB    Pickle version of same
├── vit-l-20.safetensors            917 MB    ViT-VAE encoder+decoder, FP32
├── vit-l-20.pt                     917 MB    Pickle version of same
└── media/                                    Demo gifs and PNGs (not weights)
```

**No `config.json`**. **No tokenizer.** Action input is a raw `(T, 25)` float tensor — there is no vocabulary BPE.

### 9.2 Expected (not yet dumped) tensor key prefixes in `oasis500m.safetensors`

Inferred from the PyTorch class hierarchy. Confirm by running locally:

```python
from safetensors import safe_open
with safe_open("oasis500m.safetensors", framework="pt") as f:
    for k in sorted(f.keys()): print(k)
```

Expected keys:

```
x_embedder.proj.weight                    # (1024, 16, 2, 2)
x_embedder.proj.bias                      # (1024,)
t_embedder.mlp.0.weight                   # (1024, 256)
t_embedder.mlp.0.bias                     # (1024,)
t_embedder.mlp.2.weight                   # (1024, 1024)
t_embedder.mlp.2.bias                     # (1024,)
external_cond.weight                      # (1024, 25)
external_cond.bias                        # (1024,)
blocks.{0..15}.s_norm1.{}                 # none — affine=False
blocks.{0..15}.s_attn.to_qkv.weight       # (3072, 1024) — no bias
blocks.{0..15}.s_attn.to_out.weight       # (1024, 1024)
blocks.{0..15}.s_attn.to_out.bias         # (1024,)
blocks.{0..15}.s_norm2.{}                 # none — affine=False
blocks.{0..15}.s_mlp.fc1.weight           # (4096, 1024)
blocks.{0..15}.s_mlp.fc1.bias             # (4096,)
blocks.{0..15}.s_mlp.fc2.weight           # (1024, 4096)
blocks.{0..15}.s_mlp.fc2.bias             # (1024,)
blocks.{0..15}.s_adaLN_modulation.1.weight # (6144, 1024)
blocks.{0..15}.s_adaLN_modulation.1.bias   # (6144,)
blocks.{0..15}.t_norm1.{}                  # none
blocks.{0..15}.t_attn.to_qkv.weight        # (3072, 1024) — no bias
blocks.{0..15}.t_attn.to_out.weight        # (1024, 1024)
blocks.{0..15}.t_attn.to_out.bias          # (1024,)
blocks.{0..15}.t_norm2.{}                  # none
blocks.{0..15}.t_mlp.fc1.weight            # (4096, 1024)
blocks.{0..15}.t_mlp.fc1.bias              # (4096,)
blocks.{0..15}.t_mlp.fc2.weight            # (1024, 4096)
blocks.{0..15}.t_mlp.fc2.bias              # (1024,)
blocks.{0..15}.t_adaLN_modulation.1.weight # (6144, 1024)
blocks.{0..15}.t_adaLN_modulation.1.bias   # (6144,)
final_layer.linear.weight                  # (64, 1024)        — patch_size² × out_channels = 4 × 16
final_layer.linear.bias                    # (64,)
final_layer.adaLN_modulation.1.weight      # (2048, 1024)
final_layer.adaLN_modulation.1.bias        # (2048,)
```

Total parameter count check (approximate):

- PatchEmbed: 16 × 1024 × 4 + 1024 ≈ 66 K
- TimestepEmbedder: 256 × 1024 + 1024 × 1024 + biases ≈ 1.31 M
- external_cond: 25 × 1024 + 1024 ≈ 26 K
- Per block: 2 × (QKV: 1024×3072 + Out: 1024×1024+1024 + MLP: 1024×4096+4096×1024+biases + adaLN: 1024×6144+6144) ≈ 2 × 14.9 M ≈ 29.8 M
- 16 blocks × 29.8 M ≈ 477 M
- FinalLayer: 1024 × 2048 + 2048 + 1024 × 64 + 64 ≈ 2.17 M
- **Total ≈ 481 M params** ≈ "500M" branding. ✓

### 9.3 Expected tensor key prefixes in `vit-l-20.safetensors`

```
patch_embed.proj.weight              # (1024, 3, 20, 20)
patch_embed.proj.bias                # (1024,)
encoder.{0..5}.norm1.{weight,bias}   # (1024,) each
encoder.{0..5}.attn.qkv.weight       # (3072, 1024)
encoder.{0..5}.attn.qkv.bias         # (3072,)
encoder.{0..5}.attn.proj.weight      # (1024, 1024)
encoder.{0..5}.attn.proj.bias        # (1024,)
encoder.{0..5}.norm2.{weight,bias}   # (1024,)
encoder.{0..5}.mlp.fc1.weight        # (4096, 1024)
encoder.{0..5}.mlp.fc1.bias          # (4096,)
encoder.{0..5}.mlp.fc2.weight        # (1024, 4096)
encoder.{0..5}.mlp.fc2.bias          # (1024,)
enc_norm.{weight,bias}               # (1024,)
quant_conv.weight                    # (32, 1024)
quant_conv.bias                      # (32,)
post_quant_conv.weight               # (1024, 16)
post_quant_conv.bias                 # (1024,)
decoder.{0..11}.norm1.{}             # (1024,)
decoder.{0..11}.attn.qkv.weight      # (3072, 1024)
decoder.{0..11}.attn.qkv.bias        # (3072,)
decoder.{0..11}.attn.proj.{}         # (1024, 1024)
decoder.{0..11}.norm2.{}             # (1024,)
decoder.{0..11}.mlp.fc1.{weight,bias} # (4096, 1024)
decoder.{0..11}.mlp.fc2.{weight,bias} # (1024, 4096)
dec_norm.{weight,bias}               # (1024,)
predictor.weight                     # (1200, 1024)
predictor.bias                       # (1200,)
```

`rotary_freqs` buffers are non-persistent — not in the safetensors file; HartsyInference computes them at load time from `H=18, W=32, head_dim=64, max_freq=576`.

### 9.4 Action stream format

Two on-disk formats, both PyTorch `.pt`:

- `*.actions.pt` — `List[Dict[str, int | List[int, int]]]`, one dict per frame. Each dict has all 25 keys; binary keys are `0`/`1`, the `camera` key is `[int_raw_x, int_raw_y]` in VPT bucket space. `one_hot_actions()` converts to `(T, 25)` float.
- `*.one_hot_actions.pt` — `Tensor(T, 25, float32)` already encoded.

HartsyInference's `OasisPipeline` should accept either a `float[T, 25]` array (one-hot, already encoded) or an `OasisAction[]` struct array with explicit fields. A small `OasisActionEncoder` helper handles the camera normalisation.

## Reference Implementations

**Primary** — [`github.com/etched-ai/open-oasis`](https://github.com/etched-ai/open-oasis) (MIT, ~6 files):

- [`generate.py`](https://github.com/etched-ai/open-oasis/blob/master/generate.py) — full inference loop. Quoted verbatim in § 2 / Algorithm Steps.
- [`dit.py`](https://github.com/etched-ai/open-oasis/blob/master/dit.py) — `PatchEmbed`, `TimestepEmbedder`, `FinalLayer`, `SpatioTemporalDiTBlock`, `DiT`, `DiT_S_2()` factory, `DiT_models` registry. The **only** released variant is `DiT-S/2`.
- [`vae.py`](https://github.com/etched-ai/open-oasis/blob/master/vae.py) — `DiagonalGaussianDistribution`, `Attention`, `AttentionBlock`, `AutoencoderKL`, `ViT_L_20_Shallow_Encoder()` factory, `VAE_models` registry.
- [`attention.py`](https://github.com/etched-ai/open-oasis/blob/master/attention.py) — `SpatialAxialAttention`, `TemporalAxialAttention`.
- [`utils.py`](https://github.com/etched-ai/open-oasis/blob/master/utils.py) — `sigmoid_beta_schedule`, `ACTION_KEYS`, `one_hot_actions`, `load_prompt`, `load_actions`.
- [`rotary_embedding_torch.py`](https://github.com/etched-ai/open-oasis/blob/master/rotary_embedding_torch.py) — vendored copy of `lucidrains/rotary-embedding-torch`. Provides `RotaryEmbedding`, `apply_rotary_emb`, `.get_axial_freqs()`, `.rotate_queries_or_keys()`.

**Lineage** (referenced in docstrings):

- [Facebook DiT](https://github.com/facebookresearch/DiT) — adaLN-zero modulation + sinusoidal timestep + FinalLayer. HartsyInference's existing DiT blocks match.
- [Diffusion Forcing](https://github.com/buoyancy99/diffusion-forcing) ([arXiv 2407.01392](https://arxiv.org/abs/2407.01392)) — the per-frame noise scheme and the spatial/temporal axial attention. Note the [Diffusion Forcing blog post](https://www.boyuan.space/diffusion-forcing/) explains the training paradigm; Oasis is the production-scale demonstration of it.
- [Latte](https://github.com/Vchitect/Latte) — spatio-temporal block interleaving.
- [Vision Pre-Training (VPT)](https://github.com/openai/Video-Pre-Training) — source of the 25-key Minecraft action space + camera bucketisation.

**Companion materials** (not code but reference quality):

- [Oasis blog](https://oasis-model.github.io/) — official architecture description.
- [Decart "A Universe in a Transformer"](https://decart.ai/publications/oasis-interactive-ai-video-game-model) — design philosophy + Sohu / inference cost claims.
- [Etched "Oasis: an interactive, explorable world model"](https://www.etched.com/blog-posts/oasis) — hardware perspective.
- [Chipstrat: "Etched's Oasis: Creating a Market For Sohu"](https://www.chipstrat.com/p/etcheds-oasis-creating-a-market-for) — third-party analysis.
- Press: [InfoQ](https://www.infoq.com/news/2024/11/decart-etched-oasis/), [The Decoder](https://the-decoder.com/ai-generated-game-oasis-now-turns-images-into-playable-3d-worlds/), [TechSpot](https://www.techspot.com/news/105436-oasis-interactive-ai-experience-turns-minecraft-nightmarish-hallucination.html).

**Community ports** (read-only, not source-of-truth):

- [PFWorld/open-oasis-PFWorld](https://github.com/PFWorld/open-oasis-PFWorld) — minor fork.
- [profhiggs/open-oasis-datacraft](https://github.com/profhiggs/open-oasis-datacraft) — fork wrapping Oasis around a Minecraft-mod token economy. Useful for action-stream examples.
- [XmYx/open-oasis](https://github.com/XmYx/open-oasis) — fork with QoL tweaks.
- [spikedoanz/realtime-oasis](https://github.com/spikedoanz/realtime-oasis) — real-time interactive harness; useful precedent for our `OasisLivePipeline`.

## Differences Between Implementations

There is one official reference (`etched-ai/open-oasis`) and several near-identical community forks. No serious behavioural divergence has been observed; the forks differ in:

- Real-time interactive harness vs. batch generate-and-save (spikedoanz/realtime-oasis).
- Minor tokenizer / action-loading conveniences (datacraft, PFWorld).
- Some forks add per-action sliders / UI; the model weights are untouched.

There is **no diffusers PR** for Oasis as of 2026-05-24. There is **no ComfyUI node** in the official repo. The architecture is simple enough that it has been re-implemented inside several game-engine harnesses (Unity, web demos) but always pointing back at the same safetensors.

There is also a related architectural lineage to be aware of, but they are **different models** and should NOT be confused with Oasis:

- [MineWorld (Microsoft, 2025-04)](https://arxiv.org/html/2504.08388v1) — also Minecraft, but uses a **discrete tokenizer** and a strict next-token AR transformer. Open-source.
- [Solaris (2025)](https://solaris-wm.github.io/) — multiplayer Minecraft world model, discrete.
- [DIAMOND (Alonso et al., NeurIPS 2024)](https://diamond-wm.github.io/) — Atari/CSGO, continuous, smaller scope.

These are the right precedents to study when HartsyInference's *second* world model arrives — but they are out of scope for the Oasis port itself.

## Implementation Notes

### How this maps to HartsyInference packages

A brand-new package, **`HartsyInference.World`**, is appropriate for world models (this is Phase 10 in `BUILD_ORDER.md`). Inside:

```
src/HartsyInference.World/
├── HartsyInference.World.csproj
├── Models/
│   ├── Tokenizers/
│   │   ├── IVideoTokenizer.cs                   # new abstraction (see § 6)
│   │   └── OasisVitVae.cs                        # the ViT-VAE class
│   ├── Denoisers/
│   │   ├── OasisDitConfig.cs
│   │   ├── OasisDit.cs                          # DiT-S/2 backbone
│   │   └── DitBlocks/
│   │       ├── SpatioTemporalDitBlock.cs
│   │       ├── SpatialAxialAttention.cs
│   │       └── TemporalAxialAttention.cs
│   └── Actions/
│       ├── OasisAction.cs                       # struct with 23 bools + 2 floats
│       ├── OasisActionEncoder.cs                # → float[25]
│       └── OasisActionKeys.cs                   # the constant list
├── Pipelines/
│   ├── OasisPipeline.cs                         # batch generate-N-frames API
│   └── OasisLivePipeline.cs                     # streaming real-time API (Phase 10 stretch)
├── Schedulers/
│   └── SigmoidBetaSchedule.cs                   # net-new — not in DIFFUSION_SCHEDULERS yet
└── Utilities/
    └── DiffusionForcingDdim.cs                  # v-param DDIM with per-frame noise levels
```

In **`HartsyInference.ModelAssets`**:

```
CheckpointConverters/
└── OasisCheckpointConverter.cs                  # loads oasis500m + vit-l-20 safetensors
```

### Net-new primitives required

Compared to existing HartsyInference infrastructure, Oasis introduces:

1. **`SigmoidBetaSchedule`** — not currently shipped. Easy CPU-side pre-compute (1000 floats); upload once.
2. **`DiffusionForcingDdim`** — extension of standard DDIM with **per-frame independent noise levels** in the timestep tensor (shape `(B, T)` rather than `(B,)`). The DiT forward already accepts this shape; the scheduler just needs to gather `alphas_cumprod[t]` with a per-frame index.
3. **Per-frame v-parameterisation update.** The math in § 7.3 is straight DDIM v-param; the only twist is the "freeze context frames" and "final-step force-clean" mask. Implement as a small vectorised helper.
4. **2-D axial RoPE.** The DiT's spatial attention applies RoPE separately to the H and W axes, sharing the head_dim into two halves of 32 each (`get_axial_freqs(H, W)` returns a tensor of shape `(H, W, 32)` and `apply_rotary_emb` broadcasts it correctly). HartsyInference's existing 1-D RoPE needs an axial variant. Reusable also for future video pipelines.
5. **No new attention kernels.** Both spatial and temporal use stock `F.scaled_dot_product_attention`. Map to our existing `IBackend.ScaledDotProductAttention` (with `is_causal` flag).
6. **No new conv kernels.** The VAE has only one `Conv2d` (the patch embed) — and stride == kernel, so it's a no-overlap patchify, well-served by the existing `Conv2D` path or even by a simple `(unfold + matmul)` substitute.
7. **`IVideoTokenizer` interface.** New abstraction. Continuous variant (Oasis) returns `Tensor latent` of shape `(B, T, C, H, W)`; future discrete variant returns `Tensor<int> indices` of shape `(B, T, H, W)`. The pipeline switches behaviour based on `tokenizer.IsDiscrete`. Even though Oasis is continuous, **define this abstraction now** so MagViT-v2 / MineWorld-style ports later don't force a refactor.
8. **Action-encoder utility.** Small CPU helper that maps `OasisAction` (struct) to `float[25]`. The VPT camera renormalisation lives here.

### What we already have

- `LayerNorm(eps=1e-6, elementwise_affine=False)` — used by DiT-style models everywhere.
- adaLN-zero modulation (`shift, scale, gate` from `SiLU(Linear(c))`) — `AdaLNModulation` already implemented in `DiTUtils`.
- Sinusoidal timestep embedding + 2-layer MLP — `SinusoidalTimestepEmbedding` in `DiTUtils`.
- `Mlp(in, hidden, act=GELU(tanh))` — covered by existing `GeluTanhMlp`.
- `nn.Linear` with no bias on `to_qkv`, with bias on `to_out` — standard.
- Continuous Gaussian VAE plumbing — `DiagonalGaussianDistribution` analogue exists; only the patch-embed / patch-predict heads are new shapes.
- DDIM v-param scheduler — partial; the per-frame index gather + freeze-context mask need to be added.

### Test-skipping discipline

> **Superseded 2026-08-06.** The per-model pipeline/generation tests this section specified were
> removed in the test-suite cleanup, and the rule is now the opposite: **do not add a test that
> proves a model works end to end** — a model that stops working is visible the moment anyone uses
> it. Test what breaks quietly instead (kernel numerics, cross-device equivalence, quantization and
> codec round-trips, padding/tiling geometry, format and key mapping), and put shared-component
> parity in `tests/<Project>/Parity/` with a `*ParityTests` name. See `docs/CODE_STYLE.md` §Testing.

### Layered diff harness for porting

Reuse the SD3.5 / Z-Image-style "dump each layer's input + output to JSON" harness:

1. **VAE encode** — feed sample image, dump `posterior.mean`. Compare C# against Python before touching the DiT.
2. **DiT layer 0 spatial attention** — feed a synthetic latent + zero action, dump output of `s_attn`. Iterate per layer.
3. **DiT layer 0 temporal attention** — same.
4. **DDIM step 0** — feed clean prompt latent + 1 noise frame, run 1 model forward, dump `v`, `x_start`, `x_pred`. Compare scalar-wise.
5. **Full one-step generation** — 1 DDIM step, dump final latent.
6. **Full 10-step single-frame** — full generation of frame index 1.
7. **VAE decode** — round-trip latent → pixels.
8. **Full 32-frame video** — byte-exact MP4.

### VRAM / viability per target GPU

| GPU | VRAM | Oasis-500m (FP16) |
|---|---|---|
| iGPU / 4 GB | 4 GB | Comfortable. Model ≈ 0.7 GB; activations for the default 32-frame window ≈ 1.5 GB. |
| 8 GB consumer | 8 GB | Trivial. Real-time at 20 FPS achievable with the streaming pipeline. |
| 12+ GB | — | Trivial. Headroom for batched generation. |

This is the **smallest production-quality DiT** in HartsyInference. It is the right model for: smoke-testing the world-model pipeline, validating new attention backends (since both spatial and temporal attention are exercised), demonstrating the C# real-time interactive surface, and onboarding new contributors.

### Ordering / dependencies for the build

1. **Land `OasisVitVae`** first — pure ViT, no novel primitives, validates `IVideoTokenizer` interface design.
2. **Land `SpatialAxialAttention` + `TemporalAxialAttention`** using existing `ScaledDotProductAttention` + new 2-D axial RoPE.
3. **Land `SpatioTemporalDitBlock` + `OasisDit`** — the 16-layer backbone.
4. **Land `SigmoidBetaSchedule` + `DiffusionForcingDdim`** — schedule + per-frame v-param DDIM.
5. **Land `OasisPipeline`** — batch generate-N-frames API.
6. **Validate end-to-end against Python reference (byte-exact at FP16).**
7. **Stretch (Phase 10b): `OasisLivePipeline`** — streaming surface for live demos (Unity, web). May involve KV-cache and / or pre-baked per-step kernel pipelines.
