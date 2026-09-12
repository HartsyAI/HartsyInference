# YuE2 — architecture and port contract

YuE2 shares nothing with YuE (v1) but the name. YuE1 is a 7B LLaMA stage-1 LM + 1B stage-2 LM
over xcodec RVQ tokens. YuE2 is an **AR–NAR Mixture-of-Transformers**: one autoregressive LM
plans a score and emits semantic codec tokens, and a second, separately-weighted transformer
flow-matches continuous 64-channel audio latents while attending to the AR model's per-layer
KV cache. Nothing in `YuePipeline`/`YueStage1Lm`/`YueStage2Lm`/`YueTokenizer` is reusable.

Sources of truth, in order:
1. `m-a-p/YuE2-3B` → `yue2_infer-0.1.5-py3-none-any.whl` (the official pipeline: `nar.py`,
   `sampling.py`, `protocol.py`, `pipeline.py`, `modeling_yue2.py`, `modeling_vae.py`).
2. `m-a-p/YuE2-3B` → `yue2_generation_config.json` (the release sampler contract).
3. ComfyUI `b058ec652802e1de24f1d429b3c7c1fb868f791c` (the repack layout we consume).

Weights are **CC BY-NC 4.0** (non-commercial); the code is Apache-2.0.

---

## 1. The checkpoint we support

`Comfy-Org/YuE2` → `checkpoints/yue2_3b_bf16.safetensors` (7.8 GB, 897 tensors) is a single
all-in-one file with three prefixes. There is also `checkpoints/yue2_3b_int8_convrot.safetensors`
(3.96 GB) in the per-layer `.comfy_quant` format we already read for MiniMax-H3/LTX-2.5, and
`audio_encoders/sheetsage2_bf16.safetensors` (1.39 GB) for the audio→ABC cover path.

| Prefix | Tensors | What it is |
|---|---|---|
| `text_encoders.` | 228 | The AR LM (`model.embed_tokens`, 28 layers, `model.norm`, `model.lm_head`) **plus** `text_encoders.yue2_tokenizer_json`, an 8,034,652-byte U8 tensor holding the HF `tokenizers` JSON |
| `model.diffusion_model.` | 234 | The NAR acoustic transformer (28 layers + `vae2llm`, `llm2vae`, `time_embedder`, `latent_pos_embed`) |
| `vae.` | 435 | The Oobleck audio VAE (F32, weight-norm `weight_g`/`weight_v`) |

The tokenizer riding inside the checkpoint is the notable part: no side file, no `qwen.tiktoken`.
`YueTokenizer` cannot be reused — it is a different vocabulary (184,704 vs YuE1's).

### Detection

Comfy keys on all of `vae2llm.weight`, `llm2vae.weight`, `latent_pos_embed.pe`,
`model.layers.0.self_attn.qkv_proj.weight`, `time_embedder.mlp.0.weight` under the
diffusion-model prefix. The VAE variant is keyed on `decoder.layers.6.layers.1.weight_v`
(6 stride layers, vs 5 for Stable Audio / ACE-Step).

---

## 2. Shapes — both transformers are the same Qwen3 geometry

Verified from the safetensors header, and matching `config.json`:

```
hidden_size 2048   layers 28   heads 16   kv_heads 8   head_dim 128
intermediate 6144  vocab 184704   max_position_embeddings 24576
rms_norm_eps 1e-6  rope_theta 1e6   latent_dim 64   timestep_shift 1.0
```

Qwen3 conventions apply to both stacks: merged `qkv_proj` [4096, 2048] = 2048 q ‖ 1024 k ‖ 1024 v,
merged `gate_up_proj` [12288, 2048], per-head `q_norm`/`k_norm` RMSNorm over head_dim=128, SiLU
MLP, no qkv bias, half-split (NeoX) rotary, no RoPE scaling.

The two stacks having identical geometry is *why* the port works: the AR model's per-layer K/V
is handed straight to the matching NAR layer as a prefix.

**NAR-only tensors:** `vae2llm` [2048, 64] + bias, `llm2vae` [64, 2048] + bias,
`time_embedder.mlp.{0,2}` (256→2048→2048, SiLU between), and `latent_pos_embed.pe`, a **stored**
[24576, 2048] learned buffer (not computed).

---

## 3. The three stages

### 3.1 AR stage — score planning and semantic tokens

Special ids (`protocol.py`): `EOD 151643`, `ABC_START 151847`, `ABC_END 151848`,
`MUSIC_START 151851`, `MUSIC_END 151852`, `CODEC_OFFSET 151853`, `CODEC_SIZE 32768`,
`CONTEXT 24576`, `FRAMES_PER_SECOND 25`.

Three chain-of-thought modes, each a literal instruction string prepended to the prompt:

| `cot` | Instruction |
|---|---|
| `off` | `Generate music with codec tokens from the given conditions.` |
| `melody` | `Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.` |
| `full` | `Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.` |

Prompt body: `f"{INSTRUCTION}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"`.

Prefix construction (`token_prefixes`):
- `off` → `[EOD] + encode(text) + [ABC_START, ABC_END, MUSIC_START]`
- `melody`/`full`, planning ABC → `[EOD] + encode(text) + [ABC_START]`
- `melody`/`full`, with ABC in hand → `[EOD] + encode(text) + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]`

Negative prefix for CFG (`negative_prefix`) drops style and lyrics, keeping only the instruction:
- `off` → `[EOD] + encode(INSTRUCTION) + [MUSIC_START]`
- else → `[EOD] + encode(INSTRUCTION) + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]`
  — the **exact** positive-branch ABC ids, not a re-encode.

Logit processing order (`distribution`), which must be reproduced exactly:
1. Upcast to F32 — **except** when `cot == "off"`, where logits stay BF16 (`legacy_off`).
2. Hard allow-mask: ABC phase allows `[0, EOD)`; semantic phase allows
   `[CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE)`. The phase's end token is always allowed.
3. If `step < min_tokens`, force the end token to `-inf`.
4. Windowed repetition penalty over the last `penalty_window` emitted ids, HF convention:
   `alpha = penalty ** count`, then `logit < 0 ? logit * alpha : logit / alpha`.
5. Divide by temperature (skipped when temperature == 1; temperature 0 means argmax).
6. top_k mask.
7. top_p: sort desc, softmax, `cumsum - p > top_p` removes — but the first **3** entries are
   always kept when `legacy_off`, otherwise the first **1**.

CFG, applied to raw logits before step 1: `uncond + scale * (cond - uncond)`, in BF16.

Release sampler defaults (`yue2_generation_config.json`):

| Phase | temperature | top_p | top_k | rep. penalty | window | min_tokens | max_tokens |
|---|---|---|---|---|---|---|---|
| abc | 0.7 | 0.9 | 30 | 1.005 | 100 | 32 | 4096 |
| semantic | 1.0 | 0.95 | 100 | 1.2 | 50 | 200 | 9000 |

`cfg_scale` defaults to **1.01 when `cot == "off"`**, else **1.0** (no CFG branch at all).
Default seed in the official `SongRequest` is **831001**; the same seed drives the ABC phase,
the semantic phase, and the NAR noise draw.

Semantic output is converted to codec values by subtracting `CODEC_OFFSET`; 25 tokens = 1 second.

### 3.2 NAR stage — flow matching over audio latents

`nar.py` is the authority. Per acoustic chunk:

1. **Noise**: drawn **once for the whole song** on CPU in F32 —
   `torch.randn((frames, 64), generator=Generator("cpu").manual_seed(seed))` — then sliced per
   chunk. Drawing per-chunk would change every result.
2. **AR prefill**: run the AR stack over `prefix + [codec + CODEC_OFFSET] + [MUSIC_END]`,
   capturing each layer's **post-RoPE, post-q/k-norm** K and V at AR positions `0..ar_length`.
3. **NAR forward**, per ODE evaluation:
   - `x_nar = pad(state, (0,0,1,1))` — one zero frame at each end, so `nar_length = frames + 2`.
   - `x = vae2llm(x_nar) + time_embedder(t_shifted).expand(nar_length) + latent_pos_embed(0..nar_length-1)`.
     Note the two position schemes: the learned embedding is **chunk-relative from 0**, while
     RoPE positions **continue after the prefix**, `ar_length .. ar_length + nar_length`.
   - Each layer: `q,k,v = nar_self_attn.project_qkv(nar_input_layernorm(x))`, then
     `k = cat(ar_k, k)`, `v = cat(ar_v, v)`, **non-causal** attention over prefix ‖ chunk,
     `x += nar_self_attn.o_proj(...)`, `x += nar_mlp(nar_pre_mlp_layernorm(x))`.
   - `velocity = llm2vae(norm(x))[1:-1]` — the padding frames are dropped.
4. **ODE**: 32 steps, **midpoint**, integrating `t` downward from 1.0:
   ```
   dt = 1/steps
   for step in range(steps):
       t    = 1 - step*dt
       raw  = clamp(logit(t), -20, 20)              # computed in F64 on CPU
       mid  = state - velocity(state, raw) * dt/2
       raw2 = clamp(logit(t - dt/2), -20, 20)
       state = state - velocity(mid, raw2) * dt
   ```
   The model then recovers `t` as `sigmoid(raw)` in the **compute dtype** and applies
   `shift*t/(1+(shift-1)*t)` with `shift = 1.0` (identity). So the DiT sees `t ∈ (0,1]`
   directly — which is what Comfy's `sampling_settings = {"multiplier": 1.0}` encodes — but
   parity requires the logit/sigmoid round-trip in BF16, not a raw `t`.

**Chunking**: `size = (CONTEXT - prefix_tokens - 3) // 2`, chunks of that many frames. With a
typical prefix this is ~12k frames (~8 min), so a single chunk covers the 360 s maximum.
Implement it, but single-chunk is the path that actually runs.

### 3.3 VAE — Oobleck, 48 kHz stereo

Derived from the decoder kernel widths (`k = 2*stride`): decoder strides `[6,5,4,4,2,2]`, so
encoder strides `[2,2,4,4,5,6]`, product **1920** = one latent frame per 40 ms at 48 kHz (25 fps).
`channels 64`, `c_mults [1,2,4,8,16,32]` → 64/128/256/512/1024/2048, `latent_dim 64`,
`in_channels 2` (stereo), SnakeBeta activations (`alpha`/`beta`), weight-norm parameterization,
F32 throughout.

**The odd stride — do NOT follow ComfyUI here.** The upstream `modeling_vae.py` builds every
`WNConvTranspose1d` as `(k=2s, stride=s, padding=ceil(s/2))` with **`output_padding` left at 0**, so a
layer's output length is `L·s + s − 2·ceil(s/2)`: an exact `L·s` for even strides, but `5L − 1` for the
stride-5 layer. Decoding 8 latent frames therefore yields

```
8 →(s6) 48 →(s5) 239 →(s4) 956 →(s4) 3824 →(s2) 7648 →(s2) 15296 samples
```

— **15,296, not 8 × 1920 = 15,360**, confirmed against the reference dump. The Comfy diff *adds*
`output_padding = stride % 2`, which forces the clean `L·1920` its latent framework assumes and so
produces 64 more samples than the model's own decoder. The weights are identical; only the length
differs (~1.3 ms). We follow the original, which is also what our existing
`Audio/Models/Codecs/Oobleck/OobleckDecoder` already computes —
`tUp = (t−1)·stride + 2·stride − 2·pad` with `pad = (stride+1)/2` is the same formula. **No decoder
change and no backend `output_padding` support is needed.**

- **Encode is mean-only.** YuE2 sets `sample_latent=False`: `encode` returns
  `encoder(x).chunk(2, dim=1)[0]`, not a sampled bottleneck. Only matters for the cover path.

Decode is tiled: `core_frames = 1024` (512 under a ≤12 GiB budget), `halo_frames = 16`,
halo-crop, output clamped to [-1, 1].

---

## 4. Parameter surface to expose

`MusicRequest` already carries `Seed`, `CfgScale`, `Temperature`, `TopK`, `TopP`,
`RepetitionPenalty`, `InferSteps`, `Duration`. YuE2 needs these added:

| Parameter | Default | Notes |
|---|---|---|
| `cot` | `full` | `full` \| `melody` \| `off`. Empty ABC forces `off`. |
| `abc` | `""` | External/edited ABC score; supplying one skips the planning pass. |
| `abc_temperature` / `abc_top_p` / `abc_top_k` / `abc_repetition_penalty` / `abc_max_tokens` | 0.7 / 0.9 / 30 / 1.005 / 4096 | The ABC phase samples differently from the semantic phase — one shared set of knobs is wrong. |
| `penalty_window` | 100 abc / 50 semantic | Not currently expressible. |
| `min_tokens` | 32 abc / 200 semantic | Guards against instant EOS. |
| `ode_steps` | 32 | Maps to `InferSteps`. |

`Duration` maps to `max_tokens = round(seconds * 25)` for the semantic phase, capped at 360 s.
Per `param-naming-no-vendor-prefix`, these are named for the model family, not the vendor.

---

## 5. Parity gates

Ordered so each one only fails for its own reason. Python references dumped from the official
wheel into `tests/python-reference/`, run against a scratch venv — **never** SwarmUI's bundled
ComfyUI, which is a live service.

| Gate | Checks |
|---|---|
| A1 tokenizer | Embedded `yue2_tokenizer_json` → ids for all three `cot` modes, matching `qwen.tiktoken` encode |
| A2 prefix | `token_prefixes` / `negative_prefix` byte-for-byte for `off`/`melody`/`full`, with and without external ABC |
| A3 AR logits | Greedy (temperature 0) prefill + 8 decode steps vs dumped logits — no multinomial RNG in the loop |
| A4 distribution | `distribution()` on canned logits: allow-mask, min_tokens gate, penalty window, top_k/top_p, both `legacy_off` branches |
| A5 KV export | Per-layer post-RoPE K/V from the AR prefill, all 28 layers |
| B1 NAR step | One `velocity()` call against a dumped state/t |
| B2 ODE | Full 32-step midpoint solve on a short fixed chunk |
| C1 VAE decode | Latents → 48 kHz stereo, and Stable Audio unchanged after the `output_padding` edit |
| C2 VAE encode | Mean-only encode (cover path) |
| D e2e | CLI and API, seed-fixed, against the reference render |

## 6. Deliberately out of scope for the first pass

- **SheetSage2 + MERT2** (`audio_encoders/sheetsage2_bf16.safetensors`): audio → ABC
  transcription for zero-shot covers. A separate encoder model; a follow-up.
- **`int8_convrot`**: the bf16 checkpoint lands first, then the quant variant through the
  existing per-layer `.comfy_quant` reader.
- **AudioLab / SwarmUI extension wiring**: a follow-up, and it needs an engine publish first.
