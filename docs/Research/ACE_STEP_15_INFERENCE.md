# ACE-Step 1.5 (turbo) — Inference Architecture, From Source

> Status: Complete | Last Updated: 2026-06-10 | Needed Before: SharpInference.Audio (ACE-Step 1.5 pipeline)
>
> Supersedes the v1.5 sections of [ACE_STEP_ARCHITECTURE.md](ACE_STEP_ARCHITECTURE.md) (§2.5, §8) which were
> written from configs only. This doc is extracted from the **actual shipped reference code** —
> `ACE-Step/Ace-Step1.5` on HF: `acestep-v15-turbo/modeling_acestep_v15_turbo.py` (2135 lines),
> `configuration_acestep_v15.py`, plus safetensors **header key dumps** of the main model and VAE
> (fetched via HTTP range requests — no full downloads needed).
>
> **Why this matters:** SwarmUI's only supported audio-gen model class is `ace-step-1_5`
> (T2IModelClassSorter detection key `encoder.lyric_encoder.layers.0.post_attention_layernorm.weight` —
> matches the key dump below). This is the build target for audio in the SwarmUI extension.

## 1. Corrections to prior research

The old doc assumed v1.5 is "a Qwen3-style decoder LM whose output is FSQ-quantized audio tokens" needing
dotLLM. **The shipped turbo model is NOT that.** It is:

1. **A flow-matching DiT** (`AceStepDiTModel`, velocity prediction, 8-step Euler ODE) over **continuous
   25 Hz / 64-channel Oobleck-VAE latents** — diffusion-shaped, not causal-LM-shaped.
2. The FSQ machinery (`AceStepAudioTokenizer` / `AudioTokenDetokenizer`) exists **inside the same
   checkpoint** but is used for the *LM-hints / cover* path (5 Hz codes ↔ 25 Hz latents), not as the
   model's output representation.
3. The **VAE is `AutoencoderOobleck`** — the *Stable Audio Open* VAE, verbatim (stereo, 48 kHz,
   downsampling ratios 2·4·4·6·10 = 1920× → 64-dim latents @ 25 Hz, snake activations, weight-normed
   convs). This is why ComfyUI loads `ace-step-15-vae` through its LTX *audio* VAE loader. Building
   Oobleck once serves ACE-Step 1.5, Stable Audio Open, and LTX-2 audio.
4. **Turbo uses NO CFG and no APG** — `fix_nfe=8` Euler steps over a **hardcoded timestep table**
   (see §4). The PingPong/APG/CFG-Zero★ work items only matter for v1 (3.5B) and v1.5 base/sft.
5. **Every component is a Qwen3-style block** (RMSNorm, GQA 16:8, head_dim 128, q/k-norm, RoPE θ=1e6,
   alternating sliding(128)/full attention) — the same block family SharpInference already runs for
   Anima (Qwen3-0.6B) and VibeVoice (Qwen2.5). No dotLLM dependency.

## 2. Model zoo (HF `ACE-Step/Ace-Step1.5`, one repo = all four parts)

| Part | Folder | Comfy side-file name | Role |
|---|---|---|---|
| Main model (2B) | `acestep-v15-turbo/model.safetensors` (677 keys) | the user's checkpoint (`ace-step-1_5` class) | DiT + condition encoders + FSQ tok/detok |
| Text encoder | `Qwen3-Embedding-0.6B/` | `qwen_0.6b_ace15.safetensors` | prompt/tags AND lyrics → 1024-d hidden states |
| 5 Hz planner LM | `acestep-5Hz-lm-1.7B/` | `qwen_1.7b_ace15.safetensors` | optional autoregressive audio-code generation (Comfy's `generate_audio_codes`, temperature/top_p/top_k/cfg_scale≈2) |
| VAE | `vae/diffusion_pytorch_model.safetensors` (365 keys) | `ace_step_1.5_vae` (CommonModels "ace-step-15-vae") | Oobleck: latent [B,T@25Hz,64] ↔ stereo 48 kHz wav |
| Asset | `acestep-v15-turbo/silence_latent.pt` | (bundled in Comfy's checkpoint?) | silence latent used as `src_latents` padding/base — **.pt, needs offline conversion or recompute** (encode 1 s of silence through the VAE encoder) |

## 3. Component graph (from `modeling_acestep_v15_turbo.py`)

```
prompt/tags ─► Qwen3-Embedding-0.6B ─► text_hidden_states [B,T,1024] ─► encoder.text_projector (1024→2048, no bias)
lyrics      ─► Qwen3-Embedding-0.6B ─► lyric_hidden_states [B,L,1024] ─► encoder.lyric_encoder
                                                            (Linear 1024→2048 + 8 × AceStepEncoderLayer, bidirectional)
ref audio   ─► VAE-encode → acoustic 64-d ─► encoder.timbre_encoder (Linear 64→2048 + CLS token + 4 layers)  [optional]
            └► all three packed into one cross-attn sequence (encoder_hidden_states, encoder_attention_mask)

src_latents [B,T,64]  (plain T2M: silence latent tiled to duration; cover: lm_hints)
chunk_masks [B,T,64?] (repaint mask; all-ones region = generate)
context_latents = concat([src_latents, chunk_masks], dim=-1)            # channels 128

noise x_t [B,T,64] ─► decoder (AceStepDiTModel):
    proj_in: Conv1d(in=192, out=2048, k=2, s=2)     # input = concat([x_t, context_latents]) → 192ch; patch 2 → 12.5 Hz tokens
    24 × AceStepDiTLayer (self-attn GQA sliding/full alternating + cross-attn to encoder seq + SwiGLU MLP,
                          AdaLN-6 from decoder.scale_shift_table (GLOBAL, 1 table) + per-step temb)
    time_embed + time_embed_r: TimestepEmbedding(256→2048) × 2  (t and r — turbo passes r=t)
    norm_out (RMSNorm) + adaLN shift/scale → proj_out: Conv1dTranspose-equivalent de-patchify → [B,T,64] velocity
8-step Euler: x ← x + (σ_next − σ) · v        # σ from the fixed shift table, §4

z_0 [B,T,64] ─► AutoencoderOobleck.decode ─► wav [B, 2, T·1920] @ 48 kHz  ─► mp3 (Comfy: SaveAudioMP3 V0)
```

FSQ hint path (cover / LM-codes mode only): `tokenizer` pools 25 Hz latents ×5 → FSQ (levels [8,8,8,5,5,5],
ResidualFSQ project_in/out, 64 000 codes) → 5 Hz indices; `detokenizer` expands each code ×5 with learned
special_tokens + 2 encoder layers → 25 Hz `lm_hints` used as `src_latents` when `is_covers=1`. The 1.7B LM
generates those 5 Hz indices autoregressively from lyrics/tags (Comfy's default flow sets
`generate_audio_codes=true`).

## 4. Sampling (turbo) — exact constants

`generate_audio(..., fix_nfe=8, infer_method="ode", shift=3.0)`:

```
VALID_SHIFTS = [1.0, 2.0, 3.0]    # shift snapped to nearest
SHIFT_TIMESTEPS[3.0] = [1.0, 0.9545454545, 0.9, 0.8333333333, 0.75, 0.6428571428, 0.5, 0.3]
SHIFT_TIMESTEPS[2.0] = [1.0, 0.9333333333, 0.8571428571, 0.7692307692, 0.6666666666, 0.5454545454, 0.4, 0.2222222222]
SHIFT_TIMESTEPS[1.0] = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
# final step integrates to 0; custom timestep lists are snapped to a 20-value whitelist
```

Flow convention: `x_t = t·noise + (1−t)·data`, model predicts `v = noise − data`, Euler steps toward t=0.
No CFG (turbo); `null_condition_emb` exists for base/sft CFG and for unconditional drops at train time.

## 5. Safetensors key map (ground truth, from header dumps)

### Main model (677 keys, BF16)

```
decoder.proj_in.1.{weight,bias}                    # Conv1d 192→2048 k2 s2 (Sequential index 1)
decoder.time_embed.{linear_1,linear_2,time_proj}.{weight,bias}      # ×2 with time_embed_r
decoder.condition_embedder.{weight,bias}           # 2048→2048
decoder.scale_shift_table                          # global AdaLN table
decoder.layers.{0..23}.self_attn.{q,k,v,o}_proj.weight + {q,k}_norm.weight
decoder.layers.{0..23}.cross_attn.{q,k,v,o}_proj.weight + {q,k}_norm.weight
decoder.layers.{0..23}.mlp.{gate,up,down}_proj.weight + *layernorm*  (19 keys/layer)
decoder.norm_out.weight
decoder.proj_out.1.{weight,bias}
encoder.text_projector.weight                      # 1024→2048, NO bias
encoder.lyric_encoder.embed_tokens.{weight,bias} + .norm.weight + .layers.{0..7}.* (91 keys)
encoder.timbre_encoder.* (48 keys, incl. special_token)
tokenizer.audio_acoustic_proj.{weight,bias}        # 64→2048
tokenizer.attention_pooler.* (2 layers, num_attention_pooler_hidden_layers=2)
tokenizer.quantizer.project_in/out.{weight,bias}   # ResidualFSQ
detokenizer.embed_tokens/norm/special_tokens/proj_out + .layers.{0..1}.*
null_condition_emb
```

SwarmUI's class detection key `encoder.lyric_encoder.layers.0.post_attention_layernorm.weight` ✓ present.

### VAE (365 keys) — Oobleck, weight-normed

```
encoder.conv1.{bias,weight_g,weight_v}             # weight norm: fold W = g · v/|v| at load
encoder.block.* / decoder.block.*                  # 175 keys each: res units (snake1/snake2 alpha+beta,
                                                   #   conv1/conv2 g/v) + stride up/down convs (conv_t1)
encoder.snake1.{alpha,beta}, encoder.conv2.*       # final layers; decoder mirrors
```

Snake activation: `x + (1/α)·sin²(α·x)` (with β variant: `x + (1/(β+ε))·sin²(α·x)`). Weight-norm folding
is a load-time transform (same trick as the YOLO BN-fold converter).

## 6. Config (turbo 2B, verbatim essentials)

```
hidden_size 2048 | intermediate_size 6144 | num_hidden_layers 24 | heads 16 | kv_heads 8 | head_dim 128
sliding_window 128, layer_types alternating sliding/full | rope_theta 1e6 | rms_norm_eps 1e-6
in_channels 192 (= 64 noisy + 64 src + 64 chunk_mask) | patch_size 2 | pool_window_size 5
fsq_dim 2048, fsq_input_levels [8,8,8,5,5,5], num_quantizers 1 | audio_acoustic_hidden_dim 64
text_hidden_dim 1024 (Qwen3-Embedding-0.6B) | timbre_hidden_dim 64, timbre_fix_frame 750
num_lyric_encoder_hidden_layers 8 | num_timbre_encoder_hidden_layers 4 | num_attention_pooler_hidden_layers 2
vocab_size 64003 (5Hz LM codes; main model itself has no token embedding) | is_turbo true
VAE: AutoencoderOobleck — audio_channels 2, ratios [2,4,4,6,10] (×1920), decoder_input_channels 64,
     channel_multiples [1,2,4,8,16], encoder_hidden 128, sampling_rate 48000
Latent math: T_latent = duration_s × 25 ; samples = T_latent × 1920
```

## 7. Build plan for SharpInference (everything has an in-repo precedent)

| # | Component | Reuses | New work |
|---|---|---|---|
| 1 | **Oobleck VAE decoder (+encoder)** — ✅ BUILT 2026-06-10 (`SharpInference.Audio/Models/Codecs/Oobleck/`, `OobleckVae` facade, `AceStep15` + `StableAudioOpen` presets, structural tests green; numerics validation-pending) | `IBackend.Snake` (beta variant), `WeightNormFusion`, Conv1d/ConvTranspose1d — all existed | logscale snake exp at load. **Trap found:** diffusers transpose convs use torch's default dim-0 weight_norm (g = `[C_in,1,1]`, header-verified) → use `WeightNormFusion.Fuse`, NOT EnCodec's per-out-channel `WeightNormFusionT` |
| 2 | **Qwen3-Embedding-0.6B encoder** — ✅ DONE 2026-06-10 via pure reuse: `LlamaStyleEncoderConfig.Qwen3_Embedding_0_6B` preset (= `Qwen3_0_6B with { VocabSize = 151669, EosTokenId = 151643 }`, config-verified). No pooling — ACE consumes per-token hidden states. | `LlamaStyleEncoder` (Anima's path) + embedded Qwen3 tokenizer | none — 3-line preset. Pipeline step must mirror ACE's prompt formatting (instruct template?) from the GitHub pipeline code when wiring step 5 |
| 3 | **AceStep DiT (24L)** | Qwen-style GQA attention + RMSNorm + RoPE from existing DiTs; AdaLN-6 | sliding-window attention mask; dual timestep embed; context-latent channel concat; Conv1d patchify |
| 4 | **Condition encoders** (lyric 8L, timbre 4L) | same Qwen3 encoder layer | packing of [text, lyric, timbre] cross-attn sequence |
| 5 | **Sampler** | flow-match Euler exists | trivial: fixed 8-step tables from §4, no CFG |
| 6 | **silence_latent** | — | convert .pt offline OR VAE-encode silence at first run |
| 7 | **FSQ tok/detok + 1.7B 5Hz LM** | `Fsq.cs` primitives; VibeVoice runs 24L Qwen LMs with sampling | PHASE 2 — plain T2M works without it (src=silence, is_covers=0); needed for Comfy-parity code-hints quality + cover mode |
| 8 | **Checkpoint converter** | standard prefix routing | key map in §5 — main file buckets {decoder, encoder, tokenizer, detokenizer}; VAE separate |
| 9 | **SwarmUI extension loader** | video loader pattern | params: `Text2AudioDuration/Style/BPM/TimeSignature/Language/KeyScale` (gated by `text2audio`, client-derived from the model class); output: wav → mp3 via ffmpeg (mirror `SaveAudioMP3` V0); `MediaType.AudioMp3` |

Suggested order: 1 → 2 → 3+4 → 5+6 (T2M end-to-end, hints-less) → 8 → 9 → 7 (code-hints parity).
Open question for phase 1: whether hints-less T2M (silence src_latents) matches Comfy quality — Comfy
defaults `generate_audio_codes=true`, so the LM hints may be load-bearing; validate early with the
real checkpoint before deciding phase 7 priority.
