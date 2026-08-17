# LTX 2.5 — Research Notes

> Status: **Runs end-to-end on real weights, prompt-faithful** | Last Updated: 2026-08-12

`hartsy video -m ltx-2.5` generates prompt-faithful 704×480×25f clips with a soundtrack on the official
`int8_lean_convrot` DiT + Gemma-4-12B pair (4090, ~80 s). The whole text path is verified against ComfyUI 0.32
on the real checkpoints — tokenizer ids byte-exact, Gemma-4 tower cosine 0.9999–1.0000 per layer, connector
within 0.2–0.6%, `attn2` within 0.07%, and the conditioning's prompt-to-prompt delta at ratio 1.00. The one
real defect found on the way is written up under "Prompt-independent output" below.

## Invariants worth not re-deriving

**Ungated conv VAE** (Lightricks 401s even on a byte range; both verified reachable 2026-08-12, `gated: false`):

```
https://huggingface.co/ChrisColeTech/LTX-2.5-turbo-GGUF/resolve/main/split/vae/ltx-2.5-video-vae-conv-bf16.safetensors
https://huggingface.co/dummy9996/LTX-2.5-22b-ungate/resolve/main/ltx-2.5-video-vae-conv-bf16.safetensors
```

The first is where the staged diffusion VAE came from, and that file was header-identical to the original.

**Decode through the conv VAE, not the diffusion decoder.** The diffusion decoder is parity-checked but
managed-only — there is no CUDA `Na3d` kernel — and it needs ~32 s for the smallest legal frame
(`[1,3,1,64,64]`). Real-resolution decoding is out of reach until that kernel exists, and the dominant cost is
`MatMulKernels.LinearTransB` re-casting a non-F32 weight on every call, not the attention itself.

- Gemma-4 discriminator: `conv.TextEncoder.ContainsKey("model.layers.0.layer_scalar")`. Do **not** probe for a
  missing `v_proj` — layer 0 is a sliding layer and has one.
- `Gemma4TextEncoder.EncodeMultiLayer` was written to the same signature and layer-outer `[B, seq, K·H]` layout
  as `LlamaStyleEncoder`, so the tower swap is meant to be a one-line change.
- Conditioning length: the recipe's `TokenLength` is 256, but ComfyUI pads Gemma-4 LTX conditioning to
  **1024**, and sequence length is part of the conditioning because of the connector's learnable-register
  replacement. `Gemma4Tokenizer.BuildConditioningSequence` already right-pads to 1024.
- `conv.VaeDiffusionDecoder` holds the `decoder.*` keys with the prefix kept; the latent statistics stay in
  `conv.Vae` as `latents_mean`/`latents_std`, since both decoders un-normalize identically.
- Distilled vs dev is not detectable from any checkpoint — it arrives via the `ltx-2.5-distilled` catalog id.

Both of the divergences this doc used to flag as open (the 49th state's final norm and the padding side) are
settled below — neither was a defect.

### Prompt-independent output — FIXED: prompt_adaln was driven by the raw sigma, not the scaled timestep

**Root cause**: `LtxVideo2Transformer` fed `prompt_adaln_single` / `audio_prompt_adaln_single` the raw flow sigma
(0..1) instead of the ×1000-scaled timestep every other modulator uses. The reference is explicit
(`av_model.py::_prepare_timestep`):

```python
timestep_scaled = timestep * self.timestep_scale_multiplier          # sigma * 1000
v_prompt_timestep = compute_prompt_timestep(self.prompt_adaln_single, timestep_scaled, ...)
a_prompt_timestep = compute_prompt_timestep(self.audio_prompt_adaln_single, a_timestep_scaled, ...)
```

`prompt_adaln` produces `shift_kv`/`scale_kv`, which modulate the **text keys and values** feeding every block's
cross-attention (`context * (1 + scale_kv) + shift_kv`). Evaluating a sinusoidal timestep embedding at t≈1
instead of t≈1000 lands somewhere unrelated on the embedding, so the text K/V were mis-modulated at every block
and every step. The symptom was not a dead cross-attention — magnitudes looked right — but a cross-attention
that could not *discriminate*: output followed the seed, and two unrelated prompts differed by ~1% of pixel
range. A stale comment in the code asserted the opposite convention, which is what made this survive.

**Verification** (ComfyUI 0.32 headless, real int8-convrot checkpoints, same latent + timestep + conditioning):

| quantity | before | after |
| --- | --- | --- |
| per-block prompt sensitivity vs reference (blocks 2-47) | 8-13x too weak | **ratio 1.00** |
| `\|encMod\|` at block 0 (ref 0.3906) | 0.37318 | **0.39131** |
| two unrelated prompts, one seed | ~1% pixel delta | renders each prompt |

Everything upstream was verified correct before the DiT was suspected, and none of it needed changing:
tokenizer ids byte-exact; Gemma-4 tower cosine 0.9999-1.0000 per layer including the global layers; connector
output within 0.2-0.6%; `attn2` within 0.07%; conditioning prompt-delta ratio 1.00. An earlier commit
(`5ad864c2`) blamed register padding and prescribed trimming the conditioning to the real token count — that
diagnosis was wrong and the fix would have moved this engine away from upstream, because
`Embeddings1DConnector.forward` re-pads to 1024 with tiled registers itself.

**Two traps that cost real time here**, worth knowing for the next parity harness:

- int8 weights live under the *same* `.weight` key as an ordinary tensor. A loader that checks the raw key
  before its `.weight_scale` sibling silently loads unscaled, un-rotated int8 and produces plausible garbage.
- `final_layer_norm_intermediate` is `self.layer_norm_hidden_state`, which the LTX path sets to **False** — the
  final `model.norm` is NOT applied to the 49-layer stack. Applying it makes every layer mismatch.

### Running the encoder against the int8 text encoder

`Gemma4TextEncoder`'s embedding gather reads the weight as a raw `float*` after `CastToF32IfNeeded`, so a
packed embedding table would yield silent garbage. It does not happen: the int8-convrot TE quantizes **only the
projection matrices**. From its header,

```
model.embed_tokens.weight                               BF16  [262144, 3840]
model.norm.weight                                       BF16  [3840]
model.layers.0.layer_scalar                             BF16  [1]
text_embedding_projection.video_aggregate_embed.weight  BF16  [4096, 188160]
model.layers.0.self_attn.q_proj.weight                  I8    [4096, 3840]
dtype histogram: F32 328, BF16 353, I8 328, U8 333
```

Embeddings, norms, `layer_scalar` and the connector projection all stay BF16, so the gather and the norms are
safe as written and the int8 path is confined to `Linear`.

The residual caveat is that the encoder's parity (3.9e-7 sliding / 1.1e-6 global) was measured on **F32 weights
on CPU**. Resident-int8 projections are an untested combination — not suspected broken, just unverified. Check
a couple of hidden states against the BF16 tower before treating a poor generation as a wiring bug.

### Raising the conditioning length to 1024

The pipeline was tuned at 256 tokens, so this is not a free change:

- The 49-layer feature stack is ~771 MB in F32 at 1024 tokens (`1024 × 3840 × 49 × 4`), and the prompt cache
  holds both the positive and the negative, so budget roughly 1.5 GB. The resident-prefix VRAM headroom in
  `LtxVideo2Pipeline` was sized against 256 — re-check it rather than assuming it scales.
- 1024 is a clean multiple of the connector's 128 learnable registers, so register replacement is unaffected.
- `LtxVideo2Recipe.TokenLength` is an `internal const` at 256. It has to become per-branch: bumping it globally
  changes the Gemma-3 conditioning length too, silently altering the verified 2.3 output.

## Summary

LTX 2.5 (Lightricks, released 2026-08-11) is a point release of the same dual-stream audio-video DiT the
engine already runs as LTX-2.3 (`LtxVideo2Transformer` + `LtxVideo2VaeDecoder` + `LtxAudioVaeDecoder`). The
transformer architecture is very nearly unchanged: diffing the real checkpoints' `__metadata__.config.transformer`
objects yields **exactly two functional differences** — the video-branch feed-forward drops its biases
(`ff_bias: false`), and a new learned `keyframes_abs_pos_embedding` parameter is added. Every other
field — layer count, head geometry, rope flavor, gated attention, cross-attention AdaLN, connector shape,
scheduler block — is byte-identical to 2.3.

The real work is in the surrounding components. The **text encoder changes family** from Gemma 3 12B to a
Gemma 4 12B fine-tune whose global attention layers use a different head dim, a single KV head, no `v_proj`
at all (K is reused as V), and a 25%-partial rotary — a geometry that cannot be expressed by the engine's
`LlamaStyleEncoderConfig`. Separately, 2.5 ships a second video decoder, `NADiffusionDecoder`, which replaces
convolutional upsampling with 3D neighborhood-attention transformer stages plus a single-step x0 diffusion
stage; that needs a neighborhood-attention primitive the engine does not have. The conv video VAE and the
audio VAE are structurally identical to 2.3 and need no code.

## Key numbers

### Transformer config diff — 2.3 vs 2.5 (the complete list)

Both read from the shipped checkpoints' safetensors `__metadata__.config.transformer`.
2.3 source: `Kijai/LTX2.3_comfy` `ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors` (`model_version=2.3.0`) —
the exact file the `ltx-2` catalog entry downloads. 2.5 source: `ltx-2.5-22b-distilled-transformer-bf16.safetensors`
(`model_version=2.5.0`).

| key | 2.3 | 2.5 | effect |
|---|---|---|---|
| `ff_bias` | *absent* (⇒ `true`) | `false` | video-branch `transformer_blocks.N.ff` Linears lose `.bias` |
| `use_keyframes_abs_pos_embedding` | *absent* (⇒ `false`) | `true` | adds `keyframes_abs_pos_embedding [1, 4096]` |
| `text_encoder_norm_type` | `per_token_rms` | `PER_TOKEN_RMS` | letter case only — same value |

Nothing else differs. Shared by both: `num_layers 48`, `num_attention_heads 32`, `attention_head_dim 128`,
`in_channels`/`out_channels` 128, `cross_attention_dim 4096`, `caption_channels 3840`,
`audio_num_attention_heads 32`, `audio_attention_head_dim 64`, `audio_cross_attention_dim 2048`,
`audio_out_channels 128`, `positional_embedding_theta 10000.0`,
`positional_embedding_max_pos [20, 2048, 2048]`, `audio_positional_embedding_max_pos [20]`,
`timestep_scale_multiplier 1000`, **`av_ca_timestep_scale_multiplier 1000.0`**, `rope_type split`,
`frequencies_precision float64`, `norm_eps 1e-06`, `qk_norm rms_norm`, `standardization_norm rms_norm`,
`apply_gated_attention true`, `connector_apply_gated_attention true`, `cross_attention_adaln true`,
`av_cross_ada_norm true`, `causal_temporal_positioning true`, `use_middle_indices_grid true`,
`caption_proj_before_connector true`, `caption_projection_first_linear false`,
`caption_projection_second_linear false`, `caption_proj_input_norm false`,
`connector_num_layers 8`, `connector_num_attention_heads 32`, `connector_attention_head_dim 128`,
`audio_connector_attention_head_dim 64`, `connector_num_learnable_registers 128`,
`connector_positional_embedding_max_pos [4096]`, `connector_norm_output true`.

Scheduler block is identical in both: `RectifiedFlowScheduler`, `num_train_timesteps 1000`,
`sampler LinearQuadratic`, `shifting null`, `base_resolution null`.

**`av_ca_timestep_scale_multiplier` is 1000.0 in both**, so the engine's hardcoded 1000 is correct. The
official `LTXModelConfigurator` defaults this field to `1`, which is misleading — do not "fix" the engine
toward that default.

### What the config keys do NOT say (settled by tensor keys instead)

Three flags that the ComfyUI/official plumbing exposes are **absent from both checkpoints' configs**, so
they take their defaults, and the tensor keys confirm it:

- `use_prompt_adaln_single` ⇒ `true`. Keys `prompt_adaln_single.*` and `audio_prompt_adaln_single.*` are
  **present** in 2.5. There is no timestep-independent cross-attention K/V in this checkpoint.
- `audio_ff_bias` ⇒ `true`. `transformer_blocks.N.audio_ff.net.0.proj.bias` and `.net.2.bias` are present.
- Connector FFN bias ⇒ `true`. `{video,audio}_embeddings_connector.transformer_1d_blocks.N.ff.net.{0.proj,2}.bias`
  are present.

So `ff_bias: false` removes biases from **exactly one** module family: the video branch's
`transformer_blocks.N.ff` (`ff.net.0.proj.weight [16384,4096]`, `ff.net.2.weight [4096,16384]`, no `.bias`).

### Gemma 4 12B text encoder

From `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` → `__metadata__.gemma_config.text_config`, cross-checked
against per-layer tensor shapes.

| field | value |
|---|---|
| `hidden_size` | 3840 |
| `num_hidden_layers` | 48 |
| `intermediate_size` | 15360 |
| `num_attention_heads` | 16 |
| `head_dim` (sliding) | 256 |
| `global_head_dim` | 512 |
| `num_key_value_heads` (sliding) | 8 |
| `num_global_key_value_heads` | 1 |
| `attention_k_eq_v` | `true` — global layers have no `v_proj`; V is the K projection |
| `attention_bias` | `false` |
| `sliding_window` | 1024 |
| `rms_norm_eps` | 1e-6 |
| `hidden_activation` | `gelu_pytorch_tanh` |
| `vocab_size` | 262144 |
| `tie_word_embeddings` | `true` — no `lm_head` tensor in the file |
| `bos_token_id` / `pad_token_id` / `eos_token_id` | 2 / 0 / 1 |
| `final_logit_softcapping` | 30.0 (unused for encoding) |
| `hidden_size_per_layer_input` | 0 — the Gemma-3n per-layer-input mechanism is off |
| `num_kv_shared_layers` | 0 |

`layer_types` is an explicit 48-entry list of 5×`sliding_attention` then 1×`full_attention`, cycled, so
**layer index % 6 == 5 is global**. Matches ComfyUI's `sliding_attention = [1024,1024,1024,1024,1024,False]`
indexed by `index % 6` (`comfy/text_encoders/gemma4.py:104`).

RoPE is per-layer-type (`rope_parameters`):
- `sliding_attention`: `rope_theta 1e4`, `rope_type default` — full rotary over `head_dim` 256.
- `full_attention`: `rope_theta 1e6`, `rope_type proportional`, `partial_rotary_factor 0.25`.

### Distilled sampling schedule

`packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:17-25`:

```
DISTILLED_SIGMA_VALUES  = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]   # 8 steps
STAGE_2_DISTILLED_SIGMA = [0.909375, 0.725, 0.421875, 0.0]                                          # 2x upscale stage
TDP_DISTILLED_SIGMAS    = [0.625, 0.4, 0.0]                                                          # multi-GPU tiled runner
```

> **The shipped ComfyUI 2.5 templates do NOT use that stage-2 array.** All three `video_ltx2_5_*` templates'
> refine-pass ManualSigmas literal is `0.85, 0.7250, 0.4219, 0.0` (verbatim — 0.4219, not 0.421875; ManualSigmas
> parses the text as written). The 0.909375 head above is the LTX-2.0-lineage refine, still visible in
> `video_ltx2_t2v_distilled.json`. Two more template facts the constants file does not record: the refine pass's
> noise seed is hardcoded `42, fixed` (only the base pass gets the user seed), and both passes select
> `euler_ancestral` with the RF defaults eta 1.0 / s_noise 1.0 (flf2v is the exception: single pass, eta 0).
> The engine implements the 2.5 template values, keeps derived per-generation seeds, and defaults plain Euler
> (measured: better audio — see `fca94ed2`).

Distilled runs at CFG 1 (no guidance). The **dev** checkpoint uses `PipelineParams`: 30 steps (2.3 lineage),
video CFG 3.0 / audio CFG 7.0, `rescale_scale 0.7`, `modality_scale 3.0`, STG on block 28 (2.3+) or 29 (2.0),
`skip_step 0`.

> **These are the DOCUMENTED settings, not what the reference we benchmark against actually runs (2026-08-13).**
> The captured ComfyUI graph (`Swarm/.../Data/Logs/2026-08/13-19-42.log:1423`) uses a single `SwarmKSampler`
> with **joint CFG 3.0** over the concatenated AV latent, `euler`/`normal`, 30 steps — no `LTXVDualCFGGuider`,
> no `LTXVModalityGuidance`, no `LTXVSpatioTemporalGuidance`, no CFG-rescale. ComfyUI's LTX nodes do not
> implement CFG-rescale at all, and diffusers defaults `guidance_rescale`/`stg_scale` to 0.0 and
> `modality_scale` to 1.0 (all disabled). So the machinery listed above is **opt-in and off by default**, and
> its absence from this port does NOT explain any observed gap against a ComfyUI generation. This was read as a
> missing-machinery lead once already — see the retracted audio-level section in
> `docs/Checklists/MODEL_STATUS_VIDEO.md`.

Frame/size constraints: `num_frames % 8 == 1`, width/height divisible by 32, 24 fps default.

### NADiffusionDecoder (`ltx-2.5-video-vae-bf16.safetensors`)

396 tensors: 84 `encoder.*`, 310 `decoder.*`, 2 `per_channel_statistics.*`. `_class_name: CausalDiffusionVAE`.
The **encoder half is the same conv `Encoder` as the conv VAE** (same block list, `patch_size 4`,
`latent_log_var: constant` with value −7.824046010856292, `norm_layer: pixel_norm`, `base_channels 128`) —
only the decoder is new.

Decoder config, verbatim from the checkpoint metadata (identical to the ComfyUI defaults):

| field | value |
|---|---|
| `in_channels` / `out_channels` | 128 / 3 |
| `patch_size` | 4 |
| `head_dim` | 64 |
| `stage_channels` | `[2048, 1024, 512, 512, 256]` |
| `stage_depths` | `[4, 6, 4, 2, 8]` |
| `stage_kernels` | `[(3,7,7), (3,7,7), (3,5,5), (3,5,5), (11,11,11)]` |
| `upsamples` | `[((1,2,2),2), ((2,1,1),2), ((2,2,2),1), ((2,2,2),2)]` — `(stride, out_channel_reduction)` |
| `stage5_kernel` | `(11,11,11)` |
| `timestep_scale_multiplier` | 1000.0 |
| `default_num_inference_steps` | 1 |
| `model_output_type` | `x0` — one forward pass yields pixels, no Euler loop |
| `resampler_kind` | `linear` |
| `spatial_padding_mode` | `zeros` |

Decoder tensor roots (310 keys): `conv_in` (Linear 128→2048), `conv_in_x_t` (Linear 48→256, the noised-pixel
input — this is the key that identifies a diffusion VAE), `det_stages.{stage}.{block}.{norm1,attn,norm2,mlp}`
(176 keys; fused `attn.qkv`, `attn.proj`, per-head `q_norm`/`k_norm [64]`, SwiGLU `mlp.{w_gate,w_up,w_down}`),
`upsamples.{i}.proj` (8), `t_embedder.mlp.{0,2}` (4), `shared_adaln.proj [1792, 384]` (= 7 chunks × 256),
`diff_blocks.{i}.{attn,mlp,context_proj,scale_shift_table [7,256]}` (112), `norm_out [256]`,
`conv_out` (Linear 256→48), and `type_emb [128]`.

**`type_emb` is carried by the checkpoint but used by neither reference implementation** — it appears nowhere
in ComfyUI's `na_diffusion_decoder.py` nor in the official `ltx-core` `diffusion_video_decoder.py`. Treat it as
vestigial and ignore it, but do not silently drop it without noting the divergence: if decoder parity ever comes
out subtly wrong, this is the first thing to re-examine.

Derived: temporal upscale = ∏ stride[0] = 8, spatial upscale = ∏ stride[1] × `patch_size` = 8 × 4 = 32.

**Frame-count composition — verify this before porting; an off-by-one here is a subtly wrong video, not a
crash.** `LinearPixelShuffleUpsample` drops the duplicated leading frame whenever the temporal stride is 2
(the causal temporal pixel-shuffle emits it twice). Three of the four stages have `p1 == 2`, so from `t` latent
frames:

```
stage 0  p1=1  ->  t                (no drop)
stage 1  p1=2  ->  2t - 1
stage 2  p1=2  ->  4t - 3
stage 3  p1=2  ->  8t - 7
```

which is exactly ComfyUI's `upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)` in `comfy/sd.py`, and is why
valid frame counts are `num_frames % 8 == 1` (t=1 → 1, t=2 → 9, t=3 → 17, t=4 → 25).

Trailing-pad for the NATTEN border: replicate the last latent frame `(stage_kernels[0][0] // 2) * 2 = 2` times
through stages 1-4, then crop the appendix off the context. Those 2 extra latent frames become `2 × 8 = 16`
extra output frames to remove (e.g. t=3 padded to 5 yields 33, crop 16 back to 17).

### Duration head (`ltx-2.5-duration-head-bf16.safetensors`)

15 tensors, ~10 MB. Config: `{"transformer": {"cross_attention_dim": 4096, "audio_cross_attention_dim": 2048}}`.
Keys: `duration_head.{video,audio}_input_proj.{weight [256,4096]/[256,2048], bias}`,
`{video,audio}_modality_emb [256]`, `attention_pooler.query_tokens [1,256]`,
`attention_pooler.cross_attn.{in_proj_weight [768,256], in_proj_bias [768], out_proj.weight [256,256], out_proj.bias}`,
`mlp_hidden.{weight [256,256], bias}`, `mlp_out.{weight [1,256], bias [1]}`. 4 attention heads, output is
`exp(mlp_out(...))` = seconds.

### Checkpoint inventory (HF `Lightricks/LTX-2.5`, `gated: auto`)

| file | size |
|---|---|
| `diffusion_models/ltx-2.5-22b-{distilled,dev}-transformer-bf16.safetensors` | 42.02 GB each |
| `diffusion_models/ltx-2.5-22b-{distilled,dev}-transformer-comfy-int8-convrot.safetensors` | 21.50 GB each |
| `diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` | 18.72 GB |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 26.26 GB |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15.37 GB |
| `vae/ltx-2.5-video-vae-bf16.safetensors` (diffusion decoder) | 1.47 GB |
| `vae/ltx-2.5-video-vae-conv-bf16.safetensors` | 1.45 GB |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | 0.36 GB |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | ~0.01 GB |
| `loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors` | 8.90 GB |
| `latent_upscale_models/ltx-2.5-latent-{spatial,temporal}-upscaler-x2-bf16-1.0.safetensors` | 1.00 / 0.26 GB |

Third-party repacks that are **not** gated (useful because the Lightricks repo requires auth even for a
byte-range header read): `dummy9996/LTX-2.5-22b-ungate` (bf16 mirror, no diffusion VAE),
`ChrisColeTech/LTX-2.5-turbo-GGUF` (`split/vae/*` — carries both VAEs),
`guillaume127/LTX-2.5-FP8` (`ltx-2.5-22b-distilled-transformer-fp8_e4m3fn.safetensors`, 23.49 GB),
plus assorted GGUF and NVFP4 repacks.

## Data layouts

### 2.5-only DiT tensor

```
model.diffusion_model.keyframes_abs_pos_embedding    [1, 4096]   BF16
```

Added to the token rows whose temporal start is 0, immediately after `patchify_proj`. Reference:
`comfy/ldm/lightricks/model.py:1217-1227` (`apply_keyframes_abs_pos_embedding`) and the mask builder at
`:1186` (`keyframes_abs_pos_mask`). The mask is `pixel_coords[:, 0] == 0`, with appended i2v guide tokens
excluded and generated-keyframe slots OR-ed in. With no guide tokens (plain T2V, which is what the engine
does today) this reduces to the first latent frame's token rows.

### Video FFN without bias (2.5)

```
transformer_blocks.N.ff.net.0.proj.weight   [16384, 4096]     (no .bias)
transformer_blocks.N.ff.net.2.weight        [4096, 16384]     (no .bias)
transformer_blocks.N.audio_ff.net.0.proj.{weight [8192,2048], bias [8192]}     (bias KEPT)
transformer_blocks.N.audio_ff.net.2.{weight [2048,8192], bias [2048]}          (bias KEPT)
```

### Gemma 4 per-layer keys

```
model.layers.N.input_layernorm.weight            [3840]
model.layers.N.post_attention_layernorm.weight   [3840]
model.layers.N.pre_feedforward_layernorm.weight  [3840]
model.layers.N.post_feedforward_layernorm.weight [3840]
model.layers.N.layer_scalar                      [1]
model.layers.N.mlp.gate_proj.weight              [15360, 3840]
model.layers.N.mlp.up_proj.weight                [15360, 3840]
model.layers.N.mlp.down_proj.weight              [3840, 15360]

# sliding layer (index % 6 != 5)
model.layers.N.self_attn.q_proj.weight  [4096, 3840]   q_norm.weight [256]
model.layers.N.self_attn.k_proj.weight  [2048, 3840]   k_norm.weight [256]
model.layers.N.self_attn.v_proj.weight  [2048, 3840]
model.layers.N.self_attn.o_proj.weight  [3840, 4096]

# global layer (index % 6 == 5) — note the absent v_proj
model.layers.N.self_attn.q_proj.weight  [8192, 3840]   q_norm.weight [512]
model.layers.N.self_attn.k_proj.weight  [512, 3840]    k_norm.weight [512]
model.layers.N.self_attn.o_proj.weight  [3840, 8192]
```

Non-layer keys in the TE file:

```
model.embed_tokens.weight                                  [262144, 3840]
model.norm.weight                                          [3840]
text_embedding_projection.video_aggregate_embed.weight     [4096, 188160]   bias [4096]
text_embedding_projection.audio_aggregate_embed.weight     [2048, 188160]   bias [2048]
tokenizer_json                                             [32169626]  U8
hf_asset__{chat_template.jinja,generation_config.json,processor_config.json,tokenizer_config.json}   U8
# unused multimodal tower — skip at load:
vision_model.*   audio_projector.embedding_projection.weight   multi_modal_projector.embedding_projection.weight
```

`188160 = 3840 × 49` confirms the projection consumes all 49 hidden states (embedding output + 48 block
outputs), the same all-layer harvest the engine already does for Gemma 3 12B.

## Algorithm

### Gemma 4 partial rotary on global layers

ComfyUI builds one inv-freq buffer per layer kind (`comfy/text_encoders/gemma4.py:303-312`):

```python
rope_angles_global = int(0.25 * 512 // 2)          # 64
nope_global        = 512 // 2 - rope_angles_global # 192
global_inv = 1.0 / (1e6 ** (arange(0, 128, 2).float() / 512))   # 64 values
global_inv = cat([global_inv, zeros(192)])                       # pad to 256

sliding_inv = 1.0 / (1e4 ** (arange(0, 256, 2).float() / 256))   # 128 values
```

The zero-padded lanes give `cos = 1, sin = 0`, i.e. identity, so a single unconditional rope apply
implements "rotate the first 128 of 512 dims" without branching. Note the exponent denominator is
`global_head_dim` (512), **not** `2 * rope_angles_global`.

### NADiffusionDecoder forward

Stages 1-4 deterministically upsample the (un-normalized) latent into a context volume: `conv_in` Linear →
per stage, `stage_depths[i]` pre-norm `NABlock`s (RMSNorm → 3D neighborhood attention → RMSNorm → SwiGLU,
both residual) → `LinearPixelShuffleUpsample` (Linear channel expand then channels-last pixel shuffle; when
the temporal stride is 2 the causal shuffle duplicates the leading frame, so drop it). Stage 5 runs
`DiffusionNABlock`s over patchified noised pixels `x_t`, each adding a `context_proj` of the stage-4 context
and modulating via shared AdaLN-Zero scale/shift (7 chunks from a SiLU-MLP on the timestep embedding, plus a
per-block `scale_shift_table`; the gate slots are unused/folded). Because `model_output_type` is `x0` and
`default_num_inference_steps` is 1, a single forward at t=1.0 yields the pixels.

Attention detail: fused `qkv` Linear, per-head RMS `q_norm`/`k_norm` over `head_dim` 64, then per-axis
absolute RoPE with the t/h/w dim split `default_rope_dim_split(64)`, then neighborhood attention. The
`1/sqrt(head_dim)` scale is folded into the q-norm weight (it commutes with the rotation).

### Neighborhood attention semantics (the thing most likely to be got wrong)

NATTEN semantics, which the port must reproduce exactly: each query attends to a window of **exactly**
`kernel_size` keys per axis; near a grid boundary the window is **shifted inward** rather than truncated or
zero-padded; dilation is 1. When an axis is shorter than its kernel, the window degenerates to the whole
axis. Reference behavior is `comfy_kitchen.na3d`'s eager backend, described at
`comfy/ldm/lightricks/vae/na_diffusion_decoder.py:1-17`.

## References

- ComfyUI LTX 2.5 support: commit `57ce8e1` "Add support for LTX 2.5 (#15499)" (Comfy-Org/ComfyUI), plus
  `ce4fc13` for the partner API nodes. Touched `comfy/ldm/lightricks/{model,av_model,embeddings_connector,duration_head}.py`,
  `comfy/ldm/lightricks/vae/{na_diffusion_decoder,audio_vae}.py`, `comfy/text_encoders/{lt,gemma4}.py`,
  `comfy/{sd,model_base,model_detection}.py`, `comfy_extras/nodes_lt.py`.
- `comfy/model_detection.py:400` — probes `'{}keyframes_abs_pos_embedding'.format(key_prefix) in state_dict_keys`
  and sets `use_keyframes_abs_pos_embedding` from **key presence, even when metadata config exists**. Mirror this.
- `comfy/ldm/lightricks/model.py:1186` `keyframes_abs_pos_mask`, `:1217` `apply_keyframes_abs_pos_embedding`,
  `:1120-1129` the keyframe-token/tokens-per-frame validation error.
- `comfy/text_encoders/gemma4.py:95` `Gemma4_12B_Config`, `:303-312` rope inv-freq construction.
- `comfy/text_encoders/lt.py` — `ltxav_gemma4_tokenizer` (pads `min_length` to 1024), `DualLinearProjection`,
  `sd_detect` (reads projection dims + bias presence off the checkpoint keys), `LTXAVTEModel.load_sd`.
- `comfy/sd.py` — VAE branch `elif "decoder.conv_in_x_t.weight" in sd:` selecting `CausalDiffusionVAE`,
  with `upscale_ratio = (lambda a: max(0, a * 8 - 7), 32, 32)`.
- Official package `Lightricks/LTX-2` v1.2.0 (CHANGELOG "Support for LTX 2.5"):
  `packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py` (`LTXModelConfigurator.from_metadata`,
  the authority on which config keys exist and their defaults),
  `packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py` (sigma schedules, `PipelineParams`,
  `detect_model_version`/`detect_params`).
- Checkpoint metadata itself is the primary source for every number in *Key numbers* — read via safetensors
  byte-range header requests, not by downloading the files.

### Where implementations disagree

- **`av_ca_timestep_scale_multiplier`**: the official `LTXModelConfigurator` defaults it to `1`, but both the
  2.3 and 2.5 checkpoints ship `1000.0` in metadata. The engine hardcodes 1000 and is correct for every
  shipped checkpoint. Do not "fix" toward the library default.
- **`use_prompt_adaln_single`, `audio_ff_bias`**: exposed as configurable by both ComfyUI and the official
  configurator, and widely described as 2.5 changes, but **neither shipped 2.5 checkpoint sets them** — the
  weights prove both remain in their default `true` state. Trust the tensor keys over the release narrative.
- **Keyframes flag source**: ComfyUI derives it from key presence and *overwrites* whatever the metadata
  config said (`model_detection.py:400` runs after the `dit_config.update(metadata config)` line). The
  official configurator reads it from metadata only. Follow ComfyUI: key presence wins, because repacks
  routinely strip `__metadata__` while never dropping a weight.

## Validation plan (executed 2026-08-12 — see the status banner for results)

| stage | reference to dump | compare | tolerance |
|---|---|---|---|
| Variant detection | n/a (pure) | metadata-only / keys-only / disagreement / 2.3 file / empty input | exact config equality |
| DiT with 2.5 flags | official `ltx-core` `LTXModel` built from a tiny config with `ff_bias=False`, `use_keyframes_abs_pos_embedding=True` | per-block hidden states, shared noise + shared weights (extend `tests/HartsyInference.Video.Tests/Parity/ltx2_transformer_parity_dump.py`) | per-block relL2 at the thresholds the 2.3 harness established (~1e-7 was achieved for 2.3) |
| Keyframes embedding | same harness, embedding zeroed vs non-zero | only the first `hLat*wLat` token rows may change | exact on unaffected rows |
| Gemma 4 encoder | ComfyUI `gemma4.py` on a tiny config, then real weights when available | per-layer hidden states; assert a sliding layer (0) and a global layer (5) separately | relL2 per layer, same ladder as the Gemma 3 port |
| `Na3d` op | managed default-interface fallback vs CUDA kernel, random inputs | grid < kernel, grid == kernel, grid > kernel, non-cubic kernels | ~1e-6 relL2 (f32) |
| Diffusion decoder | ComfyUI `NADiffusionDecoder` on the real 1.47 GB checkpoint, small latent, shared noise | decoded pixels | relL2 + SSIM; **and look at the image** |
| Conv / audio VAE | — | header key+shape diff vs the 2.3 files (done, identical) | n/a |

Share noise tensors, never seeds — C# Box-Muller does not match PyTorch's RNG.

Locally staged for this work: `Models/VAE/LTX-2/ltx-2.5-video-vae-bf16.safetensors`
(sha256 `847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3`, 1472223346 bytes, from
`ChrisColeTech/LTX-2.5-turbo-GGUF` `split/vae/`, header-verified identical to the Lightricks original).

## Diffusion decoder — bring-up notes

Verified against the reference at relL2 3.978e-7 (stage-1..4 context) and 2.345e-7 (final pixels), plus a real
decode of the shipped checkpoint at the smallest legal latent (309 of 310 `decoder.*` tensors consumed).

- **Make the parity fixture asymmetric.** The first one used square kernels and a cubic latent, where a T/H/W
  transposition is invisible. The committed generator uses per-axis-different kernels
  `((3,3,5),(3,5,3),(3,3,5),(3,5,3),(5,3,3))` and a `2×4×5` latent so an axis swap cannot pass.
- **`default_rope_dim_split(8)` returns `(0, 4, 4)` in the ComfyUI reference, with no assert** — a head_dim of 8
  silently drops temporal RoPE entirely. Only bites at toy sizes (the real head_dim is 64), but it makes a tiny
  test config quietly non-representative. The C# `RopeDimSplit` throws instead.
- **`MatMulKernels.LinearTransB` re-casts a non-F32 weight to F32 on every call.** Against the BF16 checkpoint
  that churns roughly 1.5 GB of transient casts per decode and is why the smallest possible frame costs 32 s.
  This, not the neighborhood attention itself, is the first thing to fix for a usable decode path.
- Regenerating the fixture needs three things absent from the repo: torch from the ComfyUI venv, comfy-kitchen's
  eager `na3d` (via `NA_EAGER_DIR`), and ltx-core's `rope_math.py` (via `LTX_ROPE_MATH`).

## Selecting the distilled variant

The dev and distilled 2.5 transformers are **indistinguishable from the checkpoint**: identical
`model_version` (`2.5.0`), zero differences across the whole `config.transformer` object, and the same 4349
tensor keys. Nothing in the file says which schedule the weights were distilled onto.

So the sampling contract cannot be detected from the tensors — it arrives as user intent, or as the filename.
The engine exposes two catalog ids, `ltx-2.5` (dev) and `ltx-2.5-distilled` (8 steps, guidance 1.0, fixed
sigmas, two-stage), each backed by its own `LtxVideo2Recipe` instance — AND
`LtxVideo2DistilledRouting.RemapFamilyId` routes a dev-family id whose checkpoint filename (or staged
directory contents) says `distilled` to the distilled contract, with a log line naming the switch. This
REVERSES the earlier "never silently switch on filename" rule (2026-08-17, by decision): SwarmUI can only
send the dev family id, so without the remap the distilled contract is unreachable from it, and the remap is
the same approach SwarmUI itself ships for Hunyuan 1.5 distilled. It is not silent — the log records it —
and a renamed repack degrades to the dev contract, exactly what it got before the remap existed.

## Verification status of the DiT-flag work

The repo's LTX-2 DiT parity harness (`tests/HartsyInference.Video.Tests/Parity/ltx2_transformer_parity_dump.py`)
depends on a diffusers install in `/home/hartsy/hfvenv`, **which no longer exists on this machine**, so it is
dormant for reasons predating this work. The two 2.5 transformer deltas are instead covered by targeted tests
that do not need it, each pinned against an independently computed expectation rather than against the
implementation's own behaviour:

- `ff_bias: false` — a bias-free video FFN is asserted to produce bit-identical output to the same weights with
  an explicitly zeroed bias, so "it loaded" is not mistaken for "it computes the right thing".
- `use_keyframes_abs_pos_embedding` — the marker's exact value is asserted on exactly the first latent frame's
  token rows and nowhere else, read from the pre-block state via the `OnBlockOutput(-1)` hook. It has to be read
  there: self-attention propagates a frame-0 change to every token within a single block, so an assertion on the
  final velocity cannot localise the mask at all.

Restoring the full harness needs either a diffusers build carrying the 2.5 flags or the official `ltx-core`
package installed; neither is present, and `ltx-core` pulls Transformers 5.8+ and CUDA 13.2 wheels that do not
fit the remaining disk.

### The reference harness that does work: headless ComfyUI 0.32

Diffusers is not the only reference. ComfyUI 0.32 loads these exact `int8_lean_convrot` checkpoints through its
own quant path and runs both the Gemma-4 tower and the AV DiT, which is what every number in the status banner
was measured against. Setup, without touching the SwarmUI install:

- SwarmUI's bundled ComfyUI is **0.28 and predates the Gemma-4-12B tower** (`Gemma4_12B_Config` does not exist
  there) — it cannot be used for 2.5. Use a 0.32+ checkout.
- 0.32 needs `comfy-kitchen==0.2.30`; the 0.28 venv pins 0.2.22. Install the newer one to a scratch directory
  and shadow it (`pip install --no-deps --target=<dir> comfy-kitchen==0.2.30`, then `PYTHONPATH=<dir>`), reusing
  the existing venv's torch. Upgrading in place would break the running SwarmUI backend.
- Point `folder_paths.add_model_folder_path` at `Models/Stable-Diffusion/LTX-2.5`, then
  `comfy.sd.load_clip([TE])` and `comfy.sd.load_diffusion_model(DIT)`. Model management handles the
  22 GB + 15 GB on a 24 GB card.
- To drive the reference DiT with **our** conditioning, pass a `[1, seq, 6144]` tensor as `c_crossattn`:
  `preprocess_text_embeds` returns it untouched when the last dim is `cross_attention_dim +
  audio_cross_attention_dim`, which bypasses its connector. `x` must be `comfy.utils.pack_latents([video,
  audio])` with the matching `latent_shapes`, not a `NestedTensor`.
- Per-block comparison: `register_forward_hook` on `diffusion_model.transformer_blocks[i]` (output `[0]` is the
  video stream) against our `LtxVideo2Transformer.OnBlockOutput` (index `-1` is post-`proj_in`).
- Measure **prompt sensitivity**, not absolute state. Two prompts through each engine, then compare
  `Δ(A,B)` per block. Absolute states drift a few percent from int8 GEMM ordering and hide the defect; the
  `Δ(A,B)` ratio pinned it to a factor of 8-13x at every block.

## Gemma 4 encoder — settled questions

**RMS-norm storage: DIRECT, not `1 + w`** (the opposite of Gemma 3). Settled by byte-ranging real tensors out
of the shipped checkpoint: `layers.0.self_attn.q_norm` is a uniform `+1.02344` across all 256 entries — under
the `1+w` convention it would store `0.02344` — `layers.5.self_attn.k_norm` is a uniform `+0.06055`,
`model.norm` has mean `+20.13` and max `+600`, and `input_layernorm` spans `[-143, +193]`.

**Tokenizer: rank-merge BPE with byte fallback**, `model.type == "BPE"`, 514,906 merges, `ignore_merges: false`,
a normalizer rewriting `" "` to `U+2581`, and a `Split(" ")` pre-tokenizer that is vestigial because
normalization has already consumed every space. Despite the family name it is *not* SentencePiece-scored. That
rules out all three existing cores — `HfTokenizerJson` and `GgufTokenizer` byte-level-remap through
`ByteLevelCodec` first, and `SpmGgufTokenizer` merges by score rather than rank — hence a dedicated
`Gemma4Tokenizer`, verified bit-exact against the HuggingFace `tokenizers` library on the real 262k blob.

**Real `layer_scalar` values are small and load-bearing**: 0.053 at layer 0 and 0.356 at layer 5, the
counterpart to `model.norm` reaching +600. Dropping the multiply is not a subtle error at real scale.

### Settled 1 — the 49th state's final norm: correctly NOT applied (this also clears the 2.3 path)

`LlamaStyleEncoder.EncodeMultiLayer` applies no final `RmsNorm` on the all-layers harvest, which used to read
like a bug because `HasFinalNorm = true` is set on the config. It is correct. The reference gates that norm on
`final_layer_norm_intermediate`, which `sd1_clip.SDClipModel` passes as `self.layer_norm_hidden_state` — and
every LTX text-encoder wrapper (`Gemma3_12BModel` in `lt.py`, `Gemma4Model` in `gemma4.py`) constructs with
`layer_norm_hidden_state=False`. So `model.norm` is **not** applied to the 49-state stack on either the 2.3
Gemma-3 path or the 2.5 Gemma-4 path.

Measured, not inferred: our layer 0 equals the reference's at **cosine 1.0000** with the norm skipped, and
0.3531 with it applied. `Gemma4TextEncoderConfig.ApplyFinalNormToLastState` stays as an escape hatch but the
default is the verified one. The earlier "two readings of ComfyUI disagree" note was a misreading of the flag,
not a real ambiguity.

### Settled 2 — padding side is numerically equivalent

ComfyUI left-pads to 1024 and masks the pads; this engine right-pads with a causal-only mask. The real tokens
therefore sit at different absolute RoPE positions (`1024-n .. 1023` vs `0 .. n-1`). That is harmless: Gemma
attention depends only on position *differences*, the masked/never-attended pads contribute nothing either way,
and the reference trims back to the real tokens before the projection.

Measured end-to-end against comfy's own conditioning for the same prompt: overall cosine **1.0012**, real rows
0.9992, register rows 1.0011, and — the sensitive test — the prompt-to-prompt delta at **ratio 1.00** on every
row band. Only row 0 (BOS) differs at cosine 0.96, which does not propagate.

## Performance (2026-08-12 pass — numbers and do-not-retry list live in `benchmarks/scoreboards/VIDEO.md`)

56.62 s warm through the SwarmUI API at 768×512×97f/30 steps against ComfyUI's 42.25 s, down from 117.13 s.
Four things did it, in descending order: a GPU reflect spatial pad in `wan_vae_build_padded` (the VAE decode
was 38 s, ~28 s of it a scalar host loop in `CausalConv3d.ReflectPadSpatial5D` that D2H-drained and
re-uploaded the activation on all 42 conv forwards); F16 block activations via `DitDtype.Act` plus a new
`ltx2_split_rope_f16` kernel (LTX-2.5 had hardcoded F32 everywhere and never opted in); `ApplyKeyframesAbsPos`
moved on-device; and a pool trim before the resident prefix is sized.

Three things worth not re-deriving:

- **The keyframe marker was silently dropped on CUDA.** `ApplyKeyframesAbsPos` aliased `hidden`'s HOST buffer,
  so the affine's result landed in the alias tensor's own device cache, which `Dispose` frees with no D2H
  write-back. It worked on the CPU backend, which is why it survived its test.
- **The step is GPU-bound, not host-launch-bound** — SM 99–100%, memory controller 78–80% under
  `nvidia-smi dmon`. CUDA graphs and launch-overhead work are not the lever for this model.
- **`Linear` is near its structural floor** (~62% of a 1.756 s step): ~0.80 s of unavoidable int8 GEMM plus
  ~0.27 s of int32 IMMA-accumulator round trip. Shrinking the row chunk to make that accumulator L2-resident
  is monotonically *slower*; the only real fix is a fused-dequant IMMA kernel.

## Open questions

- **Diffusion video decoder is managed-only.** No CUDA `Na3d` kernel, ~32 s for the smallest legal frame, so
  generation decodes through the conv VAE. `MatMulKernels.LinearTransB` re-casting a non-F32 weight per call is
  the dominant cost and the first thing to fix.
- **Encoder parity was measured on F32/CPU weights** (3.9e-7 sliding / 1.1e-6 global on a tiny config). The
  resident-int8 projection path is now exercised end-to-end and matches the reference tower at cosine
  0.9999-1.0000 per layer on real weights, so this is closed for practical purposes; a formal per-layer relL2
  ladder against the BF16 tower is still absent.
- **`LtxVideo2Recipe.TokenLength` is an `internal const` at 256**, while the 2.5 path conditions at 1024. It has
  to stay per-branch: bumping it globally would change the Gemma-3 conditioning length and silently alter
  verified 2.3 output.
- **LoRA, image-to-video conditioning, and component overrides** are deferred (`TODO(E-IMG-4/5)` in
  `LtxVideo2Recipe`).
