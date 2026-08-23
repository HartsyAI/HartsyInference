# LTX 2.5 — Research Notes

> **Stub.** This model is built and verified end-to-end (prompt-faithful video+audio confirmed 2026-08-12,
> quality parity with ComfyUI reached 2026-08-15 — see `docs/Checklists/MODEL_STATUS_VIDEO.md`), so the C# is
> the source of truth for *how it works*. What remains is what the code cannot tell you: upstream provenance,
> reference constants to diff a suspect port against, where implementations disagree, and bring-up traps.
> **Note on staleness:** the diffusion video decoder section below predates 2026-08-13/14 CUDA kernel work — it
> now has a CUDA `Na3d` kernel, decodes in ~9.8s (not "~32s, managed-only"), and its output is correctness-verified
> against ComfyUI. Current status: `MODEL_STATUS_VIDEO.md`.

`hartsy video -m ltx-2.5` generates prompt-faithful 704×480×25f clips with a soundtrack on the official
`int8_lean_convrot` DiT + Gemma-4-12B pair. The whole text path is verified against ComfyUI 0.32 on the real
checkpoints — tokenizer ids byte-exact, Gemma-4 tower cosine 0.9999–1.0000 per layer, connector within
0.2–0.6%, `attn2` within 0.07%, and the conditioning's prompt-to-prompt delta at ratio 1.00.

## Invariants worth not re-deriving

**Ungated conv VAE** (Lightricks 401s even on a byte range; both verified reachable 2026-08-12, `gated: false`):

```
https://huggingface.co/ChrisColeTech/LTX-2.5-turbo-GGUF/resolve/main/split/vae/ltx-2.5-video-vae-conv-bf16.safetensors
https://huggingface.co/dummy9996/LTX-2.5-22b-ungate/resolve/main/ltx-2.5-video-vae-conv-bf16.safetensors
```

- Gemma-4 discriminator: `conv.TextEncoder.ContainsKey("model.layers.0.layer_scalar")`. Do **not** probe for a
  missing `v_proj` — layer 0 is a sliding layer and has one.
- Conditioning length: the recipe's `TokenLength` is 256, but ComfyUI pads Gemma-4 LTX conditioning to
  **1024**, and sequence length is part of the conditioning because of the connector's learnable-register
  replacement. `Gemma4Tokenizer.BuildConditioningSequence` already right-pads to 1024.
- `conv.VaeDiffusionDecoder` holds the `decoder.*` keys with the prefix kept; the latent statistics stay in
  `conv.Vae` as `latents_mean`/`latents_std`, since both decoders un-normalize identically.
- Distilled vs dev is not detectable from any checkpoint — it arrives via the `ltx-2.5-distilled` catalog id.

### Prompt-independent output — FIXED (`prompt_adaln` fed the raw sigma, not the ×1000-scaled timestep)

`prompt_adaln_single`/`audio_prompt_adaln_single` modulate the text K/V feeding every block's cross-attention.
Evaluating their sinusoidal timestep embedding at t≈1 instead of t≈1000 (the convention every other modulator
uses) landed on an unrelated point on the embedding — cross-attention could not *discriminate*, so output
followed the seed rather than the prompt (two unrelated prompts differed by ~1% of pixel range). Fixed by
scaling the timestep before feeding it in, matching `av_model.py::_prepare_timestep`. Per-block prompt
sensitivity vs. ComfyUI went from 8-13× too weak to ratio 1.00.

Two traps that cost real time diagnosing this:
- int8 weights live under the *same* `.weight` key as an ordinary tensor — a loader that checks the raw key
  before its `.weight_scale` sibling silently loads unscaled, un-rotated int8 and produces plausible garbage.
- `final_layer_norm_intermediate` is **False** on the LTX path — the final `model.norm` is NOT applied to the
  49-layer stack. Applying it makes every layer mismatch (see "Settled" below).

### Running the encoder against the int8 text encoder

`Gemma4TextEncoder`'s embedding gather reads the weight as a raw `float*`, so a packed embedding table would
yield silent garbage — it doesn't happen, because the int8-convrot TE quantizes **only the projection
matrices** (embeddings, norms, `layer_scalar`, connector projection all stay BF16). Encoder parity (3.9e-7
sliding / 1.1e-6 global) was measured on F32/CPU weights; the resident-int8 path is now exercised end-to-end
and matches the reference tower at cosine 0.9999-1.0000/layer on real weights.

### Raising the conditioning length to 1024

Not a free change: the 49-layer feature stack is ~771 MB in F32 at 1024 tokens, and the prompt cache holds
both positive and negative (~1.5 GB budget) — the resident-prefix VRAM headroom was sized against 256, so
re-check rather than assume it scales. `LtxVideo2Recipe.TokenLength` has to become per-branch: bumping it
globally would change the Gemma-3 (2.3) conditioning length too and silently alter its verified output.

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
stage. The conv video VAE and the audio VAE are structurally identical to 2.3 and need no code.

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
- `full_attention`: `rope_theta 1e6`, `rope_type proportional`, `partial_rotary_factor 0.25`. The inv-freq
  exponent denominator is `global_head_dim` (512), **not** `2 × rope_angles_global` — easy to get backwards.

### Distilled sampling schedule

`packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:17-25`:

```
DISTILLED_SIGMA_VALUES  = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]   # 8 steps
STAGE_2_DISTILLED_SIGMA = [0.909375, 0.725, 0.421875, 0.0]                                          # 2x upscale stage
TDP_DISTILLED_SIGMAS    = [0.625, 0.4, 0.0]                                                          # multi-GPU tiled runner
```

> **The shipped ComfyUI 2.5 templates do NOT use that stage-2 array.** All three `video_ltx2_5_*` templates'
> refine-pass ManualSigmas literal is `0.85, 0.7250, 0.4219, 0.0` (verbatim — 0.4219, not 0.421875; ManualSigmas
> parses the text as written). Two more template facts the constants file does not record: the refine pass's
> noise seed is hardcoded `42, fixed` (only the base pass gets the user seed), and both passes select
> `euler_ancestral` with the RF defaults eta 1.0 / s_noise 1.0 (flf2v is the exception: single pass, eta 0).
> The engine implements the 2.5 template values, keeps derived per-generation seeds, and defaults plain Euler
> (measured: better audio — see `fca94ed2`).

Distilled runs at CFG 1 (no guidance). The **dev** checkpoint uses `PipelineParams`: 30 steps (2.3 lineage),
video CFG 3.0 / audio CFG 7.0, `rescale_scale 0.7`, `modality_scale 3.0`, STG on block 28 (2.3+) or 29 (2.0),
`skip_step 0`. **These are the documented settings, not what the ComfyUI reference we benchmark against
actually runs** — the captured graph uses a single `SwarmKSampler` at joint CFG 3.0, no dual-CFG guider, no
modality guidance, no STG, no CFG-rescale (ComfyUI's LTX nodes don't implement CFG-rescale at all). The
machinery above is opt-in and off by default; its absence from this port doesn't explain a gap against a
ComfyUI generation.

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

**Neighborhood attention semantics** (the thing most likely to be got wrong porting this): each query attends
to a window of **exactly** `kernel_size` keys per axis; near a grid boundary the window is **shifted inward**
rather than truncated or zero-padded; dilation is 1. When an axis is shorter than its kernel, the window
degenerates to the whole axis. Reference: `comfy_kitchen.na3d`'s eager backend.

### Duration head (`ltx-2.5-duration-head-bf16.safetensors`)

15 tensors, ~10 MB. Config: `{"transformer": {"cross_attention_dim": 4096, "audio_cross_attention_dim": 2048}}`.
Keys: `duration_head.{video,audio}_input_proj.{weight [256,4096]/[256,2048], bias}`,
`{video,audio}_modality_emb [256]`, `attention_pooler.query_tokens [1,256]`,
`attention_pooler.cross_attn.{in_proj_weight [768,256], in_proj_bias [768], out_proj.weight [256,256], out_proj.bias}`,
`mlp_hidden.{weight [256,256], bias}`, `mlp_out.{weight [1,256], bias [1]}`. 4 attention heads, output is
`exp(mlp_out(...))` = seconds.

### Checkpoint inventory

Full file/size table (shared with MiniMax-H3's quant family): `docs/Research/QUANTIZATION_COMFY_FORMATS.md`.
Third-party ungated repacks (Lightricks' own repo requires auth even for a byte-range header read):
`dummy9996/LTX-2.5-22b-ungate` (bf16 mirror, no diffusion VAE), `ChrisColeTech/LTX-2.5-turbo-GGUF`
(`split/vae/*` — carries both VAEs), `guillaume127/LTX-2.5-FP8`.

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

## Diffusion decoder — bring-up notes

Verified against the reference at relL2 3.978e-7 (stage-1..4 context) and 2.345e-7 (final pixels).

- **Make the parity fixture asymmetric.** The first one used square kernels and a cubic latent, where a T/H/W
  transposition is invisible. The committed generator uses per-axis-different kernels
  `((3,3,5),(3,5,3),(3,3,5),(3,5,3),(5,3,3))` and a `2×4×5` latent so an axis swap cannot pass.
- **`default_rope_dim_split(8)` returns `(0, 4, 4)` in the ComfyUI reference, with no assert** — a head_dim of 8
  silently drops temporal RoPE entirely. Only bites at toy sizes (the real head_dim is 64), but it makes a tiny
  test config quietly non-representative. The C# `RopeDimSplit` throws instead.
- Regenerating the fixture needs three things absent from the repo: torch from the ComfyUI venv, comfy-kitchen's
  eager `na3d` (via `NA_EAGER_DIR`), and ltx-core's `rope_math.py` (via `LTX_ROPE_MATH`).

## Selecting the distilled variant

The dev and distilled 2.5 transformers are **indistinguishable from the checkpoint**: identical
`model_version` (`2.5.0`), zero differences across the whole `config.transformer` object, and the same 4349
tensor keys. Nothing in the file says which schedule the weights were distilled onto.

So the sampling contract cannot be detected from the tensors — it arrives as user intent, or as the filename.
The engine exposes two catalog ids, `ltx-2.5` (dev: 20 steps, cfg 4.0 — the measured parity profile) and
`ltx-2.5-distilled` (8 steps, guidance 1.0, fixed sigmas, two-stage), both defaulting to the template
geometry (1280x736, 121f, 24 fps), each backed by its own `LtxVideo2Recipe` instance — AND
`LtxVideo2DistilledRouting.RemapFamilyId` routes a dev-family id whose checkpoint filename (or staged
directory contents) says `distilled` to the distilled contract, with a log line naming the switch. This
REVERSES the earlier "never silently switch on filename" rule (2026-08-17, by decision): SwarmUI can only
send the dev family id, so without the remap the distilled contract is unreachable from it, and the remap is
the same approach SwarmUI itself ships for Hunyuan 1.5 distilled. It is not silent — the log records it —
and a renamed repack degrades to the dev contract, exactly what it got before the remap existed.

## Verification method: headless ComfyUI 0.32

The repo's diffusers-based LTX-2 DiT parity harness is dormant (its venv no longer exists on this machine).
The 2.5 deltas are instead covered by targeted tests pinned against independently computed expectations
(`ff_bias: false` asserted bit-identical to the same weights with an explicitly zeroed bias; the keyframes
marker asserted exactly on the first latent frame's token rows via the `OnBlockOutput(-1)` hook — it has to
be read there, since self-attention propagates a frame-0 change to every token within one block).

ComfyUI 0.32 is the reference that actually produced every number in this doc's status claims, loading the
real `int8_lean_convrot` checkpoints through its own quant path:
- SwarmUI's bundled ComfyUI is 0.28 and predates the Gemma-4-12B tower — use a separate 0.32+ checkout.
- 0.32 needs `comfy-kitchen==0.2.30` (0.28's venv pins 0.2.22) — install to a scratch dir and shadow via
  `PYTHONPATH`, reusing the existing venv's torch, rather than upgrading in place and breaking the running
  SwarmUI backend.
- To drive the reference DiT with our conditioning, pass a `[1, seq, 6144]` tensor as `c_crossattn` —
  `preprocess_text_embeds` returns it untouched when the last dim matches, bypassing its connector.
- **Measure prompt sensitivity (`Δ(A,B)` across two prompts), not absolute state.** Absolute states drift a
  few percent from int8 GEMM ordering and hide the defect; the ratio pinned the timestep-scale bug to 8-13×
  at every block when absolute-state comparison saw nothing conclusive.

## Gemma 4 encoder — settled facts

- **RMS-norm storage is DIRECT, not `1 + w`** — the opposite of Gemma 3. Confirmed by byte-ranging real
  tensors (`layers.0.self_attn.q_norm` is a uniform +1.02344; under the `1+w` convention it would store 0.02344).
- **Tokenizer is rank-merge BPE with byte fallback** despite the family name — not SentencePiece-scored. Rules
  out all three existing tokenizer cores (`HfTokenizerJson`/`GgufTokenizer` byte-remap first;
  `SpmGgufTokenizer` merges by score not rank), hence a dedicated `Gemma4Tokenizer`, verified bit-exact
  against HuggingFace `tokenizers` on the real vocab.
- **The 49th state's final norm is correctly NOT applied** (also clears the 2.3 Gemma-3 path) — every LTX
  text-encoder wrapper constructs with `layer_norm_hidden_state=False`. Measured: layer 0 matches the
  reference at cosine 1.0000 with the norm skipped, 0.3531 with it applied.
- **Padding side (left vs right) is numerically equivalent** — Gemma attention depends only on position
  *differences*, so ComfyUI's left-pad-and-mask vs. this engine's right-pad-causal-mask reach the same
  result. Measured end-to-end: overall cosine 1.0012, prompt-to-prompt delta ratio 1.00.

## Performance

Numbers, the do-not-retry list, and hard ceilings live in `benchmarks/scoreboards/VIDEO.md` — not here.
Headline finding worth keeping close to the architecture notes: the keyframe marker was silently dropped on
CUDA because `ApplyKeyframesAbsPos` aliased `hidden`'s host buffer, so the affine's result landed in the
alias tensor's own device cache with no D2H write-back (worked on CPU, which is why it survived its test) —
the same aliasing defect class the diffusion decoder was later bitten by twice.

## Open items

- **LoRA, image-to-video conditioning, and component overrides** are deferred (`TODO(E-IMG-4/5)` in
  `LtxVideo2Recipe`) — see `docs/Checklists/MODEL_STATUS_VIDEO.md` Remaining work for current status.
