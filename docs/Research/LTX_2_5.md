# LTX 2.5 — Research Notes

> Status: Components complete, pipeline wiring open | Last Updated: 2026-08-12 | Two open questions below

Every component is built and checked against a reference. What is **not** done is the pipeline: `LtxVideo2Pipeline`
still constructs the Gemma-3 tower and the convolutional decoder, so `LtxVideo2Recipe` refuses a 2.5 bundle with
a targeted error rather than mis-decoding it. Wiring needs an `ILtx2TextTower` swap plus a decoder-kind branch,
and cannot be exercised on the dev box (the text encoder is 26 GB; the decoder is managed-only at ~32 s for the
smallest possible frame). This doc stays a full research doc rather than a provenance stub until that lands.

## If you are the one wiring this into the pipeline

Concrete answers to the questions that come up first, so they don't have to be re-derived.

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

**The two `LtxVideo2Recipe.Construct` guards** (`"bundles a Gemma 4 text tower"`, `"carries the LTX-2.5
diffusion video VAE"`) are deliberate placeholders. Delete them as part of wiring rather than routing around
them.

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

**Before chasing output quality, read the two divergences below** (the missing final norm, which affects the
shipping 2.3 path too, and the padding-side difference). Either can make output subtly wrong in a way that
looks like a porting bug somewhere else.

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

Distilled runs at CFG 1 (no guidance). The **dev** checkpoint uses `PipelineParams`: 30 steps (2.3 lineage),
video CFG 3.0 / audio CFG 7.0, `rescale_scale 0.7`, `modality_scale 3.0`, STG on block 28 (2.3+) or 29 (2.0),
`skip_step 0`.

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

## Validation plan

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

So the sampling contract cannot be detected — it must arrive as user intent. The engine exposes two catalog
ids, `ltx-2.5` (dev: 50 steps, guidance 3.0) and `ltx-2.5-distilled` (8 steps, guidance 1.0, the fixed sigma
list), each backed by its own `LtxVideo2Recipe` instance. A filename containing `distilled` may be logged as a
hint but must never silently switch the schedule, because repacks rename freely.

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

## Gemma 4 encoder — settled questions and two live divergences

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

### Divergence 1 — the 49th state's final norm (UNRESOLVED, affects the shipping Gemma-3 path)

`LlamaStyleEncoder.EncodeMultiLayer` — the method the LTX-2 pipeline calls — contains **no `HasFinalNorm`
branch and no `RmsNorm` call at all**, while `LlamaStyleEncoderConfig.Gemma3_12B` sets `HasFinalNorm = true`
and `RmsNormScalePlusOne = true`. So `model.norm.weight` is loaded, `1+w`-adjusted, GPU-uploaded, and never
used on that path. `Encode` (:179) and `EncodeEmbedsMrope` (:266) *do* apply it — only the all-layers harvest
skips it. The connector's own `PerTokenRmsNormMasked` does not wash the difference out, since
`RMS(RMS(x)·w) ≠ RMS(x)` for non-uniform `w`.

Whether that is a bug depends on what the reference does for the all-layers harvest, and the two readings of
ComfyUI available here disagree — one says states 0..47 raw with state 48 normed, the other that the final norm
is applied to every intermediate. **Deliberately not changed**: the LTX-2.3 Gemma-3 path is recorded as
real-weight verified (coherent video plus a decoded 48 kHz waveform), and silently altering verified shipping
behaviour on an unverified inference is the wrong trade. `Gemma4TextEncoder` defaults to the reference
behaviour and exposes `Gemma4TextEncoderConfig.ApplyFinalNormToLastState` to switch it. Resolve by dumping the
reference's 49 states once and diffing, before either path is trusted at full precision.

### Divergence 2 — padding side

ComfyUI left-pads and carries an attention mask, but its position ids are mask-blind, so a real token sits at
RoPE position `1024-n .. 1023`. This engine right-pads with a causal-only mask, putting the same token at
`0 .. n-1`. `Gemma4Tokenizer.BuildConditioningSequence` right-pads to match the engine. This is a genuine
numerical difference from upstream on every prompt shorter than the 1024-token conditioning length, not a
rounding artefact.

## Open questions

- **RMS-norm weight storage convention.** ComfyUI's Gemma 4 path uses `rms_norm_add=False` (weights applied
  directly), the opposite of Gemma 3's `1 + w`. Not yet checked against real Gemma 4 weights — a norm tensor
  whose mean sits near 1.0 confirms direct storage. Must be settled before trusting encoder parity.
- **`tokenizer_json` model type.** The 32 MB blob's `model.type` has not been inspected; the engine's
  `HfTokenizerJson` handles byte-level BPE only, and Gemma tokenizers are SentencePiece-flavored. Decides
  extend-vs-new-parser in the TE phase.
- **Conditioning sequence length.** ComfyUI pads Gemma 4 LTX conditioning to `min_length` 1024; the engine
  currently uses 256 for 2.3. Sequence length is part of the conditioning because of the connector's
  learnable-register replacement, so 1024 is the parity-correct choice, but the VRAM cost at 22B has not
  been measured.
- **End-to-end runnability on this box — the size blocker is gone as of 2026-08-12.** It used to be that
  bf16 DiT (42 GB) + bf16 TE (26 GB) fit neither the free disk nor the 24 GB + 12 GB of VRAM, and the
  official small builds were unloadable. The engine now reads `int8_tensorwise` + ConvRot resident
  ([`QUANTIZATION_COMFY_FORMATS.md`](QUANTIZATION_COMFY_FORMATS.md)), so the official pair is
  **21.50 GB DiT + 15.37 GB TE** — both fit. What remains is pipeline wiring, not size: the LTX-2 pipeline
  still constructs the Gemma-3 tower and the convolutional decoder, so a 2.5 bundle is refused rather than
  mis-decoded.
  - The official Lightricks files are **gated for download** (401 even on a byte range, though the tree API
    answers). `DmitryDB/LTX-2.5-ComfyUI-Quants` republishes the same layout ungated — its
    `int8_lean_convrot` DiT is 21.50 GB with the identical 1440 quantized matrices — and
    `dummy9996/LTX-2.5-22b-ungate` carries bf16 for reference tensors.
  - The third-party fp8_e4m3fn distilled repack (`guillaume127/LTX-2.5-FP8`, 23.49 GB) would ride the
    existing fp8 path unchanged, but it is unofficial and does nothing for the text encoder.
