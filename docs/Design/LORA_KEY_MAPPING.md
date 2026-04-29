# LoRA Key Mapping — Implementation Reference

> Authoritative table of LoRA safetensors key patterns supported in v1, and the canonical SharpInference weight keys they map to. Every mapper in `src/SharpInference.ModelHandler/Lora/Mappers/` codifies the rules in this doc.

## Supported formats (v1)

| ID | Format name | Trainer / source | Detection signal |
|---|---|---|---|
| F1 | **Kohya SD1.5** | `kohya-ss/sd-scripts`, AI Toolkit (SD1.5) | any key matches `^lora_unet_(down\|mid\|up)_blocks_` AND no `_double_blocks_` / `_single_blocks_` prefix exists |
| F2 | **Kohya SDXL** | `kohya-ss/sd-scripts`, AI Toolkit (SDXL) | same as F1 plus `lora_te2_` keys exist OR block index goes to `_2_` only (3 levels, not 4) |
| F3 | **Kohya Flux** | `kohya-ss/sd-scripts` (Flux), ComfyUI repackages | any key matches `^lora_unet_(double\|single)_blocks_` |
| F4 | **AI Toolkit Flux (legacy hybrid)** | older `ostris/ai-toolkit` builds — possibly never shipped to public | any key matches `^lora_transformer_` |
| F5 | **Diffusers Flux** (incl. **modern AI Toolkit**) | HuggingFace PEFT trainers, **`ostris/ai-toolkit` v0.1.0+**, Civitai uploads | any key matches `^transformer\.(transformer\|single_transformer)_blocks\.` |

> **Empirical correction (validated 2026-04-28 against `ostris/yearbook-photo-flux-schnell-v1.safetensors`):** the public AI Toolkit-trained LoRAs ship in **F5 (Diffusers PEFT)** format, not the F4 hybrid. The trainer's `__metadata__.software.name == "ai-toolkit"` field identifies the trainer, but the on-disk key naming is full diffusers (`transformer.transformer_blocks.{i}.attn.to_q.lora_A.weight` with dots throughout). The F4 format is kept as a defensive fallback in case any older / forked trainer build emits the `lora_transformer_` hybrid we inferred from `ai-toolkit/toolkit/network_mixins.py`, but no real F4 file has been observed in the wild yet.

**Deferred (v2):** XLabs Flux (`*.processor.*`), LoHa (`hada_w*_*`), LoKr (`lokr_w*`), DoRA (`dora_scale`), Flux.2/Z-Image/Qwen-Image LoRAs, Hunyuan-Video / Wan / Lumina2 LoRAs.

## Suffix patterns

LoRA layers come as 2 or 3 entries per target weight. The mapper groups by the layer-key root (everything before the suffix) and emits one `LoraLayer` per group.

| Suffix | Role | Tensor shape |
|---|---|---|
| `.lora_down.weight` | Down (A) | `[rank, in_dim]` |
| `.lora_up.weight` | Up (B) | `[out_dim, rank]` |
| `.lora_A.weight` | Down (A) — PEFT alias | `[rank, in_dim]` |
| `.lora_B.weight` | Up (B) — PEFT alias | `[out_dim, rank]` |
| `.alpha` | Per-layer alpha scalar | `[]` or `[1]` |

**Alpha resolution order**: explicit `.alpha` tensor → `__metadata__.ss_network_alpha` → fallback to `rank` (scale = 1.0). Always cast to `float`.

**Mixed-suffix files**: AI Toolkit Flux files force `peft_format=True` so they always use `.lora_A.weight` / `.lora_B.weight`. Kohya Flux files always use `.lora_down.weight` / `.lora_up.weight`. Detect by counting which suffix appears more often and use the dominant one; reject files that mix both as malformed.

---

## F1 — Kohya SD1.5 → diffusers UNet keys

UNet `LoadWeights(dict)` is called with empty prefix, so canonical weight keys start at the top level. CLIP `LoadWeights(dict, "text_model")` adds `text_model.` to canonical keys.

### UNet attention (`lora_unet_*` → top-level dict key)

LoRA module path → canonical weight key (append `.weight` for the actual tensor key referenced by the LoRA):

```
lora_unet_down_blocks_{N}_attentions_{M}_transformer_blocks_{T}_attn{1|2}_to_q
  → down_blocks.{N}.attentions.{M}.transformer_blocks.{T}.attn{1|2}.to_q

lora_unet_down_blocks_{N}_attentions_{M}_transformer_blocks_{T}_attn{1|2}_to_k
  → down_blocks.{N}.attentions.{M}.transformer_blocks.{T}.attn{1|2}.to_k

lora_unet_down_blocks_{N}_attentions_{M}_transformer_blocks_{T}_attn{1|2}_to_v
  → down_blocks.{N}.attentions.{M}.transformer_blocks.{T}.attn{1|2}.to_v

lora_unet_down_blocks_{N}_attentions_{M}_transformer_blocks_{T}_attn{1|2}_to_out_0
  → down_blocks.{N}.attentions.{M}.transformer_blocks.{T}.attn{1|2}.to_out.0
```

Mid block (`mid_block`) and up blocks (`up_blocks_{N}`) follow the same pattern.

### UNet feed-forward

```
lora_unet_..._transformer_blocks_{T}_ff_net_0_proj
  → ....transformer_blocks.{T}.ff.net.0.proj

lora_unet_..._transformer_blocks_{T}_ff_net_2
  → ....transformer_blocks.{T}.ff.net.2
```

These map to the SharpInference fields `_geGluProjWeight` and `_outLinearWeight` in [CrossAttentionBlock.cs:329-332](../../src/SharpInference.Diffusion/Models/Denoisers/UNetBlocks/CrossAttentionBlock.cs#L329-L332).

### CLIP-L (text encoder) — `lora_te_*`

```
lora_te_text_model_encoder_layers_{L}_self_attn_q_proj
  → text_model.encoder.layers.{L}.self_attn.q_proj

lora_te_text_model_encoder_layers_{L}_self_attn_k_proj
  → text_model.encoder.layers.{L}.self_attn.k_proj

lora_te_text_model_encoder_layers_{L}_self_attn_v_proj
  → text_model.encoder.layers.{L}.self_attn.v_proj

lora_te_text_model_encoder_layers_{L}_self_attn_out_proj
  → text_model.encoder.layers.{L}.self_attn.out_proj

lora_te_text_model_encoder_layers_{L}_mlp_fc1
  → text_model.encoder.layers.{L}.mlp.fc1

lora_te_text_model_encoder_layers_{L}_mlp_fc2
  → text_model.encoder.layers.{L}.mlp.fc2
```

These match [ClipTextEncoder.cs:289-302](../../src/SharpInference.Diffusion/Models/TextEncoders/ClipTextEncoder.cs#L289-L302) field paths.

### General algorithm (F1 / F2)

For each LoRA key:
1. Strip prefix (`lora_unet_` or `lora_te_` etc).
2. Determine `LoraTarget` from the prefix: `Unet` / `ClipL` / `ClipG`.
3. Strip suffix (`.lora_down.weight` / `.lora_up.weight` / `.alpha`) — record the role.
4. Replace **all** `_` with `.` in the body — this is naive but works because none of the canonical SD UNet/CLIP module names contain underscores in places that matter, **except** for the trailing `_0` after `to_out` which represents `.0` (a list index in PyTorch's `nn.ModuleList`). The `to_out_0` → `to_out.0` substitution happens naturally with the underscore-replace.
5. Append `.weight` to get the lookup key into the model's weight dict.

**Edge case**: `text_model_encoder_layers_{N}` correctly becomes `text_model.encoder.layers.{N}` after the underscore-replace. Block-prefix tokens like `down_blocks_{N}` correctly become `down_blocks.{N}`. No special-case logic needed for SD1.5 / SDXL UNets.

---

## F2 — Kohya SDXL → diffusers SDXL UNet keys

Same algorithm as F1 with two additions:

### Dual CLIP

| LoRA prefix | Target |
|---|---|
| `lora_te1_*` | CLIP-L (encoder1) |
| `lora_te2_*` | CLIP-G (encoder2) |
| `lora_te_*` | CLIP-L (legacy single-CLIP files — rare in SDXL training but possible) |

### UNet block depth differs

SDXL has **3** down/up levels (`down_blocks_0..2`), not 4. SDXL's transformer-block depths are `[1, 2, 10]` (down) and `[10, 2, 1]` (up). The mapper does not enforce this — bad keys produce "unknown target" warnings at apply time, not load-time errors.

---

## F3 — Kohya Flux → diffusers Flux transformer keys

The trickiest case. Kohya Flux LoRAs target **fused** weights and have to be split during the layer-build step.

### Top-level non-block keys (no QKV split)

| LoRA module path | Canonical Flux transformer key |
|---|---|
| `lora_unet_img_in` | `x_embedder` |
| `lora_unet_txt_in` | `context_embedder` |
| `lora_unet_time_in_in_layer` | `time_text_embed.timestep_embedder.linear_1` |
| `lora_unet_time_in_out_layer` | `time_text_embed.timestep_embedder.linear_2` |
| `lora_unet_vector_in_in_layer` | `time_text_embed.text_embedder.linear_1` |
| `lora_unet_vector_in_out_layer` | `time_text_embed.text_embedder.linear_2` |
| `lora_unet_guidance_in_in_layer` | `time_text_embed.guidance_embedder.linear_1` |
| `lora_unet_guidance_in_out_layer` | `time_text_embed.guidance_embedder.linear_2` |
| `lora_unet_final_layer_linear` | `proj_out` |

### Double-stream blocks (`i = 0..18`) — fused QKV requires split

`lora_unet_double_blocks_{i}_img_attn_qkv` is **fused** (Q, K, V concatenated along output dim). The mapper emits 3 `LoraLayer`s sharing the same `lora_down` matrix, with `lora_up` split along dim 0 into 3 equal chunks of `hidden_size` each.

| LoRA module path | Emit (split if QKV) | Canonical Flux key(s) |
|---|---|---|
| `..._img_attn_qkv` | **3 layers** (Q/K/V split) | `transformer_blocks.{i}.attn.to_q` / `to_k` / `to_v` |
| `..._img_attn_proj` | 1 | `transformer_blocks.{i}.attn.to_out.0` |
| `..._img_mlp_0` | 1 | `transformer_blocks.{i}.ff.net.0.proj` |
| `..._img_mlp_2` | 1 | `transformer_blocks.{i}.ff.net.2` |
| `..._img_mod_lin` | 1 | `transformer_blocks.{i}.norm1.linear` |
| `..._txt_attn_qkv` | **3 layers** (Q/K/V split) | `transformer_blocks.{i}.attn.add_q_proj` / `add_k_proj` / `add_v_proj` |
| `..._txt_attn_proj` | 1 | `transformer_blocks.{i}.attn.to_add_out` |
| `..._txt_mlp_0` | 1 | `transformer_blocks.{i}.ff_context.net.0.proj` |
| `..._txt_mlp_2` | 1 | `transformer_blocks.{i}.ff_context.net.2` |
| `..._txt_mod_lin` | 1 | `transformer_blocks.{i}.norm1_context.linear` |

### Single-stream blocks (`i = 0..37`) — fused linear1 requires 4-way split

`lora_unet_single_blocks_{i}_linear1` is fused as `[Q | K | V | mlp_proj]` along output dim. The mapper splits `lora_up` along dim 0 into 4 chunks: 3 of `hidden_size` (Q, K, V) and 1 of `mlp_inner` (mlp_proj). The dimensions are read from the actual `lora_up` shape: `out_dim = 3*hidden + mlp_inner`, where `mlp_inner = mlp_ratio * hidden = 4 * hidden` for Flux Dev.

| LoRA module path | Emit | Canonical Flux key(s) |
|---|---|---|
| `..._linear1` | **4 layers** | `single_transformer_blocks.{i}.attn.to_q` / `to_k` / `to_v` / `proj_mlp` |
| `..._linear2` | 1 | `single_transformer_blocks.{i}.proj_out` |
| `..._modulation_lin` | 1 | `single_transformer_blocks.{i}.norm.linear` |

### CLIP-L (Flux uses single CLIP, not dual)

Same as F1's `lora_te_*` rules, mapped against the Flux pipeline's CLIP-L instance.

### QKV split — concrete algorithm

For a fused `lora_up` of shape `[3*H, R]` and `lora_down` of shape `[R, in_dim]` with rank R and hidden H:

```
lora_up_split = lora_up.view(3, H, R)       # logical reshape
to_q_up = lora_up_split[0]                  # [H, R]
to_k_up = lora_up_split[1]
to_v_up = lora_up_split[2]
# All three share the same lora_down — duplicate the reference (zero-copy)
```

In C# we copy bytes into 3 owned tensors (since the safetensors source is mmap-borrowed). The down matrix is shared by reference across the 3 split layers — same mmap pointer, no copy. Alpha duplicates per split.

**Linear1 4-way**: same idea, split out dim into `[H, H, H, mlp_inner]` chunks.

---

## F4 — AI Toolkit Flux (legacy hybrid) → diffusers Flux transformer keys

> **Status:** Defensive fallback only — **modern AI Toolkit (v0.1.0+, observed via `ostris/yearbook-photo-flux-schnell`) produces F5 format, not F4.** This section is retained in case any older or forked AI Toolkit build emits the hybrid `lora_transformer_` prefix the source code suggested.

The hypothesized hybrid format combines the Kohya-style underscored prefix with PEFT-style suffixes:

- **Prefix:** `lora_transformer_` (Kohya-style underscored)
- **Suffix:** `.lora_A.weight` / `.lora_B.weight` (PEFT-style)
- **Alpha:** dropped during save (line 407 of `network_mixins.py` skips `.alpha` keys when `peft_format=True`)
- **QKV:** **already split** — AI Toolkit targets `attn.to_q`, `attn.to_k`, `attn.to_v` separately (no fused QKV at training time)

### Mapping algorithm

1. Strip `lora_transformer_` prefix.
2. Strip `.lora_A.weight` (→ Down) or `.lora_B.weight` (→ Up). No `.alpha` to strip.
3. Replace all `_` with `.` in the body.
4. Append `.weight` to get the canonical lookup key.

### Worked example

```
lora_transformer_transformer_blocks_0_attn_to_q.lora_A.weight
  → strip prefix:        transformer_blocks_0_attn_to_q
  → underscore-to-dot:   transformer.blocks.0.attn.to.q
                                  ^^ WRONG
```

The naive `_ → .` breaks `transformer_blocks` → `transformer.blocks` and `to_q` → `to.q`. Two fixes required:

**Fix 1 — protected substrings.** Before underscore-to-dot, replace these with placeholders:

| Substring | Placeholder |
|---|---|
| `transformer_blocks` | `TBLOCKS` |
| `single_transformer_blocks` | `STBLOCKS` |
| `to_q`, `to_k`, `to_v` | `TOQ`, `TOK`, `TOV` |
| `to_out` | `TOOUT` |
| `add_q_proj`, `add_k_proj`, `add_v_proj` | `ADDQP`, `ADDKP`, `ADDVP` |
| `to_add_out` | `TOADDOUT` |
| `norm_q`, `norm_k`, `norm_added_q`, `norm_added_k` | `NORMQ`, `NORMK`, `NORMADDQ`, `NORMADDK` |
| `proj_mlp`, `proj_out` | `PROJMLP`, `PROJOUT` |
| `ff_context` | `FFCONTEXT` |
| `time_text_embed`, `timestep_embedder`, `text_embedder`, `guidance_embedder` | `TTE`, `TSE`, `TXE`, `GE` |
| `linear_1`, `linear_2` | `LIN1`, `LIN2` |
| `x_embedder`, `context_embedder` | `XEMB`, `CTXEMB` |

After underscore→dot, restore placeholders with their original underscored forms.

**Fix 2 — alternative: regex replacement.** Use a single regex that matches the structured Flux paths directly. Cleaner, but the placeholder approach is more permissive for edge cases. Recommend placeholders.

### Corrected worked example

```
lora_transformer_transformer_blocks_0_attn_to_q.lora_A.weight
  → strip prefix:           transformer_blocks_0_attn_to_q
  → protect substrings:     TBLOCKS_0_attn_TOQ
  → underscore-to-dot:      TBLOCKS.0.attn.TOQ
  → restore substrings:     transformer_blocks.0.attn.to_q
  → append .weight:         transformer_blocks.0.attn.to_q.weight
  → role: Down (from .lora_A.weight)
```

This matches [FluxDoubleStreamBlock.cs:83](../../src/SharpInference.Diffusion/Models/Denoisers/DiTBlocks/FluxDoubleStreamBlock.cs#L83) which loads `weights[$"{prefix}.attn.to_q.weight"]`.

### CLIP-L in AI Toolkit Flux

When AI Toolkit trains Flux with text-encoder unfrozen (off by default), CLIP-L LoRAs use `lora_te1_*` keys with `.lora_A.weight` / `.lora_B.weight` suffixes. Same mapping rules as F1's CLIP-L, just the PEFT suffix.

---

## F5 — Diffusers PEFT Flux (incl. modern AI Toolkit) → identity

HuggingFace-published Flux LoRAs, **AI Toolkit v0.1.0+ (the dominant Flux trainer)**, and PEFT-based Civitai uploads all save with full diffusers naming, dots throughout, no prefix munging.

**Empirical validation:** `ostris/yearbook-photo-flux-schnell-v1.safetensors` has 494 layers, 988 tensor entries, all keys begin with `transformer.`, all suffixes are `.lora_A.weight` / `.lora_B.weight`, no `.alpha` entries (alpha folded at save time). Format auto-detection picks `DiffusersFlux`; every layer maps cleanly to a real `FluxTransformer` weight key (494/494 hits, 0 misses verified by `Flux_Lora_KeyCoverage_AgainstRealCheckpoint`).

```
transformer.transformer_blocks.0.attn.to_q.lora_A.weight
  → strip leading "transformer.": transformer_blocks.0.attn.to_q.lora_A.weight
  → strip suffix .lora_A.weight: transformer_blocks.0.attn.to_q
  → append .weight: transformer_blocks.0.attn.to_q.weight
```

No underscore-to-dot transformation needed — the keys already use dots correctly. The leading `transformer.` segment is stripped because [FluxTransformer.LoadWeights](../../src/SharpInference.Diffusion/Models/Denoisers/FluxTransformer.cs#L64) is called with no prefix on a dict that already has top-level keys like `transformer_blocks.0.attn.to_q.weight`.

### CLIP-L for diffusers Flux

`text_encoder.text_model.encoder.layers.{L}.self_attn.q_proj.lora_A.weight` → strip `text_encoder.` prefix → `text_model.encoder.layers.{L}.self_attn.q_proj`. CLIP `LoadWeights` is called with `"text_model"` prefix, so the canonical key is `text_model.encoder.layers.{L}.self_attn.q_proj.weight`.

---

## Detection precedence (when multiple signals match)

Run these checks **in order** on the safetensors header keys; first match wins:

1. Any key contains `transformer.transformer_blocks.` or `transformer.single_transformer_blocks.` → **F5 (Diffusers Flux)**
2. Any key starts with `lora_transformer_` → **F4 (AI Toolkit Flux)**
3. Any key starts with `lora_unet_double_blocks_` or `lora_unet_single_blocks_` → **F3 (Kohya Flux)**
4. Any key starts with `lora_unet_down_blocks_` or `lora_unet_up_blocks_` AND any `lora_te2_` key exists → **F2 (Kohya SDXL)**
5. Any key starts with `lora_unet_down_blocks_` or `lora_unet_up_blocks_` → **F1 (Kohya SD1.5)** (default; SDXL without dual-CLIP is treated as SD1.5 — apply step will fail loudly if applied to wrong model)
6. Otherwise → reject with "unsupported LoRA format" error listing the first 5 keys for diagnostic purposes.

The format is opaque to users — they just point at a file. The detector logs which format it picked.

---

## Out of scope (v1) — punt strategies

| Pattern | v1 behavior |
|---|---|
| `hada_w1_a` / `hada_w2_a` (LoHa) | Reject load with "LoHa not yet supported" |
| `lokr_w1` (LoKr) | Reject load with "LoKr not yet supported" |
| `dora_scale` suffix | Drop the dora_scale key with a warning, treat the rest as standard LoRA (lossy but visually close) |
| `*.lora_mid.weight` (LoCon Conv2d mid) | Drop the mid key, log a warning, apply down/up only (mathematically wrong but rarely used in practice) |
| `*.processor.*` (XLabs Flux) | Reject load with "XLabs format not yet supported" |
| FP8 **scaled** base (`fp8_scaled` like Krea/Kontext, `Fp8ScaleFactor != 1.0`) | Reject apply with "Cast checkpoint to F16 before applying LoRA" — the alpha-folded scale factor would interact incorrectly with the merged delta |
| Plain FP8 base (`Fp8ScaleFactor == 1.0`, e.g. Comfy-Org Schnell/Dev FP8) | **Supported.** Merge runs FP8 → F32 (cast in) → F32 accumulator → FP8 (cast out). The cast-back to FP8 is the only lossy step; precision impact is well below baseline FP8 quantization noise. Output stays FP8 so RAM matches the base. |
| Mixed `.lora_down.weight` + `.lora_A.weight` in same file | Reject as malformed |

---

## Test fixtures (Step 6 / 7 will need)

For the end-to-end tests, prefer **small** real LoRAs (rank 8-16, ~5-30 MB) so the test repo doesn't bloat. Candidates:

- **SDXL**: any 5-10 MB Civitai style LoRA. Pin via env var `SDXL_LORA_PATH`.
- **Flux Kohya** (F3): any kohya-trained Flux LoRA from Civitai. Pin via `FLUX_KOHYA_LORA_PATH`.
- **Flux AI Toolkit** (F4): an Ostris-trained Flux LoRA from HuggingFace. Pin via `FLUX_AITOOLKIT_LORA_PATH`. **This is the one that matters** — primary user training path.
- **Flux Diffusers** (F5): HF PEFT-trained Flux LoRA. Pin via `FLUX_DIFFUSERS_LORA_PATH`. Optional, lower priority since less common in the wild than F4.

Tests skip cleanly when env var is not set or path doesn't exist.
