# AuK GPU handoff

State: structural port complete (DiT, Qwen2.5-Omni thinker + audio tower, BigVGAN-flow VAE, Flash/base samplers,
`AukPipeline`, engine descriptor `auk:flash` / `auk:base`, CLI `--instruction`/`--duration`, bf16 repack recipes).
Verified only on CPU with tiny synthetic weights against double-precision references. **No real-weight run has
happened; do not claim parity.** Architecture facts: [AUK_ARCHITECTURE.md](AUK_ARCHITECTURE.md). Upstream:
`github.com/Tencent-Hunyuan/AuK` (`src/auk/infer/infer_auk.py`, `model/cfm_edit.py`, `model/flux2_edit.py`,
`model/modules.py`, `model/vae/bigvgan_flow_vae.py`).

## 1. Smoke the real weights (do first)
1. `hartsy pull auk` (both DiTs, VAE, Qwen shards 1+2, tokenizer; sha pins are in `ModelCatalog`).
2. Run the unit lane for `Auk|QwenOmni|Qwen2`, then `hartsy speak -m auk:flash --reference ref.wav --text "..."`.
3. Fix anything the load path rejects. Never loosen the fail-fast shape checks without evidence. Untested offline:
   engine load against real Qwen shards (`AudioCheckpoints.LoadAsync` `keyPrefixes` filter), Omni `tokenizer.json`
   via `HfTokenizerJson.LoadByteLevelBpe`.

## 2. Parity against the Python reference (write `tests/python-reference/auk_ref.py`, model on `f5_ref.py`)
Dump with injected noise (never rely on seeds) and compare stage by stage, bisecting before touching tolerances:
1. Whisper 128-bin features (zero-padded to 300 s) and `audio_tower` output.
2. Qwen hidden states 1..36 (HF `output_hidden_states[1:]`; the last is post-final-norm).
3. Fused text embedding (LayerNorm no-affine eps 1e-5, softmax(`layer_weights`), `layer_scale`).
4. DiT velocity at fixed x, t, ref, text (F32; cond and uncond).
5. Euler loop with injected noise (Flash 4 steps; base 32 steps, cfg 2.0, sway -1).
6. VAE decode of a given latent; VAE encode with injected eps.
7. End to end, then Whisper STT recovery of the generated speech.
Record evidence in `docs/Checklists/PARITY_VERIFICATION.md` and flip the MODEL_STATUS_AUDIO row.

## 3. Highest silent-failure risks (check these first if parity is off)
- **Layer tap off-by-one.** Taps are layers 0..34 raw plus the final-normed output; real `layer_weights` put about
  0.676 on index 0, so a one-position shift ruins ~68% of the conditioning.
- **`inv_freq`** must come from `transformer.rotary_embed.inv_freq` (bf16-rounded), not theory.
- **RMSNorm epsilon**: engine uses float32 eps 1.1920929e-7; upstream `nn.RMSNorm(eps=None)` uses the input dtype's
  eps, so a bf16 reference run differs.
- **Qwen chat framing and tokenizer**: system turn "You are a helpful assistant."; the processor may substitute its
  own default; check against `Qwen2_5OmniProcessor`. Newline and leading-space handling in the byte-level BPE.
- **Mel**: zero-padded tail (not reflect), 128 bins; `<|AUDIO|>` count = ((L-1)//2+1 - 2)//2 + 1.
- **Ref length**: pipeline trims the reference to a multiple of 480 samples; upstream encodes the whole clip and
  masks the excess. Document or match.
- **VAE**: encoder is symmetric (non-causal), decoder causal with non-causal anti-alias upsample and causal
  downsample (replicate pad 11/0); `global_log_std` is a variance (sqrt on use). `Causal`/`ActCausal` defaults come
  from the upstream dataclass; confirm against the release config. Snake divisor 1e-8 (backend) vs 1e-9 (upstream).
- Instruct-TTS template wording is inconsistent upstream (documented vs example form); both are in `AukTemplates`.
- Edit tasks reuse the same pipeline with duration = source length; 30 s budget (source + target) is enforced.

## 4. GPU / performance work not yet done
- Host loops: VAE `exp(log_std)`/noise combine, Qwen audio tower mel/position copies via `DataPointer`, F5-style
  `MishInPlace` is not used (prefer `IBackend.Mish`). Check device-resident behavior on CUDA and Vulkan.
- `Qwen2Model.ForwardEmbeds` takes no cancellation token; the thinker pass is uninterruptible.
- Residency: `AukOptions.SequentialResidency` frees device weights stage by stage; verify peak VRAM (~9-10 GB bf16
  target; Python reference peaks ~24.8 GiB) and speed. Add GpuIntegration tests for the affected backends.
- Shared code touched: `GenericTransformer.ForwardEmbeds` gained an optional `layerTap`. Run
  `tests/regression-ab.sh` (A/B against `origin/main`, identical expected) per AGENTS.md.

## 5. Repack and upload (needs the user's HF token)
- Recipes: `tools/repack/recipes/auk-base.json`, `auk-flash.json`, `auk-vae.json` (bf16 with F32 for norms, biases,
  `layer_*`, `inv_freq`; VAE stays F32 and drops training-only `flow.*`). Run `CheckpointRepacker` on the real
  files, verify against a torch cast, then upload to the **Hartsy** org with MIT LICENSE copied from tencent and a
  README stating "format/dtype repack, no weight changes". Update the catalog sources/shas afterward.
- **Do not re-host Qwen2.5-Omni** (Qwen Research License); the engine pulls `Qwen/Qwen2.5-Omni-3B` shards 1+2.
- `tools/repack/audiolab_identity.json` `tts/auk` family/variant names were invented; confirm against AudioLab.

## 6. SwarmUI-AudioLab
Branch has the `auk_tts` provider (flash/base), params, bridge binding, `_engineCompanionRepos` (Qwen repo must
also be present for "installed"), README row and bridge tests. It cannot build until this engine release is
published: then bump the `HartsyInference` and `HartsyInference.Voice` pins in `SwarmUI-AudioLab.csproj` to the
release carrying AuK (alpha.244 or later), build, run the AudioLab tests, and verify `DurationSeconds`/`Instruction`
bind to the engine types (`double?`/`string?`). Check Number() conversions and that the AuK params show only for the
AuK provider in the UI. Then run a Swarm end-to-end clone and instruct-TTS generation.

## 7. Not built (phase 3)
Editing/enhancement/separation engine service and an `auk_edit` AudioLab provider (English templates exist in
`AukTemplates`), base-model duration heuristics tuning, Chinese templates.

## Ship rules
kalebbroo is sole author: no co-author or tool-attribution trailers anywhere. Draft PR, CPU lane + affected GPU
suites, CHANGELOG `## alpha.N` + `VersionSuffix` bump per code PR, resolve all bot comments before merge.
