# Music Models — Completion Plan ("fully working + production-ready")

> **Created:** 2026-06-15 · **Scope:** every music model in PHASE_5_AUDIO §0: ACE-Step (v1 / v1.5 / XL),
> Stable Audio Open (1.0 / Small / 2), MusicGen + AudioGen, YuE, DiffRhythm, AudioLDM 2.
> **Companion:** [PHASE_5_AUDIO.md](PHASE_5_AUDIO.md) §7 (the line-item checklist this plan sequences and supersedes for music).

## 0. Definition of "fully working + production-ready"

A music model is **done** only when ALL of these hold (today *none* of them clear the bar — the gating gap is validation):

1. **Loads the real checkpoint** — every tensor resolves; converter handles the shipped layout(s).
2. **Numerically validated** — a layer-by-layer diff against a Python reference is within tolerance (~1e-3 F32 / ~1e-2 bf16) at every tap. **This is the universal missing piece.**
3. **Real-weight end-to-end** — produces correct audio (right sample rate, channels, duration), verified by a spectral/perceptual check vs the reference output, not just "finite floats."
4. **GPU path** — runs on the CUDA backend (not CPU-only), with a VRAM-probe skip and a documented footprint.
5. **Feature-complete** for the model's headline capabilities (e.g. ACE edit/repaint, YuE accompaniment, MusicGen stereo/melody) — or the un-built features are explicitly scoped out with a reason.
6. **Tests** — structural (synthetic, CPU), a real-weight load smoke, the parity diff, and an env-gated E2E.

Everything below is ordered to reach that bar with the least total work, closest-to-done first.

---

## 1. The keystone: an audio parity-harness methodology (build FIRST)

**Problem:** there are **zero** Python parity references for any audio model (`tests/python-reference/` has none for music). Every existing music test is synthetic-structural or a real-weight *smoke* — none prove numerical correctness. Without this, "fully working" is unprovable.

**Solution — replicate the Ideogram 4 harness pattern** ([dump_ideogram4_full_forward.py](../../tests/python-reference/dump_ideogram4_full_forward.py) + [diff_ideogram4_layers.py](../../tests/python-reference/diff_ideogram4_layers.py) + [Ideogram4DiffTests.cs](../../tests/HartsyInference.Diffusion.Tests/Ideogram4DiffTests.cs)). For each model:

- **Tiny synthetic checkpoint** (small dims, seeded) saved as safetensors + fixed inputs as raw bins → an **independent PyTorch reimplementation** dumps tap points → C# loads the same and runs with a `*_DEBUG_DIR` env (each music model needs a `*DebugDump` like ACE-Step does not yet have) → `diff_*.py` compares.
- **Runs on CPU/float32 — no big weights, no GPU** → executes on the local 3060 (and CI), de-risking the math before any 13 GB real-weight run.
- For models with an official Python package (MusicGen=AudioCraft/HF, Stable Audio=stable-audio-tools, AudioLDM2/DiffRhythm=diffusers), add a **second** dump that hooks the *real* upstream model for true cross-impl parity once weights are downloaded on the cloud GPU.

**Deliverable T0 (1–2 days, unblocks everything):** a shared `tests/python-reference/_audio_harness.py` helper (synthetic-checkpoint writer + tap-dump + index.json, generalized from the Ideogram script) and a C# `AudioDebugDump` mirror of `Ideogram4DebugDump`. Every model below reuses it.

> **PROGRESS (2026-06-15): the harness pattern is proven on ACE-Step v1's DiT core.** `AceStepDebugDump` + `dump_ace_step_dit.py` + `diff_ace_step_layers.py` + `AceStepDitDiffTests` are live and **match to ~1e-8 across every tap**. A useful variation was found: instead of Python generating weights (Ideogram pattern), the **C# test generates the synthetic checkpoint via the existing `*SyntheticWeights` helpers + `SafeTensorsWriter` and Python consumes it** — this avoids re-synthesizing large key surfaces (e.g. ACE's lyric Conformer) in Python. That's the template for every model below. Next: factor the shared Python utils out of `dump_ace_step_dit.py`, and add per-model `*DebugDump` classes.

---

## 2. Shared infrastructure (build once — many models block on these)

| # | Component | Needed by | Status | Effort |
|---|---|---|---|---|
| S1 | **Audio parity harness** (§1, T0) | ALL | none | S |
| S2 | **EnCodec 32 kHz support** (MusicGen) — ⚠ **bigger than a preset:** `facebook/encodec_32khz` uses `time_group_norm` (GroupNorm, not weight-norm), **non-causal** convs, ratios `[8,5,4,4]`→50 Hz, `n_filters=64`, codebook_size **2048**. The current SeaNet (weight-norm + causal only, codebook 1024) needs GroupNorm conv blocks + non-causal padding + a 2048-codebook RVQ before a 32 kHz preset works. | MusicGen | EnCodec 24 kHz only | **M** |
| S3 | **EnCodec 16 kHz support** (AudioGen) — same SeaNet additions as S2 at 16 kHz | AudioGen | "" | M |
| S4 | **`dpmpp-3m-sde` scheduler** (order-3 multistep DPM++, EDM v-pred, polyexp σ) | Stable Audio Open 1.0 | missing (PingPong/Euler/DPM++2M exist) | M |
| S5 | **`FourierFeatures1D` + `NumberConditioner`** (timing embed) | Stable Audio | missing | S |
| S6 | **DCAE encoder** (mirror of `MusicDcaeDecoder`) | ACE-Step v1 edit/repaint | decoder only | M |
| S7 | **RoBERTa BPE tokenizer + CLAP text encoder** (RoBERTa-base + proj→512) | AudioLDM 2 | missing | M |
| S8 | **GPT-2 continuous-feature decoder** (no lm_head, 8-step deterministic) | AudioLDM 2 | missing (first decoder-only in Diffusion) | M |
| S9 | **English G2P pipeline** (CMUDict trie + heteronyms + POS + OOV fallback) | DiffRhythm | research only ([G2P_PHONEMIZATION.md](../Research/G2P_PHONEMIZATION.md)) | L |
| S10 | **MuQ-MuLan style encoder** (XLM-RoBERTa text + MuQ audio → 512) | DiffRhythm (deferrable) | missing | L |
| S11 | **Vocos 16→44.1 kHz upsampler preset** | YuE (optional), ACE | Vocos exists (24 kHz) | S |

Reusable as-is (confirmed present): **T5TextEncoder** (+ `Umt5Base`, `T5Large` presets), **Oobleck VAE** (enc+dec), **XCodec/DAC/EnCodec/Mimi/Snac** codecs, **FlowMatchEuler / PingPong / DDIM / DDIMvPred / DPM++2M / Euler** schedulers, **AdaLN modulation, RoPE, SwiGLU, RMSNorm, HiFi-GAN, mel/STFT** primitives, **Qwen2Model** (YuE), **NucleusSampler**, **VaeDecoder** (mel-VAE config reuse).

---

## 3. Per-model plans (ROI-ordered: closest-to-done first)

### M1 — ACE-Step v1 (flagship; most built) → **highest priority**
**State:** DiT, lyric Conformer (8L), DCAE *decoder*, ADaMoSHiFiGAN vocoder, tokenizer, guidance (CFG/APG/CFG-Zero★), Euler/Heun/PingPong, pipeline, converter — all built; **real-weight LOAD smoke passes**; env-gated CPU E2E runs (~13 GB F32). Not numerically validated; CPU-only; no edit/repaint.
**Tasks:**
1. `AceStepDebugDump` + `dump_ace_step_forward.py` + `diff_ace_step_layers.py` (uses S1). Tap: text-enc → Conformer → patch-embed → per-DiT-layer → final velocity → DCAE → vocoder. **(the §9 line-402 checklist item)**
2. Synthetic-weight CPU parity pass (math), then real-weight single-step DiT parity on the cloud GPU (avg_err < 1e-3).
3. **DCAE encoder (S6)** → unlocks edit/repaint/reference-audio + the silence-latent path.
4. **Flow-edit / repaint** masking loop (masked dual-conditioning velocity).
5. **GPU path**: run pipeline on CUDA backend; VRAM-probe; benchmark vs the §10 target (4-min song ≤ ~120 s on 3060).
6. Optional: GGUF Q4/Q8 DiT path (drops the 13 GB F32 requirement).
**DoD:** parity green; real-weight E2E song on GPU matches reference spectrally; edit/repaint working.

### M2 — MusicGen + AudioGen (decoder done; codec-blocked) → **quick win**
**State:** decoder LM (small/med/large), delay pattern, CFG, pipeline, converter all built + structural tests. **Blocking bug:** decodes with 24 kHz EnCodec but MusicGen needs **32 kHz/50 Hz** → wrong-rate/garbled audio. AudioGen needs 16 kHz.
**Tasks:**
1. **S2 (EnCodec 32 kHz)** + wire into `MusicGenPipeline`; verify output length = 32 kHz × duration.
2. **S3 (EnCodec 16 kHz)** → AudioGen is then "MusicGen with a different codec," same pipeline.
3. Parity harness (uses S1) + real-weight validation vs HF `musicgen-small` / `audiogen-medium` (logits + waveform STFT-corr > 0.99).
4. Stereo preset (8 codebooks, delay `[0,0,1,1,2,2,3,3]`, 2-ch codec) — optional.
5. Melody conditioning (chromagram → cross-attn prepend) — optional, reuse Whisper CQT.
6. GPU path + env-gated E2E test.
**DoD:** correct-rate audio; parity vs HF; mono text-to-music + AudioGen working (stereo/melody optional).

### M3 — ACE-Step v1.5 (turbo; built, unvalidated)
**State:** Qwen3-style DiT (GQA, sliding/full), dual-timestep, condition encoder (lyric/timbre/text), Oobleck VAE, 8-step Euler pipeline — all built; synthetic tests only.
**Tasks:**
1. Parity harness (uses S1) + real-weight 8-step E2E on the public turbo checkpoint.
2. GPU path (turbo is <4 GB bf16 — fits the 3060 comfortably).
3. **Phase 2 (hints):** FSQ tok/detok (ResidualFSQ [8,8,8,5,5,5]) + planner-LM consumption + repaint/chunk-mask. Scope as a follow-on; hints-less T2M ships first.
**DoD:** parity green; hints-less turbo song on GPU; phase-2 hints tracked separately.

### M4 — Stable Audio Open (VAE done; needs the DiT)
**State:** Oobleck VAE (enc+dec) complete; T5-base + PingPong scheduler present. Missing: DiT, timing conditioning, dpmpp-3m-sde, pipeline.
**Tasks:**
1. **`StableAudioDiT`** — single-stream 1-D, config-driven variants: Open 1.0 (1536/24L/24h, dpmpp, v-pred, CFG) vs Small (1024/16L/16h, **QK-RMSNorm**, PingPong, rectified-flow, no `seconds_start`, no CFG). Cross-attn KV = 12 heads (MQA-style repeat-interleave). Reuse AdaLN/RoPE/SwiGLU.
2. **S5** timing conditioning (seconds_start/total → Fourier → cross-attn KV + global AdaLN).
3. **S4** dpmpp-3m-sde (Open 1.0); reuse PingPong (Small).
4. `StableAudioConfig` + `StableAudioPipeline` + converter (diffusers folder AND stable-audio-tools single-file layouts).
5. Parity harness (uses S1) + real-weight E2E (Open Small first — 8 steps, cheapest); GPU path.
**DoD:** parity green; Open 1.0 + Small generate correct clips on GPU; "Open 2" folds in as a config/weights variant once published.

### M5 — YuE (Stage-1 done; needs Stage-2)
**State:** Stage-1 LM (Qwen2 reuse) + XCodec decode + pipeline built; **codebooks 1–7 zero-filled**, accompaniment discarded, Vocos upsampler deferred, unvalidated.
**Tasks:**
1. **Stage-2 residual upsampler** — second `Qwen2Model` (~1.5B) consuming S1 cb0 → emits codebooks 1–7, windowed (~30 s) for KV-cache. Replace the zero-fill in `YuePipeline`.
2. **Dual-track mixing** — decode vocal + accompaniment separately via XCodec, mix.
3. Checkpoint validation — `m-a-p/YuE-s1-7B-*` + `s2-1B` + `xcodec_mini_infer`; reconcile token bases/keys; Stage-1 token-sequence parity (bit-exact first ~100 tokens).
4. **S11** Vocos 16→44.1 kHz upsampler (optional; ship 16 kHz first).
5. GPU path (7B S1 is the VRAM driver — eviction discipline / quant).
**DoD:** full 8-codebook dual-track song; Stage-1 token parity; XCodec round-trip STOI > 0.95.

### M6 — AudioLDM 2 (greenfield, high reuse)
**State:** none. ~65–75 % reusable (T5-Large, UNet 2D, HiFi-GAN, mel-VAE, DDIM/DPM++ all exist).
**Tasks:**
1. **S7** CLAP (RoBERTa + proj) + RoBERTa tokenizer.
2. **S8** GPT-2 continuous-feature decoder (8-step deterministic, KV-cache, no lm_head).
3. `AudioLDM2ProjectionModel` (trivial) + **dual-stream cross-attn routing** in the UNet Transformer2D (GPT-2 @768 ‖ T5-Large @1024 per-sublayer table).
4. Mel-VAE config (1-ch in, 8-ch latent, scale_factor 4, scaling_factor 0.41109) + SpeechT5 HiFi-GAN preset (5-stage, 64 mel, 16 kHz, log-magnitude mel).
5. DDIM (verify `set_alpha_to_one=false`, `steps_offset=1`) / DPM++2M; pipeline; converter.
6. Parity harness (uses S1) + real-weight E2E (text-to-audio SFX) + GPU.
**DoD:** parity green; correct 16 kHz audio vs diffusers reference.

### M7 — DiffRhythm (greenfield, heaviest)
**State:** none. Biggest net-new lifts: G2P (S9) + MuQ-MuLan (S10).
**Tasks:**
1. **VAE** (Stable-Audio-2-derived 1-D Conv + Snake, 44.1 kHz stereo → 64-ch @ 21.5 Hz). Convert TorchScript `.pt` → safetensors (Python pre-pass). Reuse Oobleck snake kernel.
2. **DiT** (1.1B, 16L, d2048, 32h, SwiGLU, 1-D RoPE, per-block cross-attn to lyrics, AdaLN-Zero) — reuse Flux/SD3 blocks.
3. **S9** G2P + phoneme embedding; **LRC parser** + sentence-level alignment mask (cheap).
4. Flow-match **Euler + Sway** (reuse `FlowMatchEuler` + sway remap helper) + CFG (reuse).
5. **S10** MuQ-MuLan style conditioning — **deferrable**: v1.0 ships with audio-prompt/hard-coded style; full MuLan is v1.2.
6. Parity harness + real-weight E2E + GPU.
**DoD:** parity green; full-song generation from lyrics+style on GPU (MuLan text-prompt path may trail in v1.2).

### Deferred / scope-flagged
- **ACE-Step XL (5B)** — not built; fold in after v1 is validated (shares infra). Track separately.
- **ACE-Step v1.5 hints (FSQ + planner LM)** — phase 2 of M3.
- **MusicGen melody/stereo, DiffRhythm MuLan, AudioLDM2 CLAP audio-rerank** — optional headline features, each gated behind its base model shipping first.

---

## 4. Sequencing & phasing

**Phase A — Validation foundation + quick wins (unblocks correctness everywhere)**
S1 harness → M1 ACE-Step v1 parity → M2 MusicGen/AudioGen codec fix (S2/S3) + parity → M3 ACE v1.5 parity. *Outcome: 3–4 of the existing models become genuinely validated.*

**Phase B — Finish the partially-built models**
M1 DCAE encoder (S6) + edit/repaint → M5 YuE Stage-2 + mixing → M4 Stable Audio DiT (S4/S5) + pipeline. *Outcome: feature-complete ACE, YuE, Stable Audio.*

**Phase C — Greenfield models**
M6 AudioLDM 2 (S7/S8) → M7 DiffRhythm (S9, VAE, DiT; S10 MuLan deferred).

**Phase D — GPU + perf + breadth**
GPU paths + §10 perf targets across all; ACE-Step XL; v1.5 hints; MusicGen stereo/melody; DiffRhythm MuLan.

Rough effort (sizing, not commitments): Phase A ≈ 1–2 wks · Phase B ≈ 2–3 wks · Phase C ≈ 4–6 wks (G2P + MuLan dominate) · Phase D ≈ 2–3 wks.

---

## 5. Top risks
1. **No real weights / VRAM locally** — the synthetic CPU parity harness (S1) is the mitigation: it validates math on the 3060; the 13 GB+ real-weight runs go to the cloud GPU. Every model must have the synthetic harness *before* its real-weight run.
2. **G2P accuracy (DiffRhythm)** — CMUDict covers ~99 % of normal English; OOV neural fallback + a 10k-sentence validation set needed.
3. **MuQ-MuLan (DiffRhythm)** — large net-new audio+text towers; deferred to v1.2 so DiffRhythm can ship audio-prompt-first.
4. **GPT-2 continuous-feature mode (AudioLDM 2)** — first decoder-only in the Diffusion package; validate step-by-step against diffusers.
5. **Codec sample-rate correctness (MusicGen/YuE/Stable Audio)** — wrong frame rate silently produces plausible-but-wrong audio; the codec round-trip STOI test (§9) is the guard.
6. **GPU kernels at music shapes** — long sequences (full songs) stress attention/VRAM; eviction discipline + (where available) Flash-Attention path.
