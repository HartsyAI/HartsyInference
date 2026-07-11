# Music Models — Generation Parameter Audit (engine vs official)

> **Question this answers:** for each music model the engine has *numerically verified*, do we expose **every
> generation parameter the real model supports** (so SwarmUI can surface lyrics, tags, guidance, edit modes, …)?
> **Method:** the official param list (from each model's real pipeline / repo, source-linked) matched line-by-line
> against the C# pipeline `Generate`/`Synthesize` surface + its config. Status: ✅ exposed per-call · ⚠️ present in
> config only (not a per-generation arg) · ❌ missing · ➖ n/a.
>
> **Completeness of the lineup itself** (from [MUSIC_MODELS_COMPLETION_PLAN.md](MUSIC_MODELS_COMPLETION_PLAN.md) +
> [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md)): built + verified = ACE-Step v1, ACE-Step v1.5 turbo,
> MusicGen-small, YuE stage-1, HeartMuLa. **Not built:** Stable Audio Open (1.0/Small/2), DiffRhythm, AudioLDM 2,
> ACE-Step XL, YuE **stage-2** (multi-codebook upsampler). So "all music models complete" = the 5 above are
> *numerically* done, but (a) several are **not param-feature-complete** (see below), and (b) 5+ models are unbuilt.

---

## Headline

> **Status 2026-06-30:** all P0 (correctness) + P1 (per-call sampling) items below are **implemented** and the full
> solution builds clean. Remaining gaps are P2 feature-subsystems (task modes / multi-segment / stage-2 / cover-mode
> LM) and unbuilt models, not generation-param gaps. Numerics for the new guidance/CFG paths are validation-pending
> (previous parity was teacher-forced LM logits / plain-CFG velocity).

| Model | Numerics | Param coverage | Remaining (P2 features / unbuilt) |
|---|---|---|---|
| **ACE-Step v1** | ✅ | ✅ **all generation params** (interval/decay/min, omega, dual text/lyric CFG, oss_steps, defaults 60/15) | task modes (retake/repaint/edit/extend/audio2audio) + LoRA; ERG attn-temperature hooks |
| **ACE-Step v1.5 turbo** | ✅ | ✅ **complete for T2M** (shift/seed/timbre/lyrics/hints; turbo has no CFG) | cover-mode LM sampling (via `lmHints`, phase-2), sde method, task modes |
| **MusicGen** | ✅ | ✅ **per-call** temp/top_k/top_p/cfg + use_sampling | two_step_cfg; extend_stride (>30 s); melody + continuation |
| **YuE stage-1** | ✅ (LM logits) | ✅ **CFG + per-call sampling** (temp/top_k/top_p/rep-pen/guidance) | run_n_segments, audio-prompt ICL, dual-track, stage-2 upsampler |
| **HeartMuLa** | ✅ (LM logits) | ✅ **cfg_scale + per-call sampling** (temp/top_k/top_p) | tags(text-style) conditioning wiring |

> ✅ **Cross-cutting correctness gap RESOLVED:** YuE stage-1 and HeartMuLa now run **classifier-free guidance in their
> generation loop** (YuE `guidanceScale` default 1.5 via a parallel uncond KV cache; HeartMuLa `cfgScale` 1.5 via a
> parallel backbone+depth uncond pass in `CsmModel.GenerateFrame`). The previously-verified parity was teacher-forced
> LM logits (corr 1.0), which did not exercise the CFG combine step; the CFG paths themselves are validation-pending.

---

## ACE-Step v1  (`AceStepPipeline.Generate`)
Official: `ACEStepPipeline.__call__` — https://raw.githubusercontent.com/ace-step/ACE-Step/main/acestep/pipeline_ace_step.py

C# `Generate(textEmbeds, lyricIds, durationSeconds, steps?, guidance?, guidanceMode, sampler, seed?, speakerVec?,
guidanceInterval?, guidanceIntervalDecay?, minGuidanceScale?, omegaScale?, guidanceScaleText?, guidanceScaleLyric?,
ossSteps?, onProgress?)` + `AceStepConfig`: FlowShift 3.0, NumInferenceSteps **60**, GuidanceScale **15.0**,
GuidanceInterval 0.5, GuidanceIntervalDecay 0.0, MinGuidanceScale 3.0, OmegaScale 10.0, GuidanceScaleText/Lyric 0.0.

### text2music (core)
| Official param | Default | C# | Status |
|---|---|---|---|
| prompt (tags) | — | `textEmbeds` (pre-encoded UMT5) | ✅ (caller encodes) |
| lyrics | — | `lyricIds` (pre-tokenized) | ✅ (caller tokenizes) |
| audio_duration | 60 (UI -1=rand) | `durationSeconds` | ✅ |
| infer_step | 60 | `steps` / cfg 60 | ✅ |
| guidance_scale | 15.0 | `guidance` / cfg 15.0 | ✅ |
| scheduler_type (euler/heun/pingpong) | euler | `sampler` (Euler/Heun/PingPong) | ✅ |
| cfg_type (cfg/apg/cfg_star) | apg | `guidanceMode` (Cfg/Apg/CfgZeroStar) | ✅ |
| **omega_scale** | 10.0 | `omegaScale` / cfg 10.0 | ✅ (mean-preserving Euler rescale) |
| manual_seeds | — | `seed` | ✅ |
| **guidance_interval** | 0.5 | `guidanceInterval` / cfg 0.5 | ✅ (start/end idx gate) |
| **guidance_interval_decay** | 0.0 | `guidanceIntervalDecay` / cfg 0.0 | ✅ |
| **min_guidance_scale** | 3.0 | `minGuidanceScale` / cfg 3.0 | ✅ |
| **oss_steps** | None | `ossSteps` (1-indexed subset) | ✅ |
| **guidance_scale_text** | 0.0 | `guidanceScaleText` / cfg 0.0 | ✅ (dual-condition CFG) |
| **guidance_scale_lyric** | 0.0 | `guidanceScaleLyric` / cfg 0.0 | ✅ (dual-condition CFG) |
| **use_erg_tag / lyric / diffusion** | True | omega covers ERG-diffusion; tag/lyric = attn-temperature hooks | ⚠️ omega done; tag/lyric attn hooks deferred |
| batch_size | 1 | — (caller loops) | ➖ |
| format | wav | — (engine returns float[]) | ➖ |
| (speaker/timbre) | — | `speakerVec` | ✅ extra |

### task modes — ALL missing (engine only does text2music)
| Mode | Official params | C# |
|---|---|---|
| **retake** | retake_seeds, retake_variance (0.5) | ❌ |
| **repaint** | repaint_start, repaint_end, src_audio_path | ❌ |
| **extend** | (repaint region outside audio) left/right_extend_length | ❌ |
| **edit** | edit_target_prompt, edit_target_lyrics, edit_n_min (0), edit_n_max (1), edit_n_avg (1) | ❌ |
| **audio2audio** | audio2audio_enable, ref_audio_strength (0.5), ref_audio_input | ❌ |
| **LoRA** | lora_name_or_path, lora_weight (1.0) | ❌ |

**Verdict (updated):** ACE-Step v1 text2music now exposes the **full official guidance param set** (interval/decay/min,
omega, dual text+lyric CFG, oss_steps) and defaults are reconciled to the official pipeline (60 steps / guidance 15).
ERG-diffusion is covered by `omega_scale`; the ERG tag/lyric **attention-temperature** hooks (tau≈0.01) and the
edit/repaint/retake/extend/audio2audio + LoRA task-mode subsystem remain deferred (larger features, not params).
The new guidance paths are numerically validation-pending.

---

## ACE-Step v1.5 turbo  (`AceStepPipeline15.Generate`)
Official (turbo is a distilled 2-stage LM+DiT model; UI guide): https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GRADIO_GUIDE.md

C# `Generate(textHidden, lyricHidden?, durationSeconds, shift?, seed?, timbreLatent?, lmHints?, onProgress?)` + cfg NumInferenceSteps 8, FlowShift 3.0.

| Official (DiT) | Default | C# | Status |
|---|---|---|---|
| Inference Steps | 8 (turbo) | cfg 8 (fixed) | ✅ |
| Shift | 3.0 | `shift` | ✅ |
| Seed | -1 | `seed` | ✅ |
| Guidance / CFG | — | — | ➖ (**turbo has NO CFG** — correct to omit) |
| Inference Method (ode/sde) | ode | — (Euler/ode only) | ❌ (sde not supported) |
| Custom Timesteps | — | — | ❌ |
| **LM Temperature** | 0.85 | — | ❌ |
| **LM CFG Scale** | 2.0 | — | ❌ |
| **LM Top-K / Top-P** | 0 / 0.9 | — | ❌ |
| **LM Negative Prompt** | "NO USER INPUT" | — | ❌ |
| LM Codes Strength | 1.0 | `lmHints` (partial) | ⚠️ |
| timbre conditioning | — | `timbreLatent` | ✅ |
| task modes (Remix/Repaint/Extract/…) | — | — | ❌ |

**Verdict:** the turbo T2M diffusion path is **param-complete** — `shift` (∈{1,2,3}), `seed`, `timbreLatent`,
`lyricHidden`, and `lmHints` are all exposed per-call, and the distilled turbo model has **no CFG** and a **fixed
8-step** schedule (so guidance_scale / step-count / sde would break it — correctly omitted). The remaining ❌ items
(LM temperature/CFG/top-k/top-p/negative-prompt, custom timesteps, task modes) belong to the **FSQ cover-mode LM
detokenizer** — the phase-2 subsystem reached via `lmHints`, not the T2M diffusion. Those are a deferred feature, not
a missing generation param. The engine takes pre-computed Qwen3 states, so the prompt/lyric LM front-end is external.

---

## MusicGen  (`MusicGenPipeline.Synthesize`)
Official: AudioCraft `MusicGen.set_generation_params` — https://raw.githubusercontent.com/facebookresearch/audiocraft/main/audiocraft/models/musicgen.py

C# `Synthesize(t5States, seconds, seed, guidance?, temperature?, topK?, topP?, useSampling=true)` + `MusicGenConfig`:
Temperature 1.0, TopK 250, TopP 0.0, GuidanceScale 3.0.

| Official param | Default | C# | Status |
|---|---|---|---|
| use_sampling | True | `useSampling` (false → argmax) | ✅ |
| top_k | 250 | `topK` / cfg | ✅ per-call |
| top_p | 0.0 | `topP` / cfg | ✅ per-call |
| temperature | 1.0 | `temperature` / cfg | ✅ per-call |
| duration | 30 | `seconds` | ✅ |
| cfg_coef (guidance) | 3.0 | `guidance` / cfg | ✅ per-call |
| cfg_coef_beta (double CFG) | None | — | ❌ (melody double-CFG) |
| two_step_cfg | False | — | ❌ |
| extend_stride (>30 s) | 18 | — | ❌ (no chunked extension → capped ~30 s) |
| text conditioning | — | `t5States` (pre-encoded) | ✅ |
| melody (generate_with_chroma) | — | — | ❌ |
| continuation (audio prompt) | — | — | ❌ |
| unconditional | — | — | ❌ |
| seed | — | `seed` | ✅ extra |

**Verdict (updated):** all AudioCraft `set_generation_params` sampling knobs (temp/top_k/top_p/cfg_coef + use_sampling)
are now **per-call** args. Remaining gaps are larger features: extend_stride (>30 s chunking), two_step_cfg,
melody (chroma), and audio continuation.

---

## HeartMuLa  (`HeartMulaPipeline.Generate` / `GenerateCodes`)
Samples via the shared `CsmModel.GenerateFrame` using `CsmConfig`: Temperature 0.9, TopK 50, TopP 1.0.

C# `Generate(lyricsTokens, maxFrames, seed, muqLmEmbed?, temperature?, topK?, topP?, cfgScale?)` (same for
`GenerateCodes`); sampling defaults from `HeartMulaConfig` (Temperature **1.0**, TopK 50, TopP 1.0, CfgScale 1.5).

| Param | Default | C# | Status |
|---|---|---|---|
| lyrics conditioning | — | `lyricsTokens` (pre-tokenized) | ✅ |
| style embedding (MuQ-MuLan) | — | `muqLmEmbed` | ✅ (staged) |
| duration / max frames | — | `maxFrames` | ✅ |
| temperature | 1.0 | `temperature` / cfg 1.0 | ✅ per-call (default fixed 0.9→1.0) |
| top_k | 50 | `topK` / cfg | ✅ per-call |
| top_p | 1.0 | `topP` / cfg | ✅ per-call |
| **cfg_scale** | 1.5 | `cfgScale` / cfg 1.5 | ✅ **CFG now applied** (parallel uncond backbone+depth pass in `CsmModel.GenerateFrame`; uncond = no lyrics/style) |
| repetition_penalty | (none) | — | ➖ (official doesn't expose) |
| seed | (none) | `seed` | ✅ extra |

Official: heartlib `examples/run_music_generation.py` — https://github.com/HeartMuLa/heartlib (3B/7B, HeartCodec 12.5 Hz/48 kHz).
CFG is applied per frame across **all** codebooks (backbone c0 head + each depth-decoder codebook head). The engine's
`muqLmEmbed` is the MuQ-MuLan style path; comma-separated text-**tags** conditioning is a separate wiring item (P2).

---

## YuE stage-1  (`YuePipeline.Synthesize`)
Official: `inference/infer.py` — https://github.com/multimodal-art-projection/YuE

C# `Synthesize(promptTokenIds, maxFrames, seed, temperature?, topK?, topP?, repetitionPenalty?, guidanceScale?,
uncondTokenIds?)` + `YueConfig`: Temperature 1.0, TopK 50, TopP 0.93, RepetitionPenalty 1.1, GuidanceScale 1.5;
`YueStage1Lm` applies repetition penalty + nucleus sampling + CFG.

| Official param | Default | C# | Status |
|---|---|---|---|
| genre_txt (tags) | required | `promptTokenIds` (caller tokenizes tags+lyrics) | ✅ |
| lyrics_txt ([verse]/[chorus]/[bridge]) | required | `promptTokenIds` | ✅ (caller structures) |
| max_new_tokens | 3000 | `maxFrames` | ✅ |
| temperature | 1.0 | `temperature` / cfg | ✅ per-call |
| top_k | 50 | `topK` / cfg | ✅ per-call |
| top_p | 0.93 | `topP` / cfg | ✅ per-call |
| repetition_penalty | 1.1 | `repetitionPenalty` / cfg | ✅ per-call |
| **guidance_scale (CFG)** | 1.5 (seg≤1) / 1.2 | `guidanceScale` / cfg 1.5 + `uncondTokenIds` | ✅ **CFG applied** (parallel uncond KV cache; caller passes the negative prompt + per-segment scale) |
| **run_n_segments** | 2 | — (single pass; caller loops + varies `guidanceScale`) | ⚠️ segments driven by caller |
| seed | 42 | `seed` | ✅ |
| use_audio_prompt / audio_prompt_path / prompt_start/end_time | False | — | ❌ (single-audio ICL continuation) |
| use_dual_tracks_prompt / vocal+instrumental_track_prompt_path | False | — | ❌ (dual-track ICL) |
| **stage-2 (1B upsampler, 1→8 quantizers)** | — | — | ❌ **not built** — engine emits stage-1 (1-quantizer) vocal only |
| rescale | False | — | ➖ (output clip guard) |

**Verdict (updated):** stage-1 LM now runs **CFG** (uncond negative prompt via a parallel KV cache) and exposes all
sampling params per-call; the caller drives per-segment guidance (1.5→1.2). The **stage-2 upsampler** + dual-track/ICL
remain unbuilt, so output is still coarse single-quantizer vocal, not the full song. CFG path validation-pending.

---

## Recommended engine changes (priority order)

**P0 — correctness (not just knobs; these change the audio): ✅ DONE (2026-06-30, validation-pending)**
1. ✅ **YuE stage-1 CFG.** `YueStage1Lm.GenerateCb0` runs a parallel uncond KV cache and combines
   `logits = uncond + g·(cond−uncond)`; `guidanceScale` (1.5) + `uncondTokenIds` are per-call, so the caller applies
   the per-segment schedule (1.5 for ≤1 segments, 1.2 after).
2. ✅ **HeartMuLa CFG.** `CsmModel.GenerateFrame` gained optional `cfgScale` + `uncondContext` (parallel backbone +
   depth-decoder uncond pass, CFG on every codebook). `HeartMulaConfig.CfgScale` = 1.5; `temperature` default fixed
   0.9→1.0. CSM callers omit both → unchanged single-pass behavior.
3. ✅ **ACE-Step fine guidance.** `guidance_interval` (0.5), `guidance_interval_decay` (0.0), `min_guidance_scale`
   (3.0), `omega_scale` (10.0, mean-preserving Euler rescale), dual `guidance_scale_text`/`guidance_scale_lyric`
   (double-condition CFG via a lyric-zeroed context), and `oss_steps` are all wired; defaults reconciled to 60/15.
   ERG-diffusion == omega; the ERG tag/lyric attention-temperature hooks remain deferred (P2).

**P1 — expose existing sampling per-call: ✅ DONE (2026-06-30)**
4. ✅ Per-call now: MusicGen temp/top_k/top_p/cfg_coef + `useSampling`; YuE temp/top_k/top_p/repetition_penalty;
   HeartMuLa temperature/top_k/top_p.

**P2 — feature completeness (still open):**
5. **ACE-Step task modes**: retake (retake_variance), repaint (repaint_start/end + src audio), extend, edit
   (edit_target_prompt/lyrics + edit_n_min/max), audio2audio (ref_audio_strength) + LoRA (lora_name_or_path/weight).
6. **MusicGen**: `extend_stride` (>30 s chunked generation), `two_step_cfg`, melody (chroma) + audio continuation.
7. **YuE**: `run_n_segments` (multi-segment structure), **stage-2** 1→8-quantizer upsampler (unbuilt), dual-track +
   audio-prompt ICL.
8. **ACE-Step v1.5**: `ode/sde` inference method, custom timesteps; the LM-stage sampling (temp/topP/topK/LM-cfg/
   negative prompt) if the engine is to host that stage rather than take pre-computed Qwen3 states.

**Unbuilt models** (for "all music models"): Stable Audio Open 1.0/Small/2, DiffRhythm, AudioLDM 2, ACE-Step XL.
</content>
