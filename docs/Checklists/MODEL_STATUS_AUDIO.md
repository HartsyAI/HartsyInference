# Audio models — status

[Legend](MODEL_STATUS.md) · [Numerical evidence](PARITY_VERIFICATION.md) ·
[Performance](../../benchmarks/scoreboards/AUDIO.md) · [Open work](#remaining-work).

## Consumer-path evidence and constraints

July 2026 CLI/Swarm passes confirmed word-correct speech across the major TTS paths. Component parity
alone does not establish intelligible speech: inspect/listen to audio and use a capable transcription
oracle (medium.en caught failures base.en concealed). Music requires listening, not just finite/nonzero samples.

- CosyVoice and GPT-SoVITS need a reference transcript; CosyVoice historically accepted an empty one and degraded.
- Qwen3-TTS Base uses a reference voice; select CustomVoice/VoiceDesign explicitly for those modes.
  The extension must use unpadded EncodeRaw tokens; padded Encode caused silent/overlong output.
- Dia's word-correct evidence uses Dia-1.6B-0626. CSM uses the bundled 32-codebook Mimi, I32 codec ids,
  all-codebooks-zero EOS, and the speaker/text template; a standalone 8-codebook Mimi is incompatible.
- VibeVoice's July 21 failure did not reproduce on three independent prompts; no cause was established.
- Audio cache paths vary between self-downloading models and registry-backed local families. Resolve through
  Engine; do not recreate download placement in consumers. Dangling cache symlinks require target checks.
- Demucs htdemucs was verified on real music: four distinct non-silent stems, pairwise correlation
  0.007–0.14. htdemucs_6s was wired but not individually exercised in that pass; htdemucs_ft needs its ensemble.
- Resemble-Enhance was real-weight e2e verified July 24 after module-layout/load fixes: 11 s JFK input
  produced 11 s of 44.1 kHz output, RMS 0.138 (input 0.142), word-correct transcript. Full numerical parity
  remains distinct. CPU performance is impractical; see the scoreboard.
- HeartMuLa's source-loader retention was fixed; generation-memory growth and exception masking during
  GraphStream.Dispose were separate follow-ups, not proven fixed by that load-only change.
- RVC needs a user voice checkpoint; registry resolution alone is not generation verification.

## TTS

| Model | Status | Notes |
|---|---|---|
| **GPT-SoVITS v2** | ✅ | HuBERT 1.07e-5, s1 GPT + s2 SoVITS verified, EN end-to-end → 32 kHz on real `lj1995` weights. |
| **Chatterbox** (ResembleAI) | ✅ | Full S3Gen rewrite (== CosyVoice2); enc 2.6e-6 / dec 4.4e-5 / vocoder 1.6e-5; end-to-end on CUDA. |
| **CosyVoice 2** | ✅ | Full zero-shot clone e2e on real weights, Swarm-deployed + whisper word-perfect 2026-07-17. Streaming shipped + Swarm-deployed 2026-08-11 (`CosyVoicePipeline.SynthesizeStream`); one accepted small artifact — an isolated single-word mispronunciation under adversarial content, inherent to the bounded-context-window design, not a fixable tuning knob. RTF and next perf lever tracked under Perf follow-ups. |
| **Qwen3-TTS** | ✅ | Bit-exact (RoPE split-half + byte-level tokenizer fixes). |
| **Piper** (VITS) | ✅ | corr 0.9998 vs onnxruntime; 7 VITS bugs fixed (affect all VITS). **Swarm e2e word-correct 2026-07-13** — fixed the espeak language default (`en` British → the voice's `en-us` American; it was mispronouncing vowels). |
| **Kokoro** (StyleTTS2) | ✅ | ~1e-4 on the CUDA path (added `audio_leaky_relu` / `audio_adain1d` kernels). **Swarm e2e word-correct 2026-07-13** — misaki-phoneme g2p + punctuation fix (was silently dropping words); canonical-`.pth` download fallback (was install-401). |
| **F5-TTS** (v1 Base) | ✅ | Flow-matching DiT verified bit-exact: velocity corr 1.0, full CFM sample loop (generated mel) corr 1.0, Vocos corr 0.9999. 4 bugs fixed (ConvNeXt filler-mask, ×1000 timestep scale, erf/tanh GELU split, cond-anchored CFG + end-only ref-clamp). ([details](#f5-tts)) |
| **ZipVoice** (k2-fsa) | ✅ | Zipformer backbone (`fm_decoder`+`text_encoder`) parity cosine 1.0 (2026-07-19). ([details](#zipvoice)) |
| **Kyutai TTS** (tts-1.6b-en_fr) | ✅ | **Fully intelligible e2e in pure C# 2026-07-16** (whisper medium.en: "So hello there, this is a test of the Cuta[=Kyutai] text-to-speech model" — matches the script, no clipping, 62 frames/4.96s vs moshi ref 71/4.4s). ([details](#kyutai-tts)) |
| **ResembleEnhance** | ⚠️ mixed | Real-weight enhancement e2e recorded July 24; full mel→mel numerical parity remains pending. |
| **MeloTTS** (English-v3) | ✅ | Real-weight e2e in pure C#. ([details](#melotts)) |
| **Spark-TTS-0.5B** | ✅ | Real-weight e2e bit-exact, fully in-engine (controllable mode): LM logits corr 1.0 (top-1 100%), greedy tokens 32/32 global + 179/179 semantic match Python, BiCodec wav corr 1.0 (factorized VQ, FSQ d-vector, AdaLN PreNet all corr 1.0). ([details](#spark-tts-05b)) |
| **FishSpeech 1.5** | 🔬 | DualAR LM verified: slow (24-layer) corr 1.0, fast depth-LM (4-layer) corr 0.9999. fused-key adapter + interleaved RoPE + no embed-scale + pre-norm fast input. Only the firefly-gan-vq codec remains. |
| **Dia-1.6B** | ✅ | **Swarm e2e word-correct 2026-07-15 (10/10, all 3 turns) — root cause was the WRONG CHECKPOINT.** The full transformer was already bit-exact; the "loops *Hello there* / non-verbal garbage across seeds" symptom was the engine faithfully running the **old** `nari-labs/Dia-1.6B` release. ([details](#dia-16b)) |
| **Orpheus** | ✅ | **Swarm e2e word-correct 2026-07-14** (Llama-3.2-3B + SNAC-24k). Fix was the prompt frame (missing BOS 128000 + StartOfAi 128261/StartOfSpeech 128257). Perf: 10 s via a fused BF16 lm_head GEMV (lm_head was 90% of decode). |
| **Bark / Chatterbox / VibeVoice / FishSpeech** | ✅ | Swarm e2e word-correct (2026-07-13/14). ([details](#bark--chatterbox--vibevoice--fishspeech)) |
| **VibeVoice-Realtime-0.5B** (`vibevoice:realtime`) | 🔧 | **2026-08-10**: split-LM architecture built and loads against the real checkpoint (4-layer text encoder + 20-layer TTS backbone as two genuinely separate weight-bearing Qwen2 stacks, binary EOS classifier, `tts_input_types` splice — all confirmed from the real 608-key `model.safetensors`, not the architecture doc's prose). **Cannot generate yet**: the released checkpoint ships a decode-only acoustic VAE (zero encoder keys at all), so zero-shot voice cloning is architecturally impossible against it; upstream's own precomputed per-speaker `.pt` voice caches (confirmed via pickle disassembly: nested `DynamicCache` objects, not a flat state dict) need a dedicated deserializer that doesn't exist yet. `Synthesize`/`SynthesizeStream` throw `NotSupportedException` rather than produce unconditioned audio. Not yet surfaced in the AudioLab UI. |
| **StyleTTS2** (LibriTTS, clone) | ✅ | **Clone e2e word-intelligible 2026-07-15** (in-process; Whisper recovered 5/7 content words from a reference-voice clone). ([details](#styletts2)) |
| **NeuTTS** | ✅ e2e | Default and clone paths have recorded consumer evidence; wider voices and numerical parity remain separate gates. |
| **Zonos-v0.1** (transformer) | ✅ | **Voice clone e2e word-perfect through the Swarm gallery 2026-07-17.** Installed as `Audio Models/Zonos/transformer`; `GenerateText2Image` with a real reference clip saved to the output gallery, Whisper medium.en transcribed *"Hello, this is a test of the Zonos Text-to-Speech System."* — verbatim, including the coined word "Zonos". ([details](#zonos-v01)) |

## STT

| Model | Status | Notes |
|---|---|---|
| **Whisper** (tiny → large-v3) | ✅ | JFK clip transcribes correct content words (verified 2026-07-13; the end-to-end test was removed in the 2026-08-06 suite cleanup). **Swarm e2e word-perfect 2026-07-13** on the real JFK clip; fixed the `en-US` default-language crash (locale-code normalization). |
| **Whisper streaming** (RealtimeSTT) | ✅ | LocalAgreement-2 + JFK streaming. |
| **Moonshine** | ✅ | Tests pass. **Swarm e2e word-perfect 2026-07-13** on real (JFK) + synthetic clips; ~2 s for 9 s audio on the 3060. |
| **Moonshine streaming** (tiny/small/medium) | ✅ | Real-weight parity verified 2026-07-19 (encoder/decoder cosine ~1.0); **Engine-wired 2026-07-20** as `moonshinestreaming` — JFK clip word-perfect end-to-end (CPU). Full-utterance batch only; true chunked/incremental streaming not yet implemented. |
| **Kyutai STT** (stt-1b / 2.6b) | 🔧 | Shares the moshi backbone; parity pending (no depformer). |

## Wake word

| Model | Status | Notes |
|---|---|---|
| **Wake front-end + backbone** (openWakeWord mel + Google `speech_embedding`) | ✅ | Real-weight parity vs the shipped ONNX graphs under onnxruntime 2026-08-16: mel max abs 2e-3, embedding relative L2 2e-3. Constants were read out of the graphs, not the upstream docs, which are wrong — n_fft/window is **512** (not the commonly cited 25 ms), hop 160, 32 bins, power spectrum, `10·log10`, floor `max − 80 dB`. Backbone activation is a **clipped LeakyReLU** `max(leaky(x,0.2), −0.4)`, not ReLU. Weights via `OnnxWeightLoader`; forward passes are C#. See `docs/Research/WAKE_WORD_DETECTION.md`. |
| **Wake heads** (openWakeWord family) | ✅ | All three shipped architectures load and match onnxruntime within 1e-4 (alexa agrees to 7 decimals). They genuinely differ: `alexa` has no LayerNorm, `hey_mycroft` does, `hey_jarvis` prefixes weights with `model.` **and bundles a second `verifier_model.*` that must not be loaded as the main head**. The loader discovers width and LayerNorm presence from the weight names. hey-buddy's gated/residual heads are a different architecture and are not supported. |
| **Wake streaming pipeline** | ✅ | Reproduces openWakeWord's streaming contract (1280 new samples + 480 left context per mel call), verified against a Python implementation of the same contract over 11 s of real speech: 113 consecutive scores within 1e-4. Diverges deliberately by withholding scores until 76 real mel and 16 real embedding frames exist (~1.3 s) instead of seeding buffers with random audio. |
| **Wake routing / config / webhooks** | ✅ | Per-word settings (threshold, smoothing, refractory, `route` tag, `requiredSpeaker`) persist to `{modelRoot}/wake-words.json` via `WakeWordConfigStore`, written atomically; `wake-train` records its measured threshold there so the number is not lost when the command exits. Detections POST to configured webhook URLs so out-of-process services can subscribe. `route` is opaque to the engine — it is echoed on the detection event and never interpreted, which is what lets one engine feed several agents. |
| **Wake satellite transport** | ✅ | TCP + Wyoming-style framing; device-keyed sessions, ping/pong, seq-gap reset. Hosted by the API server behind `HartsyInference__WakeEnabled`. Protocol contract in `docs/Research/WAKE_SATELLITE_PROTOCOL.md`. Cost: 11 s of audio through the full mel→backbone→head chain completes in under 2 s wall clock in Release **including** model load and test-host startup (RTF < 0.18), so one worker thread sustains at least ~5 concurrent satellites; the precise per-chunk figure has not been isolated. In Debug it is roughly 10x slower — do not benchmark this on a Debug build. |
| **Custom wake-word training** | ✅ | `hartsy wake-train "<phrase>"` synthesizes the phrase across Kokoro voices, augments (gain + noise), embeds through the frozen backbone via the real `WakeDetectionPipeline` (so training features cannot drift from inference features), and fits a ~213k-param head with hand-written Adam. Backward pass is gradient-checked against an independent double-precision forward; deleting one ReLU derivative moves it 3,400%. Verified end to end 2026-08-16: `"hey hartsy"` trained on 3 voices **fired at 0.6961 on a 4th voice it never heard** and stayed silent through 11 s of unrelated speech. ⚠️ **Two honest limits.** (1) The auto-suggested threshold overfits a small held-out set — it proposed 0.90, which would have *missed* that unseen-voice detection at 0.6961; treat it as an upper bound and lower it. (2) False accepts are reported per hour precisely because a small negative set looks fine as a percentage and is unusable in a room: with one negative recording the run reported ~1.8%/window ≈ 800/hour against openWakeWord's target of <0.5/hour. Point `--negative-audio` at hours of real room audio. |
| **Silero VAD v6** | ✅ | Ported and parity-verified against the shipped ONNX under onnxruntime: **4.77e-6 max abs over 343 chunks** of jfk.wav, non-vacuity proven by perturbing the reference. Architecture: right-only reflect pad of 64 (NOT symmetric) → fixed-DFT Conv1d STFT (kernel 256 / stride 128) → **magnitude** → 4-conv encoder k3 pad1 strides 1/2/2/1 → **ReLU on the LSTM hidden state** (absent from every written description of this model) → LSTMCell(128) → final conv → sigmoid. For the fixed 576-sample contract the encoder collapses to T=1, so the graph's trailing `ReduceMean` is vacuous and is not implemented. ⚠️ **Upstream ships two different revisions of this model**: `silero_vad_16k.safetensors` and `silero_vad.onnx` on silero-vad master have identical architecture and an identical fixed DFT basis, but every learned tensor differs (correlations 0.90–0.99, max abs up to 18). We derive our safetensors from the **ONNX**, because the ONNX is what silero's own `utils_vad.py` runs and what everyone benchmarks against. The original upstream safetensors is re-downloadable from `raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad_16k.safetensors` (md5 `8e4a09a0fa2dfa1d09441da4c110b2fd`) if that choice is ever revisited. **Not yet gated into the wake pipeline** — it exists as `SileroVad`/`SileroVadStream` and is the next integration step. |
| **Speaker identification** | ✅ | `Speakers/` — CAM++ (192-d) embedding, L2-normalized centroids on disk, cosine scoring with open-set rejection. Wired into detection: events carry `speaker`, and `WakeWordConfig.RequiredSpeaker` is now **enforced** (it fails CLOSED when identification is unavailable — silently ignoring a restriction is worse than missing a trigger). Scores the wake word together with the command that follows it, because text-independent verification degrades badly at wake-phrase length. ⚠️ Threshold **0.35 is uncalibrated** — no EER measured on this codebase or a real microphone. Every match logs the nearest name and raw score, so calibration data accumulates in production logs. CAM++ weights are read from the Chatterbox bundle (`Models/audio/tts/ResembleAI--chatterbox/s3gen.safetensors`); no standalone checkpoint is present. |
| **Wyoming / Home Assistant endpoint** | ✅ | `Wyoming/` on port 10600, off by default (`HartsyInference__WyomingEnabled`). Implements `describe`→`info`, `ping`/`pong`, `transcribe`+`audio-*`→`transcript`, `synthesize`→`audio-*`, `detect`→`detection`/`not-detected`. ASR/TTS route through the same `InferenceQueue` as HTTP routes. ⚠️ **Real Wyoming framing differs from the satellite protocol's**: it writes `data` as a separate `data_length`-prefixed block, not inline, so this needed its own codec — a test written against our own writer would have passed while HA desynced on its first request. Also, Wyoming's `info` decoder fails **silently** on a missing key, so the manifest writes every field unconditionally including explicit nulls. Streaming ASR/TTS variants are not implemented and correspondingly not advertised. Wake scoring is host-supplied via `WakeDetectorFactory` and is **not yet wired**, so HA sees ASR and TTS but not wake words. |

## Codec / voice conversion / music / separation

| Model | Status | Notes |
|---|---|---|
| **OpenVoice** (tone-color VC) | ✅ | Conv2d + GRU + speaker encoder validated. |
| **CAM++ / CamPlus** (speaker) | ✅ | From `funasr/campplus_cn_common.bin`. |
| **S3Tokenizer** | ✅ | From the `s3tokenizer` package. |
| **Vocos / vocoders** | ✅ | Test passes. |
| **GPT-SoVITS HuBERT / CosyVoice sub-encoders** | ✅ | Validated above. |
| **ACE-Step v1** (music DiT 3.5B) | ✅ | DiT ~1e-8 + DCAE decoder corr 1.0 + vocoder corr 1.0; full e2e on CUDA/3060 (bf16 + `HighPrecisionGemm`) writes finite audio. |
| **ACE-Step v1.5 turbo** (music DiT 2B) | ✅ | DiT/cond-encoder/8-step loop all corr 1.0 (~1e-6) vs torch oracle on the real Comfy-Org turbo weights; Oobleck VAE corr 0.9999999999; e2e finite tonal stereo on CUDA. ([details](#ace-step-v15-turbo)) |
| **Mimi** (codec) | 🔬 | DSM moshi-native decode has bit-exact reference evidence; CSM requires its bundled 32-codebook variant. Do not generalize across codec checkpoints. |
| **MusicGen / AudioGen** | ✅ | T5-base corr 1.0 + decoder logits corr 0.999999 + EnCodec-32k decode corr 1.0; e2e on CUDA writes music-like audio. 5 bugs fixed (T5/EnCodec). |
| **YuE** (music, Stage-1) | ✅ | Stage-1 7B LM corr 1.0 (argmax 8/8) + XCodec (SoundStream) decode corr 1.0 → generates 16 kHz vocal audio. ([details](#yue)) |
| **HeartMuLa** (oss-3B) | ✅ | LM corr 0.9996–0.9999 + HeartCodec rewritten: flow-match estimator corr 1.0 + ScalarModel corr 1.0 → generates 48 kHz audio (CPU + CUDA). ([details](#heartmula)) |
| **MiniMax Music 3** | ✅ | Prompt ids exact; condition encoder, DiT block 0 and the full 36-layer DiT match diffusers (meanAbs < 1e-3); vocoder maxAbs 1e-4 with a distinct stereo fold; window/crop geometry reproduces the reference's 529408-sample stitch. Generates real 44.1 kHz stereo on CUDA. AR parity corr 0.9999989 on CUDA, flow parity corr 0.999996, and end-to-end output confirmed by ear as real music with intelligible sung lyrics. ([details](#minimax-music-3)) |
| **RVC** (voice conversion) | 🔬 | RMVPE front-end wired as the default F0 estimator (`VcCatalog.ConvertRvc`), corr 1.000000/maxAbs 9.5e-8 vs real `rmvpe.pt` ([details](../Checklists/PARITY_VERIFICATION.md)). YIN remains selectable via `f0_method`. RVC flow/decoder + index/protect/rms_mix_rate still pending. |
| **Demucs** (separation) | ✅ e2e | Real htdemucs CPU output: four distinct stems; full numerical parity remains pending. See consumer-path evidence above. |
| **CSM** (Sesame) | ✅ | Fixed 2026-07-21 (unsloth/csm-1b key remap + bundled 32-cb Mimi + I32 codes dtype + real all-zero EOS + `[speaker]text` prompt template); Whisper word-perfect on two independent sentences. `hartsy speak -m csm`. |
| **Stable Audio Open Small** | ✅ | DiT/VAE/timing-conditioner parity cosine 1.0 each. ([details](#stable-audio-open-small)) |
| **DiffRhythm / AudioLDM 2 / ACE-Step XL** | ❌/🔧 | Music roadmap; see the [Remaining work](#remaining-work) section for the per-model build state and ROI order. |
| **PocketTTS** (continuous-latent) | ✅ | **Swarm-deployed + parity-verified 2026-07-16.** Production voiced path built on the verified cores: `PocketTtsStreamingTransformer.ForwardPrimed` (voice-KV prefix + RoPE offset), `PocketTtsFlowLm.GenerateVoiced` (LUT conditioner + out_eos stop + noise std=√temp), `PocketTtsVoice` (KV-state loader), rewritten `PocketTtsPipeline` (SentencePiece + emb_std/mean denorm). ([details](#pockettts)) |

## Notes

- Build audio with `-m:1`; the Audio test suite crashes under xunit parallel, run it sequentially
  (`-- xUnit.ParallelizeTestCollections=false`) and reuse a model cache via `HARTSYINFERENCE_MODEL_CACHE`.

## Remaining work

Distilled from the retired PHASE_5_AUDIO / MUSIC_MODELS_COMPLETION_PLAN / MUSIC_PARAM_AUDIT /
AUDIO_TTS_BRINGUP_PLAN plans. Items now ✅ above (Bark, CSM, Orpheus, Kyutai TTS, StyleTTS2 clone, Demucs,
Spark-TTS, Zonos, PocketTTS, CosyVoice 2, Chatterbox clone, Stable Audio Open Small) are omitted.
See [ROADMAP.md](ROADMAP.md) for cross-cutting infra (multi-GPU, kernel perf, quant, serving).

### Build / validation status
- [ ] Validate missing audio backend operations per target model; compiled audio kernels already ship.
- [ ] Codec STOI validation (pending checkpoints).

### Not-started ASR / TTS models
- [ ] NVIDIA NeMo family: Parakeet CTC / RNN-T / TDT, Canary, FastConformer.
- [ ] SenseVoice, FireRedASR.
- [ ] XTTS-v2, ChatTTS, Higgs Audio v2, IndexTTS 1.5 / 2, CosyVoice 1.

### Checkpoint-gated scaffolds (backbone verified, need real weights)
- [ ] Kyutai STT (no depformer).
- [ ] Fish-Speech firefly-gan-vq codec.
- [ ] Resemble Enhance numerical parity and practical runtime; real-weight loading/e2e is verified above.
- [ ] GPT-SoVITS enc_p (zero-shot clone encoder).
- [ ] StyleTTS2 random-mode (no-reference) diffusion sampler + whisper-drop quality follow-ups.

### Streaming
- [ ] Extend streaming beyond the implemented paths (including CosyVoice); validate latency/quality per provider.

### Music — not built
- [ ] Stable Audio Open 1.0 / 2 (Small is ✅).
- [ ] DiffRhythm, AudioLDM 2, ACE-Step XL.
- [ ] YuE Stage-2 upsampler.
- [ ] Shared music infra: dpmpp-3m-sde sampler, G2P, MuQ-MuLan, CLAP, GPT-2 continuous decoder.

### Music — P2 params
- [ ] ACE-Step task modes (retake / repaint / edit / audio2audio / LoRA).
- [ ] MusicGen extend_stride / melody / continuation.
- [ ] YuE stage-2 + dual-track.
- [ ] ACE-1.5 cover-mode LM.

### Remaining TTS clone paths
- [ ] Qwen3-TTS ICL mode and Kyutai clone; NeuTTS clone has recorded e2e evidence.

### Perf follow-ups
- [ ] Dia realtime (CUDA-graph decode + CFG batching).
- [ ] AudioGen duration cap 45s → 30s (unroot-caused).
- [ ] ZipVoice GPU-residency pass.
- [ ] Per-provider unload endpoint.
- [ ] CosyVoice2 streaming real-time-factor pass — currently ~8× real-time end to end, live-verified
      2026-08-11 (`CosyVoicePipeline.SynthesizeStream`). The flow+vocoder stage alone already tunes to
      ~3.45× (`CosyVoiceFlow.InferenceGrowingWindowed`, chunkSizeTokens=25/windowSizeTokens=150/
      marginFrames=40); the LM's own autoregressive speech-token decode is the dominant remaining cost
      (steady ~7.6-8.2s live per 25-token chunk vs. ~2.8-3.0s for that chunk's flow+vocoder work alone) —
      that's the next lever, not further flow/vocoder tuning. First-chunk latency 25.7s warm / 36s cold.

## Details

Verification evidence, bugs found, and caveats for the rows above. Moved out of the status
tables on 2026-08-06 so the tables stay scannable — no content was dropped.

### CosyVoice 2

**Two streaming designs measured and REJECTED before `CosyVoiceFlow.InferenceGrowingWindowed` was adopted (2026-08-11, real weights, real generation).** Both were removed from the source in the 2026-08-22 dead-code pass; their measurements are recorded here so neither gets re-attempted blind.

**(1) Self-conditioned chunking** (`InferenceChunk` — roll the model's own previous chunk mel forward as the next call's `cond` prompt): NOT VIABLE. Causes a real, progressive mel-domain **mean-level drift**. Chunk 0's returned-mel mean/std (-3.68/1.48) sits close to the matching monolithic range (-3.90/1.55 — chunk 0's lookback is the real reference clip, no self-conditioning yet); by chunk 8 the mean has drifted to **-3.23 vs the monolithic -4.23** (std stays close, 2.06 vs 1.98 — a LEVEL shift, not spectral corruption). Because mel is log-scale that ~1.0-nat drift compounds roughly exponentially in audio: per-second RMS ratio (chunked ÷ monolithic) grew **1.34× → 2.72× → 2.90× → 4.63× → 5.61× → 7.16× → 3.98× → 5.78× → 7.46×** over a 9.36 s utterance (15-token chunks, 3-token lookahead), clipping near ±1.0 well before the end. Making the lookback cumulative (MORE self-generated context, not less) made it **worse** (RMS ratio ~4.5× vs ~2.2×) — which rules out context-length truncation as the cause and confirms the mechanism is **exposure bias** from feeding generated mel back as `cond`, which CosyVoice2's flow decoder was only ever trained to receive as REAL reference-clip mel. Content stays correct throughout (Whisper word-perfect at every drift level) — a level/energy bug, not a corruption bug. A pre-drawn full-utterance noise buffer sliced per chunk (`x0Override`, tried first as the obvious suspect) does **not** fix it: the drift is in `cond`'s conditioning distribution, not the CFM noise seed. Untried design lever if anyone picks this back up: pin `cond`'s prompt region to the REAL reference-clip mel for every chunk and use only token-domain lookback for local continuity — that breaks the feedback loop, at the cost of losing mel-domain continuity with the preceding chunk (whether that trades this drift for a new boundary-discontinuity artifact is untested).

**(2) Unbounded growing recompute** (`InferenceGrowing` — recompute the full target-token history every call against the unchanged real prompt): correct but DISQUALIFYING on cost. `cond` is never self-generated so there is no drift, but it is O(current-length) per call and **O(n²) total**. On a real 26.2 s utterance (656 LM tokens, 15-token chunks) per-call wall clock grew **1.66 s (first chunk) → 18.6 s (last chunk)**, an 11.24× spread; total **360.69 s across 44 calls for 26.2 s of audio = 13.75× real time**. It looked fine on a 9.36 s test utterance and was badly disqualifying at 26.2 s — do not assume a short utterance's timing generalizes. Two properties worth knowing before re-deriving them: its final call is **bit-exact to a monolithic `Inference` call** (it was kept for a while as a harness sanity check), and it was the **decisive discriminator** for the bounded-window design's one known artifact — the "quick brown fox" → "quit brown fox" substitution reproduces byte-for-byte across every bounded config swept (windowSizeTokens ∈ {150, 200, 300} × marginFrames ∈ {40, 60} × chunkSizeTokens ∈ {15, 25}) but the unbounded path transcribes the word **correctly** on the same utterance/seed. That is what proved the artifact needs near-complete bidirectional history rather than margin/chunk-size tuning, and is why it is accepted (~1.25% WER, isolated, non-cascading) rather than chased.

### F5-TTS

Flow-matching DiT verified bit-exact: velocity corr 1.0, full CFM sample loop (generated mel) corr 1.0, Vocos corr 0.9999. 4 bugs fixed (ConvNeXt filler-mask, ×1000 timestep scale, erf/tanh GELU split, cond-anchored CFG + end-only ref-clamp). **Swarm e2e word-correct + perf pass 2026-07-13:** with a real voice ref it transcribes word-perfect (medium.en); **6.4 s** by routing the `F5ConvPosEmbed` grouped Conv1D off the host loop to `backend.Conv1d` (GPU), output bit-parity (RMS-envelope corr 1.0000).

### ZipVoice

Zipformer backbone (`fm_decoder`+`text_encoder`) parity cosine 1.0 (2026-07-19). **Engine-wired 2026-07-20** as `zipvoice` (same gap as Stable Audio — built+parity-verified, zero Engine catalog entry). Swarm `/API/GenerateText2Image` verified: clones the JFK reference voice, produces valid non-silent 24kHz mono speech. **Slow — 11.4 min for a ~10s clip**, no GPU-residency work done yet; a real perf-pass candidate (same class of host-glue likely present as pre-perf-pass Stable Audio).

### Kyutai TTS

**Fully intelligible e2e in pure C# 2026-07-16** (whisper medium.en: "So hello there, this is a test of the Cuta[=Kyutai] text-to-speech model" — matches the script, no clipping, 62 frames/4.96s vs moshi ref 71/4.4s). **ROOT-CAUSE of the earlier gibberish: the cross-attention voice source was built with only the T real voice rows.** moshi `make_condition_attributes` pads the voice to `max_speakers=5` slots → the 4 empty slots become the learned `speaker_wavs.learnt_padding` vector, and `ConditionFuser.get_cross` then adds a continuous sin pos-emb over all `5·T=625` rows (`cross_attention_pos_emb=True`). Those 500 padding rows are NOT inert — cross-attention attends over all of them — so omitting them shifted every cross-attention output (~1% error in the outlier activation dims) and, compounded over the autoregressive loop, produced non-speech + clipping + wrong length. Fix: load `speaker_wavs.learnt_padding`; `MoshiConditioner.ComputeCross` now emits `[1, 5·T, 2048]`. Also: text token must be **sampled** (temp_text 0.6/top-k 25), not argmax — the new_word/pad choice paces the words (`MoshiTtsGenerator` host-samples text). Cores verified (backbone 1.3e-4, depformer teacher-forced 27/32 argmax + ~0.99 corr = sampling-robust, conditioner ~1e-8, **Mimi decode bit-exact corr 1.0** — DSM moshi-native keys + interleaved RoPE). **Perf: 9.7 s** for the 62-frame gen on a 3060 by making the depformer's per-weight-set QKV/out/gate projections device-resident (they were sliced fresh each Block call, so the weight cache never hit and re-uploaded every op — H2D_MISS_BIG 6469→466 calls, Linear 0.356→0.045 ms/call). moshi (bf16 + CUDA-graph) is 2.3 s on the same 3060; the remaining gap is per-op launch overhead on the 32-codebook cascade (SDPA 2.7 s + Linear 2.5 s) that a per-frame CUDA-graph capture would close. **Swarm-deployed 2026-07-16** via the AudioLab extension (`kyutaitts_tts`, `KyutaiTtsModel` orchestrates `MoshiTtsGenerator` + `Mimi` + `KyutaiSttTokenizer` directly against published engine alpha.49 — no engine change; pre-embedded `kyutai/tts-voices` speaker, default `expresso/ex03-...`): live `/API/GenerateText2Image` → whisper medium.en word-perfect ("Kyutai"→"QTIE" brand mishearing), 5.52 s, peak 0.47.

### MeloTTS

Real-weight e2e in pure C#. **Swarm e2e word-correct 2026-07-13** — earlier "corr 0.9993 noise-0" was stale: the real e2e produced gibberish from a `PytorchPickleLoader` **stride bug** (bert-base-uncased Linear weights, saved as `.t()` views, loaded transposed → garbage BERT features), fixed with a stride-gather (`MakeRowMajor`, no-op for contiguous — helps all `.pth` models). Also added **number normalization** (`normalize_numbers`: years/currency/ordinals/decimals were dropped). `MeloTts` facade + gated parity test.

### Spark-TTS-0.5B

Real-weight e2e bit-exact, fully in-engine (controllable mode): LM logits corr 1.0 (top-1 100%), greedy tokens 32/32 global + 179/179 semantic match Python, BiCodec wav corr 1.0 (factorized VQ, FSQ d-vector, AdaLN PreNet all corr 1.0). `SparkTtsPipeline.LoadFromDirectory`/`LoadAsync` + `SynthesizeControllable(text, gender, pitch, speed)`; `SparkTtsTokenizer` reuses the shared BPE + ByteLevelCodec. Zero-shot cloning would need the BiCodec encoder side (wav2vec2 + ECAPA), not built.

### Dia-1.6B

**Swarm e2e word-correct 2026-07-15 (10/10, all 3 turns) — root cause was the WRONG CHECKPOINT.** The full transformer was already bit-exact; the "loops *Hello there* / non-verbal garbage across seeds" symptom was the engine faithfully running the **old** `nari-labs/Dia-1.6B` release. The current **`nari-labs/Dia-1.6B-0626`** (a drop-in: identical 343 keys + shapes, only weight *values* differ — decoder-embed corr 0.297 between the two) produces the **full 3-turn dialogue** and **emits EOS to stop itself at 11.44 s** (985 frames, doesn't run to the cap). Proven by a layer-diff A/B against the nari `dia` package (which itself hardcodes `-0626`): our forward/sampling/EOS/RoPE/masking all matched — the "divergence" was just base-vs-0626 weights, not a bug. Fix = extension repo `Dia-1.6B`→`Dia-1.6B-0626` (ships `pytorch_model.bin` → `PytorchPickleLoader`, no engine change); rebuilt + restarted Swarm → `GenerateText2Image` transcribes 10/10 (medium.en).

### Bark / Chatterbox / VibeVoice / FishSpeech

Swarm e2e word-correct (2026-07-13/14). **VibeVoice perf pass DONE 2026-07-17: RTF 0.78** — 6.47 s for an 8.27 s clip, faster than real time. Three rounds, **zero new kernels** (all reused `IBackend` ops): (1) **VAE convs** — acoustic/semantic causal convs were CPU `float*` loops (`VibeVoiceOps.CausalConv1d`/`CausalConvTranspose1d`) interleaved with GPU FFN ops (device↔host round-trip per ConvNeXt block); routed `SConv1d`/`SConvTranspose1d` → `backend.Conv1d`/`ConvTranspose1d`, channels-first RMSNorm → `Transpose2D`+`RmsNorm`+`Transpose2D`, layer-scale → groups=C kernel-1 `Conv1d` (`gamma`→`[C,1,1]`), streaming combine/tail → `Concat`/`SliceLastDim` (134→9.08 s; prefill 43.8→0.33 s / 133×, per-frame decode 944→25.5 ms). (2) **Batched CFG** — cond+uncond stacked into one N=2 diffusion-head forward (9.08→7.21 s). (3) **Head host-glue→GPU** — `SliceAlongLastDim`/`AdaLnModulate`/`AdaLnGatedAdd`/`RmsNormNoAffine` (~25 syncs/forward) → `SliceLastDim`/`AddScalar`+`Mul`+`Add`/`RmsNorm`-with-ones (7.21→6.47 s). Behavior-preserving: output corr **0.999585** vs the host path, identical Whisper transcript, 15/15 VibeVoice unit tests pass. Four stages now balanced (diffusion/LM/acoustic/semantic ≈ 31/31/24/11 %); remaining lever is CUDA-graph capture (major rewrite). `benchmarks/results/vibevoice_tts_2026-07-17.md`. Bark/Chatterbox perf still pending.

### StyleTTS2

**Clone e2e word-intelligible 2026-07-15** (in-process; Whisper recovered 5/7 content words from a reference-voice clone). Every custom piece **verified vs the Python `yl4579/StyleTTS2` reference**: StyleEncoder **corr 1.000000** (StarGAN-v2 learned-downsample + spectral-norm σ-fold + odd-width replicate-pad), new **HiFiGAN generator corr 0.999999** (`StyleHifiGanGenerator`: 4-stage 10·5·3·2 upsample + AdaIN/Snake noise-res + MRF + Snake α + conv_post/tanh, reusing `AdaSnakeResLoader`) with an exact **`StyleSineGen`** harmonic source (corr 1.0). The vocoder is `type: hifigan`, NOT Kokoro's iSTFTNet (wired via a gated `KokoroIStftNetDecoder(useHifiGan)`); the bert/text-encoder/prosody reuse Kokoro. **Bug fixed along the way:** the shared `AdaInstanceNorm1d` used a single-pass `E[x²]−E[x]²` variance that went NaN via catastrophic cancellation on the HiFiGAN's ~30 k-sample stages → switched to a stable two-pass double variance (regression-clean for Kokoro). **Swarm extension wired 2026-07-15** — `StyleTts2Model.Descriptor` (provider `styletts2_tts`, already registered in `AudioEngine`) downloads `epochs_2nd_00020.pth`, builds the pipeline via `LoadFromCheckpoint` (in-engine 178-symbol tokenizer + reference-mel front-end), and clones from `req.ReferenceMono24k` through `SynthesizeCloneFromAudio`. **Swarm e2e clone-verified 2026-07-15** — deployed via a local-engine pack (`alpha.48.2-local`, both extension pins), installed through `AudioLabInstallEngine`, generated through `/API/GenerateText2Image` with a real reference clip (jfk.wav): Whisper medium.en heard *"And so my fellow Americans ask Ned what our country can do for you"* (12/13 words; misses are ASR mishears of correctly-pronounced words), `.swarm.json` metadata sidecar present, ~0.8× wall RTF warm. Kokoro regression through the same live engine = word-perfect. **Also fixed a shared engine bug surfaced here:** `EspeakTranslator.MatchRule` indexed out of bounds (crash) when a word's pre-context scanned left past the per-word buffer start (e.g. "Americans") — guarded the boundary read as a space (espeak's clause buffer is space-padded), fixing all espeak TTS (F5/NeuTTS/Piper/StyleTTS2); purely additive (the branch previously always threw). **Quality fixes 2026-07-15 (48.4-local) after a user listen, diagnosed by A/B-ing our intermediates against the real python StyleTTS2 inference:** (1) reference-mel front-end built the filterbank at 24k/12k but `meldataset.to_mel` relies on torchaudio's DEFAULT 16k/8k (no `sample_rate`/`f_max` arg) + centers a win<n_fft window — fixed (`ComputeReferenceMel` 16k/8k + `CenterWindowInFft`), mel corr 0.93→0.9994, fixes tinny/wrong-voice timbre; (2) the "wave"/slur was PHONEMES not synthesis (proved: our phonemes → real model reproduce the garble) — extension used espeak "en" (British) not "en-us", and `PhonemizeToIpa` stripped punctuation → run-on prosody; fixed via `en-us` + new `preservePunctuation` overload (vocab already maps `.`/`,` identically), natural sentence now Whisper word-perfect. NB jfk.wav is a low-fi ~3.4 kHz reference — use a clean 24 kHz clip. Remaining: Random-mode diffusion (no-reference) still a scaffold; minor espeak-port stress quirks.

### Zonos-v0.1

**Voice clone e2e word-perfect through the Swarm gallery 2026-07-17.** Installed as `Audio Models/Zonos/transformer`; `GenerateText2Image` with a real reference clip saved to the output gallery, Whisper medium.en transcribed *"Hello, this is a test of the Zonos Text-to-Speech System."* — verbatim, including the coined word "Zonos". Every component also matches the reference on real weights: ResNet293 speaker encoder (mel corr **1.0**, 128-d embedding corr **1.0** CPU / **0.999999** CUDA vs `SpeakerEmbeddingLDA`; full `EmbedFromWav` path cos **1.000000** vs golden), prefix conditioner cond+uncond corr **1.0**, phoneme tokenizer (189-symbol) exact, backbone prefill logits corr **1.0** + argmax-exact for all 9 codebooks, decode logits bit-exact (maxAbs ~3e-5) for 33 AR steps, greedy matches Python bit-for-bit (301 frames). **THE bug that blocked e2e (fixed 2026-07-17):** `ZonosConditioning.BuildPrefix` returned the prefix **channels-first `[1, D, P]`** while `ZonosPipeline.Generate`/`Prefill` consume channels-**last `[1, P, hidden]`** (read `Shape[1]` as seq-len) — so `ZonosTts` fed the backbone a transposed prefix → garbage → instant EOS (8–18 frames). The generate parity tests passed only because they used golden `[1,P,D]` tensors, and the conditioning test's compare-helper *transposed to mask it*; added a **per-token prefix guard** (`EnginePrefix_PerToken_LocalizesDivergence`) that catches a layout regression (transpose collapses per-token corr to ~0). Also fixed the espeak stress port for `$u+` words ("this"/"that" keep primary stress in espeak-ng 1.51). Earlier fixes: 3 backbone parity bugs via parameterizing the reused `DiaAttention`/`DiaMlp` (**interleaved RoPE**, **`up·silu(gate)` MLP half**, **`1/√head_dim` scale**), delay-revert off-by-one, `Tensor.CastTo` F64→F32 (LDA doubles), single backbone (16 GB→8 GB), `PreloadWeights`, auto-**F32** (`HighPrecisionGemm`; TF32 1e-3/fwd accumulates over the AR loop). **Perf (2026-07-17): GPU-resident decode → ~6× (203→32 ms/frame stochastic on the 4090).** Replaced the host-glue attention block (host RoPE + `DiaHeads` reshapes + `RepeatKv` + host KV append, which broke the CUDA activation-residency cache and re-uploaded the O(n²) growing K/V every step) with a resident path mirroring the LLM `GenericTransformer`: `DiaAttention.SelfForwardFlash` (Q/K/V Linear straight into head-shaped tensors → `ApplyRopeInterleaved` GPU rope → `Permute0213` → `FixedKvCache` in-place `KvCacheAppend` → GQA-native `FlashAttention`, no K/V replication), gated on new `IBackend.FlashDecodeSupported` (CPU/Vulkan keep the host path). Greedy `Generate_Greedy` stays bit-parity. Now GPU-compute-bound on F32 Linear GEMVs (F16 is the next lever but risky — F32 is deliberate, TF32 degenerates over the AR loop). Speaker-encoder host syncs + MLP host-SwiGLU remain minor follow-ups. Extension `zonos_tts` wired (`ZonosModel.cs` + `ZonosTts` facade); golden `zonos_golden.py`; tests `ZonosSpeaker/Conditioning/Generate/Phoneme/E2eTests`.

### ACE-Step v1.5 turbo

DiT/cond-encoder/8-step loop all corr 1.0 (~1e-6) vs torch oracle on the real Comfy-Org turbo weights; Oobleck VAE corr 0.9999999999; e2e finite tonal stereo on CUDA. **Perf 2026-07-12:** DiT rewritten host-orchestrated → GPU-resident (device modulation/gated-residual/RoPE/KV-repeat, no per-op D2H sync); bit-identical to the pre-rewrite path (CPU golden maxAbs 0), **measured 55.3 ms/step = 0.44 s for the 8-step turbo DiT at 10 s audio on a 3060** (real weights, `AceStep15DitGpuBench`). Applies to all 9 variants. Follow-ups: F16 activations (needs a split-half F16 RoPE kernel), CUDA step-graph, XL quant.

### YuE

Stage-1 7B LM corr 1.0 (argmax 8/8) + XCodec (SoundStream) decode corr 1.0 → generates 16 kHz vocal audio. **2026-08-05: full pipeline now ACTIVE** — Stage-2 (m-a-p/YuE-s2-1B-general) + per-stem Vocos vocoders added to the weights catalog, vocoder `.pth`→safetensors auto-converts on first load (`EnsureVocoders`), Stage-1 precision is a policy (`HARTSY_AUDIO_LM_QUANT`, un-quantized bf16 when layer-split across GPUs). Verified perceptually + via Whisper STT (sung lyrics transcribe intelligibly; the old cb0-only 16 kHz draft transcribed as NOTHING — that path was the "garbled" mode and is now only a fallback when s2/vocoders are absent). **2026-08-05: the sharded-YuE Whisper check is now a committed regression test**, not a manual session — `YueLmShardingEngineTests.LmSharding_RealEngine_UnquantizedStage1_PooledAcrossGpus_ProducesAudio` generates real `[verse]/[chorus]` lyrics through the bf16 layer-split path and asserts >=50% Whisper content-word recall (real run: heard "Golden morning breaks across the ocean" for an 8.0s/400-frame clip, 2/4 target words hit — the clip length only reached the verse, not the chorus, so recall is duration-bound, not a quality ceiling). Stage-2/vocoder numerical parity vs the Python reference NOT yet run — STT + listening evidence only.

### HeartMuLa

LM corr 0.9996–0.9999 + HeartCodec rewritten: flow-match estimator corr 1.0 + ScalarModel corr 1.0 → generates 48 kHz audio (CPU + CUDA). **Perf (RTX 3060, 3b-base):** ~91 ms/frame ≈ 11 fr/s bf16 (~0.9× realtime, memory-bandwidth-bound). CUDA-graph decode of the backbone + depth steps (`HARTSY_CSM_GRAPH`, default on) is bit-identical + ~5% (launch overhead is only ~8/91 ms). Disk-cached weight quant (`HARTSY_HEARTMULA_QUANT=q8_0`) is **1.41× faster** (64.8 ms/frame ≈ 15.4 fr/s, past real-time) + ~1/2 VRAM — the fix was pinning the quant weights GPU-resident (`PreloadWeights`, quantized-only); the Q8 fused GEMV is faster than bf16 when resident.

### Stable Audio Open Small

DiT/VAE/timing-conditioner parity cosine 1.0 each. **Engine-wired 2026-07-20** as `stableaudio` (was built+parity-verified but had no `MusicCatalog` entry, and the AudioLab extension's own binding was also missing — both fixed). **GPU-residency perf pass 2026-07-20**: host RoPE/multi-head-reshape/ping-pong-scheduler loops ported to existing `IBackend` ops (`ApplyRopeSingle`/`Permute0213`/`RepeatKvHeads`/chained `Scale`+`Add`) — no new kernels needed, cosine 1.0 held after rewrite. Swarm `/API/GenerateText2Image` verified: 11.89s stereo 44.1kHz in 2.85s gen time.

### PocketTTS

**Swarm-deployed + parity-verified 2026-07-16.** Production voiced path built on the verified cores: `PocketTtsStreamingTransformer.ForwardPrimed` (voice-KV prefix + RoPE offset), `PocketTtsFlowLm.GenerateVoiced` (LUT conditioner + out_eos stop + noise std=√temp), `PocketTtsVoice` (KV-state loader), rewritten `PocketTtsPipeline` (SentencePiece + emb_std/mean denorm). All transformer layers + hidden + latents **corr 1.000000** vs `pocket_tts` 2.1.0. Voice REQUIRED (default `alba`). Weights = non-gated `kyutai/pocket-tts-without-voice-cloning` `languages/english/` (NOT the generic `tts_*.safetensors`). Whisper `medium.en` word-perfect via `/API/GenerateText2Image`.

### MiniMax Music 3

Lyrics + caption → 44.1 kHz stereo, up to six minutes. Qwen3-8B global LM (one 25 Hz semantic RVQ code per frame)
+ 0.6B depth decoder (seven residual codebooks) → the two models' **hidden states**, not their codes, condition a
2.4B flow-matching DiT whose latents a DAC-style vocoder decodes. See
`docs/Research/MINIMAX_MUSIC3_ARCHITECTURE.md` for the constants and the traps.

**Verified against diffusers PR #14456** (dump script: `tests/python-reference/dump_minimax_music3_reference.py`):

| Component | Result |
|---|---|
| Prompt assembly + token ids | exact (6 string cases + the README example's 58 ids) |
| Condition encoder | meanAbs < 1e-5 |
| DiT block 0 | meanAbs < 1e-4 |
| Full 36-layer DiT | meanAbs < 1e-3 at t=0 cond, t=0 uncond and t=0.5 |
| Vocoder | maxAbs < 1e-4; left/right provably distinct |
| Window/crop geometry | reproduces the reference's 529408-sample two-window stitch |

**Verified end to end**: a 25 s generation with the model card's Structured Caption was confirmed by listening —
real music with intelligible sung lyrics. That listening check is the gate that matters here; the numbers below
each cover one stage under forced inputs and, on their own, never distinguished music from noise.

**Both stage gates pass.** `MiniMaxMusic3ArParityTests` (teacher-forced, 8 frames) reaches corr 1.00000000 at
meanAbs 6.8e-7, and its frame-0 assertion confirms the skip rule directly. `MiniMaxMusic3FlowParityTests` runs the
whole flow stage from the reference's frame hiddens and forced noise across two windows: per-window latents
corr 0.9999990, stitched audio corr 0.9999963.

**The flow-stage divergence, found and fixed (2026-08-13)**: the stage first measured `corr 0.870` against the
reference *even when fed the reference's own frame hiddens and forced noise*. Bisecting one Euler step against
captured internals (`--stage flowprobe`) cleared the condition encoder (corr 0.99999996) and pinned it to the DiT —
which nonetheless passed on `CpuBackend`. The cause is engine-wide, not model-specific: **`Tensor.Reshape` reads
`DataPointer`, which syncs a device tensor back to the host and hands out a HOST pointer**, so the returned view has
no GPU residency. Applying rotary in place through such a view wrote to host memory while the device copy stayed
un-rotated, and CUDA then ran attention with no rotary at all. Allocating q/k/v directly at the rank-4 shape the
rotary op wants — and dropping the token-major attention entry point, which forced the reshape — cut the DiT's error
36x (meanAbs 4.1e-2 to 1.1e-3) and took the flow stage to stitched-audio corr 0.999996. **The generalizable lesson:
never `Reshape` a tensor that may be device-resident and then mutate it in place.**

**Levels**: the rotary fix is audible in the numbers. Same seed, same prompt, 17.8 s multi-window on the 3060 at
`:q4` — before the fix -19.5 dBFS peak 0.60, after it **-15.8 dBFS peak 1.0**, against the official 32 kHz asset's
-16.6. No level step at either window seam. Clips under ~10 s still measure ~20 dB quieter; that is intro material,
not a decode bug, and it is the third short-clip level scare on this machine. Do not "fix" it with normalization.

**Measured VRAM and timing** (3060, `:q4`, 30 s of audio, 7 windows): completes at ~10 GB peak in 226 s —
autoregressive 110.5 s, flow 105.0 s, vocoder 2.2 s. Two VRAM lessons are baked in here. Undisposed
`Tensor.Reshape` views used to grow the activation cache per denoising step, whose fingerprint is VRAM climbing
with the *number of forwards* rather than with tensor size. And the correctness fix that removed those reshapes had
to drop the token-major attention entry point (it needs rank-2, and reaching it from the rotary op's rank-4 layout
is what forced the reshape), which regressed the 3060 from a working 30 s generation to OOM at 12 s — fixed by
hoisting the fourteen per-block working tensors to one instance-level set reused across all 36 blocks and every
forward. Parity is byte-identical across that change.

Stage timing is emitted at `Info`; the CLI defaults to `Warning`, so use `HARTSY_LOG_LEVEL=Info` to see it.

**Performance grind plan: `MINIMAX_MUSIC3_PERF.md`** (phases, hardware protocol, out-of-scope list).

**Versus the reference** (4090, 15.0 s of audio = 375 frames = 3 windows, identical prompt/seed/steps, generation
time only with model load excluded on both sides): this engine's Q8 path takes **36.7 s** (AR 26.0, flow 10.3,
vocoder 0.4) against the diffusers reference's BF16 **49.4 s** (AR 34.9, flow+vocode 14.6) — **1.35× faster**, and
faster in both stages independently. Reproduce the baseline with `mm3-ref/stagebench.py`, which stages AR then frees
it before the flow stage exactly as the engine does; run it unstaged and the reference OOMs a 24 GB card at ~22 GB
resident.

The comparison is Q8 against BF16 because **this engine's own BF16 path does not fit 24 GB** while the reference's
does: `CudaBackend.LinearImpl` runs with `cacheWeightCast: true`, so each BF16 weight also caches a device-side
dtype cast, roughly doubling the 17.2 GB language model. That is a genuine gap, not a measurement artifact — a
like-for-like BF16 comparison is not currently possible on this hardware, and fixing the cast caching would both
close it and make the bare variant usable on a 24 GB card.
