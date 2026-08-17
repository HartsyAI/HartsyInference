# Audio Models — status

Concise status for every audio model: TTS, STT, and the codec / voice-conversion / music / separation
family. Open work (including the music-model completion list) is in the [Remaining work](#remaining-work)
section below; bring-up debugging notes live in [TROUBLESHOOTING.md](TROUBLESHOOTING.md). Parity evidence
(maxAbs, bugs found) lives in [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

> ## 🔧 Sweep-failure fix pass (2026-07-24) — the 37-model Swarm-API benchmark's failures root-caused
> The full-fleet Swarm-API sweep (27/37 generated; results in `benchmarks/swarm_audio_bench/
> swarm_audio_results.json`) surfaced 7 real failures. Fixed + verified this pass:
> - **Zonos "missing model.safetensors"** — NOT a registry bug: every Zonos weight was a symlink into
>   `~/.cache/huggingface/hub`, and a hub cache cleanup deleted the blobs. On .NET, `File.Exists` reports a
>   dangling symlink as PRESENT (link, not target), so `AudioModelCache.GetAsync` returned the dead path
>   instead of re-downloading. Fixed with a link-aware `IsUsableFile` (resolves the final target) + dangling-
>   link deletion before the atomic move; verified: Zonos self-healed (re-downloaded) and its clone output
>   transcribes word-perfect. **This protected every symlinked model in the audio cache**, not just Zonos.
> - **`distilwhisper` bare id** — v3.5 config case added (see the 2026-07-21 bullet below, now FIXED).
> - **Demucs "CUDA STFT not supported" via Swarm** — the CLI's per-command CPU forcing never applied to the
>   Engine path; `FxService` now forces a cached CPU backend for BOTH Separate and Enhance (the FX pipelines
>   are STFT-centric and GPU backends don't implement Stft). Verified through the real service path with an
>   auto→CUDA engine: 4 distinct stems.
> - **YuE "checkpoint folder not found"** — two stacked causes: (1) the checkpoint set was never downloaded —
>   completed via `AudioWeightsCatalog.EnsureAsync` into the Swarm models root (`HARTSYINFERENCE_MODELS`);
>   note the primary `HartsyAI/YuE-xcodec-mini-safetensors` repack 401s anonymously (repo private + the
>   stored `~/.cache/huggingface/token` is REVOKED — refresh it), so the public `m-a-p/xcodec_mini_infer`
>   fallback was used (loader auto-converts on first load). (2) a bare `-m yue` resolves the literal token as
>   the variant → `yue/yue` path; `AudioWeightsCatalog.NormalizeVariant` now maps bare/family-literal specs to
>   the registered default (`en-cot`; ACE-Step `turbo` same class). E2E generation still pending a free GPU
>   (7B doesn't fit the 3060; the 4090 is held by the long-running SwarmUI process).
> - **HeartMuLa OOM-killing Swarm (49 GB host RSS)** — the bf16 load path kept the full ~15 GB F32 source
>   mmaps resident alongside the ~6 GB BF16 working copy for the model's whole lifetime (the quant path
>   already disposed its sources). All tensors are now materialized into owned memory and the F32 loaders
>   disposed: measured load-time RSS **29.4 GB → 14.4 GB**. Generation-phase growth still to be profiled on a
>   free GPU. Two follow-ups noted: on the 12 GB 3060 generation fails with a CUDA error whose original
>   exception is MASKED by a throwing `GraphStream.Dispose` during unwind (CsmModel) — fix the masking; and
>   quant variants (`heartmula:*-q8`) further halve the working set.
> - **Chatterbox voice cloning wired (FULL clone)** — see the Chatterbox row; upstream `prepare_conditionals`
>   replicated end-to-end (ve.safetensors LSTM voice encoder + partials protocol + CAM++/S3-tokenizer/24k-mel
>   S3Gen reference dict, reusing the verified CosyVoice2 zero-shot components). 8/8 Chatterbox tests; cloned
>   output word-perfect via Whisper; audibly/measurably distinct from the default voice. A GPU e2e re-run
>   awaits free VRAM (CPU path fully verified).
> - **Resemble-Enhance FIXED (real weights, end-to-end)** — the "404" was the generic loader trying
>   model.safetensors/pytorch_model.bin (never existed); the deeper blocker was the module-composition
>   mismatch the 2026-07-21 note documented. The denoiser/IRMAE/UnivNet modules were rewritten to the real
>   checkpoint structure (encoder/middle/decoder blocks of pre_conv + PreactResBlocks), and
>   `FxCatalog.LoadEnhanceAsync` now fetches the DeepSpeed `enhancer_stage2/ds/G/default/
>   mp_rank_00_model_states.pt` via AudioModelCache + DeepSpeedCheckpointConverter. Verified: strict key-set
>   load on the real ~700 MB checkpoint, 4/4 ResembleEnhance tests, and e2e `fx enhance` on the 11 s JFK clip
>   → 11.0 s 44.1 kHz output, RMS 0.138 (input 0.142), Whisper transcribes the enhanced audio word-perfect.
>   **PERF CAVEAT: ~94 min wall for 11 s of audio on the single-threaded CPU path (RTF ~510×)** — usable for
>   correctness only; a perf pass (parallelize the CPU conv/CFM path or implement backend Stft on CUDA) is
>   required before this is practical. RVC's "voice model not found" is BY DESIGN (user-supplied voice
>   model; manual placement documented).

> ## 🔌 CLI wiring pass (2026-07-21) — every local audio model given a `ModelCatalog` recipe
> `hartsy speak/music/transcribe` only exposed 9/20 TTS, 2/6 STT, and 5/6 music models before this pass (the
> rest were Engine-registered in `TtsCatalog`/`SttCatalog`/`MusicCatalog` but had no CLI catalog entry, so
> `hartsy list` didn't show them and there was no download-confirm flow). Added `CatalogEntry` + `Assets` for
> the missing 12 TTS (dia, orpheus, csm, neutts, qwen3tts, chatterbox, kyutaitts, melotts, pockettts, zonos,
> gptsovits, zipvoice), 4 STT (distilwhisper, moonshinestreaming, kyutaistt, whisperstreaming), and heartmula
> (music) + fixed audiogen/stableaudio's stale entries. Added two new modalities/commands: `hartsy convert`
> (RVC/OpenVoice voice conversion) and `hartsy fx separate|enhance` (Demucs/Resemble-Enhance), backed by the
> Engine's already-built `IVoiceConversionService`/`IFxService`. Wired `--reference`/`--ref-text`/
> `--exaggeration`/`--nfe-step`/`--cfg-scale` through `hartsy speak` for clone-capable models (`SpeechRequest`/
> `TtsJob` already carried these fields end-to-end; only the CLI-side plumbing was missing).
>
> **Discovered the audio download path is a *different mechanism* than the image catalog's**: audio
> descriptors self-download via `AudioModelCache` (`~/.cache/hartsyinference/models/{owner}--{name}/`), not
> `ModelDownloader`'s `Models/<subdir>/` tree — reusing the image-style `ModelAcquisition.EnsurePresent`
> unmodified would have "confirmed" a download into a location the Engine never reads from. Added an
> audio-aware branch (`ModelAcquisition.EnsureAudioAssetsPresent`) that resolves against the real cache path
> before prompting.
>
> **Bugs found by a systematic "does every catalog id resolve against its Engine registry" sweep** (cheap,
> no-GPU, catches typos the download-confirm flow can't): 8 of the 27 wired ids didn't match their
> `TtsCatalog`/`MusicCatalog` registry key exactly (case-insensitive but hyphen-sensitive) — `fish-speech`≠
> `fishspeech`, `spark-tts`≠`sparktts`, `f5-tts`≠`f5`, `ace-step`≠`acestep` (all **pre-existing**, silently
> broken before this pass), plus `qwen3-tts`, `kyutai-tts`, `gpt-sovits`, `stable-audio` (introduced during
> this pass, then caught by the same sweep before shipping). Fixed by renaming the catalog `Id` to match.
> Also found `MusicCommand` hard-required `--model-path` for **every** music model, including the
> self-downloading catalog ones (musicgen included) — removed; only VC/FX models with no fixed HF weights
> (RVC voices, Demucs) still need it.
>
> **Exhaustive real-weight e2e pass via standalone CLI on the 3060** (GPU shared with a concurrent image-model
> session, per the turn-taking directive) — every TTS (21/21), STT (6/6), and Music (6/6) catalog model
> actually run, output inspected (Whisper `medium.en` transcript for speech, finite/non-silent/correct-rate
> waveform check for music), not just "didn't throw":
> - **All 21 TTS word-correct** except two genuinely broken (below): dia, zonos, qwen3tts, kokoro, bark,
>   melotts, orpheus, styletts2, sparktts, cosyvoice, chatterbox, neutts, fishspeech, f5, gptsovits, zipvoice
>   (degraded but intelligible — content present, some words garbled, matches its documented "no
>   GPU-residency pass yet" perf status), pockettts, kyutaitts, piper. Whisper mis-hears several coined brand
>   names (Kokoro→"Kakura", Bark→"Bach", Zonos→"Zono's", Kyutai→"Qtie", …) — expected, matches this doc's own
>   prior Kyutai→"QTIE" precedent, not a defect.
> - **`csm` FIXED 2026-07-21** (was BROKEN: `KeyNotFoundException` on `backbone.norm.weight` — the `nielsr/csm-1b`
>   mirror ships torchtune-style keys the loader never matched). Three real bugs fixed, not just the loader:
>   (1) switched to the `unsloth/csm-1b` mirror (HF `transformers`-layout keys) + a new `CsmWeightRemap` that
>   re-prefixes them and splits its two combined tensors (`audio_embeddings`, `codebooks_head`) into the
>   per-codebook slices `CsmModel.LoadWeights` reads, transposing the head to `[vocab, hidden]`; also sources
>   Mimi from the SAME checkpoint's bundled `codec_model.*` (32 codebooks) instead of the separately-published
>   8-codebook `kyutai/mimi`, which the depth decoder's 32-codebook output can't feed; (2) `CsmPipeline`'s codes
>   tensor was built `DType.F32` but `Mimi.Decode` reads codes as `Int32` — reinterpreting float bit patterns as
>   codebook indices crashed with `AccessViolationException`; (3) generation never stopped (always exactly 1024
>   frames / 81.9s regardless of input) because the EOS check compared codebook-0 to the wrong sentinel
>   (`AudioEosToken=2048`, never actually produced) instead of upstream's real condition — ALL codebooks equal 0
>   (`CodebookEosToken`) after decoding the full frame; also added the missing `[speaker]text` + BOS/EOS prompt
>   template (`AudioTextFrontend.CsmText`), which the original plain-BPE encoding was missing entirely. Verified
>   Whisper word-perfect on two independent sentences ("Hello there, how are you today?" and a 17-word sentence,
>   both exact). `CliDrivable = true`.
> - **`vibevoice` re-verified 2026-07-21 — NOT reproduced.** This same pass had flagged it BROKEN (Whisper
>   transcribing `[Hindi]`/`[speaking in native language]` on the JFK reference). Re-testing found no code
>   change to VibeVoice since the 07-17 perf pass (only a namespace rename), a clean tokenizer round-trip, and
>   3 independent prompts (short pangram, long multi-sentence, unrelated paragraph) all Whisper word-perfect
>   with the same reference clip. Root cause of the original failure unknown (unreproduced, so undiagnosable).
>   `CliDrivable = true` restored; re-flag if it recurs.
> - **`cosyvoice` needs `--ref-text` for usable quality** — `--reference` alone produced garbled
>   non-word-correct output ("And a tall fall, tear, tape, tape." for a plain sentence); `CosyVoiceModel`
>   accepts an empty transcript without throwing, but clearly needs it. With `--ref-text` set, word-perfect.
> - **`gptsovits` REQUIRES `--ref-text`** (throws `InvalidOperationException` without it, unlike CosyVoice's
>   silent degradation) — word-perfect once supplied.
> - **`qwen3tts`'s bare id defaults to Base/voice_clone** (requires `--reference`) because
>   `Qwen3TtsModel.ResolveMode` reads the whole variant string and "qwen3tts" contains neither "CustomVoice"
>   nor "VoiceDesign" — use `-m qwen3tts:1.7B-CustomVoice`/`:1.7B-VoiceDesign` for the preset/instruct modes.
> - **Fixed a real, previously-100%-broken feature**: `--voice` was wired as a bare alias of `-m|--model`
>   (`[CommandOption("-m|--model|--voice")]`), so `SpeechRequest.Voice`/`job.Voice` could never be set from the
>   CLI at all — silently breaking Kokoro non-default voice packs, Spark-TTS gender, and PocketTTS's
>   (REQUIRED) voice name for every prior CLI session. Split into a separate `--voice` option that populates
>   `parameters["voice"]`; verified by requesting Kokoro's `am_michael` pack and confirming it downloads and is
>   used (distinct from the `af_heart` default already cached).
> - **`distilwhisper`'s bare id is broken**: `SttCatalog.ResolveDistilWhisperRepo`'s no-match default
>   (`distil-whisper/distil-large-v3.5`) isn't in `WhisperPipeline.InferConfig`'s repo switch (only
>   v2/v3/medium.en/small.en) → "Unknown Whisper repo". Use `-m distilwhisper:v3` (or `:v2`/`:medium`/`:small`).
>   **FIXED 2026-07-24**: v3.5 is architecturally identical to v3 (confirmed against its config.json:
>   1280/32enc/2dec/128mel/51866vocab — the .5 is a longer-trained release); `WhisperConfig.DistilLargeV3_5`
>   + the InferConfig case added, bare `-m distilwhisper` verified word-perfect on the JFK clip (auto-downloads
>   v3.5 on first use).
> - **musicgen/audiogen/acestep/stableaudio/heartmula** all produce finite, non-silent, correct
>   sample-rate/duration/channel-count audio (heartmula's RMS ran noticeably quieter than the others — not
>   re-investigated, flagged for a follow-up listen). **`yue`** needed a real fix, not just verification (next
>   bullet).
> - **Fixed `MusicCommand` hard-requiring `--model-path` for every music model**, including the
>   self-downloading catalog ones (musicgen included) — removed; only VC/FX models with no fixed HF weights
>   (RVC voices, Demucs) still need it.
> - **Discovered and fixed a THIRD download mechanism**: unlike the `AudioModelCache`-self-downloading
>   TTS/STT/MusicGen-family models, ACE-Step and YuE are "registry-backed local-checkpoint families" resolved
>   via the (previously `internal`, now `public`) `AudioWeightsCatalog` + the STANDARD `ModelDownloader`/
>   `ModelAsset` machinery — the same one image models use — landing under `Models/audio/music/{acestep,yue}/`.
>   Worse, `YueMusicModel.LoadAsync` has **no auto-download fallback of its own** (unlike `AceStepMusicModel`,
>   which calls `AudioWeightsCatalog.EnsureAsync` internally) — before this fix, `yue` was 100%
>   manual-placement-only ("YuE checkpoint folder not found... place the m-a-p/YuE-s1-7B-anneal-* folder
>   there"). Made `AudioWeightsCatalog`'s id constants/`AssetsFor`/`IsFolderCheckpoint` public, reused
>   `AssetsFor` directly in the CLI catalog (same pattern as `SideModels` for image entries — no data
>   duplication), and special-cased `ModelAcquisition` to route `acestep`/`yue` through the standard
>   `ModelDownloader` branch instead of the `AudioModelCache` branch (with a folder-checkpoint guard so YuE's
>   resolved `LocalPath` doesn't get set to a single shard file instead of the variant directory). Verified:
>   the CLI now correctly lists YuE's 8 files + confirms + downloads into `Models/audio/music/yue/en-cot/`.
> - **VoiceConvert**: `openvoice` verified (real, non-silent, correct-rate audio via `--target`). `rvc` only
>   confirmed to *resolve* (registry lookup, no "not registered" error) — there is no default/test RVC voice
>   checkpoint anywhere on this box (RVC is inherently bring-your-own-trained-voice), so a real generation
>   pass isn't meaningfully possible without the user supplying one via `--model-path`.
> - **Fx**: **`demucs` FIXED 2026-07-21.** The real HF search for an ungated single-file htdemucs mirror was the
>   wrong approach — the canonical weights were never on HuggingFace at all; upstream ships them from Meta's own
>   public CDN (`dl.fbaipublicfiles.com`, no auth, resolved from `demucs/remote/htdemucs*.yaml` +`files.txt` on
>   GitHub). `FxCatalog` now auto-downloads `htdemucs` (4-stem) and `htdemucs_6s` (6-stem) directly from there on
>   first use. The existing `PytorchPickleLoader` parses the official `.th` with zero changes (533 tensors, keys
>   matching `HtDemucs`/`DemucsCrossTransformer`/`DemucsDConv`'s existing expectations exactly — the engine code
>   was already built against the real key layout, just never had a working download source). Fixed a real bug
>   found running it: bare `-m demucs` resolves `AudioModelSelector.Variant` to the literal string `"demucs"`
>   (not `"htdemucs"`), so `EnsureDemucsPathAsync` now treats that alias the same as unset. `htdemucs_ft` stays
>   `--model-path`-only: upstream ships it as a 4-checkpoint weight-averaged `Bag_of_models` ensemble, not a
>   single 4-stem checkpoint — out of scope. CPU-only (`DemucsSpec`'s STFT/ISTFT has no CUDA/Vulkan
>   implementation — `CudaBackend.Stft` throws `NotSupportedException`); `FxSeparateCommand` now always forces
>   the CPU backend itself so the default `hartsy fx separate <wav>` invocation just works instead of erroring
>   on the `-b auto → cuda` default. Verified real output on a music clip: 4 stems (drums/bass/other/vocals),
>   mutually distinct (pairwise sample corr 0.007–0.14 — not copies of each other or the mix) and non-silent.
>   `htdemucs_6s` is wired identically but not individually run this pass.
> - **`resemble-enhance` real-weight load fails, ROOT CAUSE NARROWED but NOT FIXED.** The DeepSpeed checkpoint
>   angle from the earlier pass was a red herring in scope, not diagnosis: `PytorchPickleLoader` (already-existing,
>   unmodified) parses `enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt` (713 MB) cleanly — 909 real
>   tensors, meaningful names — no DeepSpeed-specific reader is actually needed. The REAL blocker: the engine's
>   `ResembleDenoiser`/`ResembleIrmaeDecoder` (and likely `ResembleWnEstimator`/`ResembleUnivNet`) were built
>   against an assumed key layout (`down.{i}`/`mid.{i}`/`up.{i}`, `conv1`/`norm1`/`conv2`/`norm2`/`downsample`)
>   that does **not** match the real checkpoint (`encoder_blocks`/`middle_blocks`/`decoder_blocks`, each a
>   `pre_conv` + two `PreactResBlock`s of `GroupNorm→GELU→Conv2d` ×2 with residual add — confirmed by reading the
>   real `resemble_enhance/denoiser/unet.py` and `.../enhancer/lcfm/irmae.py` from GitHub). This is a genuine
>   forward-pass mismatch, not a naming alias — the module composition differs, so a remap dict (the CSM fix
>   pattern) isn't enough; it needs `ResembleDenoiser`'s `UNetBlock`/`PreactResBlock` (and probably the
>   `lcfm.ae`/`lcfm.cfm.net`/`vocoder` modules — not yet individually verified, but the denoiser's mismatch alone
>   rules out a quick fix) rewritten to match the real architecture. Deliberately NOT attempted this pass — a
>   half-rewritten multi-file UNet is worse than leaving this honestly `ValidationPending`/not-CliDrivable.

> ## 🏁 First e2e TTS/STT speed benchmarks (2026-07-12) — RTF on 3060 **and** 4090
> Measured through the SwarmUI+AudioLab path: `benchmarks/results/audio_tts_stt_2026-07-12.md`.
> Piper 10.4×/7.7×, Moonshine 6.5×/6.5×, Whisper-base 5.1×/5.4×, MeloTTS 1.7×/1.8× (3060/4090). **These small
> models are host/launch-bound — the 4090 barely helps; the lever is CUDA-graph capture, not a bigger GPU.**
> **Runtime outliers found (parity ✅ ≠ runnable):** ~~Kokoro install 401~~ **FIXED 07-13** (canonical-`.pth`
> download fallback); ~~Whisper `/API/ProcessSTT` rejects the default `en-US`~~ **FIXED 07-13**
> (`WhisperTokenizer.LanguageToTokenId` normalizes locale codes `en-US`→`en`); Spark-TTS install errors
> "checkpoint-reconciliation-pending" (not wired for runtime despite ✅ test parity — still open).
>
> ## ✅ TTS correctness + F5 perf pass (2026-07-13)
> Verified word-correct through the canonical `GenerateText2Image` path with **whisper `medium.en`** as the oracle
> (`base.en` dropped — it hallucinated the pangram onto broken audio) + RMS-envelope match vs a Python reference,
> across short/long/numbers/punctuation prompts: **Kokoro, Piper, MeloTTS, F5-TTS** all fixed to word-correct.
> MeloTTS root cause was a `PytorchPickleLoader` stride bug (transposed BERT weights → gibberish) + missing number
> normalization. **F5-TTS: 6.4 s** — the per-forward `F5ConvPosEmbed` grouped
> Conv1D was a host loop; routed to `backend.Conv1d` (GPU), output bit-parity. Remaining perf target: MeloTTS
> (1.4×, BERT+VITS host-flat) + a CUDA-graph pass on the host-bound small models.

> ## 🗺️ Full local-TTS Swarm runtime scoreboard (2026-07-13)
> AudioLab declares ~19 **local** engine-backed TTS providers (+ 20 cloud-API providers — ElevenLabs/Azure/OpenAI/
> etc. — which proxy to third parties and aren't engine models) and 6 local STT. "Has an install button" ≠ "engine
> runtime is wired." Verified via install → `GenerateText2Image` → medium.en. Actual runnable status:
>
> | Status | TTS |
> |---|---|
> | ✅ **verified word-correct** | Kokoro, Piper, MeloTTS, F5-TTS, **Bark**, **Chatterbox**, **VibeVoice**, **FishSpeech**, **Orpheus**, **Dia** (Swarm 10/10, all 3 turns, EOS-stops 11.4s — fixed by switching to the `Dia-1.6B-0626` checkpoint), **Kyutai TTS** (DSM, in-engine e2e word-correct 2026-07-16; **Swarm-deployed + verified word-perfect 2026-07-16**) |
> | ✅ **Qwen3-TTS all 3 modes SWARM-VERIFIED** 2026-07-18 (alpha.51) | custom_voice + voice_design word-perfect (after the extension `Encode`→`EncodeRaw` tokenizer-padding fix); **voice_clone (x-vector)** implemented + Swarm-verified word-perfect (T2I path, model `Audio Models/Qwen3TTS/1.7B-Base` + `referenceaudio` param → `mode=voice_clone`) — rewrote `EcapaSpeakerEncoder` to the real no-BatchNorm `speaker_encoder.*` layout, 128-bin slaney log-mel, ECAPA x-vector injected at the codec speaker position (Base checkpoint). Voice-responsive (corr 0.01 across two refs). ICL mode (needs the codec encoder port) is the only remaining variant. **Deploy TRAP: keep the extension TFM net8.0 (host is net8.0); a net10.0 override → `ReflectionTypeLoadException`. And purge stale local-feed/nuget-cache versions or they shadow the real publish.** NeuTTS voice-clone ✅ works e2e (the "CodecEnc.*" gate was stale). **PERF 07-18: warm RTF 0.88→~0.50 on the 3060 (1.77×, bit-identical) — root was `FixedKvCache` re-uploading ~1.9 GB of zeroed KV across PCIe every gen (device-resident-KV fix helps ALL AR audio models) + zero-alloc top-K sampler + on-device vocoder residual; talker is HBM-bandwidth-bound (FP8 is the next lever). See `benchmarks/results/qwen3_tts_perf_2026-07-18.md`.** |
> | 🚧 **build started 2026-07-15** | **StyleTTS2** (LibriTTS) — recon done: checkpoint downloaded, structure mapped, dims confirmed Kokoro-compatible (hidden 512 / style 128×2 / n_token 178 / 24 kHz / decoder 8h·3L → bert/text_encoder/predictor/decoder reuse Kokoro loaders). Remaining: reconcile the two style submodules to the real checkpoint (StyleEncoder ResBlk uses a **learned depthwise downsample** `downsample_res.conv` + `conv1` is dim_in→dim_in, not the scaffold's avgpool/dim_out; diffusion transformer uses archinetai `net.blocks`/fused-`to_kv`/`norm.fc`-AdaLN, not the scaffold's `unet.blocks`) + spectral-norm σ-fold + write `LoadFromCheckpoint`. See the Remaining work section. |
> | ✅ **Swarm-deployed 2026-07-16/17** (extension wired, whisper word-perfect) | **Kyutai TTS**, **Spark-TTS** (controllable mode), **PocketTTS** (voice-KV-primed continuous-latent flow-LM; full engine port, parity corr 1.0), **Zonos-v0.1** (transformer; voice-clone, gallery gen verbatim `Audio Models/Zonos/transformer`), **CosyVoice 2** (0.5B; zero-shot clone, gallery gen `Audio Models/CosyVoice/2-0.5b`, whisper word-perfect 2026-07-17; perf pass → **warm 5.1 s/clip** via GPU-resident CFM flow decoder, RTF 1.34) |
> | ⛔ **not wired** (install throws a clear "not runnable yet") | CSM (no runtime model) |
>
> STT (6 local): ✅ Moonshine, Whisper verified word-perfect on real (JFK) speech; Distil-Whisper / Kyutai STT /
> RealtimeSTT / Whisper Streaming not yet installed/verified. **Bark DONE 2026-07-18 → RTF 2.30** (82.62 s →
> 12.23 s on a 5.32 s clip, 6.75×; behavior-preserving, whisper word-perfect, token stream unchanged). Root
> cause was `GptBlock.ForwardStep` running the whole decode-step attention on the CPU host (24 device syncs +
> host K/V + host attention loop per step); fix = migrate Bark's decode to the shared device-resident
> `FixedKvCache` + `FlashAttention` (the same path the fast Chatterbox-T3/`GenericTransformer` use), plus
> fine-stage host-glue→device. Zero new kernels. See `benchmarks/results/bark_tts_2026-07-18.md`.
> **VibeVoice DONE 2026-07-17 → RTF 0.78** (134 s → 6.47 s on an 8.27 s clip, 20.7×,
> faster-than-real-time; VAE causal convs CPU `float*`→`backend.Conv1d`, then batched-CFG + diffusion-head
> host-glue→GPU; output corr 0.999585; see `benchmarks/results/vibevoice_tts_2026-07-17.md`).
> **Chatterbox DONE (no work needed) 2026-07-18 → RTF 0.69–0.90**, warm 3.51 s / 5.08 s audio (stages
> T3 1.49 s / Flow 1.66 s / Vocoder 0.35 s) and 12.36 s / 13.80 s audio on a long clip; whisper word-perfect.
> The old "219 s" figure predated the 07-17 CosyVoice2 S3Gen GPU-residency refactor, which Chatterbox's
> S3Gen reuses verbatim — it inherited the speedup for free. See `benchmarks/results/chatterbox_tts_2026-07-18.md`.
> All three of the previously-slow correct TTS models (VibeVoice, Chatterbox, Bark) are now faster-than- or
> near-real-time; no known unusably-slow correct TTS model remains.

> ## ⚠️ STT reality-check (2026-07-08) — parity ✅ does NOT mean intelligible speech
> The ✅/🔬 marks below are **numeric-parity** verdicts (corr 1.0 vs a Python reference on random/tap inputs).
> A real-weight end-to-end pass — generate audio → resample → Whisper-base STT → content-word recall, then
> a human listen — tells a very different story. Results so far (each writes a WAV to
> `{TmpPath}/hartsyinference_tts_to_stt/`; the round-trip tests that produced this table were removed in the
> 2026-08-06 suite cleanup — the results below stand as the recorded outcome):
>
> | Model | Doc mark | Whisper heard | Real verdict |
> |---|---|---|---|
> | **Kokoro** | ✅ | "Hello world. This is a test." (4/4) | ✅ **genuinely works** |
> | **MeloTTS** | ✅ | "Hello World, this is a test of the speech synthesizer." (5/5) | ✅ **genuinely works** |
> | **F5-TTS** | ✅ "bit-exact" | (07-08) "(laughs)" → (07-13, with a real voice ref) word-perfect | ✅ **works 2026-07-13** — the 07-08 run had no voice reference; given a reference clip + transcript through Swarm it transcribes word-perfect (medium.en) and clones the voice. Also 34× faster (host-conv→GPU). |
> | **Dia-1.6B** | ✅ | "Hello there! This is a test of the DIA text-to-speech model. It really does sound quite natural, doesn't it? Yes, the dialogue flows nicely between the two speakers." (10/10) | ✅ **fixed 2026-07-15** — the "(crickets)"/loop was the **wrong checkpoint**; `Dia-1.6B-0626` (drop-in) transcribes word-perfect through Swarm and EOS-stops at 11.4s. Not a gen-loop/DAC bug after all. |
> | **Qwen3-TTS 1.7B** | ✅ "bit-exact" | "…this is a test of the Quen Speech Synthesizer." (word-perfect) | ✅ **fixed 2026-07-18** — the earlier "RMS 0 / silent-garbage" was the extension tokenizing with `Qwen3Tokenizer.Encode` (right-pads to 512 with `<\|endoftext\|>` + no byte-level), flooding the talker text stream (a 4 s line → 32 s). Fix = `EncodeRaw`. custom_voice + voice_design word-perfect; RTF ~1.0. |
>
> Lesson: the whole audio suite's "verified" status rests on parity tests that are blind to whether the
> assembled pipeline (sampling, delay, codec decode, vocoder) makes speech. **A model is not "working" until
> Whisper recovers its words and a human confirms the WAV.** Debugging the ✗ models + STT-verifying the rest
> (Chatterbox/VibeVoice/Bark/NeuTTS/FishSpeech + the download-blocked set) is the open work.
>
> Engine changes made during this pass: `DiaPipeline.Generate` now preloads weights to **VRAM**
> (`PreloadWeights`/`FreeWeights`, like YuE) instead of streaming F32 from host RAM per op — a 6.4 GB model now
> lives on the GPU (VRAM 1.7→8.2 GB) with host RAM free, which is also what stops it OOM-crashing a
> RAM-constrained box; the Dia DAC `.pth` state-dict-unwrap load fix; bert-base-uncased converted to
> safetensors (loader can't read legacy pre-1.6 pickle). Heavy runs go through a RAM-watchdog script that
> hard-kills below 1.5 GB free.

## TTS

| Model | Status | Notes |
|---|---|---|
| **GPT-SoVITS v2** | ✅ | HuBERT 1.07e-5, s1 GPT + s2 SoVITS verified, EN end-to-end → 32 kHz on real `lj1995` weights. |
| **Chatterbox** (ResembleAI) | ✅ | Full S3Gen rewrite (== CosyVoice2); enc 2.6e-6 / dec 4.4e-5 / vocoder 1.6e-5; end-to-end on CUDA. |
| **CosyVoice 2** | ✅ | Full zero-shot clone e2e on real weights (Qwen LM `llm.pt` + OT-CFM `flow.pt` + HiFTNet `hift.pt`, default key maps correct); S3 tokenizer + CAM++ loaded from chatterbox `s3gen.safetensors` (frozen, identical — CosyVoice's own ONNX fuses Conv+BN / mangles names). Swarm-deployed + whisper word-perfect 2026-07-17. **Streaming DONE + Swarm-deployed 2026-08-11** (`CosyVoicePipeline.SynthesizeStream`, `tts_streaming` flag, native incremental `GenerateText2ImageWS` path — verified live through `swarmui.service`: cold+warm calls, 8-9 real chunks each, whisper-correct content, zero errors). True end-to-end real-time factor is ~8× (LM decode dominates, not the flow/vocoder stage, which alone tunes to ~3.45×). One accepted small artifact: an isolated single-word mispronunciation under adversarial content (~1.25% WER on the test utterance, non-cascading) inherent to the bounded-context-window design — confirmed via a parameter sweep + a decisive discriminator (an unbounded full-history variant avoids it, at disqualifying cost) that only full history resolves it, not a fixable tuning knob. |
| **Qwen3-TTS** | ✅ | Bit-exact (RoPE split-half + byte-level tokenizer fixes). |
| **Piper** (VITS) | ✅ | corr 0.9998 vs onnxruntime; 7 VITS bugs fixed (affect all VITS). **Swarm e2e word-correct 2026-07-13** — fixed the espeak language default (`en` British → the voice's `en-us` American; it was mispronouncing vowels). |
| **Kokoro** (StyleTTS2) | ✅ | ~1e-4 on the CUDA path (added `audio_leaky_relu` / `audio_adain1d` kernels). **Swarm e2e word-correct 2026-07-13** — misaki-phoneme g2p + punctuation fix (was silently dropping words); canonical-`.pth` download fallback (was install-401). |
| **F5-TTS** (v1 Base) | ✅ | Flow-matching DiT verified bit-exact: velocity corr 1.0, full CFM sample loop (generated mel) corr 1.0, Vocos corr 0.9999. 4 bugs fixed (ConvNeXt filler-mask, ×1000 timestep scale, erf/tanh GELU split, cond-anchored CFG + end-only ref-clamp). ([details](#f5-tts)) |
| **ZipVoice** (k2-fsa) | ✅ | Zipformer backbone (`fm_decoder`+`text_encoder`) parity cosine 1.0 (2026-07-19). ([details](#zipvoice)) |
| **Kyutai TTS** (tts-1.6b-en_fr) | ✅ | **Fully intelligible e2e in pure C# 2026-07-16** (whisper medium.en: "So hello there, this is a test of the Cuta[=Kyutai] text-to-speech model" — matches the script, no clipping, 62 frames/4.96s vs moshi ref 71/4.4s). ([details](#kyutai-tts)) |
| **ResembleEnhance** | 🔬 | Modules synthetic-verified + converter built; real-weight mel→mel parity pending. |
| **MeloTTS** (English-v3) | ✅ | Real-weight e2e in pure C#. ([details](#melotts)) |
| **Spark-TTS-0.5B** | ✅ | Real-weight e2e bit-exact, fully in-engine (controllable mode): LM logits corr 1.0 (top-1 100%), greedy tokens 32/32 global + 179/179 semantic match Python, BiCodec wav corr 1.0 (factorized VQ, FSQ d-vector, AdaLN PreNet all corr 1.0). ([details](#spark-tts-05b)) |
| **FishSpeech 1.5** | 🔬 | DualAR LM verified: slow (24-layer) corr 1.0, fast depth-LM (4-layer) corr 0.9999. fused-key adapter + interleaved RoPE + no embed-scale + pre-norm fast input. Only the firefly-gan-vq codec remains. |
| **Dia-1.6B** | ✅ | **Swarm e2e word-correct 2026-07-15 (10/10, all 3 turns) — root cause was the WRONG CHECKPOINT.** The full transformer was already bit-exact; the "loops *Hello there* / non-verbal garbage across seeds" symptom was the engine faithfully running the **old** `nari-labs/Dia-1.6B` release. ([details](#dia-16b)) |
| **Orpheus** | ✅ | **Swarm e2e word-correct 2026-07-14** (Llama-3.2-3B + SNAC-24k). Fix was the prompt frame (missing BOS 128000 + StartOfAi 128261/StartOfSpeech 128257). Perf: 10 s via a fused BF16 lm_head GEMV (lm_head was 90% of decode). |
| **Bark / Chatterbox / VibeVoice / FishSpeech** | ✅ | Swarm e2e word-correct (2026-07-13/14). ([details](#bark--chatterbox--vibevoice--fishspeech)) |
| **VibeVoice-Realtime-0.5B** (`vibevoice:realtime`) | 🔧 | **2026-08-10**: split-LM architecture built and loads against the real checkpoint (4-layer text encoder + 20-layer TTS backbone as two genuinely separate weight-bearing Qwen2 stacks, binary EOS classifier, `tts_input_types` splice — all confirmed from the real 608-key `model.safetensors`, not the architecture doc's prose). **Cannot generate yet**: the released checkpoint ships a decode-only acoustic VAE (zero encoder keys at all), so zero-shot voice cloning is architecturally impossible against it; upstream's own precomputed per-speaker `.pt` voice caches (confirmed via pickle disassembly: nested `DynamicCache` objects, not a flat state dict) need a dedicated deserializer that doesn't exist yet. `Synthesize`/`SynthesizeStream` throw `NotSupportedException` rather than produce unconditioned audio. Not yet surfaced in the AudioLab UI. |
| **StyleTTS2** (LibriTTS, clone) | ✅ | **Clone e2e word-intelligible 2026-07-15** (in-process; Whisper recovered 5/7 content words from a reference-voice clone). ([details](#styletts2)) |
| **NeuTTS** | 🚧 | Loads; default voice path; clone gated (X-Codec2 encoder key map). |
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
| **Wake satellite transport** | ✅ | TCP + Wyoming-style framing; device-keyed sessions, ping/pong, seq-gap reset. Hosted by the API server behind `HartsyInference__WakeEnabled`. Protocol contract in `docs/Research/WAKE_SATELLITE_PROTOCOL.md`. |
| **Custom wake-word training** | ✅ | `hartsy wake-train "<phrase>"` synthesizes the phrase across Kokoro voices, augments (gain + noise), embeds through the frozen backbone via the real `WakeDetectionPipeline` (so training features cannot drift from inference features), and fits a ~213k-param head with hand-written Adam. Backward pass is gradient-checked against an independent double-precision forward; deleting one ReLU derivative moves it 3,400%. Verified end to end 2026-08-16: `"hey hartsy"` trained on 3 voices **fired at 0.6961 on a 4th voice it never heard** and stayed silent through 11 s of unrelated speech. ⚠️ **Two honest limits.** (1) The auto-suggested threshold overfits a small held-out set — it proposed 0.90, which would have *missed* that unseen-voice detection at 0.6961; treat it as an upper bound and lower it. (2) False accepts are reported per hour precisely because a small negative set looks fine as a percentage and is unusable in a room: with one negative recording the run reported ~1.8%/window ≈ 800/hour against openWakeWord's target of <0.5/hour. Point `--negative-audio` at hours of real room audio. |
| **Silero VAD v6** | ❌ | Not built, not blocked. Weights verified (309,633 params, MIT, `Models/audio/wake/vad/`) and graph structure extracted: reflect-pad → fixed-DFT Conv1d STFT (kernel 256 / stride 128) → **magnitude** (`sqrt(re²+im²)`, not power) → 4-conv encoder k3 pad1 strides 1/2/2/1 with ReLU → LSTMCell(128) → final conv + sigmoid. The parity reference is `silero_vad.onnx` under onnxruntime (both already on disk) driven as `input` = 576 samples (64 carried + 512 new), `state` = zeros `(2,1,128)`, `sr` = 16000, threading `stateN` back — no torch install needed. Wake detection runs without it; it is needed for utterance endpointing and to stop scoring idle audio. |

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
| **Mimi** (codec) | 🔬 | SeaNet composed-weight load fixed (DSM checkpoint); DSM 32-cb decode reconcile in progress. Shared with CSM. |
| **MusicGen / AudioGen** | ✅ | T5-base corr 1.0 + decoder logits corr 0.999999 + EnCodec-32k decode corr 1.0; e2e on CUDA writes music-like audio. 5 bugs fixed (T5/EnCodec). |
| **YuE** (music, Stage-1) | ✅ | Stage-1 7B LM corr 1.0 (argmax 8/8) + XCodec (SoundStream) decode corr 1.0 → generates 16 kHz vocal audio. ([details](#yue)) |
| **HeartMuLa** (oss-3B) | ✅ | LM corr 0.9996–0.9999 + HeartCodec rewritten: flow-match estimator corr 1.0 + ScalarModel corr 1.0 → generates 48 kHz audio (CPU + CUDA). ([details](#heartmula)) |
| **MiniMax Music 3** | ✅ | Prompt ids exact; condition encoder, DiT block 0 and the full 36-layer DiT match diffusers (meanAbs < 1e-3); vocoder maxAbs 1e-4 with a distinct stereo fold; window/crop geometry reproduces the reference's 529408-sample stitch. Generates real 44.1 kHz stereo on CUDA. AR parity corr 0.9999989 on CUDA, flow parity corr 0.999996, and end-to-end output confirmed by ear as real music with intelligible sung lyrics. ([details](#minimax-music-3)) |
| **RVC** (voice conversion) | 🔬 | RMVPE front-end wired as the default F0 estimator (`VcCatalog.ConvertRvc`), corr 1.000000/maxAbs 9.5e-8 vs real `rmvpe.pt` ([details](../Checklists/PARITY_VERIFICATION.md)). YIN remains selectable via `f0_method`. RVC flow/decoder + index/protect/rms_mix_rate still pending. |
| **Demucs** (separation) | 🔧 | Built; parity pending. |
| **CSM** (Sesame) | ✅ | Fixed 2026-07-21 (unsloth/csm-1b key remap + bundled 32-cb Mimi + I32 codes dtype + real all-zero EOS + `[speaker]text` prompt template); Whisper word-perfect on two independent sentences. `hartsy speak -m csm`. |
| **Stable Audio Open Small** | ✅ | DiT/VAE/timing-conditioner parity cosine 1.0 each. ([details](#stable-audio-open-small)) |
| **DiffRhythm / AudioLDM 2 / ACE-Step XL** | ❌/🔧 | Music roadmap; see the [Remaining work](#remaining-work) section for the per-model build state and ROI order. |
| **PocketTTS** (continuous-latent) | ✅ | **Swarm-deployed + parity-verified 2026-07-16.** Production voiced path built on the verified cores: `PocketTtsStreamingTransformer.ForwardPrimed` (voice-KV prefix + RoPE offset), `PocketTtsFlowLm.GenerateVoiced` (LUT conditioner + out_eos stop + noise std=√temp), `PocketTtsVoice` (KV-state loader), rewritten `PocketTtsPipeline` (SentencePiece + emb_std/mean denorm). ([details](#pockettts)) |

## Notes

- Music models have their own definition of "production-ready" and a sequenced completion plan in
  the [Remaining work](#remaining-work) section; the universal missing piece there is
  the audio parity harness, now proven on ACE-Step's DiT.
- Build audio with `-m:1`; the Audio test suite crashes under xunit parallel, run it sequentially
  (`-- xUnit.ParallelizeTestCollections=false`) and reuse a model cache via `HARTSYINFERENCE_MODEL_CACHE`.

## Remaining work

Distilled from the retired PHASE_5_AUDIO / MUSIC_MODELS_COMPLETION_PLAN / MUSIC_PARAM_AUDIT /
AUDIO_TTS_BRINGUP_PLAN plans. Items now ✅ above (Bark, CSM, Orpheus, Kyutai TTS, StyleTTS2 clone, Demucs,
Spark-TTS, Zonos, PocketTTS, CosyVoice 2, Chatterbox clone, Stable Audio Open Small) are omitted.
See [ROADMAP.md](ROADMAP.md) for cross-cutting infra (multi-GPU, kernel perf, quant, serving).

### Build / validation status
- [ ] Overall ~70% built / ~25% validated.
- [ ] Build the PTX/Vulkan audio kernels (currently source-only, awaiting nvcc/glslc).
- [ ] Codec STOI validation (pending checkpoints).

### Not-started ASR / TTS models
- [ ] NVIDIA NeMo family: Parakeet CTC / RNN-T / TDT, Canary, FastConformer.
- [ ] SenseVoice, FireRedASR.
- [ ] XTTS-v2, ChatTTS, Higgs Audio v2, IndexTTS 1.5 / 2, CosyVoice 1.

### Checkpoint-gated scaffolds (backbone verified, need real weights)
- [ ] Kyutai STT (no depformer).
- [ ] NeuTTS clone path (X-Codec2 encoder key map).
- [ ] Fish-Speech firefly-gan-vq codec.
- [ ] Resemble Enhance real-weight load (denoiser/IRMAE/UnivNet module-composition mismatch).
- [ ] GPT-SoVITS enc_p (zero-shot clone encoder).
- [ ] StyleTTS2 random-mode (no-reference) diffusion sampler + whisper-drop quality follow-ups.

### Streaming
- [ ] Streaming pipelines deferred across the board.

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
- [ ] NeuTTS, Qwen3-TTS ICL mode, Kyutai clone.

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

