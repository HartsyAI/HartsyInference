# TTS / STT end-to-end benchmarks — 2026-07-12 (updated 2026-07-17)

> **Update 2026-07-17 — Zonos-v0.1 (transformer) voice-clone verified + given a decode perf pass: ~6× (203 → 32
> ms/frame).** Was previously ⛔ blocked (below); now Swarm-deployed (`Audio Models/Zonos/transformer`) and
> whisper-verbatim. The entire attention block ran host `Buffer.MemoryCopy`/loops on GPU tensors (host RoPE +
> `DiaHeads` reshapes + `RepeatKv` + host KV append), which broke the CUDA activation-residency cache and
> re-uploaded the O(n²) growing K/V every step. A GPU-resident decode path (mirroring the LLM `GenericTransformer`:
> `DiaAttention.SelfForwardFlash` + `FixedKvCache` in-place append + GQA-native `FlashAttention`, gated on new
> `IBackend.FlashDecodeSupported`) took the full stochastic path **203 → 32 ms/frame (RTF 17.5× → 2.9× slower than
> real time)**; greedy stays bit-parity. Now GPU-compute-bound on F32 Linear GEMVs (F16 is the next lever but risky
> — F32 is deliberate, TF32 degenerates over the AR loop). Full write-up: [`zonos_tts_2026-07-17.md`](zonos_tts_2026-07-17.md).

> **Update 2026-07-15 — Dia-1.6B verified word-correct through Swarm (10/10, all 3 turns).** Root cause of the
> long-standing "loops *Hello there* / non-verbal garbage" was the **wrong checkpoint**: the extension pulled the
> old `nari-labs/Dia-1.6B`; the current **`nari-labs/Dia-1.6B-0626`** (drop-in — identical keys/shapes) makes the
> engine produce the full dialogue and **EOS-stop at 11.44 s** (985 frames). The engine was correct all along
> (proven by a layer-diff vs the nari `dia` package). **Speed:** 11.44 s of audio in **319.7 s** on the 4090
> (`GenerateText2Image`) → **RTF ≈ 0.036×** — by far the slowest TTS here (dual-CFG stream × 18-layer decoder, F32,
> ~325 ms/frame). Correct-but-slow; a host-glue→GPU / F16 / CUDA-graph perf pass is the follow-up (like Bark/Chatterbox).


> **Update 2026-07-13 — TTS correctness + F5 perf pass.** Four TTS models now verified word-correct via a
> proven STT oracle (**whisper `medium.en`** — `base.en` was dropped after it was caught *hallucinating* the
> "quick brown fox" pangram onto broken audio) + RMS-envelope match vs a Python reference, on varied prompts
> (short / long / numbers / punctuation):
> - **Kokoro** — misaki-phoneme g2p + punctuation fix (was silently dropping words).
> - **Piper** — espeak language was defaulting to `en` (British) instead of the voice's `en-us` → mispronounced vowels; one-line fix.
> - **MeloTTS** — root cause was a `PytorchPickleLoader` **stride bug** (bert-base-uncased Linear weights, saved as
>   transposed `.t()` views, loaded transposed → garbage BERT features → gibberish); plus **number normalization**
>   was missing (`2026`/`$42.50`/`7`/`1st` dropped) — ported MeloTTS's `normalize_numbers`.
> - **F5-TTS** — verified voice-clone correct **and** given a perf pass: **174.6 s → 5–7 s (~34×)**. The per-forward
>   `F5ConvPosEmbed` grouped Conv1D was a host loop (~2.3 B MACs/forward, one CPU core pegged, GPU idle); routed to
>   the GPU `backend.Conv1d` (cuDNN + grouped kernel). Output is **bit-parity** with the host path (RMS-envelope
>   corr 1.0000). See the F5 row + outliers below.


First committed **end-to-end** speed benchmarks for the audio TTS/STT models (prior audio results were music-only:
`heartmula_music_e2e_2026-07-11.md`). Measured through the real SwarmUI + AudioLab generation path (the same
in-process C# HartsyInference engine a user hits), on **both** GPUs.

## Method
- **TTS driver: the canonical `/API/GenerateText2Image`** — the same universal generate path all of Swarm uses;
  the audio model is selected (`Audio Models/<Engine>/<variant>`), the text goes in the prompt, and the WAV lands
  in `/Output` exactly like an image gen. (An earlier revision of this file used the secondary `/API/ProcessTTS`;
  those numbers ran ~15–25% faster because they skip Swarm's param-processing + output-file pipeline — the
  `GenerateText2Image` numbers below are the honest end-user figures.)
- STT driver: `/API/ProcessSTT` for now (transcription output is text, not a file in `/Output`; wiring STT through
  the universal path is a separate item). STT input = a Piper TTS clip; transcript spot-checked for correctness.
- **RTF = generated-audio-seconds ÷ warm-gen-seconds** (higher = faster than real time). Warm = min of 3 runs after a cold warm-up (model resident).
- Engine: alpha.48 + Kokoro/pickle fixes (locally-deployed DLLs) + `HARTSY_AUDIO_CONV_CUDNN=1`. Audio device pinned per GPU via `HARTSY_AUDIO_CUDA_DEVICE` (1=3060, 0=4090).
- **No GPU-Python head-to-head**: the only torch on this box is CPU-only (`2.12.1+cpu`), so a fair GPU baseline
  isn't runnable here. Upstream reference RTF claims are noted where published (they are GPU-Python targets).

## Results (RTF, higher is better)

TTS via the canonical `GenerateText2Image` path; STT via `ProcessSTT`.

| Model | Type | 3060 RTF | 4090 RTF | Audio | Notes |
|---|---|---|---|---|---|
| **Piper** (VITS) | TTS | **8.6×** | 8.3× | 5.25 s | fastest; host-bound (4090 = 3060) |
| **Kokoro** (StyleTTS2) | TTS | 4.5× | **5.2×** | 6.45 s | **now works** (canonical-fallback fix); slightly compute-bound |
| **Moonshine** | STT | 6.5× | 6.5× | 4.33 s | **word-perfect** on real speech (JFK) + synthetic (07-13) |
| **Whisper** (base) | STT | 5.1× | 5.4× | 4.33 s | **word-perfect** on real speech (JFK); en-US default bug fixed (07-13) |
| **MeloTTS** (en-v3) | TTS | 1.4× | 1.4× | 4.45 s | **now correct** (stride + number-norm fixes); BERT+VITS, host/GPU-flat |
| **F5-TTS** (v1 base) | TTS clone | ~0.4× | — | 2.7 s | zero-shot voice clone; **174.6 s → 6.4 s (34×)** host-conv→GPU; RTF<1 (32 flow forwards), parity 1.0 |
| **StyleTTS2** (LibriTTS) | TTS clone | — | ~1.3× | 6.4 s | **new 2026-07-15**, zero-shot clone; StyleEncoder corr 1.0 + HiFiGAN corr 0.999999; Swarm e2e Whisper 12/13; ~5 s warm wall (host/launch-bound like the other small TTS) |

F5 RTF is <1 (below real-time) by design — flow-matching runs NFE×2 DiT forwards (32 at nfe=16); the **34× win**
was removing the host grouped-conv, not the DiT math (which was always GPU-resident, 87 ms/forward). Measured on
the 3060 (audio pinned there per the shared-GPU directive); the DiT loop is now GPU-bound so a 4090 would help.

## Key finding: small audio models are host/launch-bound, NOT compute-bound
The 4090 (≈3–4× the 3060's compute) gives **no meaningful speedup** here — Piper is even slightly *slower* on the
4090 (variance), Whisper/Moonshine/Melo are flat. These models spend their wall time in host orchestration + many
tiny kernel launches, not in GPU math. **Implication (answers "CUDA graphs where appropriate"): the optimization
lever for the small TTS/STT models is CUDA-graph capture / host-glue removal, which removes launch overhead —
buying a faster GPU does nothing.** This mirrors the LLM-decode graph win (dramatic on small models) and is the
opposite of the compute-bound video/music DiTs where graphs are a no-op.

## Outliers / blocked (found while sweeping — need attention)
- **Kokoro** — **FIXED 2026-07-12**. Was install-401 (`KokoroPipeline` only pulled the unpublished `Hartsy/kokoro-82m-safetensors` repack, no fallback). Now: prefer the repack, else download canonical `hexgrad/Kokoro-82M/kokoro-v1_0.pth` and do the flatten + inner-`module.`-strip in-engine (cached once). Installs + generates via the canonical path; 4.5×/5.2× above.
- **Whisper `en-US` default bug** — **FIXED 2026-07-13**. `/API/ProcessSTT` defaults language to `en-US`, which the
  engine rejected ("Unknown Whisper language code 'en-US'"). `WhisperTokenizer.LanguageToTokenId` now normalizes
  BCP-47/locale codes (lowercase + strip region subtag: `en-US`/`en_US`/`EN` → `en`), so the default works.
  Verified: omitted-language and explicit `en-US` both transcribe correctly.
- **Spark-TTS** marked ✅ (test parity) but **not runnable through Swarm** — install fails: "SparkTtsConfig token offsets + BiCodec decoder keys checkpoint-reconciliation-pending." The ✅ is parity-harness only.
- **F5-TTS** — **VERIFIED + FIXED 2026-07-13**. Zero-shot: pass a reference WAV (`referenceaudio`, base64) + its
  transcript (`referencetext`) alongside the target prompt. Was host-bound at 174.6 s (the `F5ConvPosEmbed` grouped
  Conv1D ran as a host loop, one CPU core at 100% / GPU idle); routed to `backend.Conv1d` → **6.4 s (34×)**, output
  bit-parity (RMS-envelope corr 1.0000). Voice clone transcribes word-perfect (medium.en).
- **MeloTTS** — **VERIFIED + FIXED 2026-07-13**. Two bugs: (1) `PytorchPickleLoader` ignored tensor stride →
  bert-base-uncased weights loaded transposed → gibberish; fixed with a stride-gather (`MakeRowMajor`, no-op for
  contiguous tensors, benefits all `.pth` models). (2) No number normalization → digits/currency/ordinals dropped;
  ported `normalize_numbers`. Now correct on numbers/years/ordinals/currency. Still the perf optimization target
  (1.4×, BERT+VITS host-flat).
- **Dia-1.6B** — **VERIFIED + FIXED 2026-07-15** (Swarm 10/10, EOS-stops 11.4s). Was the wrong checkpoint: repo `nari-labs/Dia-1.6B`→`nari-labs/Dia-1.6B-0626` (drop-in, ships `pytorch_model.bin`). RTF ≈ 0.036× (slowest TTS — dual-CFG 18-layer AR F32); perf pass pending.
- **Zonos-v0.1** — **VERIFIED + PERF PASS 2026-07-17** (was ⛔ blocked). Swarm-deployed (`Audio Models/Zonos/transformer`), voice-clone whisper-verbatim; GPU-resident decode → ~6× (203 → 32 ms/frame stochastic, RTF 2.9× slower than real time). See [`zonos_tts_2026-07-17.md`](zonos_tts_2026-07-17.md).
- Numerically-verified-but-no-runnable-e2e (do not benchmark yet): Kyutai TTS/STT, FishSpeech, VibeVoice, NeuTTS, StyleTTS2 (🔧/🔬); PocketTTS (⛔ blocked).

## Remaining work
- Larger verified TTS still to bench (need per-model setup/refs): Chatterbox, CosyVoice 2, Qwen3-TTS, GPT-SoVITS.
- CUDA-graph pass on Piper/Whisper/Moonshine/Melo (host-bound → high expected payoff). F5's DiT loop is now
  GPU-bound → a CUDA-graph capture (WIP `HARTSY_F5_GRAPH`, currently off — replay ILLEGAL_ADDRESS on the
  alloc/free-per-forward block scratch) is the next F5 lever.
- ~~Verify Moonshine STT transcription correctness~~ **DONE 2026-07-13** — Moonshine + Whisper both transcribe the
  real JFK human-speech clip and synthetic clips word-perfect (via ground-truth comparison + medium.en oracle).
- A real GPU-Python baseline needs GPU torch installed (currently CPU-only here).
