# Audio TTS/STT Bring-up & Fix Plan (2026-07-13)

Fix every local TTS/STT that is broken (runs → gibberish/silent) or not-runnable (install throws). Verified
runnable+correct already (do not touch): Kokoro, Piper, MeloTTS, F5-TTS, Bark, Chatterbox*, VibeVoice, FishSpeech
(TTS) + Moonshine, Whisper (STT). *Chatterbox = default voice only; clone path is Tier-2.

Method for every model (same rigor as the MeloTTS/F5 fixes): install → `GenerateText2Image` → transcribe with
**whisper `medium.en`** (never `base.en`) + compare to a Python reference (A/B stage dumps) → root-cause → fix →
re-verify word-correct. Cloud-API providers (ElevenLabs/Azure/OpenAI/etc.) are out of scope (not engine models).

---

## Tier 1 — Runs but broken (closest to done, start here)

### 1a. Orpheus — SILENT output  ← IN PROGRESS (bug localized to the LM forward)
- **Symptom:** LM emits `EndOfSpeech` after only 4 SNAC frames → 0.34 s near-silent clip (`rms 0.0006`).
- **Arch:** Llama-3.2-3B (`unsloth/orpheus-3b-0.1-ft`) via `Qwen2Model` → SNAC-24k decode.
- **DIAGNOSIS 2026-07-13 (via `HARTSY_ORPHEUS_DEBUG=1` token dump + a Python `snac` decode A/B):** everything
  downstream of the LM is CORRECT and ruled out —
  - Tokenizer ✓ (`tara: The morning sun…` → sensible Llama-3 ids; asset embedded).
  - Prompt frame ✓ (`[SOH 128259, text, EOT 128009, EOH 128260]` — matches canonical Orpheus; the model correctly
    emits `128261 start_of_ai, 128257 start_of_speech` then audio tokens).
  - Redistribution ✓ (`OrpheusCodeFrames.Redistribute` matches canonical `[0]→L1, [1,4]→L2, [2,3,5,6]→L3`).
  - **SNAC decode ✓ — bit-identical to Python `snac`**: feeding the exact 28 generated codes to `snac_24khz` in
    Python gives `rms 0.00055`, same as the engine. So the codec is not the bug.
  - **ROOT CAUSE = the LM generates DEGENERATE audio codes** (redistributed codes show heavy repetition, e.g.
    `l3=[…429,429,429,429…]`, `l2=[…3418,3418…]`) then collapses to EOS at frame 4. Config is correct (bias off,
    θ=500000, Llama3 rope-scaling IS applied — the docstring "not yet applied" is stale).
- **NEXT STEP:** LM logit A/B — build a transformers `LlamaForCausalLM` reference (imports OK; CPU, a few
  single-token forwards is tolerable) on the downloaded `unsloth/orpheus-3b-0.1-ft` weights; compare step-0 logits
  (and a few steps) to the engine's `Qwen2Model` output for the same prompt. Localize to a weight-load key mismap
  (extended 156,940-row embed/tied-head, or a per-layer key) or a forward-op detail. Fix in
  `Qwen2Model.LoadWeights` / the shared transformer forward. Re-verify: coherent speech, medium.en word-correct.
- Debug instrumentation left in place (gated `HARTSY_ORPHEUS_DEBUG=1`): token-stream + stop-reason dump in
  `OrpheusPipeline`.

### 1b. Quick wins — default-voice paths of "clone-gated" models
NeuTTS, Qwen3-TTS, Chatterbox only gate the **voice-clone** path; the **default voice** may already work. Verify
each default-voice gen (no reference) with the oracle — likely fast passes that expand the verified set before the
harder clone work.

---

## Tier 2 — Clone / synth path gated (medium; each a focused feature)
- **Chatterbox clone** — needs a PCM→40-bin-mel front-end for the voice encoder.
- **NeuTTS clone** — needs the X-Codec2 encoder (`CodecEnc.*`) key mapping.
- **Qwen3-TTS clone** — ICL/ECAPA speaker path weight-validation.
- **Kyutai TTS synth** — per-frame text-stream state machine (SentencePiece 8k + PAD/EPAD/WORD) + delayed-coordinate
  / speaker-conditioning path. (Larger than the others in this tier.)

---

## Tier 3 — Full engine bring-up (large; each a real port + weight recipe)
- **StyleTTS2** — add `StyleTts2Pipeline.LoadFromCheckpoint` (per-key load for PLBERT / text-encoder / prosody /
  decoder / StyleEncoder / StyleDenoiser from the LibriTTS checkpoint).
- **Spark-TTS** — reconcile `SparkTtsConfig` token offsets + BiCodec decoder keys to the real checkpoint (parity
  harness already ✅; runtime weight-valid load is the gap).
- **Zonos** — build the conditioning-prefix `[1,P,hidden]` (espeak phonemes + speaker emb + emotion/pitch/rate/lang).
- **PocketTTS** — reconcile placeholder config dims from the checkpoint + wire the SentencePiece tokenizer asset.
- **CosyVoice** — in-process-engine support is factory-blocked ("not yet supported"); needs the runtime wiring
  (parity harness ✅ via shared S3Gen).
- **CSM** — no runtime model descriptor at all; full wiring.

---

## Execution order
Tier 1a (Orpheus) → 1b (default-voice quick wins) → Tier 2 → Tier 3. Verify + doc-update after each model.
Perf follow-up (separate): Bark/Chatterbox/VibeVoice are correct but slow (host/AR-bound) → host-glue→GPU pass.
