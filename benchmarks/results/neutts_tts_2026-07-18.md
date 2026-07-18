# NeuTTS Air (neuphonic) — voice-clone verification, 2026-07-18

**Verdict: voice cloning already works end-to-end; no engine change needed. The "clone path gated on the
X-Codec2 encoder key map (CodecEnc.*)" concern was stale — the `NeuCodecEncoder` already maps the real
checkpoint layout.**

## What was checked
The `neuphonic/neucodec` checkpoint (811 tensors) uses the exact prefixes the engine's `NeuCodecEncoder`
expects — `acoustic_encoder.*` (BigVGAN, 146), `semantic_encoder.*` (Wav2Vec2-BERT conformer, 517, layers at
`semantic_encoder.encoder.layers.{i}.*` with `ffn1_layer_norm` / `self_attn.linear_q` / `conv_module.*` /
`self_attn.distance_embedding` sub-keys all matching), `semantic_adapter.*`, `fc_encoder.*`, `quantizer.*` —
**not** a `CodecEnc.*` prefix. So the extension's `codec.ContainsKey("acoustic_encoder.conv1.weight")` gate
passes and cloning is enabled; the `CodecEnc.*` string only appears in the never-hit fallback error message.

## Results (RTX 3060, device 1)
1. **Codec round-trip** (`<scratchpad>/nubench/`): reference wav → 16 kHz → `NeuCodecEncoder.Encode` → 317 FSQ
   codes → `NeuCodecDecoder.Decode` → 24 kHz. Whisper (medium.en):
   - reference: "Hello there, this is Style TTS2, cloning a voice with the corrected MEL front end."
   - reconstruction: "…the corrected **mal** front end." (only a whisper mishearing) — codes are correct.
   - Encode 9.9 s / 6.35 s ref (one-time per voice; semantic conformer + acoustic BigVGAN over full audio),
     decode 0.4 s.
2. **Full clone** (`<scratchpad>/nuclone/`): encode ref → espeak IPA (en-us) → `NeuTtsPromptBuilder` prompt →
   `Qwen2.5-0.5B` LM primed with ref codes → generate. Target "The quick brown fox jumps over the lazy dog."
   → whisper **word-perfect**, in the cloned voice. 196 codes → 3.92 s audio in 7.52 s, **RTF 1.92**.

## Components (all already verified elsewhere)
- `Qwen2Tokenizer.EncodeRawByteLevel` + `NeuTtsPromptBuilder`: HF-tokenizer ground-truth match
  (`NeuTtsTests.PromptPrefix_MatchesHfTokenizer_GroundTruth`).
- Espeak IPA phonemizer (en-us), NeuCodec FSQ geometry (65536 = 4^8, 50 Hz).

## Notes / possible follow-up (not blocking)
- Encoder is ~RTF 1.5 (one-shot per reference; not the hot path). If ever wanted, the 16-layer semantic
  conformer's attention is the lever.
- LM clone RTF 1.92 (Qwen2 decode, same shape as the other Qwen2-based TTS). Near real-time.
