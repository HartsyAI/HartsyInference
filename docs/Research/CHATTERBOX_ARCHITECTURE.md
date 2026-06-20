# Chatterbox TTS (Resemble AI) — Architecture

> Build-ready spec for `ResembleAI/chatterbox`. Sources: `resemble-ai/chatterbox` repo (`tts.py`,
> `models/t3/`, `models/s3gen/`, `models/s3tokenizer/`, `models/voice_encoder/`) + HF model card.
> Fetched 2026-06-19. MIT, ~0.5B, 24 kHz out.

Pipeline: `text → T3 (Llama AR LM) → S3 speech tokens (25 Hz, FSQ 6561) → S3Gen (CosyVoice2 flow + HiFTNet)
→ 24 kHz`. Two speaker embeddings: a 256-d LSTM voice-encoder (feeds T3) + an 80/192-d CAMPPlus x-vector
(feeds S3Gen).

## T3 (text → speech-token LM)
Backbone "Llama_520M" (full MHA, NOT GQA): hidden 1024, 30 layers, 16 heads, 16 KV, head_dim 64,
intermediate 4096, SwiGLU, RMSNorm 1e-5, **rope_theta 500000 + llama3 rope_scaling (factor 8 / high 4 /
low 1 / orig_max 8192)**, no bias, untied. (HF `vocab_size=8` is a placeholder; real embeds/heads in T3.)
- Modules: `tfmr.*` (Llama trunk, headless), `text_emb` (704/2454 ×1024), `speech_emb` (8194×1024),
  `text_pos_emb`/`speech_pos_emb` (**learned** position tables), `text_head`, `speech_head` (1024→8194),
  `cond_enc` (`T3CondEnc`).
- Token ids: start_text 255, stop_text 0, **start_speech 6561, stop_speech 6562** (the stop). speech vocab
  8194 (S3 codebook 6561 + specials + headroom). max_text 2048, max_speech 4096.
- Input sequence: `[cond] ++ [text_emb+text_pos] ++ [speech_emb+speech_pos]`. Cond order: `[speaker_proj
  (Linear 256→1024)] ++ [CLAP(none)] ++ [prompt_speech (perceiver resampler)] ++ [emotion_adv (Linear
  1→1024)]`. Exaggeration = scalar through `emotion_adv_fc`, default 0.5.
- Gen: CFG `cond + cfg_weight*(cond−uncond)` (cfg_weight 0.5); temp 0.8, top_p 1.0, **min_p 0.05**,
  repetition_penalty 1.2; stop on 6562; max_new 1000.

## S3Gen (speech tokens → 24 kHz) — CosyVoice2 adaptation
- Flow: `UpsampleConformerEncoder` (out 512, 8 heads, 6 blocks, token_mel_ratio 2 → 25→50 Hz) + CFM.
- CFM: `CausalConditionalCFM`, estimator = UNet1D `ConditionalDecoder` (in 320, out 80, channels [256], 12
  mid blocks, 8 heads, head_dim 64). Solver Euler, **n_timesteps 10, cosine t-schedule, cfg_rate 0.7**.
- Vocoder: **HiFTNet** (NSF HnNSF 8 harmonics + ConvRNN F0 predictor + iSTFT), upsample_rates [8,5,3],
  24 kHz, mel n_fft 1920 / hop 480 / 80 bins.
- S3 tokenizer: `speech_tokenizer_v2_25hz` FSQ D=8/L=3 = **6561**, 25 Hz, 16 kHz in.

## Voice encoder (feeds T3)
GE2E/Resemblyzer LSTM: `nn.LSTM(40, 256, 3 layers)` → `Linear(256→256)` → L2-norm. mel 40-bin, 16 kHz,
n_fft 400 / hop 160. Embedding dim **256**. (S3Gen's speaker embed is the separate CAMPPlus.)

## Weight files
`t3_cfg.safetensors` (T3), `s3gen.safetensors` (prefixes `tokenizer.`, `speaker_encoder.` CAMPPlus,
`flow.`, `mel2wav.` HiFTGenerator), `ve.safetensors` (VoiceEncoder `lstm.`/`proj.`), `tokenizer.json`,
`conds.pt` (default voice). Multilingual: `t3_mtl*`, `s3gen_v3`, `mtl_tokenizer.json`.

## C# reuse map
**Reuse 1:1:** `ConditionalCfm` (+ add cosine schedule, cfg 0.7, 10 steps), `HiFTNetVocoder` (upsample
[8,5,3], 24 kHz — matches), `CosyVoiceFlow`, `S3Tokenizer` (6561), `CamPlusSpeakerEncoder` (192-d),
`NucleusSampler` (+ min_p, done). **Backbone:** `Qwen2Model` headless (needs llama3 rope scaling — the
CSM/Orpheus-shared deferral). **Net-new (built this pass):** T3 input assembly + learned positions +
`T3CondEnc` (speaker/emotion projections) + the LSTM `ChatterboxVoiceEncoder`. **Deferred:** prompt-speech
perceiver resampler, S3Gen pipeline wiring (reuses CosyVoice flow+vocoder), Perth watermark.
