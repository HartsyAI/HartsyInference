# Audio Model Parameter Audit

Generated 2026-06-28 by a 44 agent parallel audit. Each agent fetched the real reference config (HuggingFace `config.json` / `config.yaml`, GitHub source) for every variant and diffed each architecture/inference parameter against our C# config record in `src/HartsyInference.Audio/Models`.

Status legend: **ok** present and correct, **missing** reference param has no equivalent field in our config, **wrong** field exists but the value/default disagrees with the reference, **unverified** reference value could not be confirmed from a source, **extra** we expose a field with no clear reference basis.

Totals across 44 model families: **712 ok**, **224 missing**, **88 wrong**, **39 unverified**, 49 extra.

## Summary (sorted by missing + wrong)

| # | Model | ok | missing | wrong | unverified | extra | missing+wrong |
|---|---|---|---|---|---|---|---|
| 1 | PocketTTS (Kyutai pocket-tts) | 6 | 12 | 9 | 1 | 3 | **21** |
| 2 | BiCodec (Spark) | 5 | 18 | 2 | 1 | 1 | **20** |
| 3 | HtDemucs (Hybrid Transformer Demucs v4) | 13 | 18 | 2 | 1 | 1 | **20** |
| 4 | Mimi codec (kyutai/mimi) | 12 | 14 | 3 | 1 | 1 | **17** |
| 5 | FishSpeech (Fish-Speech 1.4/1.5: DualAR text2semantic LM + Firefly-GAN-VQ codec) | 22 | 11 | 5 | 1 | 1 | **16** |
| 6 | Zonos (Zyphra Zonos-v0.1) | 22 | 9 | 4 | 1 | 3 | **13** |
| 7 | Qwen3-TTS / ECAPA / vocoder | 30 | 4 | 8 | 4 | 1 | **12** |
| 8 | CSM (Sesame csm-1b) | 14 | 9 | 3 | 1 | 3 | **12** |
| 9 | Qwen3 LM (audio backbone) | 6 | 9 | 3 | 1 | 0 | **12** |
| 10 | RVC (Retrieval-based Voice Conversion) v1/v2 + RMVPE | 22 | 6 | 6 | 0 | 1 | **12** |
| 11 | StyleTTS2 | 12 | 8 | 2 | 2 | 3 | **10** |
| 12 | Resemble-Enhance | 16 | 9 | 1 | 1 | 0 | **10** |
| 13 | SNAC codec (hubertsiuzdak/snac) | 9 | 3 | 7 | 0 | 3 | **10** |
| 14 | VITS | 22 | 7 | 2 | 2 | 1 | **9** |
| 15 | Kyutai (Moshi/STT/TTS) | 22 | 6 | 3 | 2 | 2 | **9** |
| 16 | CosyVoice (CosyVoice2-0.5B) | 28 | 5 | 4 | 1 | 2 | **9** |
| 17 | Chatterbox (ResembleAI): T3 + S3Gen + VoiceEncoder | 22 | 9 | 0 | 0 | 0 | **9** |
| 18 | WavTokenizer | 9 | 4 | 5 | 0 | 1 | **9** |
| 19 | Spark-TTS-0.5B | 27 | 7 | 0 | 0 | 1 | **7** |
| 20 | NeuTTS (neuphonic/neutts-air) | 13 | 6 | 0 | 0 | 0 | **6** |
| 21 | MusicGen (Meta AudioCraft / HF Musicgen) | 11 | 4 | 1 | 3 | 1 | **5** |
| 22 | EnCodec codec (facebook/encodec) | 18 | 5 | 0 | 2 | 2 | **5** |
| 23 | F5-TTS | 14 | 5 | 0 | 1 | 0 | **5** |
| 24 | NeuCodec (neuphonic/neucodec) | 23 | 4 | 1 | 1 | 1 | **5** |
| 25 | MeloTTS | 18 | 2 | 3 | 1 | 1 | **5** |
| 26 | Orpheus (canopylabs/orpheus-3b-0.1-ft, Llama-3.2-3B + SNAC 24kHz) | 14 | 1 | 4 | 1 | 0 | **5** |
| 27 | HuBERT | 11 | 5 | 0 | 0 | 1 | **5** |
| 28 | Bark (suno/bark) | 22 | 4 | 1 | 0 | 0 | **5** |
| 29 | Whisper | 13 | 4 | 1 | 0 | 2 | **5** |
| 30 | Vocos | 8 | 4 | 0 | 0 | 1 | **4** |
| 31 | YuE (m-a-p/YuE, two-stage LLaMA-2 lyrics2song music model) | 16 | 0 | 3 | 3 | 0 | **3** |
| 32 | Bert (audio frontends) | 8 | 3 | 0 | 0 | 0 | **3** |
| 33 | Moonshine (UsefulSensors STT, encoder-decoder) | 19 | 3 | 0 | 0 | 0 | **3** |
| 34 | VibeVoice (microsoft/VibeVoice-1.5B, VibeVoice-Large/7B, VibeVoice-Realtime-0.5B) | 42 | 1 | 2 | 0 | 1 | **3** |
| 35 | Qwen2 LM (audio backbone) | 13 | 0 | 2 | 1 | 0 | **2** |
| 36 | HeartMuLa (HeartMuLa-oss-3B) | 23 | 2 | 0 | 0 | 0 | **2** |
| 37 | Kokoro (hexgrad/Kokoro-82M) | 18 | 1 | 0 | 3 | 0 | **1** |
| 38 | GPT LM (audio backbone): GptConfig | 5 | 1 | 0 | 1 | 0 | **1** |
| 39 | Dia-1.6B (nari-labs) | 30 | 1 | 0 | 0 | 3 | **1** |
| 40 | GPT-SoVITS (stage-1 Text2Semantic AR GPT) | 10 | 0 | 1 | 0 | 2 | **1** |
| 41 | XCodec (YuE) | 13 | 0 | 0 | 2 | 0 | **0** |
| 42 | DAC codec (Descript Audio Codec) | 13 | 0 | 0 | 0 | 5 | **0** |
| 43 | Oobleck VAE (Stability AI Stable Audio autoencoder, diffusers AutoencoderOobleck) | 9 | 0 | 0 | 0 | 1 | **0** |
| 44 | OpenVoice (tone-color converter ReferenceEncoder / speaker encoder) | 9 | 0 | 0 | 0 | 0 | **0** |

## Missing variants / checkpoints (no preset in our code)

- **PocketTTS (Kyutai pocket-tts)**
  - english (a0ac5076, 6-layer, mimi.inner_dim=32)
  - english_2026-01 (b6369a24, 6-layer, mimi.inner_dim=512, insert_bos_before_voice=false, pad_with_spaces_for_short_inputs=true)
  - english_2026-04 (a0ac5076, 6-layer, mimi.inner_dim=32)
  - french_24l (709d9f84, 24-layer, remove_semicolons, model_recommended_frames_after_eos=8)
  - german (ba9816ab, 6-layer, remove_semicolons)
  - german_24l (24-layer)
  - italian (6-layer)
  - italian_24l (24-layer)
  - portuguese (6-layer)
  - portuguese_24l (24-layer)
  - spanish (6-layer)
  - spanish_24l (afd6403b, 24-layer)
- **HtDemucs (Hybrid Transformer Demucs v4)**
  - htdemucs_ft (fine-tuned 4-model bag, same architecture as htdemucs; selected per-source)
  - htdemucs_6s (6 sources: drums, bass, other, vocals, guitar, piano)
  - hdemucs_mmi / htdemucs training default (bottom_channels=0, segment differences)
- **Mimi codec (kyutai/mimi)**
  - No preset matches the published kyutai/mimi checkpoint's config-authoritative num_quantizers=32 with num_residual_layers=1 in a single clean preset. Mimi24kHz uses 8 codebooks AND wrong residual-layer count [1,1]; Mimi24kHzDsm fixes residual layers but is documented as a DSM/STT-TTS variant rather than the base kyutai/mimi checkpoint. A base preset with TotalCodebooks=32 and ResidualDilations=[1] is absent.
- **FishSpeech (Fish-Speech 1.4/1.5: DualAR text2semantic LM + Firefly-GAN-VQ codec)**
  - fish-speech-1.4 (vocab_size=32000, max_seq_len=4096, dropout=0.1)
  - OpenAudio S1 / S1-mini (modded_dac_vq codec: n_codebooks=9, semantic_codebook_size=4096, DAC-style generator decoder dim=1536 rates=[8,8,4,2], separate quantizer transformer 8L): different codec arch, likely out of scope but no preset exists
- **Zonos (Zyphra Zonos-v0.1)**
  - Zonos-v0.1-hybrid (n_layer=46, Mamba2 ssm_cfg, attn_layer_idx every 4th layer [0,4,8,...,44], plus 4 extra conditioners: vqscore_8, ctc_loss, dnsmos_ovrl, speaker_noised). We have no hybrid preset and no Mamba/SSM backbone support at all.
- **Qwen3-TTS / ECAPA / vocoder**
  - Qwen3-TTS-12Hz-0.6B (hidden 1024 / intermediate 3072 talker variant; no preset, only the 1.7B preset exists)
  - Qwen3-TTS-12Hz-1.7B-CustomVoice (built-in speaker ids + max_new_tokens 8192; no dedicated preset, ids are placeholder guesses)
  - Qwen3-TTS-12Hz-1.7B-VoiceDesign (free-form instruct mode; no preset)
- **Qwen3 LM (audio backbone)**
  - Qwen3-TTS-Flash / larger TTS size (tts_model_size other than 0b6, e.g. the Flash checkpoint) has no preset
  - Qwen3-TTS-12Hz-0.6B-Base talker as such (our Talker1_7B is mislabeled and carries wrong shapes)
- **RVC (Retrieval-based Voice Conversion) v1/v2 + RMVPE**
  - RVC v2 32k (SynthesizerTrnMs768NSFsid, upsample [10,8,2,2] / kernels [20,16,4,4], hop 320, n_mel 80, filter/win 1024)
  - RVC v1 40k (SynthesizerTrnMs256NSFsid, ContentDim 256, upsample [10,10,2,2], n_mel 125)
  - RVC v1 48k (SynthesizerTrnMs256NSFsid, ContentDim 256, 5-stage upsample [10,6,2,2,2] / kernels [16,16,4,4,4], hop 480, n_mel 128)
  - RVC v1 32k (SynthesizerTrnMs256NSFsid, ContentDim 256, 5-stage upsample [10,4,2,2,2] / kernels [16,16,4,4,4], hop 320, n_mel 80, filter/win 1024)
- **VITS**
  - jaywalnut310 LJS base preset (single-speaker, resblock 1, [8,8,2,2], 512ch, 22.05kHz, hop 256, vocab ~178): we have no preset whose architecture matches the canonical VITS reference; both our presets are Piper-shaped
  - VCTK multi-speaker preset (gin_channels=256, n_speakers=109, resblock 1, [8,8,2,2], 512ch): no multi-speaker preset; GinChannels defaults to 0 and there is no factory that sets it
  - MMS-TTS preset (16 kHz sampling_rate, vocab_size=38, resblock 1, [8,8,2,2], 512ch, SDP): no preset for the HF MMS arch (16kHz)
  - Piper low/x_low presets are not distinguished from medium in code (acceptable since arch is identical), but no high-vs-medium SampleRate variance exists
- **Kyutai (Moshi/STT/TTS)**
  - moshiko (moshi v0.1, dim 4096 / 32L / 32 heads / dep_q 8 / n_q 16 / text_card 32000 / context 3000 / max_period 10000): no preset
  - moshika (female-voice sibling of moshiko, same arch): no preset
  - tts-1.6b-en_fr depformer_num_layers in HF JSON is 4 which we match, but no preset exists for the older/streaming-server Moshi LM variants that share KyutaiTtsConfig shape
- **CosyVoice (CosyVoice2-0.5B)**
  - CosyVoice-300M (v1, FunAudioLLM/CosyVoice-300M): different arch with 4096 speech tokens, LLM = a custom TransformerLM (not Qwen2), 50 Hz tokens, mask-based flow; no preset
  - CosyVoice-300M-SFT (v1)
  - CosyVoice-300M-Instruct (v1)
  - CosyVoice-300M-25Hz (v1 at 25 Hz)
- **Chatterbox (ResembleAI): T3 + S3Gen + VoiceEncoder**
  - multilingual: T3Config.multilingual() with text_tokens_dict_size=2454 (we hardcode TextVocab=704 with only a comment mentioning 2454; no factory/preset exists)
  - turbo: ChatterboxTurboTTS uses llama_config_name=GPT2_medium (24 layers, max_position_embeddings 8196, no rope), text_tokens_dict_size=50276, speech_tokens_dict_size=6563, use_perceiver_resampler=False, emotion_adv=False; no preset exists
- **WavTokenizer**
  - frame40 (40 tokens/s): downsamples [6,5,5,4], n_fft=2400, hop_length=600, dim 768/intermediate 2304/12 layers (config wavtokenizer_smalldata_frame40_...yaml)
  - large/medium speech v2 checkpoints (novateur/WavTokenizer-large-speech-75token, WavTokenizer-medium-speech-75token, *_v2.ckpt) share frame75 architecture but are distinct presets users may want to select
- **NeuTTS (neuphonic/neutts-air)**
  - neuphonic/neutts-nano (Llama backbone, hidden_size 576, 24 layers, vocab 194256, different special-token ids: speech_0=128262, gen_end=128261; this is the DEFAULT backbone in the reference code)
  - neuphonic/neutts-nano-german (Llama, vocab 194256)
  - neuphonic/neutts-nano-french (Llama)
  - neuphonic/neutts-nano-spanish (Llama)
  - neuphonic/neutts-air-q4-gguf and neutts-air-q8-gguf (quantized GGUF backbones)
  - neuphonic/neutts-nano-q4-gguf / q8-gguf (and per-language gguf variants)
  - neuphonic/distill-neucodec (alternate codec)
  - neuphonic/neucodec-onnx-decoder / -int8 (alternate codec decoders)
- **MusicGen (Meta AudioCraft / HF Musicgen)**
  - musicgen-melody (chroma/melody conditioning: num_chroma=12, chroma_length=235; same decoder dims as medium but needs melody conditioner)
  - musicgen-stereo-small/medium/large (audio_channels=2, num_codebooks=8, delay pattern [0,0,1,1,2,2,3,3])
  - audiogen-medium has hidden=1536/layers=48/heads=24 which matches our AudioGen preset, but our AudioGen preset is unverified against a real config.json (audiogen-medium/raw 404'd)
- **EnCodec codec (facebook/encodec)**
  - encodec_48khz (stereo, time_group_norm, normalize=true, chunk_length_s=1.0, overlap=0.01, target_bandwidths=[3,6,12,24])
- **F5-TTS**
  - F5TTS_Base (v0 DiT: text_mask_padding=False, pe_attn_head=1)
  - E2TTS_Base (UNetT backbone: depth=24, ff_mult=4, no ConvNeXt text encoder, no text_dim/conv_layers, pe_attn_head=1)
- **NeuCodec (neuphonic/neucodec)**
  - neuphonic/distill-neucodec (DistillNeuCodec: SQCodec acoustic encoder + DistillHubert semantic encoder; fc_prior Linear(768+768 -> 2048), extra fc_sq_prior Linear(512 -> 768), SemanticEncoder(768,768,1024); identical Vocos decoder). No preset exists for this checkpoint's encoder.
  - neuphonic/neucodec-onnx-decoder (decoder-only ONNX export; same decoder architecture, no separate preset needed but worth noting)
- **MeloTTS**
  - Chinese (num_languages=4, num_tones=11, symbols~122)
  - Japanese (num_languages=10, num_tones=16)
  - Spanish (num_languages=10, num_tones=16)
  - French (num_languages=10, num_tones=16)
  - Korean (num_languages=10, num_tones=16)
  - English (EN base multi-speaker, n_speakers>1: EN-US/EN-BR/EN-INDIA/EN-AU/EN-Default)
- **Orpheus (canopylabs/orpheus-3b-0.1-ft, Llama-3.2-3B + SNAC 24kHz)**
  - orpheus-3b-0.1-pretrained (base/un-finetuned 3B; eos differs, 128001/128009)
  - multilingual / smaller research_release checkpoints (use the <custom_token_3/4/5> 'smaller' prompt format, different framing token layout)
- **HuBERT**
  - facebook/hubert-large-ll60k (hidden 1024, 24 layers, 16 heads, intermediate 4096, conv_bias true, feat_extract_norm=layer, do_stable_layer_norm=true)
  - facebook/hubert-base-ls960 (same dims as our base, but is the canonical English base checkpoint)
  - ContentVec / lengyue233/content-vec-best (HubertModelWithFinalProj: base dims + final_proj 768->256, used by RVC/so-vits-svc)
- **Whisper**
  - base.en / small.en / medium.en / tiny.en (English-only stock checkpoints; identical arch shape to their multilingual counterparts but distinct checkpoints; no dedicated preset, though shapes are covered by Base/Small/Medium/Tiny)
  - distil-large-v3.5 (newer distil checkpoint, 2-layer decoder, 128 mel; same shape as DistilLargeV3 so functionally covered)
- **Vocos**
  - charactr/vocos-encodec-24khz (EncodecFeatures: encodec_24khz, bandwidths [1.5,3.0,6.0,12.0]; input_channels=128, dim=384, intermediate_dim=1152, num_layers=8, adanorm_num_embeddings=4; head n_fft=1280, hop_length=320, padding=same)
- **YuE (m-a-p/YuE, two-stage LLaMA-2 lyrics2song music model)**
  - m-a-p/YuE-s1-7B-anneal-en-icl (English in-context-learning s1)
  - m-a-p/YuE-s1-7B-anneal-zh-cot (Chinese CoT s1)
  - m-a-p/YuE-s1-7B-anneal-zh-icl (Chinese ICL s1)
  - m-a-p/YuE-s1-7B-anneal-jp-kr-cot (Japanese/Korean CoT s1)
  - m-a-p/YuE-s1-7B-anneal-jp-kr-icl (Japanese/Korean ICL s1)
  - m-a-p/YuE-s1-0.5B (experimental 0.5B s1)
- **Bert (audio frontends)**
  - cl-tohoku/bert-base-japanese-v3 (MeloTTS/Bert-VITS2 Japanese frontend): hidden 768, 12 layers, 12 heads, intermediate 3072, vocab 32768
  - hfl/chinese-roberta-wwm-ext (base, non-large): hidden 768, 12 layers, 12 heads, intermediate 3072, vocab 21128
  - Bert-VITS2 v2.x Chinese DeBERTa-v2 frontend (relative-position attention, not representable by this standard-BERT config record at all)
- **Qwen2 LM (audio backbone)**
  - No dedicated preset for plain Qwen2.5-1.5B / 0.5B standalone (their max_position_embeddings is 131072 for 1.5B and 32768 for 0.5B, different from the VibeVoice decoder_config values our presets encode). Only relevant if these are ever used as a standalone LM rather than the VibeVoice backbone.
- **HeartMuLa (HeartMuLa-oss-3B)**
  - No real checkpoint gap. heartlib FLAVORS also defines llama-7B (32 layers / 4096 dim / 32 q / 8 kv / 14336 FFN) and llama-400M (4 layers / 8 q / 4 kv) but no public checkpoint ships them; the only released checkpoints (HeartMuLa-oss-3B and HeartMuLa-oss-3B-happy-new-year) both use llama-3B + llama-300M which our Oss3B preset already matches exactly.
- **GPT-SoVITS (stage-1 Text2Semantic AR GPT)**
  - V1 (s1longer.yaml: phoneme_vocab_size=512, top_k=5)
  - V2/V2Pro/V3/V4 (s1longer-v2.yaml: phoneme_vocab_size=732, top_k=15) - note our single V2 preset is mislabeled and uses the V1 vocab value

## Top critical findings

### PocketTTS (Kyutai pocket-tts)
- DModel default 0 should be 1024, NumHeads 0 should be 16, FfnDim 0 should be 4096 (hidden_scale=4 * d_model). All real public dims ARE published in the GitHub YAMLs; the 'NOT public' assumption in the doc comment is incorrect.
- LatentDim should be 32 (mimi.quantizer.dimension), not 0; FlowHeadDim should be 512 (flow.dim) and the flow head is a fixed-depth (flow.depth=6) SimpleMLPAdaLN, not a single MLP width.
- Norm is LayerNorm eps 1e-5 throughout the FlowLM transformer (and out_norm), NOT RMSNorm eps 1e-6 as our ToTransformerConfig emits via Qwen2; FFN is a plain GELU MLP (linear1->gelu->linear2, no bias), NOT SwiGLU. Reusing Qwen2Config (RMSNorm+SwiGLU) is an architecture mismatch.
- LsdDecodeSteps default should be 1 (DEFAULT_LSD_DECODE_STEPS=1), not 4.
- No latent CFG exists in the reference inference path; LatentCfgScale is an extra field with no reference basis (sampling uses temp/noise_clamp/eos_threshold only).
- Missing 24-layer ('_24l') variants entirely: french/spanish/german/italian/portuguese _24l all use num_layers=24, so a single 6-layer assumption is wrong for half the languages.
- VocabSize/n_bins is 4000 and lookup_table.dim is 1024 (text embed projected to d_model); our VocabSize default 0 plus TieWordEmbeddings=true is wrong (text is a LUTConditioner with a separate output projection, embeddings are not tied to a logits head; the only head is out_eos: Linear(dim,1)).

### BiCodec (Spark)
- GlobalFsqLevels default is WRONG: our [8,8,8,5,5] (5 dims, 12800 vocab) vs reference fsq_levels [4,4,4,4,4,4] (6 dims, 4096 vocab). Global codes will not match Spark weights.
- Global (speaker) encoder is WRONG-ARCH: reference takes a 128-bin MEL spectrogram into ECAPA_TDNN_GLOB_c512 + PerceiverResampler + ResidualFSQ (input_dim=128, latent_dim=128, out_dim=1024, fsq_num_quantizers=1). Our code runs cross-attention over 1024-dim w2v-BERT features; the keys we load (speaker_encoder.queries, attn.q_proj, fsq_proj) do not exist in the real checkpoint.
- Semantic encoder is WRONG-ARCH: reference encoder is a 12-layer Vocos backbone (vocos_dim=384, intermediate=2048), our code uses a single Linear projection, so semantic latents will not match real weights.
- Entire decode path is absent from config: mel_params, decoder (WaveGenerator), prenet, postnet have no config fields. Encode-only is by design but no records exist even for round-trip.
- GlobalQueryHeads (8) has no basis in config.yaml (PerceiverResampler head count is internal); it is an artifact of the wrong cross-attention path.

### HtDemucs (Hybrid Transformer Demucs v4)
- Our FreqEmbScale (0.2) is correctly the reference freq_emb (0.2), but we are MISSING emb_scale (=10) and emb_smooth (=true) which scale/smooth the frequency positional embedding and affect the spec-branch tensor that is added at encoder layer 0.
- No preset for htdemucs_6s: it changes Sources to 6 stems (drums, bass, other, vocals, guitar, piano), which changes the final decoder output channel count (NumSources). Our only preset hardcodes 4 stems.
- Missing several architecture-affecting encoder/decoder params that our .cs likely hardcodes silently: time_stride (2), context (1), context_enc (0), norm_starts (4), norm_groups (4), dconv_depth (2), dconv_comp (8), dconv_init (1e-3), multi_freqs_depth (3), rewrite (true). If any are hardcoded wrong, weights will not load.
- Missing cac (complex-as-channels = true). We hardcode SpecInChannels = 2*AudioChannels = 4 which matches cac=true, but there is no field to express cac=false, and cac also affects the decoder output (2 complex channels per source).
- Missing transformer detail params: t_max_positions (10000), t_max_period (10000), t_emb (sin), t_norm_in/t_norm_first/t_norm_out (all true), t_layer_scale (true), t_gelu (true). These affect attention/positional-embedding tensors and norm layers, not just training.
- bottom_channels: our 512 matches the RELEASED htdemucs/htdemucs_ft/6s checkpoints, but the training default in config.yaml/htdemucs.py is 0. Document that 512 is the release value (correct for loading published weights).

### Mimi codec (kyutai/mimi)
- num_residual_layers reference default is 1, but Mimi24kHz default ResidualDilations=[1,1] implies 2 residual blocks per stage, requesting weight keys absent from the real 1-block checkpoint (same bug the DSM preset doc warns about). Base preset should be ResidualDilations=[1].
- num_quantizers (total codebooks) reference default is 32 and config.json does not override it, so the published checkpoint is 32 total (1 semantic + 31 acoustic). Mimi24kHz hardcodes 8 total; there is no clean preset reflecting the 32-codebook checkpoint as shipped.
- frame_rate reference is 12.5 Hz but our computed FrameRate returns 24000/960 = 25 Hz, contradicting both the reference and our own XML comment (the compress=2 internal stride is not modeled).
- dilation_growth_rate=2 has no config field (harmless at 1 residual layer but unmodeled).
- Many architecturally relevant params are missing as fields and not clearly hardcoded: compress=2, sliding_window=250, kernel_size=7, last_kernel_size=3, residual_kernel_size=3, use_conv_shortcut=false, upsample_groups=512, layer_scale_initial_scale=0.01, max_position_embeddings=8000, head_dim=64, num_key_value_heads=8, trim_right_ratio=1.0, pad_mode=constant, hidden_act=gelu.

### FishSpeech (Fish-Speech 1.4/1.5: DualAR text2semantic LM + Firefly-GAN-VQ codec)
- FireflyConfig is missing n_groups=8 (grouped FSQ). The shipped codec is GFSQ with n_groups=8, n_codebooks=1, levels=[8,5,5,5]; the 8 'codebooks' come from the 8 groups, not from n_codebooks. Without n_groups the quantizer group split of the 512-dim latent cannot be reconstructed correctly.
- FireflyConfig omits the ConvNeXt encoder backbone and LogMel spec_transform entirely: input_channels/n_mels=160, depths=[3,3,9,3], dims=[128,256,384,512], kernel_size=7, n_fft=2048, hop_length=512, win_length=2048. Acceptable if decode-only, required for encode (audio->tokens).
- Sampling defaults disagree with fish-speech-1.5 CLI: ours Temperature=1.0/TopP=0.9/RepetitionPenalty=1.1/MaxNewTokens=1500 vs reference 0.7/0.7/1.2/0 (0 means auto). TopK=30 is not a reference CLI default (sampler is top_p based, no top_k option).
- No fish-speech-1.4 preset. 1.4 differs from 1.5 in vocab_size (32000 vs 102048) and max_seq_len (4096 vs 8192); the single V1_5 preset cannot load 1.4 weights (embedding/head vocab shape mismatch).
- FireflyConfig is missing HiFiGAN head input dim num_mels=512 (distinct from upsample_initial_channel), quantizer input_dim=512, hop_length=512, and pre/post_conv_kernel_size=13.

### Zonos (Zyphra Zonos-v0.1)
- No hybrid variant preset and no Mamba2/SSM support: Zonos-v0.1-hybrid uses n_layer=46 with ssm_cfg.layer=Mamba2 and attention only on layers [0,4,8,...,44]. Our ZonosConfig is transformer-only (all-attention) with no attn_layer_idx field, so the hybrid checkpoint cannot be expressed or loaded.
- The hybrid checkpoint adds 4 conditioners we model nowhere: vqscore_8 (Fourier input_dim=8, 0.5..0.8), ctc_loss (Fourier -1..1000), dnsmos_ovrl (Fourier 1..5), speaker_noised (Integer 0..1). These change the prefix length and projection input dim.
- pad_vocab_to_multiple_of=8 (config.py default, ZonosConfig.pad_vocab_to_multiple_of) is not represented; the real model pads per-codebook embedding/head vocab to a multiple of 8 (1026->1032, 1025->1032), so our hardcoded 1026/1025 mismatch real weight tensor shapes.
- Speaker encoder defaults are a deliberately compact stand-in that do not match real ResNet293: ref in_planes=64 with block counts [10,20,64,3]; ours are BaseWidth=32 and StageBlocks=[3,4,6,3], StageWidths=[32,64,128,256]. Real weights will not load against these defaults.
- NumLanguages=105 is wrong: the language_id IntegerConditioner range is -1..126 (128 raw ids, 127 valid).

### Qwen3-TTS / ECAPA / vocoder
- Vocoder AcousticCodebookDim is 256 in our config but the reference decoder codebook_dim is 512 (semantic stays 256). This changes the acoustic codebook embedding tensor shape and will mis-load real weights.
- Vocoder RmsNormEps is 1e-6f but the reference codec decoder rms_norm_eps is 1e-05. Talker stays 1e-6, but the codec transformer norm differs.
- Qwen3TtsConfig.MaxNewTokens is 2048 but the Base generation_config sets max_new_tokens to 8192 (research doc explicitly notes config wins). Generation truncates too early.
- EcapaConfig.EmbeddingDim is 192 (SpeechBrain default) but the real Qwen3-TTS speaker_encoder.fc outputs enc_dim 2048; InputChannels is 80 but the real stem conv input is 128. Our ECAPA preset will not match the shipped speaker encoder.
- CustomVoiceSpeakerIds are placeholder guesses [2075..2083]; the verified real ids are scattered (Ryan 3061, Serena 3066, Ono-Anna 2873, Sohee 2864) and live in a different codec range.
- No separate MTP/sub-talker sampling fields; the code predictor must run greedy (the C reference uses temp 0/top_k 1) while the talker samples temp 0.9. A single shared sampling block cannot express both.
- position_id_per_seconds (=13) is not represented as a config field; it feeds mRoPE position assignment.
- LanguageIdCount is 25 (implies ids 2050..2074) but the verified language id list tops out at portuguese 2071 (about 22 ids).

### CSM (Sesame csm-1b)
- num_codebooks: reference = 32 (Mimi set to 32 codebooks via mimi.set_num_codebooks(32), CsmConfig num_codebooks default 32). Our NumCodebooks = 8. This is a core architecture mismatch: the backbone predicts codebook 0 and the depth decoder predicts the remaining 31, not 7. Output tensor shapes (input_ids last dim, codebook heads, depth-decoder seq) are all wrong with 8.
- Depth decoder Decoder.MaxPositionEmbeddings = 64 in ours, but reference CsmDepthDecoderConfig.max_position_embeddings default = 33 (one position per codebook + the backbone hidden state). With 32 codebooks the depth decoder runs over 33 positions, so 64 over-allocates and the value disagrees with the reference.
- Llama-3.2 RoPE scaling (scale_factor = 32, the Llama3 rope type) is present in the reference FLAVORS for BOTH backbone and decoder but is NOT modeled in our config (plain RoPE theta=500k only). Already flagged in the XML doc comment as pending; this is a real numeric divergence above ~8k effective context and should be a config field (RopeScaling/RopeType).
- Codebook special-token ids do not match: reference codebook_eos_token_id = 0 and codebook_pad_token_id = 2050, but our AudioEosToken = 2048. There is no field for codebook pad. Verify the actual EOS semantics against the checkpoint before trusting AudioEosToken = 2048.

### Qwen3 LM (audio backbone)
- Talker1_7B preset has WRONG hidden_size (2048 vs real 1024) and intermediate_size (6144 vs real 3072). Those 2048/6144 values are the base Qwen3-1.7B TEXT LLM, not the Qwen3-TTS talker. The real talker is hidden 1024 / intermediate 3072 (text_hidden_size=2048 is a separate field for the text embedding projection). Confirmed by the official Qwen/Qwen3-TTS-12Hz-0.6B-Base/config.json.
- MaxPositionEmbeddings default 32768 is correct for the talker but WRONG for the CodePredictor: the real code_predictor_config uses max_position_embeddings=65536. Our CodePredictor preset relies on the default 32768.
- Missing vocab_size entirely: talker vocab_size=3072, code_predictor vocab_size=2048. Tensor shapes (embedding + lm_head) depend on this so it must be a config field, not hardcoded elsewhere unverified.
- Missing num_code_groups=16 (number of codebooks the talker/predictor model), which drives the multi-codebook head shapes.
- Missing text_vocab_size (151936) and text_hidden_size (2048) used by the talker text input embedding path.
- Missing special/codec token ids (codec_bos_id=2149, codec_eos_token_id=2150, tts_bos/eos/pad, codec_language_id map) needed for correct generation.

### RVC (Retrieval-based Voice Conversion) v1/v2 + RMVPE
- RMVPE Mel front-end is WRONG: real rmvpe.py uses MelSpectrogram(is_half, 128, 16000, 1024, 160, None, 30, 8000), so NFft and WinLength must be 1024, not 2048. Our 2048 changes the spectrogram and breaks parity with rmvpe.pt.
- RMVPE VoicingThreshold is WRONG: real inference (pipeline.py and rmvpe __main__) calls decode with thred=0.03, not 0.3. Our 0.3 (10x too high) will mark nearly all frames unvoiced.
- RMVPE UNet is structurally undersized vs rmvpe.pt: real E2E(4,1,(2,2)) has en_de_layers=5 encoder/decoder stages with channels 16->32->64->128->256 (out 512), n_blocks=4 residual blocks each, inter_layers=4. Our EncoderChannels [32,64,128] (3 stages) + IntermediateBlocks 2 cannot load the real checkpoint. Add EnDeLayers=5, NBlocks=4, InterLayers=4, KernelSize=(2,2), NGru=1 and fix the channel schedule (doc already flags it as a stand-in).
- RMVPE uses BatchNorm2d, not GroupNorm: our NormGroups field has no reference basis (the real ConvBlockRes/Encoder use nn.BatchNorm2d). Replace with BatchNorm or drop the field.
- Missing 4 reference variants: v2 32k, v1 40k, v1 48k, v1 32k. v1 needs ContentDim 256 (HuBERT 256-d) and v1 48k/32k use 5-stage upsamplers; v2 32k uses [10,8,2,2]. Only V2_40k and V2_48k presets exist.
- Minor RMVPE FirstFreqHz is slightly off: real cents_mapping base 1997.3794 yields 10*2^(1997.3794/1200) = 31.70 Hz for bin 0, not 32.70 Hz (C1). Use 31.70 to match rmvpe.pt cents grid.

### StyleTTS2
- Decoder type is variant-specific in the reference but our config forces the Kokoro iSTFTNet decoder for BOTH variants. The LibriTTS checkpoint actually uses a HiFi-GAN decoder (type 'hifigan', upsample_rates [10,5,3,2], upsample_kernel_sizes [20,10,6,4]). Only LJSpeech uses iSTFTNet (upsample_rates [10,6], kernels [20,12]). Reusing KokoroConfig.V1 (iSTFTNet) for LibriTTS is an architecture mismatch that will not load LibriTTS weights correctly.
- The diffusion style transformer (StyleTransformer1d) shape params are entirely missing from the config: num_layers=3, num_heads=8, head_features=64, multiplier=2. These set attention head count and FFN width of the style-diffusion estimator. StyleTransformerLayer.cs hardcodes the structure with no head-count or multiplier field, so a fork with different heads cannot be expressed.
- The inference style-blend knobs alpha (default 0.3 for cloning, 0.7 long-form) and beta (default 0.7) are absent. They interpolate the diffusion-sampled style with the reference-encoded style and directly affect the style vector fed to predictor and decoder, so they are inference-shaping params, not training-only.
- Our LjSpeech preset sets EmbeddingScale=1.5, but the canonical reference inference default (both notebooks) is embedding_scale=1. The 1.5/2.0 values are demo-specific expressiveness tweaks, not the documented default.

### Resemble-Enhance
- Denoiser is entirely absent from the config. The reference ships a denoiser UNet (input/output dim, hidden_dim=16, num_blocks=4, num_middle_blocks=2) used both standalone (--denoise_only) and as the lambd-blend conditioner inside the enhancer. Our config has no denoiser fields at all, yet the XML doc claims an optionally-denoised mel.
- UnivNet vocoder params are missing: univnet_nc=96 (vocoder base channels) and vocoder_extra_dim=32 (the extra conditioning dim from the CFM latent). Without these the vocoder cannot be sized to match the checkpoint.
- IRMAE num_irms=4 (the stack of 4 latent linear layers that defines the implicit-rank-minimizing bottleneck) is missing. AeResBlocks=4 we have is the ResBlock count, a separate quantity; num_irms is what makes this an IRMAE rather than a plain AE.
- CFM perturbation sigma=1e-4 is missing (hardcoded in cfm.py). It affects the OT-CFM target and must match for parity.
- Lambd default mismatch: the user-facing CLI default is lambd=1.0 (full denoise), the enhance() helper default is 0.5, and the module init is 0.0. We use 0.5. Confirm which entry point we emulate; the CLI default users actually get is 1.0.
- STFT/mel front-end params win_size=2048, stft_magnitude_min=1e-4, preemphasis=0.97 are missing. preemphasis in particular changes the mel features fed to the CFM and vocoder.

### SNAC codec (hubertsiuzdak/snac)
- The 32 kHz preset is wrong on almost every architecture field: reference snac_32khz uses encoder_rates [2,3,8,8], decoder_rates [8,8,3,2], encoder_dim 64, decoder_dim 1536, vq_strides [8,4,2,1], attn_window_size 32. Our Snac32kHz uses encoder_rates [2,4,8,8], decoder_rates [8,8,4,2], encoder_dim 48 (inherited default), decoder_dim 1024 (inherited default), vq_strides [4,2,1,1]. These produce completely different tensor shapes and will not load the real weights.
- The 44.1 kHz preset is wrong: reference snac_44khz uses encoder_rates [2,3,8,8], decoder_rates [8,8,3,2], vq_strides [8,4,2,1] (4 codebooks, NCodebooks 4), attn_window_size 32. Our Snac44kHz uses encoder_rates [3,3,7,7], decoder_rates [7,7,3,3], vq_strides [8,4,2] (3 codebooks). Note: [3,3,7,7]/[7,7,3,3] are the snac.py constructor library defaults, NOT the values in the actual snac_44khz checkpoint config.json. Our preset copied the source-file defaults instead of the published 44 kHz config.
- attn_window_size is completely missing from our config. It is null for 24 kHz but 32 for both 32 kHz and 44 kHz, where it inserts a LocalMHA windowed-attention block (dim_head 64, rotary pos emb) in the encoder bottleneck and decoder. This is architecture-affecting (adds attention weights and tensors) and must be modeled.
- noise (bool, default true) is missing. When true, DecoderBlock inserts a NoiseBlock after each upsample transposed-conv; this adds learned parameters and changes the decode graph. All three checkpoints set noise=true.
- depthwise (bool, default true) is missing. When true, convolutions use groups=channels (depthwise separable), changing weight tensor shapes for the conv layers. All three checkpoints set depthwise=true.

### VITS
- Posterior encoder WaveNet depth (HF posterior_encoder_num_wavenet_layers=16, jaywalnut310 n_layers_q=3) has NO field in VitsConfig. Pure TTS inference does not run the posterior encoder, but the encode path (voice conversion / spectrogram->latent, used by OpenVoice and GPT-SoVITS SoVITS half which the doc comment claims to back) needs it. If any of those use the posterior encoder it will be wrong or hardcoded.
- Stochastic duration predictor spline params are missing: duration_predictor_flow_bins=10 (num_bins) and duration_predictor_tail_bound=5.0 have no fields. These directly affect the rational-quadratic spline transform shapes and outputs in the SDP, which runs on every inference. If hardcoded they must be verified.
- The default VitsConfig record does NOT match the canonical VITS (jaywalnut310 LJS / HF MMS) architecture: defaults are resblock="2", upsample_rates=[8,8,4], upsample_initial_channel=256, which is the Piper medium/low decoder. The reference LJS/MMS/VCTK all use resblock="1", [8,8,2,2], 512ch. There is no preset that reproduces the reference decoder, only the two Piper-shaped configs.
- Depth-separable conv params for the SDP DDSConv are missing: depth_separable_channels (HF default 2, =flow hidden) and depth_separable_num_layers (HF default 3). The SDP convolutional flow stack uses these; absence means they are implicitly hardcoded.
- use_spectral_norm (false in all refs) has no field; if the HiFi-GAN discriminator/decoder norm is hardcoded this is fine for inference, but it is not represented.

### Kyutai (Moshi/STT/TTS)
- STT MaxPositionEmbeddings (reference `context`) is SWAPPED between presets: Stt1B sets 375 but HF stt-1b-en_fr config says context=750; Stt2_6B sets 750 but HF stt-2.6b-en says context=375. Both are wrong vs reference and will mis-size the RoPE/KV window.
- STT IntermediateSize is 11264 (hidden_scale 5.5) but HF hidden_scale is 4.125 which gives 8448; verify against checkpoint MLP shapes, this is likely wrong for both STT presets.
- Moshiko/Moshika have NO preset: their LM is dim 4096 / 32L / 32 heads / dep_q 8 / n_q 16 / text_card 32000 / context 3000 / max_period 10000 / depformer 6 layers / ffn 4224, totally different from the STT and TTS presets we ship.
- TTS sampling default Temperature is 0.8 but HF tts-1.6b lm_gen_config temp is 0.6 (text_temp 0.6); TopP 0.95 / TopK 0 have no reference basis.
- TTS conditioners under-modeled: HF defines cfg LUT (n_bins 7, dim 16, CFG 1.0..4.0), control LUT (dim 2048), speaker_wavs cross-attention, a fuser routing, plus second_stream_ahead=2 and demux_second_stream=true; we only carry SpeakerDim=512 and an unfounded MaxSpeakers=5.

### CosyVoice (CosyVoice2-0.5B)
- UnetChannels = [256, 256] is WRONG: the reference CFM estimator uses channels: [256] (a single level, with num_mid_blocks=12 carrying the depth). Two levels changes the UNet downsample/upsample tensor shapes and will not match real flow.pt weights.
- RAS sampling threshold is modeled wrong: we use RasMaxRepeat = 4, but the reference triggers on rep_num >= win_size * tau_r = 10 * 0.1 = 1. The effective max-repeat is 1, not 4, so our RAS almost never fires when it should.
- Sampling.Temperature = 0.8 is EXTRA / not used: CosyVoice2 ras_sampling has no temperature argument (nucleus + top_k only). RepetitionPenalty = 1.1 is also not part of the reference LM sampling path.
- Estimator in_channels (320) and CFM decoder in_channels (240) are not represented as config fields; only the flow InputSize/MelBins are exposed. The 320 = mel(80) + spk_proj(80) + mu(80) + cond(80) channel stack and 240 (decoder in) are load-bearing for shape matching and currently implicit.
- Flow encoder linear_units (2048 FFN) and pre_lookahead_len (3) / token_mel_ratio (2) are not config fields (linear_units likely hardcoded; token_mel_ratio is implied by MelFrameRateHz/TokenRateHz = 50/25 = 2).

### Chatterbox (ResembleAI): T3 + S3Gen + VoiceEncoder
- No multilingual preset: reference T3Config.multilingual() sets text_tokens_dict_size=2454; our config only hardcodes 704 with a comment, so the multilingual checkpoint cannot be loaded with correct text embedding/head sizing.
- No Turbo preset: ChatterboxTurboTTS swaps the T3 backbone to GPT2_medium (24 layers vs 30, max_pos 8196, no RoPE), text vocab 50276, speech vocab 6563, and disables perceiver_resampler + emotion_adv; none of this is expressible in our single config.
- Missing speech_cond_prompt_len (reference default 150) which controls the T3 speech-conditioning prompt length.
- Missing use_perceiver_resampler (default True) and emotion_adv (default True): architecture-affecting toggles that differ between standard and Turbo checkpoints.
- Missing input_pos_emb (default "learned") and the VoiceEncoder front-end params (num_mels 40, sample_rate 16000, n_fft 400, hop 160) are not parameterized.

### WavTokenizer
- CodebookDim=8 is WRONG: WavTokenizer's encoder is EnCodec's ResidualVectorQuantizer operating at the full latent dimension 512, NOT a DAC-style factorized 8-dim codebook. There is no codebook projection in WavTokenizer; this will produce wrong tensor shapes for the quantizer.
- HeadConvNeXtBlocks=8 is WRONG: the VocosBackbone num_layers is 12 (both frame75 and frame40 YAMLs set num_layers: 12).
- EncoderDim=64 is WRONG: EnCodec encodec_24khz uses n_filters (base channels) = 32, not 64.
- ResidualKernelSize=7 and ResidualDilations=[1,3,9] are WRONG: WavTokenizer uses EnCodec's SEANet residual unit with residual_kernel_size=3, n_residual_layers=1, dilation_base=2 (dilations [1, 2] per block), not the DAC [1,3,9] triple at kernel 7.
- Encoder LSTM (2 layers) is not represented anywhere: EnCodec SEANetEncoder has lstm=2 between conv stack and output projection, which affects weights/inference but has no config field.
- No frame40 (40 tokens/s) variant preset: needs downsamples [6,5,5,4], n_fft=2400, hop_length=600.
- adanorm_num_embeddings=4 (VocosBackbone AdaLayerNorm) has no config field.

### Spark-TTS-0.5B
- No wrong values found: all 7 token-ID offset fields (GlobalTokenBase 151665, SemanticTokenBase 155761, EosTokenId 151645, StartGlobalTokenId 165150, EndGlobalTokenId 165156, StartSemanticTokenId 165151, EndSemanticTokenId 165157) exactly match added_tokens.json ground truth, and the Qwen25_0_5B preset matches LLM/config.json exactly (hidden 896, 24 layers, 14 heads, 2 kv, intermediate 4864, max_pos 32768, tie=true) with VocabSize correctly overridden to 166000.
- All BiCodec decode params match BiCodec/config.yaml: codebook_size 8192, codebook_dim 8 -> SemanticDim 1024, fsq_levels [4,4,4,4,4,4], latent_dim 128, token_num 32, speaker out_dim 1024, vocos_dim 384, vocos_intermediate 2048, prenet 12 layers, decoder channels 1536, rates [8,5,4,2], kernel_sizes [16,11,8,4]. Sampling defaults temp 0.8 / top_k 50 / top_p 0.95 match cli/SparkTTS.py.
- Encode-side mel_params are absent from our decode config (n_fft 1024, win_length 640, hop_length 320, num_mels 128, mel_fmin 10). These do not affect the TTS generation path (decode does not run mel) but are needed for any audio-tokenization / voice-prompt encode path. Add a mel sub-config if the encoder is implemented.
- Controllable-TTS control tokens are not modeled anywhere in the config: <|task_tts|>=165137, <|task_controllable_tts|>=165143, <|start_content|>=165146, <|end_content|>=165152, plus the attribute token banks (age/gender/pitch/speed). These drive prompt construction for the controllable mode; the basic clone path works without them but controllable synthesis cannot be built until they are added.
- postnet (vocos_num_layers 6) in BiCodec/config.yaml is correctly NOT modeled: bicodec.py uses it only in forward() for training, never in detokenize/inference, so excluding it is correct.

### NeuTTS (neuphonic/neutts-air)
- Missing variants: our config only models neutts-air (Qwen2). The entire neutts-nano family (nano, nano-german, nano-french, nano-spanish) uses a DIFFERENT Llama backbone (hidden_size 576, vocab 194256) with a completely different special-token id block (speech_0=128262, SPEECH_GENERATION_END=128261, TEXT_PROMPT_START=128257). nano is even the reference code's DEFAULT backbone_repo, so none of our hardcoded Air ids apply to it.
- Missing config fields for two special tokens used by the reference torch chat-template assembly: <|TEXT_REPLACE|> (151665) and <|SPEECH_REPLACE|> (151668). _apply_chat_template splices the prompt at TEXT_REPLACE and replaces SPEECH_REPLACE with SPEECH_GENERATION_START. We expose TextPromptStart/End but not these two replace markers.
- Missing max_context = 2048 (hardcoded in reference NeuTTS.__init__, passed as generate(max_length=2048)). We have no max-length/context field, so our generation could run past the reference cap.
- All Air sampling defaults are CORRECT and verified against _infer_torch (temperature=1.0, top_k=50, min_new_tokens=50, do_sample=True, eos=SPEECH_GENERATION_END). The doc comment is right: generation_config.json (temp 0.7, top_p 0.8, rep 1.1, top_k 20) is NOT used by the torch path, so our TopP=0/no-rep-penalty matches.
- All hardcoded Air token ids verified bit-exact against the real tokenizer_config.json: SpeechTokenBase=151671, SpeechGenStart=151669, SpeechGenEnd=151670, TextPromptStart=151666, TextPromptEnd=151667, CodebookSize=65536 (speech_0=151671 .. speech_65535=217206).

### MusicGen (Meta AudioCraft / HF Musicgen)
- No stereo variant presets: stereo checkpoints use num_codebooks=8 and audio_channels=2 with delay pattern [0,0,1,1,2,2,3,3]. Our config has only mono presets (NumCodebooks=4) and no AudioChannels field, so stereo models cannot be configured correctly.
- No musicgen-melody preset: melody needs chroma conditioning (num_chroma=12, chroma_length=235) which has no field anywhere in our config or model.
- TopK default is 50 per the HF transformers/generation_config path, but our config hardcodes TopK=250 (the AudioCraft original-codebase default). This is OK for AudioCraft parity but WRONG if matching HF generation_config; document which reference you target. Same for TopP (HF=1.0 vs our 0.0).
- AudioGen preset (16 kHz) is unverified: facebook/audiogen-medium config.json returned 404, so its exact upsampling_ratios/sampling_rate path could not be confirmed from a config file (only from AudioCraft docs: 16 kHz, 4 codebooks, 50 Hz).

### EnCodec codec (facebook/encodec)
- No use_conv_shortcut field: HF default is True (used by 24kHz), but encodec_32khz sets use_conv_shortcut=false. This changes the SEANet residual block shortcut path (Conv1d shortcut vs identity) and will mismatch real 32kHz weights if hardcoded one way.
- No encodec_48khz preset: that variant differs in many architecture-affecting ways (audio_channels=2 stereo, norm_type=time_group_norm instead of weight_norm, normalize=true, chunk_length_s=1.0 with overlap=0.01, use_causal_conv=false, codebook_size=1024 not 2048). It cannot be represented by any current preset.
- normalize is not modeled: 24/32 kHz are false but 48 kHz is true. normalize=true rescales audio by its volume and stores a scale per chunk, affecting decode output. Missing.
- chunk_length_s / overlap not modeled: 48 kHz uses chunked processing (chunk_length_s=1.0, overlap=0.01); 24/32 kHz are null (whole-input). Affects framing/inference tensor layout for 48 kHz.
- 16 kHz preset is unverified: there is no standalone facebook/encodec_16khz HF EncodecModel repo (AudioGen's 16 kHz codec lives inside audiocraft). Our NFilters=64 and codebook=2048 assumptions are unconfirmed against any published config.json.

### F5-TTS
- No preset for E2TTS_Base: it uses a different backbone (UNetT, depth=24, ff_mult=4) with NO ConvNeXt text encoder (no text_dim/conv_layers) and pe_attn_head=1. The current single-config record cannot represent it.
- No preset for the older F5TTS_Base (v0): differs from v1 by text_mask_padding=False and pe_attn_head=1, which alter attention/masking behavior at inference.
- Missing text_mask_padding field. v1 base = True, v0/E2 = False. This controls whether padded text positions are masked in attention. Currently not represented and effectively hardcoded.
- Missing pe_attn_head field. v1 = null (RoPE on all heads), v0/E2 = 1 (RoPE applied to a single head). This changes the rotary-embedding application and is an architecture-affecting param with no field.
- Missing CFM/inference defaults (nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, target_rms=0.1, ode_method=euler). If these live only in the pipeline, fine, but they are not in the config record.

### NeuCodec (neuphonic/neucodec)
- CodebookSize: our config hardcodes 65536 (= FSQ 4^8, the true effective code count), but the reference CodecDecoderVocos is constructed with codebook_size=16384 and codebook_dim=16. Those are legacy/unused residual-VQ args that the ResidualFSQ (levels=[4]*8) overrides, so 65536 is functionally correct, but if any loader/test reads codebook_size to size a tensor it will disagree with the checkpoint's nominal 16384. Flag and document.
- Decoder default hop_length in the reference dataclass is 320 (n_fft would be 1280), but the NeuCodec class is instantiated as cls(24_000, 480) so the real decoder uses hop_length=480 and n_fft=1920. Our HopLength=480 / NFft=1920 / Win=1920 match the instantiated model, NOT the dataclass default. This is correct but a footgun: do not copy the 320 default.
- Missing the DistillNeuCodec variant entirely. neuphonic/distill-neucodec ships an SQCodec acoustic encoder + DistillHubert semantic encoder with a different fc_prior (Linear(768+768 -> 2048)) and an extra fc_sq_prior (Linear(512 -> 768)). Our NeuCodecEncoderConfig only models the full BigCodec encoder, so it cannot load the distill checkpoint.
- Missing semantic-branch config (Wav2Vec2-BERT-large) and the SemanticEncoder(1024,1024,1024) projection plus fc_post_a (Linear(2048 -> 1024)). The decoder NeuCodecConfig has QuantizerDim/BackboneDim that imply fc_post_a but there is no explicit field for the semantic encoder dims; encoder code documents the semantic branch as an intentional no-op.
- FcPostA dim (Linear 2048 -> 1024) is implied by QuantizerDim=2048 and BackboneDim=1024 but never named as its own field; the encoder's fc_prior is Linear(hidden_dim=1024 -> FSQ), and the project-in to FSQ dim 8 lives inside ResidualFSQ. Naming alignment is fine numerically but worth a comment.

### MeloTTS
- EnglishV3 preset sets NumLanguages=10 but the EN-v3 reference config.json has num_languages=8. The embedding table (language_emb) is sized wrong, which mismatches the checkpoint shape and will break weight load or index mapping.
- Only one variant preset (EnglishV3) exists. The other five published checkpoints (Chinese, Japanese, Spanish, French, Korean) plus the EN multi-speaker base are not represented. Chinese in particular differs sharply: num_languages=4, num_tones=11.
- n_speakers differs per checkpoint: EN-v3 has n_speakers=1 (matches), but ZH/JP/ES/FR/KR all use n_speakers=256 in the model (speaker embedding table sized 256) even though spk2id lists one logical speaker. Our NumSpeakers=1 and Core.NumSpeakers=1 would size emb_g wrong for those checkpoints.
- Core.NumVocab=256 does not match EN-v3 symbols length (~231 to 249 depending on commit). Hardcoding 256 oversizes the phoneme embedding vs the actual checkpoint and will mismatch on load.

### Orpheus (canopylabs/orpheus-3b-0.1-ft, Llama-3.2-3B + SNAC 24kHz)
- rope_scaling is MISSING entirely: reference is rope_type=llama3, factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192. Our Qwen2Config has no rope_scaling field and the doc comment admits the llama3 NTK-by-parts rescale is not applied. This changes the actual RoPE frequencies (inference numerics), so positions past 8192 and even short-context attention drift from reference.
- pad_token_id is WRONG: our PadTokenId=128263, reference config.json ft = 128004 (baseten/pretrained lineage uses 128001). 128263 has no basis in any reference config.
- eos_token_id mismatch vs config.json: our EosTokenId=128258 (end-of-speech). The HF config.json declares eos_token_id=128009 (ft mirror) or 128001 (pretrained). 128258 is the correct AR *stop* token used at inference (streaming example stop_token_ids=[128258]), so the value is functionally right but it is NOT the config.json eos; document this so a converter does not overwrite it from config.json.
- Sampling defaults disagree with the canonical engine: engine_class.py uses top_p=0.8 and repetition_penalty=1.3 (we have top_p=0.95, rep=1.1). Low severity (caller-tunable), but our defaults match neither the engine nor the streaming example exactly.

### HuBERT
- No preset for hubert-large (hidden 1024, 24 layers, 16 heads, FFN 4096, conv_bias=true, feat_extract_norm=layer, do_stable_layer_norm=true). The current record cannot represent it because the architecture-switching fields do not exist.
- Missing feat_extract_norm field: base uses group norm (single GroupNorm after conv0), large uses layer norm (LayerNorm after every conv). This changes the feature-extractor weight layout and is hardcoded implicitly to the group variant.
- Missing do_stable_layer_norm field: base is post-LN (false), large is pre-LN (true) with a final encoder LayerNorm. This changes transformer block structure; our doc comment hardcodes post-LayerNorm.
- Missing conv_bias field: false for base/ContentVec, true for large. Affects conv weight loading.
- Missing ContentVec final projection (final_proj 768->256, classifier_proj_size=256). ContentVec/RVC tap the projected output, not raw last_hidden_state, so no preset can reproduce it.

### Bark (suno/bark)
- CoarseInferToken is 12_051 in our config but the upstream bark/generation.py constant COARSE_INFER_TOKEN is 12_050. This is an off-by-one bug in a token-offset constant that directly drives coarse-stage inference (wrong infer marker fed to the coarse GPT).
- min_eos_p (0.2) for the semantic stage early-stop is MISSING. Without it, semantic generation does not terminate on the EOS probability threshold the reference uses, which changes output length and content.
- max_coarse_history (630) is MISSING. The coarse stage uses this to bound how much semantic history is carried per sliding window; absence changes the coarse conditioning context.
- n_codes_given (=1) is MISSING. The fine stage treats the first codebook as given (from coarse) and predicts the remaining 7; our config only has NumCodebooks=8 and NumCoarseCodebooks=2 with no n_codes_given field.

### Whisper
- pad_token_id for large-v3 is 50256 in the reference config (distinct from eos 50257), but our config exposes only EndOfTextTokenId = 50257 and treats it as both eos and pad; for large-v3 this means an incorrect pad id if pad is ever used. tiny/base/small/medium/large-v2/turbo all correctly use pad_token_id = 50257, so the issue is large-v3 specific.
- No field for scale_embedding (reference = false for all variants). It is safe only if the model code never scales embeddings; confirm the WhisperDecoder hardcodes no sqrt(d_model) scaling, otherwise it is a silent bug.
- No field for activation_function (reference = 'gelu' for all variants). OK only if the FFN/conv activation is hardcoded to GELU; should be confirmed in WhisperBlock/encoder code.
- Reference begin_suppress_tokens = [220, 50257] and large per-variant suppress_tokens / forced_decoder_ids are not represented anywhere in the config; if logit suppression and forced language/task prefix are needed for correct decoding they must be sourced elsewhere.
- All architectural shape parameters (vocab_size, num_mel_bins, layers, d_model, heads, ffn_dim, max positions) match the reference exactly across tiny, large-v2, large-v3, and turbo.

### Vocos
- No preset exists for the charactr/vocos-encodec-24khz variant, which differs in nearly every dimension (input_channels 128, dim 384, intermediate_dim 1152, n_fft 1280, hop_length 320) and uses an EncodecFeatures front-end plus AdaLayerNorm conditioning. The current record cannot represent it.
- Missing field adanorm_num_embeddings. The encodec variant sets it to 4, which switches the ConvNeXt blocks from plain LayerNorm to AdaLayerNorm (conditional on bandwidth id). This is an architecture-changing parameter with no field.
- Missing field layer_scale_init_value (reference default 1/num_layers when None). Affects the learned per-block residual scale gamma; if hardcoded incorrectly it changes inference outputs.
- Missing field padding (center for mel-24khz, same for encodec-24khz). This governs STFT/iSTFT centering and output length alignment and differs across variants.

### YuE (m-a-p/YuE, two-stage LLaMA-2 lyrics2song music model)
- Stage2 NumHiddenLayers is WRONG: our preset = 22, real m-a-p/YuE-s2-1B-general config.json = 32. This changes the layer count / tensor shapes and would mismatch real weights.
- Stage2 IntermediateSize is WRONG: ours = 5632, real = 5504 (FFN gate/up/down proj shapes differ).
- Stage2 VocabSize is WRONG: ours = 100000, real = 83840 (embed + lm_head shapes differ). Note s2 vocab (83840) differs from s1 vocab (83968).
- Audio-token base IDs (VocalTokenBase=45334, AccompTokenBase=46358, AudioEosToken=32002) are UNVERIFIED: YuE assigns soa/eoa/codec offsets dynamically from the SentencePiece tokenizer at runtime, so no static numeric constant exists in the reference source to confirm them.
- Six reference s1 variants have no preset (only the en-cot s1 + general s2 are wired): en-icl, zh-cot, zh-icl, jp-kr-cot, jp-kr-icl, and the 0.5B.

### Bert (audio frontends)
- No preset for cl-tohoku/bert-base-japanese-v3 (vocab 32768), the MeloTTS/Bert-VITS2 Japanese frontend BERT. Both existing presets use the wrong vocab size for Japanese, which would mismatch the embedding table shape.
- hidden_act is not a config field: exact GELU is hardcoded in the doc comment/model. All target checkpoints use hidden_act=gelu (exact erf GELU), so this is currently correct, but it is an implicit assumption with no field to override if a future variant uses gelu_new/gelu_fast.
- position_embedding_type is not a config field: absolute learned positions are assumed. bert-base-uncased and chinese-roberta-wwm-ext-large both use absolute, so OK for current presets, but a DeBERTa-v2 Chinese frontend (relative attention) cannot be represented by this record.
- pad_token_id (0 in all references) is not stored. Only matters if attention masking is derived from pad id rather than an explicit mask; verify the model receives an explicit attention mask.

### Moonshine (UsefulSensors STT, encoder-decoder)
- pad_head_dim_to_multiple_of=8 is present in both checkpoint configs and is architecture-affecting: for base head_dim=52 it pads the attention compute head_dim up to 56 (round up to multiple of 8). It is documented in our XML comment but there is no config field and HeadDim is computed as raw HiddenSize/NumHeads. If the attention path uses the padded head dim (as HF does), our model must pad too, so this needs a field or explicit handling.
- tie_word_embeddings=True has no field. Moonshine ties the decoder token embedding with the lm_head, so the output projection must reuse the input embedding weights. This is an inference-affecting parameter with no representation in our config.
- decoder_start_token_id=1 has no dedicated field. It happens to equal our BosTokenId (1) so generation is currently correct, but it is a distinct concept and should be exposed to avoid coupling assumptions.
- encoder/decoder layer counts and heads are correctly split in the reference (separate encoder_* and decoder_* keys) and our presets match the real tiny (6/6) and base (8/8) checkpoints. Note the HF dataclass default max_position_embeddings is 512, but BOTH real checkpoints set 194, which is what our config uses, so OK.

### VibeVoice (microsoft/VibeVoice-1.5B, VibeVoice-Large/7B, VibeVoice-Realtime-0.5B)
- Streaming05B preset has MaxPositionEmbeddings = 32768 but microsoft/VibeVoice-Realtime-0.5B config.json sets max_position_embeddings = 8192. This is a wrong value that changes the RoPE/context assumptions for the streaming variant.
- Streaming05B Decoder uses TieWordEmbeddings = true (Qwen25_0_5B preset), but the Realtime-0.5B config.json sets tie_word_embeddings = false. With tying assumed true, lm_head.weight would not be loaded as a separate tensor and decode would use the wrong output projection.
- corpus_normalize (= 0.0 in both acoustic and semantic tokenizer configs) has no field in VibeVoiceTokenizerConfig. It is 0.0 (disabled) on the public checkpoints so it is inert for now, but it is a real reference param with no equivalent field (missing).
- Naming drift: our preset is Streaming05B and doc comments say VibeVoice-Streaming-0.5B / org vibevoice/, but the real repo is microsoft/VibeVoice-Realtime-0.5B (model_type vibevoice_streaming is correct). No tensor impact, but the doc comments and any path-based loader keying on the repo name will 404.

### Qwen2 LM (audio backbone)
- Qwen25_0_5B preset MaxPositionEmbeddings = 32768 is WRONG for the VibeVoice-Realtime-0.5B backbone it claims to mirror: that decoder_config has max_position_embeddings = 8192. (32768 only matches the standalone Qwen2.5-0.5B, not the TTS backbone.)
- Qwen25_0_5B preset TieWordEmbeddings = true is WRONG for VibeVoice-Realtime-0.5B: its decoder_config has tie_word_embeddings = false (untied lm_head). The XML comment also asserts 0.5B is tied, which is incorrect for the realtime backbone.
- EosTokenId default = 151645 is the Qwen2.5-Instruct <|im_end|> value; the base models and the VibeVoice decoder_config use/imply 151643. This is fine for instruct/chat use but mismatched for base-model AR loops. Informational only since the AR loop owns sampling.

### HeartMuLa (HeartMuLa-oss-3B)
- text_eos_id (128001) is MISSING from HeartMulaConfig. Reference gen_config + pipeline (music_generation.py: text_eos_id: int = 128001) appends this token to tags and lyrics before encoding. Only AudioEosToken (8193) is modeled. Without text_eos_id the prompt assembly cannot terminate text segments correctly.
- Inference sampling defaults (topk=50, temperature=1.0, cfg_scale=1.5) are not surfaced anywhere in the config record. These are reference defaults from examples/run_music_generation.py and the pipeline. They are runtime knobs not tensor-shape params, so lower priority, but worth carrying as defaults for parity.
- All architecture params are CORRECT: backbone llama-3B (28 layers, 3072 dim, 24 q / 8 kv heads, 8192 FFN, max_seq 8192) and decoder llama-300M (3 layers, 3072 dim, 8 q / 4 kv heads, 8192 FFN, max_seq 2048), rope_base 500000, scale_factor 32, norm_eps 1e-5, all match torchtune llama3_2_3B / llama3_2_300M verbatim. audio_vocab_size 8197, audio_num_codebooks 8, text_vocab_size 128256, muq_dim 512 match HF config.json. Codec num_quantizers 8 / codebook_size 8192 / codebook_dim 32 match HeartCodecConfig.

### Kokoro (hexgrad/Kokoro-82M)
- No wrong values and no missing architecture-shaping params: every numeric field in config.json matches our record exactly.
- plbert.embedding_size (EmbeddingSize=128) and plbert layer_norm_eps (LayerNormEps=1e-12) are NOT in config.json; they are HuggingFace AlbertConfig defaults applied via AlbertConfig(**config['plbert']). Our values match the defaults but are config-derived, not config.json keys.
- config.json multispeaker:true has no field in our record; model.py never reads it, so it is benign metadata.
- config.json vocab (178-slot sparse IPA->ID map) is not in KokoroConfig; it is owned by the tokenizer in our port. Confirm the tokenizer uses identical sparse IDs.
- Single architecture variant only (one config.json, kokoro-v1_0.pth); no missing presets.

### GPT LM (audio backbone): GptConfig
- No Bias field: upstream GPTConfig dataclass defaults bias=True, but the actual Bark HF checkpoints (suno/bark and suno/bark-small) ship with bias=false on attention/MLP/LayerNorm. If GptBackbone hardcodes bias on (matching the dataclass default) it will not match real weights, which have no bias tensors. This determines which weight tensors exist, so it is shape/load affecting. Confirm what GptBackbone actually does and make Bias a config field (default false to match HF checkpoints).
- LayerNorm epsilon (reference 1e-5) is not exposed in the config; verify GptBackbone hardcodes 1e-5 (the upstream value). This is an architecture-affecting numeric not currently visible in GptConfig.

### Dia-1.6B (nari-labs)
- No correctness defects in architecture params: every encoder/decoder/vocab/delay/rope value in DiaConfig matches both the original nari-labs config.json (v0.1) and the transformers Dia-1.6B-0626 config.json exactly.
- Our generation defaults (CfgScale 3.0, Temperature 1.2, TopP 0.95, TopK 45) match the canonical nari-labs model.generate() defaults exactly. The HF transformers generation_config.json (0626) uses different values (temperature 1.8, top_p 0.90, top_k 50, guidance_scale 3.0); these are HF's repackaged sampling presets, not authoritative architecture, so no change is required but worth a one-line note.
- Minor: reference has rope_min_timescale=1 (original config.json). We hardcode RopeTheta=10000 (rope_max_timescale) but expose no min-timescale field. If our rope implementation assumes min=1 this is OK; classify as missing-but-likely-defaulted.

### GPT-SoVITS (stage-1 Text2Semantic AR GPT)
- WRONG: PhonemeVocab=512 is the V1 value. The reference V2/V2Pro/V3/V4 s1 model uses phoneme_vocab_size=732 (confirmed in s1longer-v2.yaml and corroborated by the Russian fork extending 732->753). Since the only preset is named V2 but carries the V1 vocab, every V2+ checkpoint will mismatch the ar_text_embedding tensor shape [732,512]. The in-code comment asserting 512 is correct for v2 is factually wrong.
- MISSING VARIANTS: only one preset (V2) exists, and it conflates V1 and V2. There is no distinct V1 preset (phoneme_vocab_size=512, top_k=5) and the V2 preset has the wrong phoneme vocab. Add separate V1 and V2 factories.
- TopK default mismatch by variant: V1 yaml sets top_k=5, V2 yaml sets top_k=15. Our TopK=15 matches V2 but a V1 preset should use 5.

### XCodec (YuE)
- No correctness gaps found: every inference/architecture parameter in our XCodecConfig matches the YuE xcodec_mini_infer reference (config.yaml generator + SoundStream.__init__ + dac2.Decoder).
- The reference SoundStream default D=128 is OVERRIDDEN to D=256 by final_ckpt/config.yaml (D: 256). Our AcousticDim default is correctly 256 (the checkpoint value), not the dataclass default 128. This is the one place where reading only the Python default would be wrong; our value tracks the actual checkpoint config, which is correct.
- n_q (NCodebooks=12) is correct: reference computes int(1000*6 // (ceil(16000/320)*10)) = 12 from target_bandwidths[-1]=6. We hardcode 12 rather than deriving it; values agree but the derivation (target_bandwidths) is not represented in our config (low impact since decode only uses cb0).

---

## Per model detail

## PocketTTS

Reference: Kyutai pocket-tts (github.com/kyutai-labs/pocket-tts, weights hf://kyutai/pocket-tts). Architecture is YAML-defined (no config.json); a FlowLM streaming transformer over continuous Mimi latents plus a per-frame SimpleMLPAdaLN flow head with Lagrangian Self-Distillation. Variants are per-language YAMLs that share dims except num_layers (6 for base languages, 24 for the `_24l` variants) and mimi.inner_dim (512 for english_2026-01/b6369a24, 32 elsewhere). The public dims ARE published in the repo YAMLs, contrary to our config's "NOT public" doc comment.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| flow_lm.transformer.d_model | DModel | 1024 | 0 | wrong | All YAMLs: d_model=1024 (config/english.yaml). |
| flow_lm.transformer.num_layers | NumLayers | 6 (base) / 24 (`_24l`) | 6 | wrong | 6 ok for english/german/etc; 24 for french_24l/spanish_24l/etc. Single value cannot cover both. |
| flow_lm.transformer.num_heads | NumHeads | 16 | 0 | wrong | english.yaml num_heads=16; our fallback d_model/64=16 happens to match, but the explicit default is 0. |
| (no GQA in ref) | NumKvHeads | n/a (MHA) | 0 | ok | Reference uses plain MHA; falling back to NumHeads is correct. |
| hidden_scale -> ffn = d_model*hidden_scale | FfnDim | 4096 (1024*4) | 0 | wrong | from_pydantic_config: dim_feedforward=int(d_model*hidden_scale), hidden_scale=4. Our fallback DModel*4 matches only if DModel set. |
| flow_lm.transformer.max_period | RopeTheta | 10000 | 10000 | ok | max_period=10000 -> RoPE theta. |
| norm eps (LayerNorm) | RmsNormEps | 1e-5 (LayerNorm) | 1e-6 (RMSNorm) | wrong | mimi_transformer.py norm1/norm2 = nn.LayerNorm(eps=1e-5); flow_lm out_norm LayerNorm eps 1e-5. Reference is LayerNorm not RMSNorm, and eps differs. |
| FFN type | (via Qwen2 SwiGLU) | GELU MLP (linear1->gelu->linear2, bias=False) | SwiGLU | wrong | _ff_block uses F.gelu, two bias-free linears. ToTransformerConfig builds a Qwen2 SwiGLU body, an architecture mismatch. |
| LayerScale | (none) | layer_scale present in FlowLM blocks (LayerScale module) | absent | missing | FlowLM TransformerLayer wraps both sublayers in LayerScale; not modeled. |
| lookup_table.n_bins (text vocab) | VocabSize | 4000 | 0 | wrong | LUTConditioner n_bins=4000. |
| lookup_table.dim | (none) | 1024 | absent | missing | Text LUT embedding dim before projection to d_model; no field. |
| tie embeddings | TieWordEmbeddings | false (no tied LM head; only out_eos Linear(dim,1)) | true | wrong | Reference has no token-logits head to tie; only a 1-d EOS head and the flow head. |
| context (FlowLM attn window) | MaxSequenceLength | None (full causal) | 4096 | unverified | FlowLM transformer is built with no `context` (full causal, unbounded). 4096 is our own KV-cache cap, not a reference value. |
| mimi.quantizer.dimension (ldim) | LatentDim | 32 | 0 | wrong | flow_lm ldim = mimi quantizer.dimension = 32. |
| mimi.inner_dim | (none) | 512 (english_2026-01) / 32 (others) | absent | missing | Per-variant; affects the continuous-latent Mimi path. No field. |
| mimi.outer_dim | (none) | 512 | absent | missing | No field; Mimi config covers seanet.dimension but not outer_dim. |
| flow.dim | FlowHeadDim | 512 | 0 | wrong | SimpleMLPAdaLN model_channels = flow.dim = 512. |
| flow.depth | (none) | 6 | absent | missing | Number of ResBlocks in the flow head; no field (we only have a single width). |
| num_time_conds | (none) | 2 | absent | missing | Flow head uses 2 time conditions (start s, target t), averaged; not 1. |
| time freq embed size | TimeEmbedDim | 256 (frequency_embedding_size) | 0 | wrong | TimestepEmbedder frequency_embedding_size=256; model_channels=flow.dim. |
| DEFAULT_LSD_DECODE_STEPS | LsdDecodeSteps | 1 | 4 | wrong | default_parameters.py = 1. |
| DEFAULT_TEMPERATURE | (none) | 0.7 | absent | missing | Sampling temperature (noise std = temp**0.5); no field. |
| DEFAULT_EOS_THRESHOLD | (none) | -4.0 | absent | missing | out_eos > eos_threshold; no field. |
| DEFAULT_NOISE_CLAMP | (none) | None | absent | missing | trunc-normal clamp on flow noise; no field. |
| MAX_TOKEN_PER_CHUNK | (none) | 50 | absent | missing | Text chunking bound; no field. |
| insert_bos_before_voice | (none) | true (most) / false (english_2026-01) | absent | missing | Per-variant BOS-before-voice flag; no field. |
| pad_with_spaces_for_short_inputs | (none) | true (english_2026-01 only) | absent | missing | Per-variant; no field. |
| remove_semicolons | (none) | true (french_24l, german) | absent | missing | Per-variant text cleanup; no field. |
| model_recommended_frames_after_eos | (none) | 8 (french_24l) | absent | missing | Per-variant trailing frames; no field. |
| sample_rate | SampleRate | 24000 | 24000 | ok | Matches. |
| frame_rate | FrameRateHz | 12.5 | 12 | wrong | mimi.frame_rate=12.5 (int field truncates to 12; should be float 12.5). |
| LatentCfgScale | LatentCfgScale | n/a | 1.0 | extra | No latent CFG in reference inference (only temp/noise_clamp/eos_threshold). |
| MaxFrames | MaxFrames | n/a (dynamic by text) | 1500 | extra | No fixed frame cap in reference; ours is a safety bound. |
| Voices list | Voices | embeddings dir (per-voice .safetensors) | hardcoded 26 names | extra | Reference resolves voices from embeddings/ on the hub plus DEFAULT_VOICE_FOR_LANGUAGE; the hardcoded list is a convenience, not a config param. |

### Action items
- Set real backbone defaults: DModel=1024, NumHeads=16, FfnDim=4096 (or derive from hidden_scale=4), LatentDim=32, FlowHeadDim=512, TimeEmbedDim=256, VocabSize=4000, LsdDecodeSteps=1; the "dims NOT public" comment is wrong, all are in the repo YAMLs.
- Fix architecture: the FlowLM body is LayerNorm (eps 1e-5) plus a plain bias-free GELU MLP plus LayerScale, not Qwen2 RMSNorm+SwiGLU; either stop routing through Qwen2Config or add LayerNorm/GELU/LayerScale options. Set TieWordEmbeddings=false (there is no tied LM head, only out_eos Linear(dim,1)).
- Change FrameRateHz to a float 12.5 (currently truncated to 12).
- Add per-frame flow-head structural fields: flow.depth (6 ResBlocks) and num_time_conds (2); a single FlowHeadDim is insufficient.
- Add inference fields: Temperature (0.7), EosThreshold (-4.0), NoiseClamp (nullable), MaxTokenPerChunk (50); drop or zero-default the unsupported LatentCfgScale.
- Add per-variant flags: InsertBosBeforeVoice, PadWithSpacesForShortInputs, RemoveSemicolons, ModelRecommendedFramesAfterEos, plus Mimi inner_dim/outer_dim.
- Add presets per language YAML, especially the missing 24-layer variants (french_24l, spanish_24l, german_24l, italian_24l, portuguese_24l use num_layers=24) and the english_2026-01/b6369a24 variant (mimi.inner_dim=512, insert_bos_before_voice=false, pad_with_spaces_for_short_inputs=true).
- Verify the FlowLM attention context: reference builds the transformer with no `context` (full causal); our MaxSequenceLength=4096 is a KV-cache cap, not a reference window, so document it as ours rather than a reference value.

<details><summary>Sources consulted</summary>

- https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/config/english.yaml
- https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/config/english_2026-01.yaml
- https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/config/french_24l.yaml
- https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/config/spanish_24l.yaml
- https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/utils/config.py
- https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/models/flow_lm.py
- https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/modules/mimi_transformer.py
- https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/modules/mlp.py
- https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/default_parameters.py
- https://huggingface.co/kyutai/pocket-tts/tree/main

</details>

---

## BiCodec (Spark)

Reference: SparkAudio/Spark-TTS-0.5B `BiCodec/config.yaml` (single checkpoint, no size variants) plus `sparktts/models/bicodec.py` and `sparktts/modules/speaker/speaker_encoder.py`. There is exactly one BiCodec variant in the official release, so no per-variant presets are missing, but our single `BiCodecConfig.Default` diverges from it on the global (speaker) path and omits the whole decode/synthesis side.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| quantizer.input_dim | SemanticHiddenDim | 1024 | 1024 | ok | semantic VQ in_proj input (config.yaml) |
| quantizer.codebook_size | SemanticCodebookSize | 8192 | 8192 | ok | config.yaml |
| quantizer.codebook_dim | SemanticCodebookDim | 8 | 8 | ok | config.yaml |
| quantizer.use_l2_normlize | (hardcoded L2NormalizeRows) | True | True (hardcoded) | ok | cosine lookup matches L2-normalized codebook |
| quantizer.threshold_ema_dead_code | none | 0.2 | absent | ok | training/EMA only, no inference shape impact, safe to skip |
| encoder.input_channels | FeatureDim | 1024 | 1024 | ok | w2v-BERT feature width (config.yaml) |
| speaker_encoder.token_num | GlobalTokens | 32 | 32 | ok | config.yaml |
| speaker_encoder.fsq_levels | GlobalFsqLevels | [4,4,4,4,4,4] | [8,8,8,5,5] | wrong | 6 dims/4096 vocab vs our 5 dims/12800 vocab; global codes cannot match real weights (config.yaml) |
| speaker_encoder pooling heads | GlobalQueryHeads | not exposed (PerceiverResampler internal) | 8 | unverified | head count is internal to PerceiverResampler, not a config.yaml key; our value is an artifact of the wrong cross-attn path |
| speaker_encoder.input_dim | none | 128 (mel bins) | absent (we feed 1024 w2v-BERT) | missing | reference speaker encoder consumes a 128-bin mel, not w2v-BERT features |
| speaker_encoder.out_dim | none | 1024 | absent | missing | speaker embedding output dim |
| speaker_encoder.latent_dim | none | 128 | absent | missing | PerceiverResampler latent / FSQ pre-proj dim |
| speaker_encoder.fsq_num_quantizers | none | 1 | absent | missing | ResidualFSQ stage count |
| speaker_encoder backbone | none | ECAPA_TDNN_GLOB_c512 | absent (cross-attn over w2v-BERT) | missing | whole global-encoder architecture differs |
| mel_params.sample_rate | none | 16000 | absent | missing | needed to build mel for speaker encoder + output rate |
| mel_params.n_fft | none | 1024 | absent | missing | mel front-end |
| mel_params.win_length | none | 640 | absent | missing | mel front-end |
| mel_params.hop_length | none | 320 | absent | missing | 50 Hz frame rate (16000/320) |
| mel_params.num_mels | none | 128 | absent | missing | mel bins = speaker_encoder.input_dim |
| mel_params.mel_fmin / mel_fmax | none | 10 / null | absent | missing | mel front-end band |
| encoder.vocos_dim / vocos_intermediate_dim / vocos_num_layers | none | 384 / 2048 / 12 | absent | missing | semantic encoder is a Vocos backbone (12 layers, 2048 FFN), not a single Linear proj as we model it |
| encoder.sample_ratios | none | [1,1] | absent | missing | encoder downsample ratios |
| decoder.input_channel / channels / rates / kernel_sizes | none | 1024 / 1536 / [8,5,4,2] / [16,11,8,4] | absent | missing | WaveGenerator (HiFi-GAN) decode path, not implemented |
| prenet.* (vocos_dim/intermediate/num_layers/condition_dim/use_tanh_at_final) | none | 384/2048/12/1024/False | absent | missing | conditioning prenet for decode |
| postnet.* (vocos_dim/intermediate/num_layers/use_tanh_at_final) | none | 384/2048/6/False | absent | missing | 6-layer postnet for decode |
| (none) | SemanticCodebookDim used as semantic proj: fine; but semantic encoder modeled as 1 Linear | Vocos 12-layer encoder | single ProjectLinear | extra/wrong-arch | our semantic encoder.proj is one Linear; reference encoder is a 12-layer Vocos block, so semantic latents will not match real weights either |

Action items:
- Fix `GlobalFsqLevels` default to `[4, 4, 4, 4, 4, 4]` (4096 vocab) to match `speaker_encoder.fsq_levels`; the current `[8,8,8,5,5]` is wrong.
- Add a real config record (and rewrite `BiCodecGlobalEncoder`) for the reference speaker encoder: mel input (`input_dim=128`), `latent_dim=128`, `out_dim=1024`, `fsq_num_quantizers=1`, ECAPA_TDNN_GLOB_c512 backbone + PerceiverResampler pooling. The current w2v-BERT cross-attention with `speaker_encoder.queries/attn.*_proj/fsq_proj` keys does not exist in the real checkpoint.
- Add `mel_params` fields (sample_rate 16000, n_fft 1024, win_length 640, hop_length 320, num_mels 128, mel_fmin 10, mel_fmax null); the speaker encoder needs a mel front-end, and these set the 50 Hz frame rate.
- Add encoder Vocos params (`vocos_dim=384`, `vocos_intermediate_dim=2048`, `vocos_num_layers=12`, `sample_ratios=[1,1]`) and rewrite the semantic encoder as a Vocos block rather than a single Linear; otherwise semantic latents will not match real weights.
- If round-trip decode is ever needed, add decoder/prenet/postnet records (decoder channels 1536, rates [8,5,4,2], kernel_sizes [16,11,8,4]; prenet/postnet Vocos 12/6 layers, intermediate 2048).
- Remove or replace `GlobalQueryHeads` once the global encoder is rewritten; it has no basis in the reference config.

<details><summary>Sources consulted</summary>

- https://huggingface.co/SparkAudio/Spark-TTS-0.5B/raw/main/BiCodec/config.yaml
- https://github.com/SparkAudio/Spark-TTS/blob/main/sparktts/models/bicodec.py
- https://github.com/SparkAudio/Spark-TTS/blob/main/sparktts/modules/speaker/speaker_encoder.py
- https://arxiv.org/abs/2503.01710

</details>

---

## HtDemucs

Reference: facebookresearch/demucs (htdemucs.py HTDemucs.__init__ + conf/config.yaml). Released variants: htdemucs (4 stems), htdemucs_ft (4-model fine-tuned bag), htdemucs_6s (6 stems). Our config exposes a single Htdemucs preset (4 stems, bottom_channels=512).

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| audio_channels | AudioChannels | 2 | 2 | ok | |
| sources | Sources | [drums,bass,other,vocals] | same | ok | 6s adds guitar,piano (no preset) |
| channels | Channels | 48 | 48 | ok | |
| growth | Growth | 2 | 2 | ok | |
| depth | Depth | 4 | 4 | ok | |
| nfft | NFft | 4096 | 4096 | ok | hop = nfft/4 = 1024 |
| (hop_length) | HopLength | 1024 | 1024 | ok | derived in ref (nfft//4) |
| kernel_size | KernelSize | 8 | 8 | ok | |
| stride | Stride | 4 | 4 | ok | |
| bottom_channels | BottomChannels | 0 (train default); 512 in released ckpts | 512 | ok | 512 correct for loading published weights; train default is 0 (config.yaml/htdemucs.py) |
| t_layers | TLayers | 5 | 5 | ok | |
| t_heads | THeads | 8 | 8 | ok | |
| t_hidden_scale | HiddenScale | 4.0 | 4.0 | ok | FFN = 512*4 = 2048 |
| samplerate | SampleRate | 44100 | 44100 | ok | |
| cac | (hardcoded SpecInChannels=4) | true | implied true | ok-implied | complex-as-channels; no explicit field, cannot express cac=false; also affects decoder out channels |
| freq_emb | FreqEmbScale | 0.2 | 0.2 | ok | our name says "scale" but value equals ref freq_emb (the emb weight) |
| emb_scale | (none) | 10 | absent | missing | scales freq positional embedding indices |
| emb_smooth | (none) | true | absent | missing | smooths freq emb across bins |
| time_stride | (none) | 2 | absent | missing | last temporal-encoder stride; affects shapes |
| multi_freqs | (none) | [] | absent | missing | per-band freq split list (empty default) |
| multi_freqs_depth | (none) | 3 | absent | missing | |
| rewrite | (none) | true | absent | missing | adds 1x1 rewrite conv in enc/dec blocks (weight layout) |
| context | (none) | 1 | absent | missing | decoder context conv |
| context_enc | (none) | 0 | absent | missing | encoder context |
| norm_starts | (none) | 4 | absent | missing | group-norm starts at this layer index |
| norm_groups | (none) | 4 | absent | missing | GroupNorm group count |
| dconv_mode | (none) | 1 | absent | missing | which blocks get DConv |
| dconv_depth | (none) | 2 | absent | missing | residual DConv branch depth (weight layout) |
| dconv_comp | (none) | 8 | absent | missing | DConv channel compression |
| dconv_init | (none) | 1e-3 | absent | missing | LayerScale init in DConv (init only, but defines a learned tensor) |
| rescale | (none) | 0.1 | absent | missing | weight rescale init (init-time only) |
| wiener_iters | (none) | 0 | absent | missing | 0 = use cac iSTFT path, not Wiener; affects inference output path |
| t_emb | (none) | "sin" | absent | missing | transformer positional-embedding type (sin vs cape/scaled) |
| t_max_positions | (none) | 10000 | absent | missing | only used for scaled-emb |
| t_max_period | (none) | 10000.0 | absent | missing | sinusoidal period |
| t_norm_in / t_norm_first / t_norm_out | (none) | true/true/true | absent | missing | LayerNorm placement; changes which norm tensors exist |
| t_layer_scale | (none) | true | absent | missing | per-layer LayerScale gammas (learned tensors) |
| t_gelu | (none) | true | absent | missing | GELU vs ReLU in transformer FFN |
| t_dropout | (none) | 0.0 | absent | excluded | inference no-op (dropout) |
| t_cape_* / t_sparse_* / t_mask_* | (none) | defaults off | absent | excluded | only active for cape emb / sparse attention (not used by released sin-emb models) |
| segment | (none) | 10 (code) / 11 (config) | absent | unverified | inference chunk seconds; code default 10, config.yaml 11; released checkpoints carry their own; not a tensor-shape param but drives valid_length |
| t_weight_pos_embed | (none) | 1.0 | absent | extra-side | scalar multiplier on pos-embed; minor |
| NormEps | NormEps | n/a (PyTorch default 1e-5) | 1e-5 | extra | not a demucs hparam; matches torch GroupNorm/LayerNorm default eps |

Action items:
- Add emb_scale (10) and emb_smooth (true) fields; they shape the frequency positional embedding added at spec encoder layer 0.
- Add the encoder/decoder structural params so they are not silently hardcoded wrong: time_stride (2), context (1), context_enc (0), norm_starts (4), norm_groups (4), rewrite (true), multi_freqs_depth (3), dconv_depth (2), dconv_comp (8), dconv_init (1e-3). Verify each against what HtDemucs model .cs currently hardcodes; any mismatch breaks weight loading.
- Add an explicit cac flag (default true) instead of inferring SpecInChannels = 4, so cac=false models are expressible.
- Add transformer params t_emb (sin), t_max_period (10000), t_norm_in/t_norm_first/t_norm_out (true), t_layer_scale (true), t_gelu (true); these define which norm and LayerScale tensors exist in the cross-domain transformer.
- Add wiener_iters (0) to select the cac iSTFT inference path explicitly.
- Add presets: HtdemucsFt (same arch, used as a 4-model bag) and Htdemucs6s (Sources = drums,bass,other,vocals,guitar,piano), since 6s changes NumSources and the decoder output channel count.
- Confirm/segment: record the per-checkpoint segment (training segment) value; not a tensor shape but it drives valid_length and overlap-add chunking.

<details><summary>Sources consulted</summary>

- https://raw.githubusercontent.com/facebookresearch/demucs/main/conf/config.yaml
- https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/htdemucs.py
- https://github.com/facebookresearch/demucs (htdemucs_6s sources: drums, bass, other, vocals, guitar, piano)

</details>

---

## Mimi codec

Reference: kyutai/mimi (HF transformers MimiModel / MimiConfig). Single published 24 kHz checkpoint; the config dataclass default is num_quantizers=32 total codebooks. Our config exposes 2 presets (Mimi24kHz with 8 total codebooks, Mimi24kHzDsm with 32 total). Reference values from the HF raw config.json and configuration_mimi.py constructor defaults.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| sampling_rate | SampleRate | 24000 | 24000 | ok | |
| audio_channels | Channels | 1 | 1 | ok | |
| num_filters | EncoderDim | 64 | 64 | ok | |
| upsampling_ratios | EncoderRates | [8,6,5,4] | [8,6,5,4] | ok | |
| hidden_size | TransformerDim / LatentDim | 512 | 512 | ok | Both LatentDim and TransformerDim map to hidden_size=512. |
| num_hidden_layers | TransformerLayers | 8 | 8 | ok | |
| num_attention_heads | TransformerHeads | 8 | 8 | ok | |
| intermediate_size | TransformerFfnDim | 2048 | 2048 | ok | |
| rope_theta | TransformerRopeTheta | 10000.0 | 10000 | ok | rope_theta is in config.json (10000.0). |
| codebook_size | CodebookSize | 2048 | 2048 | ok | |
| codebook_dim | CodebookDim | 256 | 256 | ok | |
| vector_quantization_hidden_dimension | (CodebookDim reused) | 256 | 256 | ok | Same value; no separate field but matches. |
| num_residual_layers | ResidualDilations.Count | 1 | 2 (default [1,1]) | wrong | Base Mimi24kHz default [1,1] => 2 blocks; reference checkpoint has 1. DSM preset correctly uses [1]. Base preset should be [1]. Source: configuration_mimi.py default num_residual_layers=1 and config.json num_residual_layers:1. |
| num_quantizers (total codebooks) | TotalCodebooks (=1+AcousticCodebooks) | 32 | 8 (Mimi24kHz) / 32 (Dsm) | wrong | Reference default 32; config.json does not override, so checkpoint is 32. Mimi24kHz uses 8 (common Moshi/CSM truncation) but no clean base-32 preset exists. Source: configuration_mimi.py num_quantizers=32. |
| frame_rate | FrameRate (computed) | 12.5 | 25 (computed) | wrong | Property returns 24000/960=25; reference is 12.5 (extra /2 from compress/internal stride not modeled). XML comment also says 12.5, contradicting the property. |
| num_semantic_quantizers | (implicit 1 in TotalCodebooks) | 1 | 1 | ok | Hardcoded as the +1 in TotalCodebooks. |
| dilation_growth_rate | (none) | 2 | (absent) | missing | No field. Harmless at num_residual_layers=1 but unmodeled. |
| compress | (none) | 2 | (absent) | missing | Affects encoder/decoder downsample factor and effective frame rate (12.5 vs 25). |
| kernel_size | (none) | 7 | (likely hardcoded in SEANet) | missing | Not a config field; verify it is hardcoded =7 in the encoder/decoder. |
| last_kernel_size | (none) | 3 | (likely hardcoded) | missing | Final conv kernel; confirm hardcoded =3. |
| residual_kernel_size | (none) | 3 | (likely hardcoded) | missing | Residual block kernel; confirm hardcoded =3. |
| sliding_window | (none) | 250 | (absent) | missing | Transformer-of-codecs attention sliding window; affects attention masking. |
| max_position_embeddings | (none) | 8000 | (absent) | missing | RoPE position cap. |
| head_dim | (none) | 64 | (implicit 512/8=64) | missing | Implicitly 64 via hidden/heads; no explicit field but value matches. |
| num_key_value_heads | (none) | 8 | (implicit MHA) | missing | Equals num_attention_heads (full MHA, no GQA); behavior matches but unmodeled. |
| use_conv_shortcut | (none) | false | (absent) | missing | Residual shortcut conv toggle. |
| upsample_groups | (none) | 512 | (absent) | missing | Grouped transpose-conv for the upsample stage. |
| layer_scale_initial_scale | (none) | 0.01 | (absent) | missing | LayerScale init for transformer blocks (inference-relevant: loaded weight, not just init). |
| trim_right_ratio | (none) | 1.0 | (absent) | missing | Causal conv right-trim; affects streaming output length. |
| pad_mode | (none) | constant | (likely hardcoded) | missing | Conv padding mode; confirm hardcoded constant. |
| hidden_act | (none) | gelu | (likely hardcoded) | missing | Transformer FFN activation; confirm gelu. |
| norm_eps | (none) | 1e-5 | (likely hardcoded) | unverified | LayerNorm/RMSNorm eps; could not confirm whether hardcoded 1e-5 in the model code. |
| attention_bias | (none) | false | (n/a) | ok | No bias; default behavior, not a shape param. |
| use_causal_conv | (none) | true | (causal by design) | ok | Encoder/decoder are causal by construction. |
| EncoderDim alias note | LatentDim=512 | n/a | 512 | extra | LatentDim duplicates hidden_size; not a distinct reference param (low priority). |

Action items:
- Fix the base Mimi24kHz preset to use ResidualDilations=[1] (num_residual_layers=1) to match the real checkpoint and avoid requesting a nonexistent 2nd residual block's weights.
- Add a config field for total/num_quantizers semantics and provide a preset matching the shipped kyutai/mimi checkpoint (32 total codebooks, ResidualDilations=[1]); keep an 8-codebook preset for Moshi/CSM truncated use, but label it as the truncated variant rather than the default.
- Correct the FrameRate computation to 12.5 Hz (model the extra /2 from compress / internal stride) so it agrees with both the reference and the XML comment.
- Add explicit fields (or document the hardcoded values) for: compress=2, sliding_window=250, last_kernel_size=3, residual_kernel_size=3, kernel_size=7, use_conv_shortcut=false, upsample_groups=512, layer_scale_initial_scale=0.01, max_position_embeddings=8000, trim_right_ratio=1.0, dilation_growth_rate=2, head_dim=64, num_key_value_heads=8.
- Verify and document norm_eps (reference 1e-5) and hidden_act=gelu / pad_mode=constant against the SEANet and transformer code.

<details><summary>Sources consulted</summary>

- https://huggingface.co/kyutai/mimi/raw/main/config.json
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/mimi/configuration_mimi.py

</details>

---

## FishSpeech

Reference: fishaudio/fish-speech 1.4/1.5, DualAR text2semantic LM (config.json, model_type=dual_ar) plus Firefly-GAN-VQ codec (firefly_gan_vq.yaml at tag v1.5.1). Our presets: FishSpeechConfig.V1_5 and FireflyConfig.V1_5 only. Verified against HF config.json (1.5 and 1.4), the v1.5.1 firefly YAML, and the v1.5.1 inference CLI defaults.

### DualAR LM (FishSpeechConfig, vs fish-speech-1.5 config.json)

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| dim | Backbone.HiddenSize | 1024 | 1024 | ok | |
| n_layer | Backbone.NumHiddenLayers | 24 | 24 | ok | |
| n_head | Backbone.NumAttentionHeads | 16 | 16 | ok | |
| n_local_heads | Backbone.NumKeyValueHeads | 2 | 2 | ok | GQA kv heads |
| head_dim | (derived 1024/16) | 64 | 64 | ok | matches config.json head_dim=64 |
| intermediate_size | Backbone.IntermediateSize | 4096 | 4096 | ok | |
| vocab_size | Backbone.VocabSize / TextVocab | 102048 | 102048 | ok | 1.5 value |
| max_seq_len | Backbone.MaxPositionEmbeddings | 8192 | 8192 | ok | 1.5 value |
| rope_base | Backbone.RopeTheta | 1000000.0 | 1000000 | ok | |
| norm_eps | Backbone.RmsNormEps | 1e-6 | 1e-6 | ok | |
| attention_qkv_bias | Backbone.AttentionBias | false | false | ok | |
| tie_word_embeddings | Backbone.TieWordEmbeddings | false | false | ok | |
| fast_dim | Fast.HiddenSize | 1024 | 1024 | ok | |
| n_fast_layer | Fast.NumHiddenLayers | 4 | 4 | ok | |
| fast_n_head | Fast.NumAttentionHeads | 16 | 16 | ok | |
| fast_n_local_heads | Fast.NumKeyValueHeads | 2 | 2 | ok | |
| fast_head_dim | (derived) | 64 | 64 | ok | |
| fast_intermediate_size | Fast.IntermediateSize | 4096 | 4096 | ok | |
| fast_attention_qkv_bias | Fast.AttentionBias | false | false | ok | |
| num_codebooks | NumCodebooks | 8 | 8 | ok | |
| codebook_size | CodebookSize / Fast.VocabSize | 1024 | 1024 | ok | |
| dropout | (none, inference-disabled) | 0.0 (1.5) / 0.1 (1.4) | n/a | ok | training-only, excluded |
| initializer_range | (none) | 0.02 | n/a | ok | training-only, excluded |
| is_reward_model | (none) | false | n/a | ok | not relevant for TTS inference |
| model_type | (implicit) | dual_ar | n/a | ok | |
| temperature | Temperature | 0.7 | 1.0 | wrong | v1.5.1 inference.py CLI default is 0.7 |
| top_p | TopP | 0.7 | 0.9 | wrong | CLI default is 0.7 |
| repetition_penalty | RepetitionPenalty | 1.2 | 1.1 | wrong | CLI default is 1.2 |
| max_new_tokens | MaxNewTokens | 0 (auto/unbounded) | 1500 | wrong | CLI default 0 means compute from max_seq_len |
| top_k | TopK | (none in CLI) | 30 | extra | sampler is top_p based; no top_k click option in v1.5.1 |
| chunk_length | (none) | 100 | n/a | unverified | text chunking length for long-form; affects segmentation not tensor shape; not modeled |
| scale_codebook_embeddings | ScaleCodebookEmbeddings | false (per checkpoint) | false | ok | matches runtime-verified note |

### Firefly-GAN-VQ codec (FireflyConfig, vs v1.5.1 firefly_gan_vq.yaml)

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| quantizer.levels | FsqLevels | [8,5,5,5] | [8,5,5,5] | ok | per-group FSQ levels (product 1000) |
| quantizer.downsample_factor | QuantizerUpsampleFactors | [2,2] | [2,2] | ok | |
| head.upsample_initial_channel | UpsampleInitialChannel | 512 | 512 | ok | |
| head.upsample_rates | UpsampleRates | [8,8,2,2,2] | [8,8,2,2,2] | ok | |
| head.upsample_kernel_sizes | UpsampleKernelSizes | [16,16,4,4,4] | [16,16,4,4,4] | ok | YAML value (firefly.py code default [16,16,8,2,2] is unused) |
| head.resblock_kernel_sizes | ResBlockKernelSizes | [3,7,11] | [3,7,11] | ok | |
| head.resblock_dilation_sizes | ResBlockDilations | [[1,3,5]x3] | [[1,3,5]x3] | ok | |
| spec_transform.sample_rate | SampleRate | 44100 | 44100 | ok | |
| quantizer.n_groups | (none) | 8 | n/a | missing | grouped FSQ; 8 groups times 1 codebook give the 8 codebooks; required for quantizer shape |
| quantizer.n_codebooks | (none) | 1 | n/a | missing | note: this is 1, not 8; the "8 codebooks" are the groups |
| quantizer.input_dim | (none) | 512 | n/a | missing | FSQ input channels |
| head.num_mels | (none) | 512 | n/a | missing | HiFiGAN head input dim (separate from upsample_initial_channel) |
| head.hop_length | (none) | 512 | n/a | missing | also win/hop of spec |
| head.pre_conv_kernel_size | (none) | 13 | n/a | missing | |
| head.post_conv_kernel_size | (none) | 13 | n/a | missing | |
| backbone.input_channels / n_mels | (none) | 160 | n/a | missing | ConvNeXt encoder input mel bins (encode path) |
| backbone.depths | (none) | [3,3,9,3] | n/a | missing | ConvNeXt encoder (encode path) |
| backbone.dims | (none) | [128,256,384,512] | n/a | missing | ConvNeXt encoder (encode path) |
| backbone.kernel_size | (none) | 7 | n/a | missing | ConvNeXt encoder (encode path) |
| spec_transform n_fft/win/hop | (none) | 2048/2048/512 | n/a | missing | LogMel front-end (encode path) |

### Action items

- Add `NGroups = 8` (and `QuantizerNCodebooks = 1`, `QuantizerInputDim = 512`) to FireflyConfig; the grouped-FSQ layout is currently underspecified.
- Add HiFiGAN head params to FireflyConfig: `NumMels = 512`, `HopLength = 512`, `PreConvKernelSize = 13`, `PostConvKernelSize = 13` (only `UpsampleInitialChannel` is present today).
- If the encode (audio to tokens) path is in scope, add the ConvNeXt encoder block (input_channels 160, depths [3,3,9,3], dims [128,256,384,512], kernel_size 7) and the LogMel spec_transform (n_mels 160, n_fft 2048, hop_length 512, win_length 2048).
- Correct sampling defaults in FishSpeechConfig to the v1.5.1 CLI: Temperature 0.7, TopP 0.7, RepetitionPenalty 1.2, MaxNewTokens 0 (auto). Reconsider TopK (no reference CLI default; either drop it or document it as an engine-specific extra).
- Add a FishSpeechConfig.V1_4 preset: VocabSize/TextVocab 32000, MaxPositionEmbeddings 4096 (dropout 0.1 is training-only). The current single V1_5 preset cannot load 1.4 checkpoints.
- Decide scope for OpenAudio S1 / S1-mini: its codec is modded_dac_vq (n_codebooks 9, semantic_codebook_size 4096, DAC decoder dim 1536 rates [8,8,4,2], extra quantizer transformer), a different architecture with no preset here.

<details><summary>Sources consulted</summary>

- https://huggingface.co/fishaudio/fish-speech-1.5/raw/main/config.json
- https://huggingface.co/fishaudio/fish-speech-1.4/raw/main/config.json
- https://raw.githubusercontent.com/fishaudio/fish-speech/v1.5.1/fish_speech/configs/firefly_gan_vq.yaml
- https://raw.githubusercontent.com/fishaudio/fish-speech/v1.5.1/fish_speech/models/text2semantic/inference.py

</details>

---

## Zonos

Reference: Zyphra Zonos-v0.1, two checkpoints (Zonos-v0.1-transformer = pure attention 26-layer; Zonos-v0.1-hybrid = 46-layer Mamba2/attention hybrid). Our config covers only the transformer variant; values were checked against the HF config.json for both checkpoints plus the GitHub source (config.py, model.py, sampling.py, speaker_cloning.py).

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| backbone.d_model | Hidden | 2048 | 2048 | ok | HF transformer config.json |
| backbone.n_layer | NumLayers | 26 (transformer) / 46 (hybrid) | 26 | ok (transformer only) | hybrid not covered, see missing variants |
| attn_cfg.num_heads | NumHeads | 16 | 16 | ok | |
| attn_cfg.num_heads_kv | NumKvHeads | 4 | 4 | ok | GQA 4:1 |
| attn_cfg.rotary_emb_dim | HeadDim | 128 | 128 | ok | rotary_emb_dim doubles as head_dim |
| backbone.attn_mlp_d_intermediate | FfnIntermediate | 8192 | 8192 | ok | |
| backbone.norm_epsilon | NormEps | 1e-5 | 1e-5f | ok | rms_norm=false so LayerNorm, matches our doc |
| attn_cfg.rotary_emb_interleaved | (hardcoded interleaved RoPE) | true | true (DiaAttention interleaved) | ok | transformer config; note hybrid config.json omits this key |
| rope_theta | RopeTheta | 10000 (Mamba/flash default) | 10000f | unverified | not in config.json; flash-attn rotary default is 10000, not explicitly set in Zonos source |
| attn_cfg.qkv_proj_bias / out_proj_bias | (hardcoded no bias) | false / false | no bias | ok | |
| MaxPositions / max_seqlen | MaxPositions | InferenceParams.max_seqlen (runtime, no static default) | 16384 | ok (extra-ish) | runtime-supplied in ref; our 16384 is a reasonable cap |
| eos_token_id | EosToken | 1024 | 1024 | ok | |
| masked_token_id | MaskedToken | 1025 | 1025 | ok | |
| per-codebook embedding vocab | InputVocab | nn.Embedding(1026, dim) | 1026 | ok | model.py |
| per-codebook head vocab | OutputVocab | nn.Linear(dim, 1025) | 1025 | ok | model.py |
| pad_vocab_to_multiple_of | (none) | 8 | (none) | missing | config.py default 8; real head/embed tensors pad 1025/1026 up to 1032 |
| num_codebooks | Channels | 9 | 9 | ok | autoencoder.num_codebooks |
| codebook size | (implicit in vocabs) | 1024 | 1024 implied | ok | |
| autoencoder (DAC, 44.1 kHz) | Codec = DacConfig.Dac44kHz | descript-audio-codec 44khz, 9 cb | Dac44kHz | ok | model.py DACAutoencoder |
| delay pattern | BuildDelays() -> [1..9] | offset delay per codebook | [1,2,...,9] | ok | apply_delay_pattern in model.py |
| speaker cond_dim | SpeakerDim | 128 | 128 | ok | PassthroughConditioner cond_dim |
| emotion input_dim | EmotionDim | 8 | 8 | ok | FourierConditioner emotion |
| fmax max_val | FmaxMax | 24000 | 24000f | ok | |
| pitch_std max_val | PitchStdMax | 400 | 400f | ok | |
| speaking_rate max_val | SpeakingRateMax | 40 | 40f | ok | |
| language_id range | NumLanguages | min -1, max 126 (127 ids) | 105 | wrong | IntegerConditioner -1..126 means 128 raw / 127 valid; 105 does not match |
| conditioner min_vals (emotion/fmax/pitch/rate=0) | (none) | 0 | (none) | missing | only max stored; mins are 0 so low impact, but undocumented |
| prefix_conditioner.projection | (none) | "linear" | (none) | missing | conditioner output projection type not a field |
| ssm_cfg.layer (Mamba2) | (none) | "Mamba2" (hybrid) | (none) | missing | no SSM/Mamba support; blocks hybrid checkpoint |
| attn_layer_idx | (none) | [0..25] transformer / [0,4,...,44] hybrid | (none, all-attn implied) | missing | needed to express hybrid layer placement |
| conditioner: vqscore_8 | (none) | Fourier input_dim 8, 0.5..0.8 (hybrid) | (none) | missing | hybrid-only conditioner |
| conditioner: ctc_loss | (none) | Fourier -1..1000 (hybrid) | (none) | missing | hybrid-only conditioner |
| conditioner: dnsmos_ovrl | (none) | Fourier 1..5 (hybrid) | (none) | missing | hybrid-only conditioner |
| conditioner: speaker_noised | (none) | Integer 0..1 (hybrid) | (none) | missing | hybrid-only conditioner |
| cfg_scale | CfgScale | 2.0 | 2.0f | ok | model.py sampling_params / generate default |
| temperature | Temperature | 1.0 | 1.0f | ok | sampling.py |
| min_p | MinP | 0.1 | 0.1f | ok | model.py default sampling_params(min_p=0.1) |
| repetition_penalty | RepetitionPenalty | 3.0 | 3.0f | ok | sampling.py |
| repetition_penalty_window | RepetitionWindow | 2 | 2 | ok | sampling.py |
| max_new_tokens | MaxNewTokens | 86*30 = 2580 | 2580 | ok | model.py |
| linear / conf / quad sampling | (none) | 0.0 each (unified sampler) | (none) | extra-absent | optional sampler, defaults off; fine to omit |
| --- Speaker encoder (ZonosSpeakerConfig) --- | | | | | speaker_cloning.py SpeakerEmbeddingLDA / ResNet293 |
| sample_rate | SampleRate | 16000 | 16000 | ok | |
| n_mels | NumMels | 80 | 80 | ok | |
| base width (in_planes) | BaseWidth | 64 | 32 | wrong | ResNet in_planes default 64 |
| block counts per stage | StageBlocks | [10, 20, 64, 3] | [3, 4, 6, 3] | wrong | ResNet293 layer schedule; ours is a compact stand-in (documented) |
| stage widths | StageWidths | base-doubling (approx [64,128,256,512]) | [32,64,128,256] | wrong | follows from base_width 64 |
| bottleneck embd_dim | BottleneckDim | 256 | 256 | ok | |
| LDA final embed dim | EmbedDim | 128 | 128 | ok | SpeakerEmbeddingLDA LDA-128 |
| SimAM lambda | SimAmLambda | 1e-4 (typical) | 1e-4f | ok | SimAM e_lambda standard default |
| NormGroups / PooledFreq | NormGroups / PooledFreq | (BatchNorm in ref, not GroupNorm) | 32 / 10 | extra | ref ResNet uses BatchNorm2d, not GroupNorm; our GN+groups is an impl choice |

Action items:
- Add a Zonos hybrid variant preset and an `AttnLayerIdx` (or equivalent layer-type schedule) field plus Mamba2/SSM backbone support so Zonos-v0.1-hybrid (n_layer=46, attn on [0,4,8,...,44]) can be expressed and loaded.
- Add the 4 hybrid-only conditioners (vqscore_8 Fourier 0.5..0.8 input_dim=8, ctc_loss Fourier -1..1000, dnsmos_ovrl Fourier 1..5, speaker_noised Integer 0..1) to the conditioning config.
- Add `PadVocabToMultipleOf = 8` and apply it to per-codebook embedding/head vocab so tensor shapes (1026->1032, 1025->1032) match real weights.
- Fix NumLanguages: the IntegerConditioner range is -1..126 (128 raw ids, 127 valid). Replace the 105 default with the correct count, or store min/max directly.
- Reconcile ZonosSpeakerConfig defaults against real ResNet293: BaseWidth 32 -> 64, StageBlocks [3,4,6,3] -> [10,20,64,3], StageWidths -> base-doubling ([64,128,256,512]); confirm BatchNorm vs the GroupNorm currently used.
- Add explicit conditioner `min_val` fields (currently only max stored) and a `projection` field ("linear") for completeness.
- Confirm RopeTheta=10000 against the actual flash-attn rotary default used by Zonos (not set in config.json), or load it from the checkpoint metadata if present.

<details><summary>Sources consulted</summary>

- https://huggingface.co/Zyphra/Zonos-v0.1-transformer/raw/main/config.json
- https://huggingface.co/Zyphra/Zonos-v0.1-hybrid/raw/main/config.json
- https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/config.py
- https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/model.py
- https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/sampling.py
- https://raw.githubusercontent.com/Zyphra/Zonos/main/zonos/speaker_cloning.py

</details>

---

## Qwen3-TTS / ECAPA / vocoder

Reference: Qwen/Qwen3-TTS-12Hz-1.7B-Base (talker + code_predictor + speaker_encoder) and the standalone Qwen/Qwen3-TTS-Tokenizer-12Hz codec (encoder/decoder). Our presets cover the 1.7B talker, the MTP code predictor, the codec decoder, and an ECAPA stub. The 0.6B, CustomVoice, and VoiceDesign variants have no dedicated presets. Values below cross-checked against the raw HF config.json / generation_config.json and the in-repo real-weight reconciliation in docs/Research/QWEN3_TTS_ARCHITECTURE.md.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| talker hidden_size | Talker.HiddenSize | 2048 | 2048 | ok | config.json |
| talker num_hidden_layers | Talker.NumHiddenLayers | 28 | 28 | ok | |
| talker num_attention_heads | Talker.NumAttentionHeads | 16 | 16 | ok | |
| talker num_key_value_heads | Talker.NumKeyValueHeads | 8 | 8 | ok | |
| talker head_dim | Talker.HeadDim | 128 | 128 | ok | |
| talker intermediate_size | Talker.IntermediateSize | 6144 | 6144 | ok | |
| talker rope_theta | Talker.RopeTheta | 1e6 | 1e6 | ok | |
| talker rms_norm_eps | Talker.RmsNormEps | 1e-6 | 1e-6 | ok | |
| max_position_embeddings | Talker.MaxPositionEmbeddings | 32768 | 32768 | ok | |
| text vocab_size | TextVocabSize | 151936 | 151936 | ok | |
| codec vocab_size | CodecVocabSize / CodecHeadOut | 3072 | 3072 | ok | |
| num_code_groups | NumCodeGroups | 16 | 16 | ok | |
| mrope_section | MropeSection | [24,20,20] | [24,20,20] | ok | rope_scaling.mrope_section, interleaved=true |
| position_id_per_seconds | (none) | 13 | absent | missing | feeds mRoPE position assignment; reduces to sequential for text-only TTS but still a real config value |
| code_predictor hidden_size | CodePredictor.HiddenSize | 1024 | 1024 | ok | |
| code_predictor num_hidden_layers | CodePredictor.NumHiddenLayers | 5 | 5 | ok | |
| code_predictor intermediate_size | CodePredictor.IntermediateSize | 3072 | 3072 | ok | |
| code_predictor head_dim | CodePredictor.HeadDim | 128 | 128 | ok | |
| code_predictor vocab_size | MtpVocabSize | 2048 | 2048 | ok | |
| MTP codebooks (1..15) | MtpCodebooks | 15 | 15 | ok | |
| codec_pad_id | CodecPad | 2148 | 2148 | ok | |
| codec_bos_id | CodecBos | 2149 | 2149 | ok | |
| codec_eos_token_id | CodecEos | 2150 | 2150 | ok | |
| codec_think_id | CodecThink | 2154 | 2154 | ok | |
| codec_nothink_id | CodecNoThink | 2155 | 2155 | ok | |
| codec_think_bos_id | CodecThinkBos | 2156 | 2156 | ok | |
| codec_think_eos_id | CodecThinkEos | 2157 | 2157 | ok | |
| language id base | LanguageIdBase | 2050 (english) | 2050 | ok | |
| language id count / max | LanguageIdCount | observed max 2071 (~22 ids) | 25 (implies 2074) | wrong | verified list ends at portuguese 2071; 25 overshoots. Confirm full set from CustomVoice checkpoint |
| custom-voice speaker ids | CustomVoiceSpeakerIds | Ryan 3061, Serena 3066, Ono-Anna 2873, Sohee 2864 (9 total, scattered) | [2075..2083] | wrong | placeholders; real ids verified from C reference + research doc, different range/spacing |
| talker do_sample | (implicit) | true | true | ok | generation_config.json |
| talker temperature | Temperature | 0.9 | 0.9 | ok | |
| talker top_k | TopK | 50 | 50 | ok | |
| talker top_p | TopP | 1.0 | 1.0 | ok | |
| talker repetition_penalty | RepetitionPenalty | 1.05 | 1.05 | ok | |
| max_new_tokens | MaxNewTokens | 8192 | 2048 | wrong | Base generation_config = 8192; research doc notes "config wins" |
| min_new_tokens | MinNewTokens | 2 | 2 | ok | |
| subtalker/MTP sampling (temp/top_k) | (none, shares talker fields) | greedy in C ref (temp 0/top_k 1); config temp 1.0 rep 1.0 | shares talker block | missing | need a separate MTP sampling block; greedy decode for cb1..15 |
| codec num_quantizers (decoder) | AcousticCodebooks (+1 semantic) | 16 | 15 + 1 | ok | matches 16 |
| semantic_codebook_size | SemanticCodebookSize | 4096 | 4096 | ok | tokenizer config |
| semantic codebook_dim | SemanticCodebookDim | 256 | 256 | ok | encoder codebook_dim 256 |
| acoustic codebook_size | AcousticCodebookSize | 2048 | 2048 | ok | |
| acoustic codebook_dim | AcousticCodebookDim | 512 | 256 | wrong | decoder codebook_dim=512; our XML comment wrongly claims "same 256". Acoustic embed tensor shape mismatch |
| decoder_dim | DecoderInChannels | 1536 | 1536 | ok | |
| codec latent_dim | PreConvOut | 1024 | 1024 | ok | reference latent_dim=1024 maps to pre_conv out |
| vector_quantization_hidden_dimension | LatentDim | 512 | 512 | ok | quantizer output proj width |
| codec hidden_size (transformer) | TransformerDim | 512 | 512 | ok | |
| codec num_hidden_layers | TransformerLayers | 8 | 8 | ok | |
| codec num_attention_heads | TransformerHeads | 16 | 16 | ok | |
| codec head_dim | TransformerHeadDim | 64 | 64 | ok | |
| codec intermediate_size | TransformerFfnDim | 1024 | 1024 | ok | |
| codec rope_theta | TransformerRopeTheta | 10000 | 10000 | ok | |
| codec sliding_window | SlidingWindow | 72 | 72 | ok | |
| layer_scale_initial_scale | LayerScaleInit | 0.01 | 0.01 | ok | |
| codec rms_norm_eps | RmsNormEps | 1e-05 | 1e-6 | wrong | decoder rms_norm_eps=1e-05 (talker is 1e-6) |
| upsample_rates | UpsampleRates | [8,5,4,3] | [8,5,4,3] | ok | |
| upsampling_ratios (ConvNeXt) | ConvNeXtUpsampleRates | [2,2] | [2,2] | ok | total upsample 1920 |
| sampling_rate | SampleRate | 24000 | 24000 | ok | |
| residual_kernel_size | ResidualKernel | 3 (encoder); decoder unspecified | 7 | unverified | reference config lists residual_kernel_size=3 for encoder only; decoder DAC residual kernel not in config. Our 7 is a DAC default, unconfirmed |
| residual dilations | ResidualDilations | not in config | [1,3,9] | unverified | decoder residual dilations absent from config; DAC-style default assumed, confirm from weights |
| pre_conv kernel | PreConvKernel | not in config | 3 | unverified | research doc says pre_conv k3; not a config key |
| decoder out kernel | OutKernel / DecoderInKernel | not in config | 7 | unverified | DAC default k7; not a config key |
| speaker_encoder enc_dim | EmbeddingDim | 2048 | 192 | wrong | real speaker_encoder.fc outputs enc_dim 2048 (verified from real weights); 192 is the SpeechBrain default for the wrong variant |
| speaker_encoder input channels | InputChannels | 128 (real stem conv in) | 80 | wrong | real stem = conv 128->512 k5 (research doc real-weight reconcile) |
| speaker_encoder stem channels | StemChannels | 512 | 512 | ok | blocks.0 conv out 512 |
| speaker_encoder sample_rate | (none) | 24000 | absent | missing | speaker_encoder config sample_rate=24000 not modeled |
| ECAPA ConditioningDim | ConditioningDim | n/a (talker side) | 2048 | extra | not a reference codec/encoder field; coincidentally equals enc_dim. Keep but document |

Action items:
- Set Qwen3TtsVocoderConfig.AcousticCodebookDim to 512 (semantic stays 256) and fix the XML comment that claims 256.
- Set Qwen3TtsVocoderConfig.RmsNormEps to 1e-5f (codec decoder), leaving the talker at 1e-6f.
- Set Qwen3TtsConfig.MaxNewTokens to 8192 to match the Base/CustomVoice generation_config.
- Fix EcapaConfig for the shipped variant: EmbeddingDim 2048 (enc_dim), InputChannels 128, add a stem kernel of 5, and add a SampleRate (24000) field. The current 192/80 SpeechBrain defaults do not match the real speaker_encoder weights.
- Replace the placeholder CustomVoiceSpeakerIds with the verified scattered ids (Ryan 3061, Serena 3066, Ono-Anna 2873, Sohee 2864, plus the remaining 5 from the CustomVoice checkpoint).
- Recheck LanguageIdCount: the verified id list tops out at 2071 (about 22 ids), so 25 is likely too high.
- Add a separate MTP/sub-talker sampling block (greedy: temp 0/top_k 1 per the C reference) instead of reusing the talker sampling fields.
- Add a PositionIdPerSeconds field (=13) for mRoPE position assignment.
- Add presets for the 0.6B talker (hidden 1024 / intermediate 3072), CustomVoice (speaker ids + 8192 tokens), and VoiceDesign variants.
- Verify the decoder residual kernel/dilations (currently 7 and [1,3,9]) against the real decoder weight shapes; the config does not specify them.

<details><summary>Sources consulted</summary>

- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/main/generation_config.json
- https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/raw/main/config.json
- docs/Research/QWEN3_TTS_ARCHITECTURE.md (real-weight reconciliation of Qwen/Qwen3-TTS-12Hz-1.7B-Base model.safetensors + speech_tokenizer/model.safetensors)

</details>

---

## CSM

Reference: Sesame csm-1b, GitHub SesameAILabs/csm (models.py FLAVORS llama-1B backbone + llama-100M depth decoder) plus HF Transformers CsmConfig / CsmDepthDecoderConfig defaults. One variant only (csm-1b); our single preset CsmConfig.V1B maps to it.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| num_codebooks | NumCodebooks | 32 | 8 | wrong | mimi.set_num_codebooks(32) in generator.py; CsmConfig num_codebooks default 32 (HF doc line 354). Core shape mismatch. |
| codebook vocab_size (audio) | AudioVocab | 2051 | 2051 | ok | CsmConfig.vocab_size / CsmDepthDecoderConfig.vocab_size = 2051 (HF doc 356/442). |
| text_vocab_size | TextVocab | 128256 | 128256 | ok | HF doc 358; FLAVORS vocab_size 128_256. |
| backbone hidden_size | Backbone.HiddenSize | 2048 | 2048 | ok | llama-1B embed_dim 2048. |
| backbone num_hidden_layers | Backbone.NumHiddenLayers | 16 | 16 | ok | llama-1B num_layers 16. |
| backbone num_attention_heads | Backbone.NumAttentionHeads | 32 | 32 | ok | llama-1B num_heads 32. |
| backbone num_key_value_heads | Backbone.NumKeyValueHeads | 8 | 8 | ok | llama-1B num_kv_heads 8. |
| backbone intermediate_size | Backbone.IntermediateSize | 8192 | 8192 | ok | llama-1B intermediate_dim 8192. |
| backbone max_position_embeddings | Backbone.MaxPositionEmbeddings | 2048 | 2048 | ok | llama-1B max_seq_len 2048. |
| backbone rope_theta | Backbone.RopeTheta | 500000 | 500000 | ok | llama-1B rope_base 500_000. |
| backbone rms_norm_eps | Backbone.RmsNormEps | 1e-5 | 1e-5 | ok | norm_eps 1e-5. |
| backbone attention_bias / mlp_bias | Backbone.AttentionBias | false | false | ok | Llama has no qkv bias (HF doc 396/400). |
| hidden_act | (hardcoded silu in Qwen2 body) | silu | silu (assumed) | ok | HF doc 370; Llama MLP is SwiGLU/silu. |
| depth hidden_size | Decoder.HiddenSize | 1024 | 1024 | ok | llama-100M embed_dim 1024. |
| depth num_hidden_layers | Decoder.NumHiddenLayers | 4 | 4 | ok | llama-100M num_layers 4. |
| depth num_attention_heads | Decoder.NumAttentionHeads | 8 | 8 | ok | llama-100M num_heads 8. |
| depth num_key_value_heads | Decoder.NumKeyValueHeads | 2 | 2 | ok | llama-100M num_kv_heads 2. |
| depth intermediate_size | Decoder.IntermediateSize | 8192 | 8192 | ok | llama-100M intermediate_dim 8192. |
| depth max_position_embeddings | Decoder.MaxPositionEmbeddings | 33 | 64 | wrong | CsmDepthDecoderConfig.max_position_embeddings default 33 (HF doc 456) = num_codebooks 32 + 1. |
| depth rope_theta | Decoder.RopeTheta | 500000 | 500000 | ok | llama-100M rope_base 500_000. |
| rope_scaling / rope_parameters (Llama3, scale_factor 32) | (none) | scale_factor 32, llama3 type | not modeled | missing | FLAVORS scale_factor=32 for both bodies (models.py); HF rope_parameters (HF doc 394/470). Doc comment already flags this. |
| backbone_hidden_size (depth) | (implicit via Backbone.HiddenSize) | 2048 | 2048 | ok | CsmDepthDecoderConfig.backbone_hidden_size 2048 (HF doc 440); projection lives on outer model. |
| tie_codebooks_embeddings | (none) | true | not modeled | missing | HF doc 404; affects how depth-decoder codebook embeddings are shared with backbone (weight loading). |
| codebook_pad_token_id | (none) | 2050 | not modeled | missing | HF doc 382. |
| codebook_eos_token_id | AudioEosToken (mismatched) | 0 | 2048 | wrong | HF doc 384 says 0; our AudioEosToken=2048 does not match. Verify semantics vs checkpoint. |
| audio_token_id | (none) | 128002 | not modeled | missing | HF doc 390; placeholder id in the text stream. |
| audio_eos_token_id (text stream) | (none) | 128003 | not modeled | missing | HF doc 392. |
| bos_token_id | (none) | 128000 | not modeled | missing | HF doc 386. |
| pad_token_id | (none) | 128002 | not modeled | missing | HF doc 380. |
| head_dim | (derived in body) | hidden/heads (64) | derived | ok | HF doc 402; default hidden_size//num_heads. |
| Mimi sample rate | SampleRate | 24000 | 24000 | unverified | Mimi is 24 kHz by convention; generator.py reads mimi.sample_rate dynamically and did not print the literal. Consistent with Kyutai Mimi. |
| frame samples / frame rate | FrameSamples 1920 (12.5 Hz) | 1920 @ 12.5 Hz | 1920 | ok | 24000/12.5 = 1920; Mimi 12.5 Hz frame rate. |
| Temperature | Temperature | 0.9 (sampling default) | 0.9 | extra | Sampling default lives in generation config, not the model config; acceptable. |
| TopK | TopK | 50 | 50 | extra | Sampling default; reference generator uses topk=50. |
| TopP | TopP | 1.0 | 1.0 | extra | Sampling default; not a model-config field. |

Action items:
- Fix NumCodebooks from 8 to 32 (this is the load-bearing bug; codebook heads, input_ids last dim, and the depth-decoder loop length all depend on it).
- Change Decoder.MaxPositionEmbeddings from 64 to 33 (num_codebooks + 1).
- Add RoPE Llama3 scaling (scale_factor 32 / rope_type llama3) to both Backbone and Decoder, or add a RopeScaling field on Qwen2Config and set it in V1B; remove the doc-comment caveat once modeled.
- Add codebook special-token fields: codebook_pad_token_id (2050) and codebook_eos_token_id (0); reconcile AudioEosToken (currently 2048) against the checkpoint, the reference codebook EOS is 0.
- Add text-stream special tokens: audio_token_id (128002), audio_eos_token_id (128003), bos_token_id (128000), pad_token_id (128002).
- Add tie_codebooks_embeddings (true) so weight loading matches the shared codebook embeddings.
- Confirm Mimi sample_rate literal (24000) from the Mimi checkpoint rather than relying on the dynamic mimi.sample_rate read.

<details><summary>Sources consulted</summary>

- https://github.com/SesameAILabs/csm/blob/main/models.py (FLAVORS llama-1B / llama-100M, ModelArgs)
- https://github.com/SesameAILabs/csm/blob/main/generator.py (mimi.set_num_codebooks(32))
- https://huggingface.co/docs/transformers/en/model_doc/csm (CsmConfig + CsmDepthDecoderConfig defaults)

</details>

---

## Qwen3 LM (audio backbone)

Reference: official Qwen3-TTS config (`Qwen/Qwen3-TTS-12Hz-0.6B-Base/config.json`, `talker_config` + nested `code_predictor_config`), cross-checked against base Qwen3-1.7B / Qwen3-0.6B text LLM configs and the mlx-audio / qwen3-tts.cpp reimplementations. Our config exposes two presets: `Talker1_7B` and `CodePredictor`. The biggest issue: `Talker1_7B` carries the base text-LLM shapes (2048/6144), not the actual TTS talker shapes (1024/3072).

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| talker hidden_size | HiddenSize (Talker1_7B) | 1024 | 2048 | wrong | Real talker is 1024; 2048 is `text_hidden_size`/base-LLM hidden. Source: Qwen3-TTS-12Hz-0.6B-Base config.json talker_config.hidden_size=1024 |
| talker intermediate_size | IntermediateSize (Talker1_7B) | 3072 | 6144 | wrong | Real talker SwiGLU is 3072, not 6144. Same config.json talker_config.intermediate_size=3072 |
| talker num_hidden_layers | NumHiddenLayers (Talker1_7B) | 28 | 28 | ok | matches |
| talker num_attention_heads | NumAttentionHeads | 16 | 16 | ok | |
| talker num_key_value_heads | NumKeyValueHeads | 8 | 8 | ok | GQA group 2 |
| head_dim | HeadDim | 128 | 128 | ok | both talker + predictor |
| rope_theta | RopeTheta | 1000000 | 1000000 | ok | |
| rms_norm_eps | RmsNormEps | 1e-6 | 1e-6 | ok | |
| talker max_position_embeddings | MaxPositionEmbeddings (Talker) | 32768 | 32768 (default) | ok | talker is 32768 |
| code_predictor hidden_size | HiddenSize (CodePredictor) | 1024 | 1024 | ok | |
| code_predictor num_hidden_layers | NumHiddenLayers (CodePredictor) | 5 | 5 | ok | |
| code_predictor intermediate_size | IntermediateSize (CodePredictor) | 3072 | 3072 | ok | |
| code_predictor max_position_embeddings | MaxPositionEmbeddings (CodePredictor) | 65536 | 32768 (default) | wrong | predictor uses 65536; our default 32768 understates it |
| talker vocab_size | (none) | 3072 | absent | missing | embedding/lm_head shape; codec token vocab |
| code_predictor vocab_size | (none) | 2048 | absent | missing | per-codebook output vocab |
| num_code_groups | (none) | 16 | absent | missing | number of codebooks (multi-codebook heads) |
| text_vocab_size | (none) | 151936 | absent | missing | text input embedding vocab |
| text_hidden_size | (none) | 2048 | absent | missing | text-embedding projection dim into talker |
| attention_bias | (none) | false | absent (assumed false) | missing | Qwen3 drops qkv bias; we likely hardcode no-bias but no field/assert |
| hidden_act | (none) | "silu" | absent (assumed SwiGLU/silu) | missing | almost certainly hardcoded silu; add for completeness |
| attention_dropout | (none) | 0 | absent | ok-ish | inference-disabled, excluded |
| codec_bos_id / codec_eos_token_id | (none) | 2149 / 2150 | absent | missing | generation control tokens |
| tts_bos/eos/pad + im_start/im_end + assistant ids | (none) | 151672/151673/151671/151644/151645/77091 | absent | missing | prompt/template tokens (may live in tokenizer/pipeline, verify) |
| codec_language_id map | (none) | 10-lang map | absent | missing | per-language codec id selection |
| q_norm / k_norm (per-head RMSNorm) | (model code, not config) | enabled | docstring-noted | unverified | Qwen3 hallmark; verify it is wired in Qwen3 block, not just documented |
| sliding_window / use_sliding_window | (none) | null / false | absent | ok | not used (full attention); fine to omit |
| tie_word_embeddings | (none) | true (base LLM) | absent | unverified | base Qwen3 ties; talker codec head likely untied, verify against checkpoint |

Action items:
- Fix `Talker1_7B`: set `HiddenSize = 1024` and `IntermediateSize = 3072` (current 2048/6144 are the wrong/base-LLM values). Consider renaming the preset to `Talker0_6B` or `Talker` since the real talker is the 0.6B-base talker, and update the XML doc that says hidden 2048.
- Fix `CodePredictor`: set `MaxPositionEmbeddings = 65536` explicitly (do not inherit the 32768 default).
- Add config fields: `VocabSize` (talker 3072 / predictor 2048), `NumCodeGroups` (16), `TextVocabSize` (151936), `TextHiddenSize` (2048).
- Add `AttentionBias` (false) and `HiddenAct` (silu) fields or assertions to lock the architecture, and confirm per-head `q_norm`/`k_norm` is actually implemented (not just documented).
- Add codec/special token id fields (or confirm they are owned by the tokenizer/pipeline): codec_bos_id 2149, codec_eos_token_id 2150, tts_bos/eos/pad, codec_language_id map.
- Add a preset for the larger/Flash TTS checkpoint (tts_model_size other than 0b6) if that variant is in scope.

<details><summary>Sources consulted</summary>

- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
- https://raw.githubusercontent.com/Blaizzy/mlx-audio/main/mlx_audio/tts/models/qwen3_tts/config.py
- https://deepwiki.com/predict-woo/qwen3-tts.cpp/2.2-model-setup

</details>

---

## RVC

Reference: RVC-Project/Retrieval-based-Voice-Conversion-WebUI (configs/v1 and v2 JSON, infer/lib/rmvpe.py, infer/modules/vc/pipeline.py), fetched 2026-06-28. Variants we cover: V2_40k, V2_48k, plus a single RMVPE Default. Reference ships six synthesizer variants (v1/v2 x 32k/40k/48k; v2 40k is generated at runtime) and one RMVPE model.

### Synthesizer (SynthesizerTrnMs768NSFsid / Ms256NSFsid)

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| inter_channels | Core.InterChannels | 192 | 192 | ok | all variants |
| hidden_channels | Core.HiddenChannels | 192 | 192 | ok | |
| filter_channels | Core.FilterChannels | 768 | 768 | ok | |
| n_heads | Core.NumHeads | 2 | 2 | ok | |
| n_layers | Core.NumEncoderLayers | 6 | 6 | ok | |
| kernel_size | Core.EncoderKernelSize | 3 | 3 | ok | |
| resblock | Core.ResBlock | "1" | "1" | ok | v1 and v2 both resblock 1 |
| resblock_kernel_sizes | Core.ResBlockKernelSizes | [3,7,11] | [3,7,11] | ok | |
| resblock_dilation_sizes | Core.ResBlockDilations | [[1,3,5]x3] | [[1,3,5]x3] | ok | |
| upsample_rates (v2 40k) | Core.UpsampleRates | [10,10,2,2] | [10,10,2,2] | ok | hop 400 |
| upsample_kernel_sizes (v2 40k) | Core.UpsampleKernelSizes | [16,16,4,4] | [16,16,4,4] | ok | |
| upsample_rates (v2 48k) | Core.UpsampleRates | [12,10,2,2] | [12,10,2,2] | ok | hop 480, V2_48k preset correct |
| upsample_kernel_sizes (v2 48k) | Core.UpsampleKernelSizes | [24,20,4,4] | [24,20,4,4] | ok | |
| upsample_initial_channel | Core.UpsampleInitialChannel | 512 | 512 | ok | |
| gin_channels | Core.GinChannels | 256 | 256 | ok | |
| spk_embed_dim | SpkEmbedDim | 109 | 109 | ok | |
| sampling_rate (v2 40k/48k) | Core.SampleRate | 40000 / 48000 | 40000 / 48000 | ok | |
| ContentVec/HuBERT dim (v2) | ContentDim | 768 | 768 | ok | v1 = 256, no preset (see below) |
| f0_min | F0Min | 50 | 50 | ok | pipeline.py |
| f0_max | F0Max | 1100 | 1100 | ok | pipeline.py |
| f0_bin (coarse) | PitchBins | 256 | 256 | ok | 255 scaling, 0 reserved |
| upsample (v2 32k) | (none) | [10,8,2,2] | n/a | missing | no V2_32k preset |
| upsample (v1 48k) | (none) | [10,6,2,2,2] k [16,16,4,4,4] | n/a | missing | 5-stage; no V1_48k preset |
| upsample (v1 32k) | (none) | [10,4,2,2,2] k [16,16,4,4,4] | n/a | missing | 5-stage; no V1_32k preset |
| upsample (v1 40k) | (none) | [10,10,2,2] | n/a | missing | needs ContentDim 256; no V1_40k preset |
| filter_length / win_length | (none) | 2048 (40k/48k) / 1024 (32k) | n/a | missing | enc_q posterior is inference-unused, so training-only for VC (low priority) |
| n_mel_channels | (none) | 125/128/80 | n/a | missing | enc_q only; inference-unused |
| mel_fmin / mel_fmax | (none) | 0.0 / null | n/a | missing | enc_q only; training-only |

NSF inference scales (noise scale 0.66666 applied as z_p = m_p + exp(logs_p)*randn*0.66666) are hardcoded in the pipeline/research doc and match the reference.

### RMVPE pitch extractor (E2E + MelSpectrogram)

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| sampling_rate | SampleRate | 16000 | 16000 | ok | MelSpectrogram(...,16000,...) |
| n_mel_channels | MelBins | 128 | 128 | ok | |
| hop_length | HopLength | 160 | 160 | ok | |
| win_length | WinLength | 1024 | 2048 | wrong | MelSpectrogram(is_half,128,16000,1024,160,None,30,8000) |
| n_fft | NFft | 1024 (= win_length) | 2048 | wrong | rmvpe.py n_fft defaults to win_length=1024 |
| mel_fmin | Fmin | 30 | 30 | ok | |
| mel_fmax | Fmax | 8000 | 8000 | ok | |
| clamp / log floor | MelConfig LogFloor | 1e-5 | 1e-5 | ok | MelSpectrogram clamp=1e-5 |
| en_out_channels | StemChannels | 16 | 16 | ok | E2E en_out_channels=16 |
| encoder channels | EncoderChannels | 16,32,64,128,256 (5 stages, out 512) | [32,64,128] (3 stages) | wrong | en_de_layers=5; our compact stand-in cannot load rmvpe.pt |
| en_de_layers | (none) | 5 | n/a | missing | add EnDeLayers=5 |
| n_blocks (ResEncoderBlock depth) | (none) | 4 | n/a | missing | E2E(4,...); add NBlocks=4 |
| inter_layers | IntermediateBlocks | 4 | 2 | wrong | DeepUnet inter_layers=4 |
| kernel_size (pool) | (none) | (2,2) | n/a | missing | E2E(...,(2,2)); add KernelSize |
| n_gru | (none) | 1 | n/a | missing | E2E(4,1,...); BiGRU layer count |
| gru hidden | GruHidden | 256 | 256 | ok | BiGRU(3*128,256,n_gru) |
| output bins | NumBins | 360 | 360 | ok | Linear(512,360) |
| cents per bin | CentsPerBin | 20 | 20 | ok | cents_mapping = 20*arange(360)+1997.3794 |
| first bin freq | FirstFreqHz | 31.70 Hz | 32.70 Hz | wrong | 10*2^(1997.3794/1200) = 31.70 Hz, not C1 32.70 |
| local average window | LocalAverageWindow | 4 (pad 4,4 -> 9-wide) | 4 | ok | to_local_average_cents |
| decode threshold (thred) | VoicingThreshold | 0.03 | 0.3 | wrong | pipeline.py and rmvpe __main__ call decode(thred=0.03) |
| frame pad multiple | (none) | 32 | n/a | n/a | mel2hidden pads to multiple of 32; derivable, low priority |
| normalization type | NormGroups (GroupNorm) | BatchNorm2d | GroupNorm groups=8 | extra | ConvBlockRes/Encoder use nn.BatchNorm2d, not GroupNorm |

### Action items
- Fix RMVPE Mel front-end: set NFft and WinLength to 1024 (real MelSpectrogram uses win_length 1024 with n_fft defaulting to it).
- Fix RMVPE VoicingThreshold from 0.3 to 0.03 (10x error that would silence most frames).
- Rebuild the RMVPE UNet config to match rmvpe.pt: encoder channel schedule 16,32,64,128,256 (out 512) over 5 stages; add EnDeLayers=5, NBlocks=4, InterLayers=4, KernelSize=(2,2), NGru=1; this is required to load the real checkpoint.
- Replace GroupNorm/NormGroups with BatchNorm2d to match the reference normalization.
- Correct FirstFreqHz to 31.70 Hz (derived from cents base 1997.3794), not 32.70.
- Add synthesizer presets for v2 32k ([10,8,2,2] k [20,16,4,4]), v1 40k (ContentDim 256), v1 48k ([10,6,2,2,2] k [16,16,4,4,4], ContentDim 256), and v1 32k ([10,4,2,2,2] k [16,16,4,4,4], ContentDim 256); v1 variants use the 256-d HuBERT tap and the Ms256 synthesizer.

<details><summary>Sources consulted</summary>

- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/configs/v1/32k.json
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/configs/v1/40k.json
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/configs/v1/48k.json
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/configs/v2/32k.json
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/configs/v2/48k.json
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/lib/rmvpe.py
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/infer/modules/vc/pipeline.py

</details>

---

## StyleTTS2

Reference: yl4579/StyleTTS2, model_params + diffusion + preprocess from Configs/config_libritts.yml (multispeaker LibriTTS, decoder type 'hifigan') and Configs/config.yml (single-speaker LJSpeech, decoder type 'istftnet'); inference defaults from the Demo/Inference_*.ipynb notebooks. Our config wraps the shared KokoroConfig backbone and exposes a Karras + ADPM2 style sampler.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| hidden_dim | Backbone.HiddenDim | 512 | 512 | ok | |
| n_token | Backbone.NumTokens | 178 | 178 | ok | |
| style_dim | Backbone.StyleDim / StyleDim | 128 (full vec 256) | 128 / 256 | ok | StyleDim => StyleDim*2 = 256 matches acoustic+prosodic concat |
| n_layer | Backbone.TextEncoderNumLayers | 3 | 3 | ok | |
| n_mels | Backbone.NumMels | 80 | 80 | ok | |
| max_dur | Backbone.MaxDuration | 50 | 50 | ok | |
| max_conv_dim | Backbone.MaxConvDim | 512 | 512 | ok | |
| dim_in | Backbone.DimIn | 64 | 64 | ok | |
| multispeaker | MultiSpeaker | true (LibriTTS) / false (LJSpeech) | true / false | ok | both presets correct |
| preprocess sr | Backbone.SampleRate | 24000 | 24000 | ok | |
| diffusion.dist.sigma_data | SigmaData | 0.2 (estimated 0.199) | 0.2 | ok | HF-resolved estimate is 0.199; nominal 0.2 |
| sampler sigma_min | SigmaMin | 0.0001 (KarrasSchedule) | 1e-4 | ok | notebook DiffusionSampler |
| sampler sigma_max | SigmaMax | 3.0 | 3.0 | ok | |
| sampler rho | SigmaRho | 9.0 | 9.0 | ok | |
| diffusion_steps (inference) | DiffusionSteps | 5 | 5 | ok | |
| decoder.type | (none, forced iSTFTNet) | hifigan (LibriTTS) / istftnet (LJSpeech) | iSTFTNet for both | wrong | Backbone.IStftNet hardcodes the Kokoro iSTFTNet decoder for the LibriTTS preset too, but LibriTTS uses HiFi-GAN. No decoder-type selector exists. (config_libritts.yml decoder.type: hifigan) |
| decoder.upsample_rates / kernel_sizes | Backbone.IStftNet.UpsampleRates/KernelSizes | LibriTTS [10,5,3,2]/[20,10,6,4]; LJSpeech [10,6]/[20,12] | [10,6]/[20,12] for both | wrong | LibriTTS upsample chain wrong (4-stage HiFi-GAN vs 2-stage iSTFTNet) |
| diffusion.transformer.num_layers | (none) | 3 | not present | missing | style-diffusion StyleTransformer1d depth; StyleTransformerLayer.cs has no count field |
| diffusion.transformer.num_heads | (none) | 8 | not present | missing | attention head count of style estimator |
| diffusion.transformer.head_features | (none) | 64 | not present | missing | per-head dim of style estimator |
| diffusion.transformer.multiplier | (none) | 2 | not present | missing | FFN expansion of style estimator |
| inference alpha | (none) | 0.3 (clone) / 0.7 (long-form) | not present | missing | blends sampled style vs ref-encoded acoustic style; shapes the decoder style |
| inference beta | (none) | 0.7 | not present | missing | blends sampled style vs ref-encoded prosodic style; shapes the predictor style |
| diffusion.dist.mean | (none) | -3.0 | not present | missing | log-sigma sampling mean (used if estimate_sigma_data drives schedule) |
| diffusion.dist.std | (none) | 1.0 | not present | missing | log-sigma sampling std |
| embedding_scale (LJSpeech) | LjSpeech.EmbeddingScale | 1 (notebook default) | 1.5 | wrong | canonical default is 1; 1.5/2.0 are demo expressiveness tweaks |
| diffusion.embedding_mask_proba | (none) | 0.1 | not present | unverified | CFG dropout proba; training-only, only matters at inference if uncond path requires the trained null embedding (excluded as train-only) |
| diffusion.dist.estimate_sigma_data | (none) | true | not present | unverified | training-time sigma_data estimation flag; affects the trained sigma_data (0.199) rather than an inference tensor |
| slm.* (wavlm-base-plus, 13 layers) | (none) | discriminator | not present | extra/skip | SLM is the adversarial discriminator, training-only, no inference tensors |
| EmbeddingScale (LibriTTS) | EmbeddingScale | 1 | 1.0 | extra | our doc frames it as (1-s)*uncond + s*cond CFG; value matches default |
| SigmaData as fixed field | SigmaData | estimated at train time | fixed 0.2 | extra | acceptable inference fixed value |

Action items:
- Add a decoder-type selector so the LibriTTS preset uses the HiFi-GAN decoder (upsample_rates [10,5,3,2], upsample_kernel_sizes [20,10,6,4], 4 upsample stages) instead of the Kokoro iSTFTNet decoder; keep iSTFTNet only for the LJSpeech preset. This is the highest-impact fix (LibriTTS weights will not map onto the current iSTFTNet head).
- Add diffusion style-transformer shape fields: TransformerNumLayers=3, TransformerNumHeads=8, TransformerHeadFeatures=64, TransformerMultiplier=2, and wire them into StyleTransformer1d / StyleTransformerLayer instead of hardcoding.
- Add inference style-blend knobs Alpha (default 0.3) and Beta (default 0.7) that interpolate the diffusion-sampled style against the reference-encoded style halves.
- Correct LjSpeech.EmbeddingScale from 1.5 to 1 (the reference default) or document it explicitly as a non-default expressiveness override.
- Optionally record diffusion dist mean=-3.0 / std=1.0 and the trained sigma_data 0.199 for completeness; embedding_mask_proba and estimate_sigma_data can stay omitted as training-only.

<details><summary>Sources consulted</summary>

- https://raw.githubusercontent.com/yl4579/StyleTTS2/main/Configs/config_libritts.yml
- https://raw.githubusercontent.com/yl4579/StyleTTS2/main/Configs/config.yml
- https://huggingface.co/yl4579/StyleTTS2-LibriTTS/raw/main/Models/LibriTTS/config.yml
- https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LibriTTS.ipynb
- https://github.com/yl4579/StyleTTS2/blob/main/Demo/Inference_LJSpeech.ipynb

</details>

---

## Resemble-Enhance

Reference: resemble-ai/resemble-enhance (GitHub main). Single shipped checkpoint (enhancer stage including a denoiser UNet, IRMAE latent AE, latent OT-CFM with a WaveNet velocity net, and a UnivNet vocoder); no small/medium/large or per-sample-rate variants exist, so our single Default preset is correct. The base front-end runs at 44.1 kHz, n_fft=2048, hop=420, num_mels=128. Sources: hparams.py, enhancer/hparams.py, enhancer/lcfm/{cfm,wn,irmae}.py, denoiser/unet.py.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| wav_rate | SampleRate | 44100 | 44100 | ok | hparams.py |
| num_mels | NMels | 128 | 128 | ok | hparams.py |
| n_fft | NFft | 2048 | 2048 | ok | hparams.py |
| hop_size | HopLength | 420 | 420 | ok | hparams.py |
| win_size | (none) | 2048 | n/a | missing | hparams.py; equals n_fft here but should be explicit |
| stft_magnitude_min | (none) | 1e-4 | n/a | missing | hparams.py; mel log-floor, affects features |
| preemphasis | (none) | 0.97 | n/a | missing | hparams.py; pre-emphasis filter on input wav, changes mel |
| lcfm_latent_dim | LatentDim | 64 | 64 | ok | enhancer/hparams.py |
| lcfm_z_scale | LatentScale | 5 | 5 | ok | enhancer/hparams.py |
| WN n_layers | WnLayers | 30 | 30 | ok | enhancer/lcfm/wn.py |
| WN hidden_dim | WnHidden | 512 | 512 | ok | wn.py |
| WN kernel_size | WnKernel | 3 | 3 | ok | wn.py |
| WN dilation_cycle | WnDilationCycle | 5 | 5 | ok | wn.py |
| time_emb_dim | TimeEmbDim | 128 | 128 | ok | cfm.py |
| IRMAE hidden_dim | AeHidden | 1024 | 1024 | ok | irmae.py |
| IRMAE ResBlock count | AeResBlocks | 4 | 4 | ok | irmae.py (range(4)) |
| ResBlock dilations | AeDilations | [1,2,4,8] | [1,2,4,8] | ok | irmae.py default |
| IRMAE num_irms | (none) | 4 | n/a | missing | irmae.py; the 4 latent linear layers that define the IRMAE bottleneck; distinct from AeResBlocks |
| IRMAE GroupNorm groups | (none) | 32 | n/a | missing | irmae.py uses GroupNorm(32); only NormEps exposed |
| GroupNorm eps | NormEps | 1e-5 | 1e-5 | ok | torch GroupNorm default (not set in code) |
| cfm sigma (perturb) | (none) | 1e-4 | n/a | missing | cfm.py hardcoded; OT-CFM target noise level |
| cfm_solver_method | Solver | midpoint | midpoint | ok | hparams.py |
| cfm_time_mapping_divisor | TimeMappingDivisor | 4 | 4 | ok | hparams.py |
| cfm_solver_nfe / --nfe | Nfe | 64 (CLI) / 32 (enhance fn) | 64 | ok | hparams.py + __main__.py default 64; note enhance() helper uses 32 |
| lambd | Lambd | 1.0 (CLI) / 0.5 (enhance fn) / 0.0 (init) | 0.5 | wrong | __main__.py default is 1.0, enhance() helper is 0.5; we match the helper, not the CLI users see |
| tau | Tau | 0.5 | 0.5 | ok | __main__.py / inference.py |
| univnet_nc | (none) | 96 | n/a | missing | enhancer/hparams.py; UnivNet base channel count |
| vocoder_extra_dim | (none) | 32 | n/a | missing | enhancer/hparams.py; extra conditioning dim into vocoder |
| force_gaussian_prior | (none) | False | n/a | missing | enhancer/hparams.py; toggles prior type, affects sampling |
| denoiser UNet hidden_dim | (none) | 16 | n/a | missing | denoiser/unet.py |
| denoiser num_blocks / num_middle_blocks | (none) | 4 / 2 | n/a | missing | denoiser/unet.py; whole denoiser path absent from config |
| lcfm_training_mode | (none) | "ae" | n/a | unverified | enhancer/hparams.py; training-only selector, likely inference-irrelevant but not confirmed against our solver path |

### Action items
- Add denoiser fields (UNet hidden_dim=16, num_blocks=4, num_middle_blocks=2, kernel=3) or a nested DenoiserConfig; the config currently has no denoiser at all despite the doc comment.
- Add UnivNet vocoder params UnivNetNc=96 and VocoderExtraDim=32.
- Add IRMAE NumIrms=4 (the 4 latent linear layers) and the GroupNorm group count (32); do not conflate with AeResBlocks.
- Add CFM perturbation Sigma=1e-4.
- Add front-end WinSize=2048, StftMagnitudeMin=1e-4, Preemphasis=0.97.
- Reconcile the Lambd default: CLI users get 1.0; pick 1.0 to match the shipped CLI, or document why 0.5 (enhance() helper) is chosen.
- Add ForceGaussianPrior=false for prior-type parity.
- Confirm whether the enhance() helper nfe=32 vs CLI nfe=64 matters for our default; we currently use 64 (CLI), which is fine if that is the intended entry point."

<details><summary>Sources consulted</summary>

- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/hparams.py
- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/enhancer/hparams.py
- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/enhancer/lcfm/cfm.py
- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/enhancer/lcfm/wn.py
- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/enhancer/lcfm/irmae.py
- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/enhancer/enhancer.py
- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/enhancer/inference.py
- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/enhancer/__main__.py
- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/denoiser/unet.py
- https://raw.githubusercontent.com/resemble-ai/resemble-enhance/main/resemble_enhance/denoiser/hparams.py

</details>

---

## SNAC codec

Reference: hubertsiuzdak/snac (GitHub hubertsiuzdak/snac); three published checkpoints snac_24khz, snac_32khz, snac_44khz. Our config exposes Snac24kHz, Snac32kHz, Snac44kHz. The 24 kHz preset is correct; the 32 kHz and 44 kHz presets are substantially wrong, and three architecture-affecting fields (attn_window_size, noise, depthwise) are missing entirely.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| sampling_rate | SampleRate | 24000 / 32000 / 44100 | 24000 / 32000 / 44100 | ok | Matches per variant. |
| encoder_dim | EncoderDim | 48 (24k), 64 (32k), 64 (44k) | 48, 48, 64 | wrong | 32 kHz must be 64, not the inherited default 48. See snac_32khz config.json. |
| decoder_dim | DecoderDim | 1024 (24k), 1536 (32k), 1536 (44k) | 1024, 1024, 1536 | wrong | 32 kHz must be 1536, not inherited 1024. snac_32khz config.json. |
| encoder_rates | EncoderRates | [2,4,8,8] (24k), [2,3,8,8] (32k), [2,3,8,8] (44k) | [2,4,8,8], [2,4,8,8], [3,3,7,7] | wrong | 32k should be [2,3,8,8]; 44k should be [2,3,8,8] (we used the library source default [3,3,7,7], not the checkpoint). |
| decoder_rates | DecoderRates | [8,8,4,2] (24k), [8,8,3,2] (32k), [8,8,3,2] (44k) | [8,8,4,2], [8,8,4,2], [7,7,3,3] | wrong | 32k should be [8,8,3,2]; 44k should be [8,8,3,2]. |
| vq_strides | VqStrides | [4,2,1] (24k), [8,4,2,1] (32k), [8,4,2,1] (44k) | [4,2,1], [4,2,1,1], [8,4,2] | wrong | 32k should be [8,4,2,1]; 44k should be [8,4,2,1] (4 codebooks, not 3). |
| (n_codebooks = len(vq_strides)) | NCodebooks | 3 / 4 / 4 | 3 / 4 / 3 | wrong | 44 kHz has 4 codebooks (len of [8,4,2,1]); our preset sets NCodebooks 3. Reference derives count from vq_strides length, no separate field. |
| codebook_size | CodebookSize | 4096 | 4096 | ok | All variants. |
| codebook_dim | CodebookDim | 8 | 8 | ok | All variants. |
| attn_window_size | (none) | null (24k), 32 (32k), 32 (44k) | absent | missing | When non-null, inserts LocalMHA windowed attention (dim_head 64, 16 heads at dim 1024, rotary pos emb) in encoder bottleneck and decoder. Architecture-affecting; must add an int? field. attention.py. |
| noise | (none) | true (all) | absent | missing | DecoderBlock adds a NoiseBlock per upsample when true (extra learned params). layers.py DecoderBlock. Must add bool, default true. |
| depthwise | (none) | true (all) | absent | missing | Convs use groups=channels (depthwise separable) when true, changing conv weight shapes. layers.py. Must add bool, default true. |
| latent_dim | LatentDim (derived) | None -> encoder_dim * 2^len(encoder_rates) | derived EncoderDim * 2^len(EncoderRates) | ok | Same derivation; reference allows explicit override (always None in published configs). |
| residual kernel = 7 | ResidualKernelSize | 7 (hardcoded) | 7 | ok | ResidualUnit kernel default 7. |
| stem/final kernel = 7 | StemKernelSize / DecoderFinalKernelSize | 7 (hardcoded) | 7 | ok | Encoder/Decoder kernel_size=7, padding=3. |
| residual dilations [1,3,9] | ResidualDilations | [1,3,9] (hardcoded) | [1,3,9] | ok | EncoderBlock/DecoderBlock hardcode dilations 1,3,9. |
| channels (mono) | Channels | 1 (WNConv1d input 1) | 1 | extra (ok) | Reference hardcodes mono input; our field matches, no reference config key. |
| (none) | DecoderFinalKernelSize | n/a | 7 | extra | Splits the single reference kernel into multiple fields; harmless, value correct. |
| (none) | ResidualKernelSize / StemKernelSize | n/a | 7 | extra | Same: convenience splits of one hardcoded 7. |

### Action items
- Fix Snac32kHz: set EncoderDim 64, DecoderDim 1536, EncoderRates [2,3,8,8], DecoderRates [8,8,3,2], VqStrides [8,4,2,1], NCodebooks 4, attn_window_size 32.
- Fix Snac44kHz: set EncoderRates [2,3,8,8], DecoderRates [8,8,3,2], VqStrides [8,4,2,1], NCodebooks 4, attn_window_size 32 (keep EncoderDim 64, DecoderDim 1536). Do not use [3,3,7,7]/[7,7,3,3]; those are the library constructor defaults, not the published checkpoint.
- Add AttnWindowSize (int?) field: null for 24 kHz, 32 for 32 kHz and 44 kHz, and wire the LocalMHA block (dim_head 64, rotary pos emb) into the encoder/decoder when non-null.
- Add Noise (bool, default true) and wire NoiseBlock into DecoderBlock.
- Add Depthwise (bool, default true) and apply groups=channels to the relevant convs.
- After adding these, re-derive NCodebooks from VqStrides length to avoid the 44 kHz count drift.

<details><summary>Sources consulted</summary>

- https://huggingface.co/hubertsiuzdak/snac_24khz/raw/main/config.json
- https://huggingface.co/hubertsiuzdak/snac_32khz/raw/main/config.json
- https://huggingface.co/hubertsiuzdak/snac_44khz/raw/main/config.json
- https://raw.githubusercontent.com/hubertsiuzdak/snac/main/snac/snac.py
- https://raw.githubusercontent.com/hubertsiuzdak/snac/main/snac/layers.py
- https://raw.githubusercontent.com/hubertsiuzdak/snac/main/snac/attention.py

</details>

---

## VITS

Reference: jaywalnut310/vits (LJS + VCTK base configs) and HuggingFace `VitsConfig` (facebook/mms-tts-eng), with Piper (rhasspy/piper `piper_train/vits/config.py`) for the medium/high decoder presets. Our config records the SynthesizerTrn arch and exposes two presets (PiperMedium, PiperHigh) plus a default record. The default record reproduces the Piper medium/low decoder, not the canonical LJS/MMS decoder.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| hidden_size / hidden_channels | HiddenChannels | 192 | 192 | ok | |
| inter_channels / flow_size | InterChannels | 192 | 192 | ok | |
| ffn_dim / filter_channels | FilterChannels | 768 | 768 | ok | |
| num_attention_heads / n_heads | NumHeads | 2 | 2 | ok | |
| num_hidden_layers / n_layers | NumEncoderLayers | 6 | 6 | ok | |
| ffn_kernel_size / kernel_size | EncoderKernelSize | 3 | 3 | ok | |
| window_size | WindowSize | 4 | 4 | ok | rel-pos clip, HF VitsConfig default 4 |
| use_stochastic_duration_prediction / use_sdp | UseSdp | true | true | ok | jaywalnut310 LJS sets use_sdp true, Piper config.py default true |
| duration_predictor_filter_channels | DpFilterChannels | 256 | 256 | ok | |
| duration_predictor_kernel_size | DpKernelSize | 3 | 3 | ok | |
| duration_predictor_num_flows | SdpFlows | 4 | 4 | ok | |
| duration_predictor_flow_bins | (none) | 10 | (hardcoded?) | missing | SDP rational-quadratic spline num_bins; HF default 10. Affects spline shapes every inference. https://huggingface.co/facebook/mms-tts-eng/raw/main/config.json |
| duration_predictor_tail_bound | (none) | 5.0 | (hardcoded?) | missing | SDP spline tail bound; HF default 5.0 |
| depth_separable_channels | (none) | 2 | (hardcoded?) | missing | SDP DDSConv hidden multiplier, HF default 2 |
| depth_separable_num_layers | (none) | 3 | (hardcoded?) | missing | SDP DDSConv depth, HF default 3 |
| prior_encoder_num_flows | FlowFlows | 4 | 4 | ok | HF prior_encoder_num_flows=4 == our coupling+flip count |
| prior_encoder_num_wavenet_layers | FlowLayers | 4 | 4 | ok | HF=4 WN layers per coupling |
| wavenet_kernel_size | FlowKernelSize | 5 | 5 | ok | |
| wavenet_dilation_rate | FlowDilationRate | 1 | 1 | ok | |
| posterior_encoder_num_wavenet_layers / n_layers_q | (none) | 16 (HF) / 3 (jaywalnut310) | (none) | missing | Posterior encoder depth. Not used in pure TTS decode, but the doc comment claims this config backs OpenVoice and GPT-SoVITS SoVITS which use the encode (VC) path. Note HF=16 vs jaywalnut310 n_layers_q=3 differ. https://raw.githubusercontent.com/jaywalnut310/vits/main/configs/ljs_base.json |
| resblock | ResBlock | "1" (LJS/MMS/VCTK), "2" (Piper low/medium) | "2" default, "1" in PiperHigh | wrong (default) | Default record uses "2"; canonical VITS uses "1". Correct for Piper medium, wrong for LJS/MMS reference. |
| resblock_kernel_sizes | ResBlockKernelSizes | [3,7,11] (ref) | [3,5,7] default, [3,7,11] PiperHigh | wrong (default) | Same root cause as resblock |
| resblock_dilation_sizes | ResBlockDilations | [[1,3,5],[1,3,5],[1,3,5]] (ref) | [[1,2],[2,6],[3,12]] default | ok (Piper-medium) / mismatch vs LJS | Matches Piper medium/low; differs from canonical VITS. Captured under the wrong default rows above. |
| upsample_rates | UpsampleRates | [8,8,2,2] (LJS/MMS/VCTK) | [8,8,4] default, [8,8,2,2] PiperHigh | ok (Piper) | Default is Piper medium hop 256 via [8,8,4]; reference LJS also hop 256 via [8,8,2,2]. No preset reproduces the reference 4-stage decoder at 256ch. |
| upsample_kernel_sizes | UpsampleKernelSizes | [16,16,4,4] (ref) | [16,16,8] default | ok (Piper) | matches our upsample_rates choice |
| upsample_initial_channel | UpsampleInitialChannel | 512 (LJS/MMS/VCTK) | 256 default, 512 PiperHigh | ok (Piper-medium) | Piper medium uses 256; reference uses 512 |
| use_spectral_norm | (none) | false | (hardcoded?) | missing | Decoder/discriminator norm flag; inference uses generator only, low risk but unrepresented |
| leaky_relu_slope | (none) | 0.1 | (hardcoded?) | unverified | HiFi-GAN leaky ReLU slope, HF default 0.1; likely hardcoded in decoder, not confirmed from .cs |
| gin_channels | GinChannels | 0 (LJS) / 256 (VCTK) | 0 | ok | single-speaker default correct; no multi-speaker preset sets 256 |
| num_speakers / n_speakers | NumSpeakers | 1 (LJS/MMS) / 109 (VCTK) | 1 | ok | per-checkpoint |
| speaker_embedding_size | GinChannels | 0 (single) | 0 | ok | HF aliases gin_channels as speaker_embedding_size |
| sampling_rate | SampleRate | 22050 (LJS/Piper) / 16000 (MMS) | 22050 | ok | MMS 16kHz has no preset |
| vocab_size / num_vocab | NumVocab | 38 (MMS) / ~178 (LJS) / 256 (Piper) | 256 | ok | per-checkpoint; default = Piper |
| noise_scale | NoiseScale | 0.667 | 0.667 | ok | |
| noise_scale_duration / noise_w | NoiseScaleW | 0.8 | 0.8 | ok | |
| speaking_rate / length_scale | LengthScale | 1.0 | 1.0 | ok | |
| hop_length | HopLength (computed) | 256 | 256 (8*8*4) | ok | derived from upsample_rates |
| layer_norm_eps | (none) | 1e-05 | (hardcoded?) | unverified | HF default 1e-5; LayerNorm eps not a field, presumed backend default |
| (n/a) | SdpPrefix | n/a | "sdp"/"dp" | extra | C# state-dict prefix helper, no reference param basis; benign |

### Action items
- Add SDP spline fields: `DurationPredictorFlowBins` (default 10) and `DurationPredictorTailBound` (default 5.0), wire them into the stochastic duration predictor instead of hardcoding.
- Add `DepthSeparableChannels` (default 2) and `DepthSeparableNumLayers` (default 3) for the SDP DDSConv stack.
- Add a posterior encoder depth field (`PosteriorEncoderNumWavenetLayers`, default 16 for HF / 3 for jaywalnut310) since OpenVoice and GPT-SoVITS use the encode path; pick the value per checkpoint, do not hardcode.
- Add a canonical reference preset (e.g. `LjsBase`: resblock "1", ResBlockKernelSizes [3,7,11], ResBlockDilations [[1,3,5]x3], UpsampleRates [8,8,2,2], UpsampleKernelSizes [16,16,4,4], UpsampleInitialChannel 512, NumVocab per phoneme set, SampleRate 22050) so the default-shaped Piper config is not the only option.
- Add an `MmsTts` preset (SampleRate 16000, NumVocab 38, resblock "1", [8,8,2,2], 512ch, SDP).
- Add a multi-speaker preset (e.g. `VctkBase`: GinChannels 256, NumSpeakers 109, resblock "1", [8,8,2,2], 512ch) since GinChannels currently only ever defaults to 0.
- Add `UseSpectralNorm` (default false) if the decoder norm is configurable; otherwise document that it is fixed for inference.
- Confirm `leaky_relu_slope` (0.1) and `layer_norm_eps` (1e-5) are correctly hardcoded in the decoder/encoder implementations, then either expose or document them.

<details><summary>Sources consulted</summary>

- https://huggingface.co/facebook/mms-tts-eng/raw/main/config.json
- https://raw.githubusercontent.com/jaywalnut310/vits/main/configs/ljs_base.json
- https://raw.githubusercontent.com/jaywalnut310/vits/main/configs/vctk_base.json
- https://raw.githubusercontent.com/rhasspy/piper/master/src/python/piper_train/vits/config.py
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/vits/configuration_vits.py
- https://huggingface.co/speaches-ai/piper-en_US-kristin-medium/raw/main/config.json

</details>

---

## Kyutai (Moshi/STT/TTS)

Reference: kyutai/stt-1b-en_fr, kyutai/stt-2.6b-en, kyutai/tts-1.6b-en_fr (HF config.json) plus the moshi repo LMConfig defaults (loaders.py) for moshiko/moshika. We have presets for the two STT variants and the one TTS variant (with a separate depformer config); we have no moshiko/moshika preset.

| Reference param | Our field | Reference value | Our value | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| dim | Helium/Temporal.HiddenSize | 2048 (all 3) | 2048 | ok | |
| num_layers (stt-1b) | Stt1B Helium.NumHiddenLayers | 16 | 16 | ok | stt-1b-en_fr config.json |
| num_heads (stt-1b) | Stt1B Helium.NumAttentionHeads | 16 | 16 | ok | head_dim 128 |
| context (stt-1b) | Stt1B Helium.MaxPositionEmbeddings | 750 | 375 | wrong | stt-1b-en_fr config.json says context=750; we set 375 (swapped with 2.6b) |
| text_card (stt-1b) | Stt1B TextVocab | 8000 | 8001 | unverified | we use 8001 (text_card+1 shared-table slot); confirm the +1 is intended vs off-by-one |
| num_layers (stt-2.6b) | Stt2_6B Helium.NumHiddenLayers | 48 | 48 | ok | stt-2.6b-en config.json |
| num_heads (stt-2.6b) | Stt2_6B Helium.NumAttentionHeads | 32 | 32 | ok | head_dim 64 |
| context (stt-2.6b) | Stt2_6B Helium.MaxPositionEmbeddings | 375 | 750 | wrong | stt-2.6b-en config.json says context=375; we set 750 (swapped with 1b) |
| text_card (stt-2.6b) | Stt2_6B TextVocab | 4000 | 4001 | unverified | same +1 convention question |
| max_period (STT) | Helium.RopeTheta | 100000.0 | 100000 | ok | both STT configs |
| hidden_scale | IntermediateSize | 4.125 (8448 for TTS @ dim 2048; 11264 used for STT) | STT 11264 / TTS 8448 | wrong | 4.125*2048 = 8448; STT presets use 11264 which is hidden_scale 5.5, not 4.125 per HF (STT hidden_scale is also 4.125) |
| existing_text_padding_id | TextPad / TextPad | 3 | 3 | ok | |
| n_q | NumCodebooks | 32 | 32 | ok | |
| card | CodebookVocab | 2048 | STT 2049 / TTS 2048 | ok | STT 2049 = card+1 pad/BOS slot (AudioBos 2048); intentional |
| gating=silu, norm=rms_norm_f32, positional_embedding=rope, causal | (hardcoded via Qwen2) | silu/rms/rope/causal | silu/rms/rope/causal | ok | RmsNormEps 1e-8 matches rms_norm_f32 |
| stt_config.audio_delay_seconds (1b) | Stt1B AudioDelaySeconds | 0.5 | 0.5 | ok | |
| stt_config.audio_silence_prefix_seconds (1b) | Stt1B AudioSilencePrefixSeconds | 0.0 | 0.0 | ok | |
| stt_config.audio_delay_seconds (2.6b) | Stt2_6B AudioDelaySeconds | 2.5 | 2.5 | ok | |
| stt_config.audio_silence_prefix_seconds (2.6b) | Stt2_6B AudioSilencePrefixSeconds | 1.0 | 1.0 | ok | |
| lm_gen_config.top_k / top_k_text (STT) | (none) | 250 / 50 | n/a | missing | STT presets carry no sampling defaults; greedy temp 0 + top_k 250/text 50 |
| TTS dim/num_layers/num_heads | Temporal.* | 2048/16/16 | 2048/16/16 | ok | tts-1.6b config.json |
| TTS context | Temporal.MaxPositionEmbeddings | 500 | 500 | ok | |
| TTS max_period | Temporal.RopeTheta | 10000 | 10000 | ok | |
| dep_q (TTS) | NumCodebooks / Depth.DepQ | 32 | 32 | ok | |
| delays (TTS) | AcousticDelay + BuildDelays | [0,0,2,2,...,2] | AcousticDelay 2 | ok | BuildDelays reproduces 0,0,then 2s |
| tts_config.audio_delay (s) | StreamDelaySteps | 1.28 s -> 16 frames | 16 | ok | 1.28*12.5=16 |
| tts_config.second_stream_ahead | (none) | 2 | n/a | missing | second-stream lookahead unrepresented |
| demux_second_stream | (none) | true | n/a | missing | not modeled |
| depformer_dim | Depth.Dim | 1024 | 1024 | ok | |
| depformer_num_heads | Depth.NumHeads | 16 | 16 | ok | head_dim 64 |
| depformer_num_layers (TTS) | Depth.NumLayers | 4 | 4 | ok | tts-1.6b says 4 |
| depformer_dim_feedforward (TTS) | Depth.FfnDim | 3072 | 3072 | ok | TTS-specific; base moshi uses 4224 |
| depformer_low_rank_embeddings | Depth.LowRankEmb | 128 | 128 | ok | |
| depformer_weights_per_step_schedule | Depth.WeightSetSchedule | [0..7,8x8,9x8,10x8] | BuildSchedule(32) identical | ok | 11 weight sets |
| depformer_multi_linear / weights_per_step / pos_emb=none | (modeled in depformer) | true/true/none | yes | ok | |
| conditioners.speaker_wavs.dim | SpeakerDim | 512 | 512 | ok | |
| conditioners.cfg (LUT n_bins 7, dim 16, values 1.0..4.0) | (none) | CFG LUT | n/a | missing | CFG-coefficient conditioner not modeled |
| conditioners.control (LUT dim 2048) | (none) | control LUT | n/a | missing | not modeled |
| fuser (sum control+cfg, cross speaker_wavs, cross_attention_pos_emb) | (none) | fuser spec | n/a | missing | conditioner fusion routing not modeled |
| lm_gen_config.temp / text_temp (TTS) | Temperature | 0.6 / 0.6 | 0.8 | wrong | HF tts-1.6b lm_gen_config temp=0.6 |
| (no ref) | TopP / TopK | n/a | 0.95 / 0 | extra | reference TTS samples temp-only; STT uses top_k 250 |
| (no ref) | MaxSpeakers | n/a | 5 | extra | no `max_speakers` key in HF config; speaker_wavs is a single tensor cond |
| moshiko/moshika LM (dim 4096, 32L, 32 heads, dep_q 8, n_q 16, text_card 32000, context 3000, max_period 10000, depformer_num_layers 6, dim_feedforward 4224) | (none) | full moshi base | n/a | missing | no preset for the Moshi dialogue models |

### Action items
- Fix the swapped `context`/MaxPositionEmbeddings: set Stt1B MaxPositionEmbeddings=750 and Stt2_6B MaxPositionEmbeddings=375 (HF stt configs).
- Reconcile IntermediateSize with hidden_scale 4.125: STT presets should use 8448 (4.125*2048), not 11264, unless an internal source proves otherwise; verify against the actual STT checkpoint MLP weight shapes.
- Correct TTS default Temperature to 0.6 (and add a text temperature of 0.6); reconsider exposing TopP=0.95/TopK=0 since reference TTS sampling is temperature-only.
- Verify the TextVocab +1 convention (8001/4001 vs HF text_card 8000/4000) against the real shared-embedding row count to rule out an off-by-one in the sampling range.
- Add explicit fields for the TTS conditioners: cfg LUT (n_bins 7, dim 16, CFG values 1.0..4.0), control LUT (dim 2048), and the fuser routing (sum control+cfg, cross-attend speaker_wavs with positional emb), plus `second_stream_ahead`=2 and `demux_second_stream`=true.
- Add STT sampling defaults (top_k 250, top_k_text 50, greedy temp 0) if inference uses them.
- Add a moshiko/moshika preset (Moshi base LM): dim 4096, 32 layers, 32 heads, n_q 16, dep_q 8, text_card 32000, context 3000, max_period 10000, depformer_num_layers 6, depformer_dim_feedforward 4224; do not reuse the TTS depformer FfnDim 3072 for it.
- Drop or document MaxSpeakers=5 (no reference basis).

<details><summary>Sources consulted</summary>

- https://huggingface.co/kyutai/stt-1b-en_fr/raw/main/config.json
- https://huggingface.co/kyutai/stt-2.6b-en/raw/main/config.json
- https://huggingface.co/kyutai/tts-1.6b-en_fr/raw/main/config.json
- https://raw.githubusercontent.com/kyutai-labs/moshi/main/moshi/moshi/models/loaders.py
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/moshi/configuration_moshi.py

</details>

---

## CosyVoice

Reference: FunAudioLLM/CosyVoice2-0.5B (cosyvoice2.yaml on HF + the cosyvoice repo GitHub source). Our config models only the CosyVoice2 0.5B variant; the older CosyVoice v1 300M family (different LLM and 4096 speech tokens) has no preset.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| llm_input_size / output_size | Llm.HiddenSize (Qwen25_0_5B) | 896 | 896 | ok | Qwen2.5-0.5B backbone 24L/896/14:2 GQA, RoPE 1e6 |
| speech_token_size | SpeechTokenSize | 6561 | 6561 | ok | FSQ 3^8 |
| (sos/eos/task speech slots) | SpeechTokenExtra | (3 control ids) | 3 | ok | llm_decoder out = 6561+3 |
| token_frame_rate | TokenRateHz | 25 | 25 | ok | |
| sample_rate | SampleRate | 24000 | 24000 | ok | |
| flow output_size (mel) | Flow.MelBins | 80 | 80 | ok | |
| token_mel_ratio (=2) | Flow.MelFrameRateHz (50) | 2 | 50/25=2 | ok | implied, not a direct field |
| flow input_size | Flow.InputSize | 512 | 512 | ok | |
| encoder output_size | Flow.EncoderOutputSize | 512 | 512 | ok | |
| encoder attention_heads | Flow.EncoderNumHeads | 8 | 8 | ok | |
| encoder num_blocks (pre) | Flow.EncoderNumPreBlocks | 6 | 6 | ok | YAML num_blocks=6 |
| up_encoders count (post) | Flow.EncoderNumPostBlocks | 4 (hardcoded range(4)) | 4 | ok | not in YAML; hardcoded in upsample_encoder.py |
| encoder linear_units (FFN) | (none) | 2048 | (implicit) | missing | FFN width not a field; verify it is not hardcoded wrong |
| estimator channels | Flow.UnetChannels | [256] | [256, 256] | wrong | reference is a SINGLE level [256]; depth comes from num_mid_blocks=12. Two levels breaks UNet shapes/weight load |
| estimator num_mid_blocks | Flow.NumMidBlocks | 12 | 12 | ok | |
| estimator n_blocks | Flow.NumBlocks | 4 | 4 | ok | |
| estimator num_heads | Flow.NumHeads | 8 | 8 | ok | |
| estimator attention_head_dim | Flow.AttentionHeadDim | 64 | 64 | ok | |
| estimator in_channels | (none) | 320 | (implicit) | missing | mu+spk+cond+x stack; load-bearing for shapes |
| estimator out_channels | Flow.MelBins (80) | 80 | 80 | ok | |
| cfm decoder in_channels | (none) | 240 | (implicit) | missing | decoder input channel stack |
| spk_embed_dim | Flow.SpeakerEmbedDim | 192 | 192 | ok | CAM++ 192-dim |
| cfm spk_emb_dim (internal) | (none) | 80 | (implicit) | missing | 192 is projected to 80 before fusion; only 192 exposed |
| pre_lookahead_len | (none) | 3 | (implicit) | missing | chunk-aware lookahead; affects causal masking |
| sigma_min | (none) | 1e-6 | (none) | unverified | not exposed; standard OT-CFM sigma_min, assume default 1e-6 used in code |
| solver | (Euler implied) | euler | Euler | ok | NumEulerSteps=10 |
| t_scheduler | (none) | cosine | (linear?) | wrong | reference uses cosine timestep schedule; verify our Euler step uses cosine, not uniform/linear t |
| inference_cfg_rate | Flow.CfgRate | 0.7 | 0.7 | ok | |
| NFE / num steps | Flow.NumEulerSteps | 10 (default flow) | 10 | ok | |
| hift sampling_rate | Hift.SampleRate | 24000 | 24000 | ok | |
| hift in_channels (mel) | Hift.MelBins | 80 | 80 | ok | |
| hift base_channels | Hift.UpsampleInitialChannel | 512 | 512 | ok | |
| hift nb_harmonics | Hift.HarmonicNum | 8 | 8 | ok | |
| upsample_rates | Hift.UpsampleRates | [8,5,3] | [8,5,3] | ok | |
| upsample_kernel_sizes | Hift.UpsampleKernelSizes | [16,11,7] | [16,11,7] | ok | |
| resblock_kernel_sizes | Hift.ResBlockKernelSizes | [3,7,11] | [3,7,11] | ok | |
| resblock_dilation_sizes | Hift.ResBlockDilationSizes | [[1,3,5]x3] | [[1,3,5]x3] | ok | |
| istft n_fft | Hift.IstftNFft | 16 | 16 | ok | |
| istft hop_len | Hift.IstftHopSize | 4 | 4 | ok | |
| ras top_k | Sampling.TopK | 25 | 25 | ok | |
| ras top_p | Sampling.TopP | 0.8 | 0.8 | ok | |
| ras win_size | Sampling.RasWindow | 10 | 10 | ok | |
| ras tau_r | Sampling.RasMaxRepeat | 0.1 (-> threshold win_size*tau_r=1) | 4 | wrong | reference triggers re-roll at rep_num >= 1; our integer max-repeat of 4 is the wrong semantics and wrong value |
| (LM temperature) | Sampling.Temperature | none in ras_sampling | 0.8 | extra | CosyVoice2 ras_sampling takes no temperature |
| (LM repetition_penalty) | Sampling.RepetitionPenalty | none | 1.1 | extra | not in reference LM path; RAS is the de-dup mechanism |
| mix_ratio [text,speech] | StreamingTextChunk/SpeechChunk | [5,15] | 5 / 15 | ok | interleave ratio |

Action items:
- Fix Flow.UnetChannels from [256, 256] to [256] to match the reference single-level CFM estimator (depth is carried by num_mid_blocks=12). This is the highest-risk weight-shape mismatch.
- Fix RAS: change RasMaxRepeat=4 to model tau_r=0.1 with threshold rep_num >= win_size*tau_r (effective 1). Either rename to RasTauR (float 0.1) or set the trigger count to 1.
- Remove or stop using Sampling.Temperature (0.8) and Sampling.RepetitionPenalty (1.1) for the LM path: reference CosyVoice2 uses ras_sampling (top_k=25, top_p=0.8) with no temperature/rep-penalty.
- Add (or document as intentionally hardcoded) the missing load-bearing flow fields: estimator in_channels=320, cfm decoder in_channels=240, internal spk_emb_dim=80, encoder linear_units=2048, pre_lookahead_len=3, sigma_min=1e-6, and t_scheduler=cosine.
- Verify the CFM timestep schedule is cosine (t_scheduler: cosine), not uniform/linear Euler t.
- Add presets for the CosyVoice v1 300M family (CosyVoice-300M / -SFT / -Instruct / -25Hz) if v1 support is in scope; it has a different LLM (non-Qwen TransformerLM) and 4096 speech tokens, so it needs its own config shape.

<details><summary>Sources consulted</summary>

- https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/raw/main/cosyvoice2.yaml
- https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/transformer/upsample_encoder.py
- https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml
- https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/cosyvoice/utils/common.py

</details>

---

## Chatterbox

Reference: ResembleAI/chatterbox (T3 Llama-style AR LM + S3Gen CosyVoice2 flow + HiFTNet vocoder + LSTM VoiceEncoder). Reference variants: english-only (T3Config.english_only, text vocab 704), multilingual (T3Config.multilingual, text vocab 2454), and Turbo (ChatterboxTurboTTS, GPT2_medium backbone). Our config exposes only the english-only Default. T3 backbone numerics, special tokens, vocab sizes, rates, and generation defaults all match; the gaps are missing conditioning/architecture toggle fields and missing multilingual + Turbo presets.

| Reference param | Our field | Reference value | Our value | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Llama_520M hidden_size | T3.HiddenSize | 1024 | 1024 | ok | llama_configs.py |
| Llama_520M num_hidden_layers | T3.NumHiddenLayers | 30 | 30 | ok | llama_configs.py |
| Llama_520M num_attention_heads | T3.NumAttentionHeads | 16 | 16 | ok | |
| Llama_520M num_key_value_heads | T3.NumKeyValueHeads | 16 | 16 | ok | MHA (no GQA) |
| Llama_520M head_dim | (derived 1024/16) | 64 | 64 | ok | |
| Llama_520M intermediate_size | T3.IntermediateSize | 4096 | 4096 | ok | SwiGLU |
| Llama_520M max_position_embeddings | T3.MaxPositionEmbeddings | 131072 | 131072 | ok | |
| Llama_520M rope_theta | T3.RopeTheta | 500000.0 | 500000 | ok | |
| rope_scaling (llama3) | T3.RopeScaling | factor 8.0, low 1.0, high 4.0, orig 8192, type llama3 | same | ok | exact match |
| Llama_520M rms_norm_eps | T3.RmsNormEps | 1e-05 | 1e-5 | ok | |
| Llama_520M attention_bias | T3.AttentionBias | False | false | ok | |
| Llama_520M tie_word_embeddings | T3.TieWordEmbeddings | False | false | ok | |
| text_tokens_dict_size (english) | TextVocab | 704 | 704 | ok | english_only() |
| speech_tokens_dict_size | SpeechVocab / T3.VocabSize | 8194 | 8194 | ok | |
| speaker_embed_size | SpeakerEmbedDim | 256 | 256 | ok | VoiceEncoder LSTM dim |
| max_text_tokens | MaxTextTokens | 2048 | 2048 | ok | |
| max_speech_tokens | MaxSpeechTokens | 4096 | 4096 | ok | |
| start_text_token | StartTextToken | 255 | 255 | ok | |
| stop_text_token | StopTextToken | 0 | 0 | ok | |
| start_speech_token | StartSpeechToken | 6561 | 6561 | ok | |
| stop_speech_token | StopSpeechToken | 6562 | 6562 | ok | |
| S3_TOKEN_RATE | SpeechTokenRate | 25 | 25 | ok | s3tokenizer.py |
| SPEECH_VOCAB_SIZE (FSQ 3^8) | S3CodebookSize | 6561 | 6561 | ok | s3tokenizer.py |
| S3GEN_SR | SampleRate | 24000 | 24000 | ok | s3gen/const.py |
| temperature | Temperature | 0.8 | 0.8 | ok | tts.py |
| top_p | TopP | 1.0 | 1.0 | ok | tts.py |
| min_p | MinP | 0.05 | 0.05 | ok | tts.py |
| repetition_penalty | RepetitionPenalty | 1.2 | 1.2 | ok | tts.py |
| cfg_weight | CfgWeight | 0.5 | 0.5 | ok | tts.py |
| exaggeration | Exaggeration | 0.5 | 0.5 | ok | tts.py |
| max_new_tokens | MaxNewTokens | 1000 | 1000 | ok | tts.py |
| speech_cond_prompt_len | (none) | 150 | absent | missing | t3_config.py default; sets T3 speech-conditioning prompt length |
| use_perceiver_resampler | (none) | True (False for Turbo) | absent | missing | architecture toggle; differs per checkpoint |
| emotion_adv | (none) | True (False for Turbo) | absent | missing | enables the exaggeration/emotion conditioning channel |
| input_pos_emb | (none) | "learned" | absent (comment only) | missing | learned position embeddings on top of backbone |
| encoder_type | (none) | "voice_encoder" | absent | missing | selects conditioning encoder type |
| llama_config_name | (implicit, fixed to Llama_520M) | "Llama_520M" / "GPT2_medium" (Turbo) | hardcoded Llama_520M | missing | no way to select GPT2_medium for Turbo |
| VoiceEncoder num_mels | (none) | 40 | absent | missing | voice_encoder/config.py; affects mel input shape |
| VoiceEncoder sample_rate | (none) | 16000 | absent | missing | VE runs at 16 kHz (distinct from 24 kHz output) |
| VoiceEncoder n_fft/hop/win/fmax | (none) | 400 / 160 / 400 / 8000 | absent | missing | VE mel-spectrogram front-end params |
| text_tokens_dict_size (multilingual) | (none) | 2454 | absent | missing | T3Config.multilingual(); no preset |
| Turbo text/speech vocab | (none) | 50276 / 6563 | absent | missing | Turbo checkpoint; no preset |

Action items:
- Add a multilingual preset (factory) with TextVocab=2454, mirroring T3Config.multilingual().
- Add a Turbo preset: GPT2_medium backbone (24 layers, max_position_embeddings 8196, no RoPE scaling), TextVocab=50276, SpeechVocab=6563, plus UsePerceiverResampler=false and EmotionAdv=false.
- Add the missing T3 conditioning fields: SpeechCondPromptLen (default 150), UsePerceiverResampler (default true), EmotionAdv (default true), InputPosEmb (default "learned"), EncoderType (default "voice_encoder").
- Add a backbone selector (LlamaConfigName) or a second backbone preset so the engine can pick Llama_520M vs GPT2_medium rather than hardcoding Llama_520M.
- Add VoiceEncoder config fields (NumMels=40, VeSampleRate=16000, NFft=400, HopSize=160, WinSize=400, Fmax=8000) so the 16 kHz LSTM speaker encoder front-end is parameterized instead of implicit.

<details><summary>Sources consulted</summary>

- https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/models/t3/modules/t3_config.py
- https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/models/t3/llama_configs.py
- https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/tts.py
- https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/models/voice_encoder/config.py
- https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/models/s3gen/const.py
- https://raw.githubusercontent.com/resemble-ai/chatterbox/master/src/chatterbox/models/s3tokenizer/s3tokenizer.py
- https://deepwiki.com/resemble-ai/chatterbox/3.1-chatterboxturbotts

</details>

---

## WavTokenizer

Reference: novateur/WavTokenizer (Ji et al. 2024, ICLR 2025), upstream jishengpeng/WavTokenizer. Encoder = EnCodec encodec_24khz SEANet + single-quantizer RVQ; decoder = VocosBackbone (ConvNeXt) + ISTFTHead. Two architecture variants exist: frame75 (75 tok/s, n_fft 1280, hop 320, ratios [8,5,4,2]) and frame40 (40 tok/s, n_fft 2400, hop 600, downsamples [6,5,5,4]). We only model frame75.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| sample_rate | SampleRate | 24000 | 24000 | ok | both YAMLs |
| audio_channels | Channels | 1 | 1 | ok | mono |
| feature_extractor downsamples (frame75) | EncoderRates | [8,5,4,2] | [8,5,4,2] | ok | frame75 YAML downsamples; product 320 |
| n_filters (EnCodec base channels) | EncoderDim | 32 | 64 | wrong | feature_extractors.py SEANetEncoder n_filters=32; arxiv "C=32" |
| feature_extractor dimension / backbone input_channels | LatentDim | 512 | 512 | ok | EnCodec dimension=512; backbone input_channels=512 |
| vq bins / codebook_size | CodebookSize | 4096 | 4096 | ok | YAML vq_bins=4096 (overrides EnCodec default 16384) |
| RVQ codebook dimension | CodebookDim | 512 (no factorization) | 8 | wrong | EnCodec RVQ operates at full 512; no DAC-style projection. CodebookDim=8 is invalid for this arch |
| num_quantizers (n_q) | (none, implied 1) | 1 | n/a | ok | single codebook; CodebookSize alone implies nq=1 |
| residual_kernel_size | ResidualKernelSize | 3 | 7 | wrong | EnCodec SEANet residual_kernel_size=3 |
| residual dilations (dilation_base=2, n_residual_layers=1) | ResidualDilations | [1, 2] per block | [1,3,9] | wrong | EnCodec uses dilation_base=2, 1 residual layer; DAC [1,3,9] is wrong arch |
| stem kernel_size | StemKernelSize | 7 | 7 | ok | EnCodec first/last conv kernel 7 |
| encoder lstm layers | (none) | 2 | absent | missing | SEANetEncoder lstm=2; affects weights/inference |
| backbone num_layers (ConvNeXt) | HeadConvNeXtBlocks | 12 | 8 | wrong | both YAMLs num_layers: 12 |
| backbone dim | HeadDim | 768 | 768 | ok | VocosBackbone dim=768 |
| backbone intermediate_dim / ffn ratio | HeadFfnRatio | 2304 (=768*3) | 3 | ok | 768*3=2304 matches |
| adanorm_num_embeddings | (none) | 4 | absent | missing | VocosBackbone AdaLayerNorm bandwidth embeddings |
| head n_fft (frame75) | NFft | 1280 | 1280 | ok | frame75 ISTFTHead |
| head hop_length (frame75) | HopLength | 320 | 320 | ok | frame75 ISTFTHead |
| head padding | (none) | "same" | absent | missing | ISTFTHead padding="same" (minor) |
| target_bandwidths / bandwidths | (none) | [6.6]*4 (kbps) | absent | extra-absent | YAML bandwidths; informational, single nq fixes rate to 0.9 kbps. Low priority |
| FrameRate (derived) | FrameRate | 75 | 75 (derived) | extra | computed property, fine |

Action items:
- Fix CodebookDim: WavTokenizer has no factorized codebook; the RVQ runs at LatentDim=512. Remove/repurpose CodebookDim (set to 512 or drop it) so the quantizer shape is correct.
- Set HeadConvNeXtBlocks (backbone num_layers) to 12, not 8.
- Set EncoderDim (EnCodec n_filters) to 32, not 64.
- Fix the EnCodec residual unit params: ResidualKernelSize=3 and replace ResidualDilations=[1,3,9] with the EnCodec dilation_base=2 / n_residual_layers=1 pattern (dilations [1, 2]).
- Add an encoder LSTM representation (2 layers) and an AdaNorm embeddings field (adanorm_num_embeddings=4) so decoder weights map.
- Add a frame40 (40 tokens/s) preset: EncoderRates [6,5,5,4], NFft 2400, HopLength 600, same backbone (768/2304/12).
- Consider distinct presets for the large/medium speech v2 checkpoints (same frame75 arch, different weights).

<details><summary>Sources consulted</summary>

- https://github.com/jishengpeng/WavTokenizer/blob/main/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml
- https://github.com/jishengpeng/WavTokenizer/blob/main/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml
- https://raw.githubusercontent.com/jishengpeng/WavTokenizer/main/decoder/feature_extractors.py
- https://arxiv.org/html/2408.16532v1
- https://huggingface.co/novateur/WavTokenizer

</details>

---

## Spark-TTS

Reference: SparkAudio/Spark-TTS-0.5B (LLM/config.json + BiCodec/config.yaml + LLM/added_tokens.json + cli/SparkTTS.py). Single released checkpoint (0.5B); our `SparkTtsConfig.V0_5B` is the only preset and it is the only variant that exists, so no presets are missing. Audit result: config is in very good shape, every value present matches ground truth; the only gaps are encode-side mel params and the controllable-TTS control tokens.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| LLM hidden_size | Llm.HiddenSize | 896 | 896 | ok | LLM/config.json |
| LLM num_hidden_layers | Llm.NumHiddenLayers | 24 | 24 | ok | LLM/config.json |
| LLM num_attention_heads | Llm.NumAttentionHeads | 14 | 14 | ok | head_dim=64 |
| LLM num_key_value_heads | Llm.NumKeyValueHeads | 2 | 2 | ok | GQA |
| LLM intermediate_size | Llm.IntermediateSize | 4864 | 4864 | ok | LLM/config.json |
| LLM vocab_size | Llm.VocabSize | 166000 | 166000 | ok | overridden in V0_5B preset |
| LLM max_position_embeddings | Llm.MaxPositionEmbeddings | 32768 | 32768 | ok | |
| LLM rope_theta | Llm (Qwen2Config default) | 1000000.0 | 1000000.0 (default) | ok | standard Qwen2.5; confirm Qwen2Config default |
| LLM rms_norm_eps | Llm (Qwen2Config default) | 1e-06 | 1e-06 (default) | ok | standard Qwen2.5 |
| LLM tie_word_embeddings | Llm.TieWordEmbeddings | true | true | ok | tied lm_head |
| LLM eos_token_id | EosTokenId | 151645 | 151645 | ok | <\|im_end\|>, added_tokens.json |
| bicodec_global_0 id | GlobalTokenBase | 151665 | 151665 | ok | added_tokens.json |
| bicodec_global_4095 id | (GlobalTokenBase+4095) | 155760 | 155760 | ok | range 0..4095 confirmed |
| bicodec_semantic_0 id | SemanticTokenBase | 155761 | 155761 | ok | added_tokens.json |
| bicodec_semantic_8191 id | (SemanticTokenBase+8191) | 163952 | 163952 | ok | range 0..8191 confirmed |
| start_global_token id | StartGlobalTokenId | 165150 | 165150 | ok | added_tokens.json |
| end_global_token id | EndGlobalTokenId | 165156 | 165156 | ok | added_tokens.json |
| start_semantic_token id | StartSemanticTokenId | 165151 | 165151 | ok | added_tokens.json |
| end_semantic_token id | EndSemanticTokenId | 165157 | 165157 | ok | generation stop marker |
| quantizer codebook_size | SemanticVocab | 8192 | 8192 | ok | BiCodec/config.yaml |
| quantizer codebook_dim | (implicit via SemanticDim) | 8 | 8 (implied) | ok | factorized VQ 8-D -> 1024; not an explicit field |
| quantizer input_dim / SemanticDim | SemanticDim | 1024 | 1024 | ok | out_project target |
| speaker fsq_levels | FsqLevels | [4,4,4,4,4,4] | [4,4,4,4,4,4] | ok | 4096 global codes |
| speaker latent_dim | LatentDim | 128 | 128 | ok | BiCodec/config.yaml |
| speaker token_num | TokenNum / NumGlobalTokens | 32 | 32 / 32 | ok | two fields, consistent |
| speaker out_dim | GlobalDim | 1024 | 1024 | ok | d-vector dim |
| prenet vocos_dim | VocosDim | 384 | 384 | ok | |
| prenet vocos_intermediate_dim | VocosIntermediate | 2048 | 2048 | ok | |
| prenet vocos_num_layers | PrenetLayers | 12 | 12 | ok | |
| prenet sample_ratios | DownsampleStages | [1,1] | 2 (count) | ok | encoded as stage count |
| decoder channels | WaveGenChannels | 1536 | 1536 | ok | |
| decoder rates | UpsampleRates | [8,5,4,2] | [8,5,4,2] | ok | product 320 = hop |
| decoder kernel_sizes | UpsampleKernelSizes | [16,11,8,4] | [16,11,8,4] | ok | |
| mel sample_rate | SampleRate | 16000 | 16000 | ok | |
| inference temperature | Temperature | 0.8 | 0.8 | ok | cli/SparkTTS.py default |
| inference top_k | TopK | 50 | 50 | ok | cli/SparkTTS.py default |
| inference top_p | TopP | 0.95 | 0.95 | ok | cli/SparkTTS.py default |
| mel n_fft | (none) | 1024 | absent | missing | encode-side only; needed for voice-prompt audio tokenization, not decode |
| mel win_length | (none) | 640 | absent | missing | encode-side only |
| mel hop_length | (none) | 320 | absent | missing | encode-side; equals product of UpsampleRates so decode is unaffected |
| mel num_mels | (none) | 128 | absent | missing | encode-side only |
| mel mel_fmin / mel_fmax | (none) | 10 / null | absent | missing | encode-side only |
| speaker_encoder input_dim | (none) | 128 | absent | missing | encoder mel-feature dim; encode-side |
| fsq_num_quantizers | (none) | 1 | absent (implied) | missing | single FSQ quantizer; structurally assumed |
| control tokens (task_tts 165137, task_controllable_tts 165143, start_content 165146, end_content 165152, attribute banks) | (none) | see added_tokens.json | absent | missing | required only for controllable-TTS prompt construction; basic clone path works without them |
| postnet (vocos_num_layers 6) | (none) | 6 layers | absent | ok (correctly excluded) | training-only; bicodec.py detokenize never calls postnet |
| PrenetLayers reused as backbone count | PrenetLayers comment | n/a | 12 | extra | comment says "main-backbone layer count"; no separate backbone exists, prenet==12 only |

Action items:
- Confirm `Qwen2Config` defaults for rope_theta (1000000.0) and rms_norm_eps (1e-06) so the V0_5B preset inherits the correct values; they are not set explicitly in the Spark preset.
- (Encoder path) Add a mel sub-config (n_fft 1024, win_length 640, hop_length 320, num_mels 128, mel_fmin 10, mel_fmax null) plus speaker_encoder input_dim 128 if/when the voice-prompt audio tokenizer (encode side) is implemented; not required for pure TTS decode.
- Add the controllable-TTS control token IDs (task_tts 165137, task_controllable_tts 165143, start_content 165146, end_content 165152, and the gender/pitch/speed/age attribute token banks) before building controllable synthesis; the basic zero-shot clone path does not need them.
- Optional: make codebook_dim (8) and fsq_num_quantizers (1) explicit fields for documentation, even though they are currently implied correctly.
- No variants to add: 0.5B is the only released checkpoint and is fully covered.

<details><summary>Sources consulted</summary>

- https://huggingface.co/SparkAudio/Spark-TTS-0.5B/raw/main/LLM/config.json
- https://huggingface.co/SparkAudio/Spark-TTS-0.5B/raw/main/BiCodec/config.yaml
- https://huggingface.co/SparkAudio/Spark-TTS-0.5B/resolve/main/LLM/added_tokens.json
- https://huggingface.co/SparkAudio/Spark-TTS-0.5B/raw/main/config.yaml
- https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/cli/SparkTTS.py
- https://raw.githubusercontent.com/SparkAudio/Spark-TTS/main/sparktts/models/bicodec.py

</details>

---

## NeuTTS

Reference: neuphonic/neutts-air (Qwen2.5-0.5B `Qwen2ForCausalLM`, vocab 217652, tied head) emitting a single NeuCodec FSQ stream decoded to 24 kHz. Inference path is `NeuTTS._infer_torch` in github.com/neuphonic/neutts. Our config models only the Air variant; the nano family (Llama backbone) and the gguf/onnx/distill variants have no preset. All Air token ids and sampling values verified against the live tokenizer_config.json and neutts.py source.

| Reference param | Our field | Reference value | Our value | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| hidden_size | Llm.HiddenSize (Qwen25_0_5B) | 896 | inherited preset | ok | config.json |
| num_hidden_layers | Llm.NumLayers | 24 | inherited preset | ok | config.json |
| num_attention_heads | Llm.NumHeads | 14 | inherited preset | ok | config.json |
| num_key_value_heads | Llm.NumKvHeads | 2 | inherited preset | ok | config.json |
| intermediate_size | Llm.IntermediateSize | 4864 | inherited preset | ok | config.json |
| rope_theta | Llm.RopeTheta | 1000000.0 | inherited preset | ok | config.json |
| rms_norm_eps | Llm.RmsNormEps | 1e-06 | inherited preset | ok | config.json |
| max_position_embeddings | Llm.MaxPositionEmbeddings | 32768 | inherited preset | ok | config.json (note: runtime is capped by max_context=2048, see below) |
| tie_word_embeddings | (Qwen2 tied head) | true | tied | ok | config.json; doc comment notes tied head |
| vocab_size | Llm.VocabSize (Air override) | 217652 | 217652 | ok | config.json; Air sets `with { VocabSize = 217_652 }` |
| `<\|speech_0\|>` id (speech token base) | SpeechTokenBase | 151671 | 151671 | ok | tokenizer_config added_tokens_decoder |
| codebook size | CodebookSize | 65536 | 65536 | ok | speech_0=151671 .. speech_65535=217206 (65536 codes); NeuCodec FSQ single codebook, 16 bits/token (neucodec README) |
| `<\|SPEECH_GENERATION_START\|>` id | SpeechGenStart | 151669 | 151669 | ok | tokenizer_config |
| `<\|SPEECH_GENERATION_END\|>` id (eos) | SpeechGenEnd | 151670 | 151670 | ok | tokenizer_config; _infer_torch uses this as eos_token_id |
| `<\|TEXT_PROMPT_START\|>` id | TextPromptStart | 151666 | 151666 | ok | tokenizer_config |
| `<\|TEXT_PROMPT_END\|>` id | TextPromptEnd | 151667 | 151667 | ok | tokenizer_config |
| temperature (`_infer_torch`) | Temperature | 1.0 | 1.0 | ok | neutts.py _infer_torch (NOT generation_config.json's 0.7) |
| top_k (`_infer_torch`) | TopK | 50 | 50 | ok | neutts.py _infer_torch |
| min_new_tokens | MinNewTokens | 50 | 50 | ok | neutts.py _infer_torch |
| do_sample / top_p / rep_penalty | TopP=0 (disabled), none | do_sample=True, no top_p, no rep_penalty in torch path | TopP=0, no rep field | ok | torch path passes only temp/top_k; generation_config.json values are unused by the code |
| `<\|TEXT_REPLACE\|>` id | (none) | 151665 | absent | missing | tokenizer_config; used by _apply_chat_template to splice the phonemized prompt into the chat string |
| `<\|SPEECH_REPLACE\|>` id | (none) | 151668 | absent | missing | tokenizer_config; _apply_chat_template replaces this marker with SPEECH_GENERATION_START before appending ref codes |
| max_context (generate max_length) | (none) | 2048 | absent | missing | neutts.py NeuTTS.__init__ self.max_context=2048, passed as generate(max_length=2048) |
| sample_rate | Codec (NeuCodecConfig) | 24000 | in codec config | ok | neutts.py self.sample_rate=24000; lives in NeuCodecConfig, audit separately |
| hop_length | Codec (NeuCodecConfig) | 480 | in codec config | ok | neutts.py self.hop_length=480; lives in NeuCodecConfig |
| streaming_frames_per_chunk / lookforward / lookback / overlap | (none) | 25 / 5 / 50 / 1 | absent | missing (low priority) | neutts.py; only used by the GGUF streaming path which we do not implement |
| nano backbone family | (no preset) | Llama hidden_size=576, layers=24, vocab=194256, speech_0=128262, gen_end=128261 | absent | missing | nano/nano-de/fr/es are a DIFFERENT arch + token layout; nano is the reference DEFAULT backbone |

### Action items
- Add `TextReplace` (151665) and `SpeechReplace` (151668) fields so the prompt assembly can match `_apply_chat_template` exactly (splice at TEXT_REPLACE, swap SPEECH_REPLACE for SPEECH_GENERATION_START).
- Add a `MaxContext` (or MaxLength) field defaulting to 2048 and enforce it as the generation cap (reference passes generate(max_length=2048)).
- Add a `Nano` preset (and per-language nano-de/fr/es) for the Llama backbone variant: hidden_size 576, 24 layers, num_attention_heads 9, num_kv_heads 3, intermediate_size 2304, rope_theta 500000, rope_scaling linear factor 32, vocab 194256, head_dim 64, max_position_embeddings 2048, with the nano token block (SpeechTokenBase 128262, SpeechGenStart 128260, SpeechGenEnd 128261, TextPromptStart 128257, TextPromptEnd 128258, TextReplace 128256, SpeechReplace 128259). This is the reference's DEFAULT model so it is high priority.
- Decide whether to model the gguf (q4/q8) backbones and the alternate codecs (distill-neucodec, neucodec-onnx-decoder/-int8); at minimum document them as out-of-scope if not ported.
- Optionally add the streaming constants (frames_per_chunk 25, lookforward 5, lookback 50, overlap 1) only if a streaming decode path is implemented (currently GGUF-only in the reference).

<details><summary>Sources consulted</summary>

- https://huggingface.co/neuphonic/neutts-air/raw/main/config.json
- https://huggingface.co/neuphonic/neutts-air/raw/main/generation_config.json
- https://huggingface.co/neuphonic/neutts-air/resolve/main/tokenizer_config.json
- https://huggingface.co/neuphonic/neutts-air/resolve/main/special_tokens_map.json
- https://raw.githubusercontent.com/neuphonic/neutts/main/neutts/neutts.py
- https://huggingface.co/neuphonic/neucodec/resolve/main/README.md
- https://huggingface.co/neuphonic/neutts-nano/resolve/main/config.json
- https://huggingface.co/neuphonic/neutts-nano/resolve/main/tokenizer_config.json
- https://huggingface.co/neuphonic/neutts-nano-german/resolve/main/config.json

</details>

---

## MusicGen

Reference: Meta AudioCraft MusicGen / HF `MusicgenForConditionalGeneration` (decoder + frozen T5-base text encoder + frozen EnCodec). Variants checked: small (1024/24/16), medium (1536/48/24), large (2048/48/32), melody (== medium dims + chroma), stereo-* (8 codebooks, 2 channels), audiogen-medium (16 kHz). Our presets: Small, Medium, Large, AudioGen.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| decoder.hidden_size | Hidden | 1024/1536/2048 | 1024/1536/2048 | ok | matches small/medium/large config.json |
| decoder.num_hidden_layers | NumLayers | 24/48/48 | 24/48/48 | ok | matches all three |
| decoder.num_attention_heads | NumHeads | 16/24/32 | 16/24/32 | ok | matches all three |
| decoder.ffn_dim | FfnDim (=4*Hidden) | 4096/6144/8192 | 4096/6144/8192 | ok | 4*hidden holds for every variant |
| decoder.max_position_embeddings | MaxPositions | 2048 | 2048 | ok | small/medium/large/melody/stereo all 2048 |
| decoder.num_codebooks | NumCodebooks | 4 (mono), 8 (stereo) | 4 | wrong/missing | mono OK; no preset has 8 for stereo checkpoints |
| decoder.vocab_size (codebook) | CodebookSize | 2048 | 2048 | ok | per-codebook vocab; matches |
| decoder.pad_token_id / bos_token_id | SpecialToken | 2048 | 2048 | ok | == codebook size, used as BOS/blank in delay pattern |
| decoder.audio_channels | (none) | 1 (mono), 2 (stereo) | implied 1 | missing | no AudioChannels field; stereo doubling only encoded indirectly via DelayPattern |
| text_encoder.d_model | TextDim | 768 (t5-base) | 768 | ok | T5-base cross-attn dim |
| audio_encoder.sampling_rate | CodecSampleRate | 32000 (music), 16000 (audiogen) | 32000 / 16000(AudioGen) | ok | AudioGen preset sets 16000 |
| frame_rate | CodecFrameRate | 50 Hz | 50 | ok | EnCodec token rate; 50 Hz for all |
| decoder delay pattern | DelayPattern | [0,1,2,3] mono; [0,0,1,1,2,2,3,3] stereo | [0,1,2,3] | ok/partial | mono correct; stereo pattern documented in comment but no stereo preset uses it |
| guidance_scale (generation_config) | GuidanceScale | 3.0 | 3.0 | ok | matches generation_config.json |
| temperature (AudioCraft default) | Temperature | 1.0 | 1.0 | ok | HF generation_config omits it (defaults 1.0) |
| top_k | TopK | AudioCraft 250 / HF default 50 | 250 | wrong | HF generation_config.json does NOT set top_k (HF default 50); 250 is the AudioCraft original-repo default. Pick a reference and document |
| top_p | TopP | AudioCraft 0.0 / HF 1.0 | 0.0 | ok(AudioCraft) | 0.0 disables nucleus (AudioCraft sampling default) |
| max_length (generation_config) | (none) | 1500 (music) / 1503 stereo | not stored | missing | generation horizon not a config field (likely set at call site) |
| activation_function | (hardcoded GELU) | "gelu" | GELU (in model) | ok | comment notes GPT-2-family GELU MLP |
| scale_embedding | (none) | false | n/a | ok | false == no scaling; safe to omit |
| layerdrop | (none) | 0.0 | n/a | ok | inference-disabled, correctly excluded |
| chroma_length / num_chroma (melody) | (none) | 235 / 12 | n/a | missing | melody conditioning unsupported; no field or preset |
| HeadDim | HeadDim (=Hidden/NumHeads) | 64 (all) | 64 | extra/ok | derived; small 1024/16=64, medium 1536/24=64, large 2048/32=64 (consistent) |
| audiogen-medium exact dims | AudioGen preset | unconfirmed (404) | 1536/48/24 @16kHz | unverified | facebook/audiogen-medium/raw/main/config.json returned 404; dims inferred from AudioCraft docs (medium-class, 16 kHz, 4 codebooks) |

Action items:
- Add a stereo path: introduce an `AudioChannels` field (1/2) and stereo presets (StereoSmall/Medium/Large) with `NumCodebooks=8` and `DelayPattern=[0,0,1,1,2,2,3,3]`; current presets cannot represent stereo checkpoints.
- Add a `Melody` preset (medium dims) plus chroma-conditioning fields (`NumChroma=12`, `ChromaLength=235`) or explicitly document melody as out of scope.
- Reconcile `TopK`/`TopP` defaults: 250/0.0 matches AudioCraft original repo, but HF generation_config implies 50/1.0; add a doc comment naming the targeted reference so parity tests use the right baseline.
- Verify the AudioGen preset against a real config (the HF raw config 404'd); confirm 16 kHz EnCodec upsampling_ratios and that hidden=1536/layers=48/heads=24 are correct for audiogen-medium, and consider an audiogen-small preset if needed.
- Optionally store `max_length` / generation horizon (1500 music, 1503 stereo) if the pipeline relies on a config-level cap rather than a call-site argument.

<details><summary>Sources consulted</summary>

- https://huggingface.co/facebook/musicgen-small/raw/main/config.json
- https://huggingface.co/facebook/musicgen-medium/raw/main/config.json
- https://huggingface.co/facebook/musicgen-large/raw/main/config.json
- https://huggingface.co/facebook/musicgen-melody/raw/main/config.json
- https://huggingface.co/facebook/musicgen-stereo-small/raw/main/config.json
- https://huggingface.co/facebook/musicgen-small/raw/main/generation_config.json
- https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md
- https://huggingface.co/docs/transformers/model_doc/musicgen

</details>

---

## EnCodec codec

Reference: Meta EnCodec via HuggingFace `EncodecConfig` (facebook/encodec_24khz, encodec_32khz, encodec_48khz; the standalone encodec_16khz repo does not exist, AudioGen's 16 kHz codec is bundled in audiocraft). Our config exposes 3 presets (24/32/16 kHz) and is missing the 48 kHz multi-band stereo variant plus several inference-shaping fields.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| sampling_rate | SampleRate | 24000 / 32000 / 48000 | 24000 / 32000 / (16000) | ok | 24/32 match; 48 kHz preset absent |
| audio_channels | Channels | 1 (24/32k), 2 (48k) | 1 | ok for 24/32k | 48 kHz is stereo (2); no preset covers it |
| hidden_size | LatentDim | 128 | 128 | ok | maps to codebook input dim |
| codebook_dim | (LatentDim) | null -> defaults to hidden_size (128) | 128 | ok | HF codebook_dim null means = hidden_size |
| num_filters | NFilters | 32 (24k), 64 (32k) | 32 / 64 | ok | matches per config.json |
| upsampling_ratios | Ratios | [8,5,4,2] (24k), [8,5,4,4] (32k), [8,5,4,2] (48k) | [8,5,4,2] / [8,5,4,4] | ok | 24/32 match |
| num_residual_layers | NResidualLayers | 1 | 1 | ok | |
| kernel_size | KernelSize | 7 | 7 | ok | |
| last_kernel_size | LastKernelSize | 7 | 7 | ok | |
| residual_kernel_size | ResidualKernelSize | 3 | 3 | ok | |
| dilation_growth_rate | DilationBase | 2 | 2 | ok | |
| compress | Compress | 2 | 2 | ok | |
| num_lstm_layers | LstmLayers | 2 | 2 | ok | |
| use_causal_conv | Causal | true (24k), false (32k/48k) | true / false / false | ok | matches per variant |
| pad_mode | PadMode | "reflect" | "reflect" | ok | |
| norm_type | Norm | "weight_norm" (24/32k), "time_group_norm" (48k) | "weight_norm" | ok for 24/32k | 48 kHz uses time_group_norm; not representable |
| codebook_size | VqCodebookSize | 1024 (24k/48k), 2048 (32k) | 1024 / 2048 | ok | note 48 kHz is 1024 not 2048 |
| target_bandwidths | KbpsPerCodebook + ActiveCodebooks | [1.5,3,6,12,24] (24k), [2.2] (32k), [3,6,12,24] (48k) | derived (0.75/cb, 8 active) / (0.55/cb, 4 active) | ok (modeled differently) | We compute kbps per codebook instead of an explicit list; 32k 2.2/4=0.55 checks out, 24k log2(1024)*75/1000=0.75 checks out. No explicit allowed-bandwidth set. |
| use_conv_shortcut | (none) | true (HF default, 24k), false (32k) | not modeled | missing | Selects SEANet residual shortcut: Conv1d vs identity. 32k config sets false; if hardcoded, 32k weights mismatch. Source: encodec_32khz/config.json + configuration_encodec.py |
| normalize | (none) | false (24/32k), true (48k) | not modeled | missing | true rescales audio by volume and stores per-chunk scale; affects 48 kHz decode |
| chunk_length_s | (none) | null (24/32k), 1.0 (48k) | not modeled | missing | 48 kHz processes in 1.0s chunks |
| overlap | (none) | null (24/32k), 0.01 (48k) | not modeled | missing | chunk overlap for 48 kHz; stride=int((1-overlap)*chunk) |
| trim_right_ratio | (none) | 1.0 | not modeled | missing | trims transposed-conv output on right under causal conv; 1.0 = all-right trim. Only meaningful when Causal=true. |
| (ELU alpha) | EluAlpha | not a config key (hardcoded 1.0 in source) | 1.0 | extra | EnCodec uses ELU default alpha=1.0; fine but not a reference config field |
| (per-codebook kbps) | KbpsPerCodebook | derived from log2(bins)*frame_rate | 0.75 / 0.55 | extra | helper, no direct reference key |
| frame_rate | FrameRate (computed) | ceil(sampling_rate / prod(ratios)) | sampling_rate / prod(ratios) | ok | HF uses ceil; for these ratios division is exact so no difference |
| 16 kHz preset values | NFilters=64, codebook=2048, ratios=[8,5,4,2] | no published HF config.json | inferred | unverified | facebook/encodec_16khz returns 401/does-not-exist; AudioGen codec is internal to audiocraft. Values are assumptions per the doc comment. |

Action items:
- Add a `UseConvShortcut` (use_conv_shortcut) bool field, default true, and set it false in the EnCodec32kHz preset; wire it into the SEANet residual block shortcut path.
- Add an EnCodec48kHz preset: SampleRate=48000, Channels=2, NFilters=32, Ratios=[8,5,4,2], Causal=false, Norm="time_group_norm", VqCodebookSize=1024, plus Normalize=true, ChunkLengthS=1.0, Overlap=0.01, and target bandwidths [3,6,12,24].
- Add Normalize, ChunkLengthS, Overlap, and TrimRightRatio fields (inference-affecting for chunked 48 kHz and causal trimming) even if 24/32 kHz leave them at the null/false defaults.
- Confirm "time_group_norm" is supported by the norm fusion path; currently Norm only documents "weight_norm"/"none".
- Verify the 16 kHz preset against the actual AudioGen codec weights on first load (no public config.json exists to confirm NFilters/codebook_size).

<details><summary>Sources consulted</summary>

- https://huggingface.co/facebook/encodec_24khz/raw/main/config.json
- https://huggingface.co/facebook/encodec_32khz/raw/main/config.json
- https://huggingface.co/facebook/encodec_48khz/raw/main/config.json
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/encodec/configuration_encodec.py

</details>

---

## F5-TTS

Reference: SWivid/F5-TTS (GitHub configs F5TTS_v1_Base.yaml, F5TTS_Base.yaml, E2TTS_Base.yaml + infer/utils_infer.py). Our config has one preset (V1Base); reference ships three checkpoints (F5TTS_v1_Base DiT, F5TTS_Base v0 DiT, E2TTS_Base UNetT).

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| arch.dim | Dim | 1024 | 1024 | ok | F5TTS_v1_Base.yaml |
| arch.depth | Depth | 22 (DiT v1/v0), 24 (E2TTS) | 22 | ok | v1; E2TTS=24 needs a preset |
| arch.heads | Heads | 16 | 16 | ok | head_dim=64 |
| arch.ff_mult | FfMult | 2 (DiT), 4 (E2TTS) | 2 | ok | v1; E2TTS=4 needs a preset |
| arch.text_dim | TextDim | 512 (DiT); absent for E2TTS | 512 | ok | E2TTS has no text encoder |
| arch.conv_layers | TextConvLayers | 4 (DiT); absent for E2TTS | 4 | ok | ConvNeXtV2 block count |
| arch.text_mask_padding | (none) | True (v1), False (v0/E2) | not represented | missing | masks padded text in attention; v1=True per F5TTS_v1_Base.yaml |
| arch.pe_attn_head | (none) | null (v1), 1 (v0/E2) | not represented | missing | null=RoPE on all heads; 1=single-head RoPE; arch-affecting |
| arch.qk_norm | (none) | null (v1) | not represented | missing | null means no QK-norm in v1; if a variant sets it (rmsnorm) it changes attention |
| arch.backbone | (none) | DiT (F5) vs UNetT (E2TTS) | DiT only (implicit) | missing | E2TTS uses UNetT, a different block topology |
| mel_spec.n_mel_channels | MelDim | 100 | 100 | ok | |
| mel_spec.target_sample_rate | (none) | 24000 | implied (Vocos) | ok | noted in doc comment, vocoded at 24k |
| mel_spec.hop_length | (none) | 256 | not in config | ok | DSP lives in vocoder/pipeline; mentioned in comment |
| mel_spec.win_length / n_fft | (none) | 1024 / 1024 | not in config | ok | DSP, pipeline-side |
| tokenizer / vocab size | TextNumEmbeds | pinyin, 2545 chars (v1 vocab.txt) | 2545 | ok | +1 filler = 2546 rows |
| ConvPositionEmbedding kernel | ConvPosKernel | 31 | 31 | ok | hardcoded in F5 DiT source |
| ConvPositionEmbedding groups | ConvPosGroups | 16 | 16 | ok | F5 DiT source |
| time freq_embed_dim | TimeFreqEmbedDim | 256 | 256 | ok | SinusPosEmbed dim in F5 source |
| RoPE theta | RopeTheta | 10000 | 10000 | ok | |
| text max positions | TextMaxPos | (no explicit ref; precompute cap) | 8192 | unverified | reference precomputes freqs to max len at runtime, no fixed 8192 in config; value is an implementation cap, not a reference constant |
| infer nfe_step | (none) | 32 | not in config | missing | utils_infer.py default 32 |
| infer cfg_strength | (none) | 2.0 | not in config | missing | utils_infer.py |
| infer sway_sampling_coef | (none) | -1.0 | not in config | missing | utils_infer.py |
| infer ode_method | (none) | euler | euler (pipeline) | ok | mentioned in doc comment |
| infer target_rms | (none) | 0.1 | not in config | missing | utils_infer.py |
| infer cross_fade_duration | (none) | 0.15 | not in config | missing | utils_infer.py |

Action items:
- Add fields text_mask_padding (bool), pe_attn_head (int?), and qk_norm (string/enum) to the config record; default them to the v1 values (True, null, null).
- Add an F5TTS_Base (v0) preset that sets text_mask_padding=False and pe_attn_head=1.
- Add an E2TTS_Base preset/representation: backbone=UNetT, depth=24, ff_mult=4, no ConvNeXt text encoder (no text_dim/conv_layers), text_mask_padding=False, pe_attn_head=1. This likely needs a separate model path, not just a config preset.
- Surface inference defaults (nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1.0, target_rms=0.1, cross_fade_duration=0.15) either as config fields or documented pipeline constants matching utils_infer.py.
- Re-examine TextMaxPos=8192: the reference computes RoPE/text position frequencies up to actual sequence length at runtime rather than a fixed 8192 cap; confirm our cap never truncates valid sequences.

<details><summary>Sources consulted</summary>

- https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/configs/F5TTS_v1_Base.yaml
- https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/configs/F5TTS_Base.yaml
- https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/configs/E2TTS_Base.yaml
- https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/utils_infer.py

</details>

---

## NeuCodec

Reference: neuphonic/neucodec (GitHub neuphonic/neucodec, package modules model.py / codec_encoder.py / codec_decoder_vocos.py / module.py; weights pytorch_model.bin, no config.json). Single-codebook 0.8 kbps FSQ codec, 16 kHz in / 24 kHz out, 50 Hz, FSQ levels [4]^8 = 65536 codes. Variants: full NeuCodec (BigCodec acoustic + Wav2Vec2-BERT semantic), DistillNeuCodec (SQCodec + DistillHubert), and an ONNX decoder export. Our presets: NeuCodecConfig.Default (decoder) and NeuCodecEncoderConfig.Default (BigCodec encoder only).

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| sample_rate (output) | SampleRate | 24000 | 24000 | ok | model.py cls(24_000, 480) |
| input sample rate (semantic feat) | InputSampleRate | 16000 | 16000 | ok | 16 kHz feature extraction |
| hop_length | HopLength | 480 (instance; dataclass default 320) | 480 | ok | NeuCodec passes 480; do NOT use the 320 dataclass default (codec_decoder_vocos.py) |
| frame rate | FrameRate | 50 | 50 | ok | 16000 / 320 down = 50 Hz |
| FSQ levels | FsqLevels | [4,4,4,4,4,4,4,4] | [4,4,4,4,4,4,4,4] | ok | ResidualFSQ levels (codec_decoder_vocos.py) |
| FSQ dim (code-vector) | FsqDim | 8 | 8 | ok | len(levels) |
| vq_dim (quantizer) | QuantizerDim | 2048 | 2048 | ok | fc_prior Linear(2048,2048), vq_dim=2048 |
| hidden_dim (backbone) | BackboneDim | 1024 | 1024 | ok | VocosBackbone hidden_dim |
| depth | Depth | 12 | 12 | ok | decoder transformer depth |
| heads (decoder) | NumHeads | 16 | 16 | ok | CodecDecoderVocos heads=16 |
| pos_meb_dim (head dim) | HeadDim (derived) | 64 | 64 | ok | BackboneDim/NumHeads = 64 == pos_meb_dim |
| embed conv kernel | EmbedKernel | 7 (pad 3) | 7 | ok | Conv1d k7 p3 |
| resnet kernel | ResnetKernel | 3 | 3 | ok | ResnetBlock convs |
| GroupNorm groups | GroupNormGroups | 32 | 32 | ok | num_groups=32 |
| GroupNorm / LN eps | NormEps | 1e-6 | 1e-6 | ok | eps=1e-6 |
| prior_net resnet blocks | PriorResnetBlocks | 2 | 2 | ok | two ResnetBlocks |
| post_net resnet blocks | PostResnetBlocks | 2 | 2 | ok | two ResnetBlocks |
| n_fft | NFft | hop*4 = 1920 | 1920 | ok | n_fft = hop_length*4 |
| win_length | Win | 1920 (= n_fft) | 1920 | ok | win_length = n_fft |
| rope theta | RopeTheta | (not explicit in source) | 10000 | unverified | RoPE used (pos_meb_dim=64) but theta not shown in fetched source; 10000 is the standard default, confirm against pytorch_model.bin |
| codebook_size | CodebookSize | 16384 (constructor arg) / 65536 effective (FSQ 4^8) | 65536 | wrong/ambiguous | Reference passes codebook_size=16384, codebook_dim=16 to CodecDecoderVocos but ResidualFSQ(levels=[4]*8) overrides to 65536 codes. 65536 is the true count; flag the 16384 nominal mismatch for any code reading this field |
| fc_post_a (Linear 2048->1024) | (implied by QuantizerDim+BackboneDim) | 2048 -> 1024 | implied | ok | no dedicated field, numerically covered |
| SemanticEncoder dims (full) | (none) | (1024,1024,1024) | absent | missing | Wav2Vec2-BERT semantic branch; encoder treats it as documented no-op |
| semantic encoder model | (none) | Wav2Vec2-BERT-large | absent | missing | needed for full-fidelity encode; out of scope per code comment |
| ngf (encoder base width) | Ngf | 48 | 48 | ok | CodecEncoder ngf=48 |
| up_ratios (downsample) | DownRatios | [2,2,4,4,5] | [2,2,4,4,5] | ok | product 320 -> 50 Hz |
| residual units / stage | ResidualUnitsPerStage | 3 | 3 | ok | EncoderBlock has 3 ResidualUnits |
| dilations | ResidualDilations | (1,3,9) | [1,3,9] | ok | module.py |
| residual conv1 kernel | ResidualKernel | 7 | 7 | ok | ResidualUnit conv1 k7 dilated |
| residual conv2 kernel | (hardcoded 1) | 1 | 1 (pad 0) | ok | pointwise conv hardcoded in NeuCodecEncoder |
| stem conv kernel | StemKernel | 7 (pad 3) | 7 | ok | WNConv1d(1,d,k7,p3) |
| final conv kernel | FinalKernel | 3 (pad 1) | 3 | ok | WNConv1d(d,hidden,k3,p1) |
| encoder hidden_dim (final conv out) | FcInDim | 1024 | 1024 | ok | feeds fc_prior |
| downsample conv kernel/stride/pad | (hardcoded) | k=2*stride, stride=ratio, pad=stride//2+stride%2 | k=2*ratio, stride=ratio, pad=(ratio+1)/2 | ok | matches module.py EncoderBlock |
| encoder depth/heads | (none) | depth=12, heads=12 | absent | extra/na | CodecEncoder declares depth=12/heads=12 but they are unused placeholders (no attention in encoder); safe to omit |
| weight normalization on convs | (none) | WNConv1d everywhere | absent | missing | reference uses weight_norm on all conv layers; ensure loader fuses g/v parametrization (WeightNormFusion) when reading pytorch_model.bin |

Action items:
- Document or reconcile CodebookSize: keep 65536 (effective FSQ count) but add a comment that the reference constructor nominal is codebook_size=16384 / codebook_dim=16 (unused under FSQ), so no loader sizes a tensor from 16384.
- Add a DistillNeuCodec encoder preset (or a separate config): SQCodec acoustic encoder + DistillHubert semantic, fc_prior Linear(768+768 -> 2048), extra fc_sq_prior Linear(512 -> 768), SemanticEncoder(768,768,1024). Our current NeuCodecEncoderConfig cannot load neuphonic/distill-neucodec.
- Add semantic-branch fields (Wav2Vec2-BERT-large for full, DistillHubert for distill) and SemanticEncoder(1024,1024,1024) dims, even if the branch stays a no-op, so the config records the true architecture and the eventual semantic front-end has a home.
- Confirm RopeTheta=10000 against the checkpoint (not stated in the fetched source); leave as unverified until checked.
- Confirm the weight-norm parametrization (g/v) on encoder convs is fused on load (WeightNormFusion), since the reference uses WNConv1d throughout.
- Add an explicit FcPostA field (Linear 2048 -> 1024) or a comment, since it is currently only implied by QuantizerDim and BackboneDim.

<details><summary>Sources consulted</summary>

- https://github.com/neuphonic/neucodec/blob/main/neucodec/model.py
- https://raw.githubusercontent.com/neuphonic/neucodec/main/neucodec/codec_decoder_vocos.py
- https://raw.githubusercontent.com/neuphonic/neucodec/main/neucodec/codec_encoder.py
- https://raw.githubusercontent.com/neuphonic/neucodec/main/neucodec/module.py
- https://huggingface.co/neuphonic/neucodec
- https://huggingface.co/neuphonic/distill-neucodec
- https://huggingface.co/neuphonic/neucodec-onnx-decoder
- https://arxiv.org/html/2509.09550v1

</details>

---

## MeloTTS

Reference: myshell-ai MeloTTS, a VITS2 (SynthesizerTrn) text-to-speech core with tone-id, language-id, and dual BERT feature streams. Published checkpoints: English-v3, English (base multispeaker), Chinese, Japanese, Spanish, French, Korean. Our config exposes a single `EnglishV3` preset over a shared `VitsConfig` Core; reference values below are from each checkpoint's raw config.json.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| inter_channels | Core.InterChannels | 192 | 192 | ok | EN-v3 model section |
| hidden_channels | Core.HiddenChannels | 192 | 192 | ok | |
| filter_channels | Core.FilterChannels | 768 | 768 | ok | |
| n_heads | Core.NumHeads | 2 | 2 | ok | |
| n_layers | Core.NumEncoderLayers | 6 | 6 | ok | |
| n_layers_trans_flow | Core.FlowFlows / FlowLayers | 3 | FlowFlows=4, FlowLayers=4 | wrong | Reference flow uses 3 transformer-coupling layers (n_layers_trans_flow=3) for all variants; our Core defaults are 4. MeloTTS uses TransformerCouplingBlock, not the plain WN flow, so this controls flow depth. https://huggingface.co/myshell-ai/MeloTTS-English-v3/raw/main/config.json |
| kernel_size | Core.EncoderKernelSize | 3 | 3 | ok | |
| n_layers_q | (none) | 3 | absent | ok | Posterior encoder, inference-unused on decode path (no shape impact at inference); acceptable to omit |
| resblock | Core.ResBlock | "1" | "1" | ok | EnglishV3 overrides Core to "1" |
| resblock_kernel_sizes | Core.ResBlockKernelSizes | [3,7,11] | [3,7,11] | ok | |
| resblock_dilation_sizes | Core.ResBlockDilations | [[1,3,5],[1,3,5],[1,3,5]] | same | ok | |
| upsample_rates | Core.UpsampleRates | [8,8,2,2,2] | [8,8,2,2,2] | ok | product 512 = hop |
| upsample_initial_channel | Core.UpsampleInitialChannel | 512 | 512 | ok | |
| upsample_kernel_sizes | Core.UpsampleKernelSizes | [16,16,8,2,2] | [16,16,8,2,2] | ok | |
| gin_channels | Core.GinChannels | 256 | 256 | ok | |
| sampling_rate | Core.SampleRate | 44100 | 44100 | ok | all current MeloTTS checkpoints are 44.1 kHz |
| filter_length / hop_length | (HopLength derived) | 2048 / 512 | hop=512 derived, filter_length absent | ok | hop derived from upsample product (512). filter_length (n_fft) only matters for the mel posterior at train time |
| num_tones | NumTones | 16 (EN/JP/ES/FR/KR), 11 (ZH) | 16 | wrong | Correct for EN-v3 but the ZH checkpoint uses num_tones=11. A ZH preset must override. https://huggingface.co/myshell-ai/MeloTTS-Chinese/raw/main/config.json |
| num_languages | NumLanguages | 8 (EN-v3); 10 (JP/ES/FR/KR); 4 (ZH) | 10 | wrong | EnglishV3 preset uses default 10 but EN-v3 reference is 8. language_emb table sized wrong for EN-v3. ZH is 4. https://huggingface.co/myshell-ai/MeloTTS-English-v3/raw/main/config.json |
| n_speakers | NumSpeakers / Core.NumSpeakers | 1 (EN-v3); 256 (ZH/JP/ES/FR/KR model) | 1 | wrong | OK for EN-v3 only. The non-EN model sections declare n_speakers=256 (emb_g table), so single-speaker presets undersize the speaker embedding for those checkpoints. https://huggingface.co/myshell-ai/MeloTTS-Japanese/raw/main/config.json |
| symbols length (n_vocab) | Core.NumVocab | ~231 to 249 (EN-v3, commit-dependent) | 256 | unverified | Reported symbols length varied (231/234/249) across fetches of the same file; we hardcode 256 which oversizes the phoneme embedding. Confirm exact symbols list length against the actual downloaded checkpoint. |
| use_spk_conditioned_encoder | (implicit) | true | implied | ok | Encoder consumes gin/spk; structurally present |
| use_noise_scaled_mas | (none) | true | absent | ok | Training-only (MAS), no inference shape impact |
| sdp_ratio | SdpRatio | 0.2 | 0.2 | ok | api.py default. https://github.com/myshell-ai/MeloTTS/blob/main/melo/api.py |
| noise_scale | NoiseScale | 0.6 | 0.6 | ok | api.py default |
| noise_scale_w | NoiseScaleW | 0.8 | 0.8 | ok | api.py default |
| length_scale (1/speed) | LengthScale | 1.0 | 1.0 | ok | speed=1.0 default |
| bert_dim (EN/multi via DeBERTa/XLM-R) | BertDim | 1024 | 1024 | ok | XLM-R / Chinese-RoBERTa large hidden=1024 |
| ja_bert_dim | JaBertDim | 768 | 768 | ok | Japanese BERT base hidden=768 |
| WindowSize (rel-pos clip) | Core.WindowSize | 4 | 4 | extra-ish | Not in config.json; VITS rel-pos window=4 is the canonical hardcoded value, so OK |

Action items:
- Fix EnglishV3 preset: set NumLanguages=8 (currently 10) to match EN-v3 config.json.
- Correct flow depth: set Core.FlowFlows (n_layers_trans_flow) to 3 for MeloTTS, do not inherit the 4 default.
- Confirm and set Core.NumVocab to the exact symbols list length of the downloaded EN-v3 checkpoint (likely ~231 to 249, not 256).
- Add per-language presets: Chinese (NumLanguages=4, NumTones=11, NumSpeakers=256), Japanese, Spanish, French, Korean (each NumLanguages=10, NumTones=16, NumSpeakers=256), and an English base multispeaker preset (n_speakers>1).
- Make NumSpeakers / Core.NumSpeakers a per-variant value (256 for ZH/JP/ES/FR/KR model sections; 1 only for EN-v3).

<details><summary>Sources consulted</summary>

- https://huggingface.co/myshell-ai/MeloTTS-English-v3/raw/main/config.json
- https://huggingface.co/myshell-ai/MeloTTS-Chinese/raw/main/config.json
- https://huggingface.co/myshell-ai/MeloTTS-Japanese/raw/main/config.json
- https://huggingface.co/myshell-ai/MeloTTS-French/raw/main/config.json
- https://huggingface.co/myshell-ai/MeloTTS-Spanish/raw/main/config.json
- https://huggingface.co/myshell-ai/MeloTTS-Korean/raw/main/config.json
- https://github.com/myshell-ai/MeloTTS/blob/main/melo/api.py

</details>

---

## Orpheus

Reference: canopylabs/orpheus-3b-0.1-ft, a Llama-3.2-3B (LlamaForCausalLM) fine-tune with vocab extended to 156,940 for SNAC 24kHz audio tokens, decoded by hubertsiuzdak/snac_24khz. Ground truth taken from public ft/pretrained config mirrors (canopylabs repo is gated) and the canopyai/Orpheus-TTS GitHub. We ship one preset (Orpheus3B); no pretrained or multilingual/"smaller"-format variants.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| hidden_size | Llm.HiddenSize | 3072 | 3072 | ok | |
| num_hidden_layers | Llm.NumHiddenLayers | 28 | 28 | ok | |
| num_attention_heads | Llm.NumAttentionHeads | 24 | 24 | ok | |
| num_key_value_heads | Llm.NumKeyValueHeads | 8 | 8 | ok | GQA |
| intermediate_size | Llm.IntermediateSize | 8192 | 8192 | ok | |
| vocab_size | Llm.VocabSize | 156940 | 156940 | ok | extended audio vocab |
| head_dim | (derived 3072/24) | 128 | 128 (derived) | ok | reference sets explicit head_dim=128; matches derivation |
| hidden_act | (Qwen2 silu) | silu | silu | ok | |
| max_position_embeddings | Llm.MaxPositionEmbeddings | 131072 | 131072 | ok | |
| rope_theta | Llm.RopeTheta | 500000.0 | 500000 | ok | |
| rms_norm_eps | Llm.RmsNormEps | 1e-5 | 1e-5 | ok | |
| attention_bias | Llm.AttentionBias | false | false | ok | |
| mlp_bias | (Qwen2, none) | false | n/a | ok | Qwen2 path has no MLP bias |
| tie_word_embeddings | Llm.TieWordEmbeddings | true | true | ok | |
| bos_token_id | Llm.BosTokenId | 128000 | 128000 | ok | |
| rope_scaling | (none) | {rope_type: llama3, factor: 32.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_position_embeddings: 8192} | absent | missing | Llama-3.2 NTK-by-parts rescale not applied; alters RoPE frequencies and thus inference numerics. Source: unsloth/baseten config.json. |
| pad_token_id | Llm.PadTokenId | 128004 (ft); 128001 (pretrained lineage) | 128263 | wrong | 128263 has no reference basis. Source: unsloth ft config.json pad_token_id=128004. |
| eos_token_id (config.json) | Llm.EosTokenId | 128009 (ft) / 128001 (pretrained) | 128258 | wrong | 128258 is the correct *inference* stop (end-of-speech, streaming stop_token_ids=[128258]) but is NOT the config.json eos; flag so a converter does not overwrite from config.json. |
| temperature (engine default) | Temperature | 0.6 (engine_class) / 0.4 (streaming ex) | 0.6 | ok | matches engine_class. |
| top_p (engine default) | TopP | 0.8 (engine_class) / 0.9 (streaming ex) | 0.95 | wrong | low severity (caller-tunable); matches neither reference default. |
| repetition_penalty (engine default) | RepetitionPenalty | 1.3 (engine_class) / 1.1 (streaming ex) | 1.1 | wrong | low severity; matches streaming example, not the engine default. |
| top_k | TopK | not set (None) | 0 (disabled) | ok | reference does not pass top_k; 0=disabled is equivalent. |
| max_tokens | (not in config) | 1200 (engine) / 2000 (streaming) | n/a | unverified | inference cap lives in pipeline, not this config record; not confirmed in OrpheusConfig. |
| stop_token_ids / end-of-speech | EndOfSpeech | 128258 | 128258 | ok | |
| start_of_human | StartOfHuman | 128259 | 128259 | ok | |
| end_of_human | EndOfHuman | 128260 | 128260 | ok | |
| end_of_text | EndOfText | 128009 | 128009 | ok | |
| code start | CodeStart | 128257 | 128257 | ok | |
| audio code base | AudioCodeBase | 128266 | 128266 | ok | matches decoder offset (token-10-(i%7)*4096 with custom_token base). |
| tokens per frame | TokensPerFrame | 7 | 7 | ok | |
| SNAC sample_rate | Codec.SampleRate | 24000 | 24000 | ok | hubertsiuzdak/snac_24khz |
| SNAC codebook_size | Codec.CodebookSize | 4096 | 4096 | ok | |
| SNAC vq_strides | Codec.VqStrides | [4,2,1] | [4,2,1] | ok | 1/2/4 redistribution matches. |

Action items:
- Add rope_scaling support to the Llama/Qwen2 path (rope_type=llama3, factor 32.0, low_freq_factor 1.0, high_freq_factor 4.0, original_max_position_embeddings 8192) and wire it for Orpheus3B; this is the only architecture-affecting gap.
- Correct PadTokenId from 128263 to 128004 (or read pad from the loaded config.json).
- Keep EosTokenId=128258 for inference stop, but add a note/field so a config.json-driven converter does not overwrite it with 128009/128001 (the declared eos).
- Align sampling defaults with engine_class.py if "reference engine" parity is the intent: top_p 0.8, repetition_penalty 1.3 (currently 0.95 / 1.1).
- Add presets for orpheus-3b-0.1-pretrained (eos 128001/128009) and the multilingual/"smaller" research checkpoints that use the <custom_token_3/4/5> prompt framing.

<details><summary>Sources consulted</summary>

- https://huggingface.co/unsloth/orpheus-3b-0.1-ft/resolve/main/config.json
- https://huggingface.co/baseten/orpheus-3b-0.1-ft/resolve/main/config.json
- https://github.com/canopyai/Orpheus-TTS/blob/main/orpheus_tts_pypi/orpheus_tts/engine_class.py
- https://github.com/canopyai/Orpheus-TTS/blob/main/orpheus_tts_pypi/orpheus_tts/decoder.py
- https://github.com/canopyai/Orpheus-TTS/blob/main/realtime_streaming_example/main.py
- https://huggingface.co/hubertsiuzdak/snac_24khz

</details>

---

## HuBERT

Reference: HuggingFace `transformers` HubertConfig as published for TencentGameMate/chinese-hubert-base (our only preset), facebook/hubert-base-ls960, facebook/hubert-large-ll60k, and lengyue233/content-vec-best (ContentVec, `HubertModelWithFinalProj`). Our config has one preset (ChineseHubertBase) and is structurally locked to the base, group-norm, post-LayerNorm, no-bias, no-final-proj variant.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| hidden_size | Hidden | 768 (base/cnhubert/ContentVec), 1024 (large) | 768 | ok | Base matches. Large differs (no preset). |
| num_hidden_layers | NumLayers | 12 (base), 24 (large) | 12 | ok | Base matches; large=24 has no preset. |
| num_attention_heads | NumHeads | 12 (base), 16 (large) | 12 | ok | Base matches; large=16 has no preset. |
| intermediate_size | FfnDim | 3072 (base), 4096 (large) | 3072 | ok | Base matches; large=4096 has no preset. |
| conv_dim (per-layer) | ConvDim (scalar 512) | [512]*7 (all) | 512 | ok | All seven conv layers are 512; scalar is fine. |
| conv_kernel | ConvKernels | [10,3,3,3,3,2,2] | [10,3,3,3,3,2,2] | ok | Matches all variants. |
| conv_stride | ConvStrides | [5,2,2,2,2,2,2] | [5,2,2,2,2,2,2] | ok | Matches all variants; product 320 (Downsample). |
| num_feat_extract_layers | (derived from ConvKernels.Count) | 7 | 7 | ok | Implicit via list length. |
| num_conv_pos_embeddings | PosConvKernel | 128 | 128 | ok | Matches all variants. |
| num_conv_pos_embedding_groups | PosConvGroups | 16 | 16 | ok | Matches all variants. |
| layer_norm_eps | NormEps | 1e-05 | 1e-5 | ok | Matches all variants. |
| sampling_rate (preprocessor) | SampleRate | 16000 | 16000 | ok | 16 kHz mono for all variants. |
| conv_bias | (none) | false (base/cnhubert/ContentVec), true (large) | (hardcoded false) | missing | Required to load large; reference https://huggingface.co/facebook/hubert-large-ll60k/raw/main/config.json. |
| feat_extract_norm | (none) | "group" (base/cnhubert/ContentVec), "layer" (large) | (hardcoded group) | missing | Group: one GroupNorm after conv0. Layer: LayerNorm after each of 7 convs. Changes weight layout. |
| do_stable_layer_norm | (none) | false (base/cnhubert/ContentVec), true (large) | (hardcoded post-LN) | missing | true = pre-LN blocks + final encoder LayerNorm. Doc comment hardcodes "post-LayerNorm". |
| classifier_proj_size / final_proj | (none) | 256 (ContentVec final_proj 768->256; base has classifier_proj_size 256 but unused for features) | (none) | missing | ContentVec is `HubertModelWithFinalProj`; RVC/so-vits-svc read the 256-dim projected output, not last_hidden_state. |
| hidden_act / feat_extract_activation | (none) | "gelu" (all) | (hardcoded gelu) | missing (low) | All listed variants use gelu; safe to keep implicit but undocumented. |
| HeadDim | HeadDim (derived) | Hidden/NumHeads (=64 base, 64 large) | 64 | extra | Derived helper, not a reference field; fine. |

Action items:
- Add `ConvBias` (bool, default false) so the large variant (true) can load.
- Add `FeatExtractNorm` (enum/string: group vs layer) to select single-GroupNorm vs per-conv-LayerNorm feature extractor; default group.
- Add `DoStableLayerNorm` (bool, default false) to switch between post-LN blocks and pre-LN blocks plus a final encoder LayerNorm; update the doc comment which currently asserts post-LayerNorm only.
- Add a `FinalProjDim` (int?, default null) and a HubertLargeLl60k preset, plus a ContentVec preset that enables the 768->256 final projection (classifier_proj_size 256) used by RVC/so-vits-svc.
- Add presets: `HubertBaseLs960` (English base, same dims), `HubertLargeLl60k` (1024/24/16/4096, conv_bias true, feat_extract_norm layer, do_stable_layer_norm true), and `ContentVec`.

<details><summary>Sources consulted</summary>

- https://huggingface.co/TencentGameMate/chinese-hubert-base/raw/main/config.json
- https://huggingface.co/facebook/hubert-base-ls960/raw/main/config.json
- https://huggingface.co/facebook/hubert-large-ll60k/raw/main/config.json
- https://huggingface.co/lengyue233/content-vec-best/raw/main/config.json

</details>

---

## Bark

Reference: suno/bark (HF config.json) + suno-ai/bark GitHub bark/generation.py. Variants: Full (hidden 1024 / 24L / 16H) and Small (hidden 768 / 12L / 12H). We have presets for both; no reference checkpoint is missing a preset.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| semantic hidden_size | Stage.Hidden (Full/Small) | 1024 / 768 | 1024 / 768 | ok | HF config.json semantic_config |
| semantic num_layers | Stage.NumLayers | 24 / 12 | 24 / 12 | ok | |
| semantic num_heads | Stage.NumHeads | 16 / 12 | 16 / 12 | ok | |
| block_size | BlockSize (GptConfig) | 1024 | 1024 | ok | also CONTEXT_WINDOW_SIZE=1024 in generation.py |
| semantic input_vocab_size | SemanticInputVocab | 129600 | 129600 | ok | |
| semantic output_vocab_size | SemanticOutputVocab | 10048 | 10048 | ok | |
| coarse input/output_vocab_size | CoarseVocab | 12096 / 12096 | 12096 | ok | in == out |
| fine input/output_vocab_size | FineVocab | 1056 | 1056 | ok | per-codebook |
| n_codes_total | NumCodebooks | 8 | 8 | ok | fine_acoustics_config; also N_FINE_CODEBOOKS=8 |
| N_COARSE_CODEBOOKS | NumCoarseCodebooks | 2 | 2 | ok | generation.py |
| CODEBOOK_SIZE | CodebookSize | 1024 | 1024 | ok | EnCodec acoustic codebook entries |
| SAMPLE_RATE | SampleRate | 24000 | 24000 | ok | |
| TEXT_ENCODING_OFFSET | TextEncodingOffset | 10048 | 10048 | ok | |
| SEMANTIC_PAD_TOKEN | SemanticPadToken | 10000 | 10000 | ok | also semantic EOS |
| SEMANTIC_INFER_TOKEN | SemanticInferToken | 129599 | 129599 | ok | |
| TEXT_PAD_TOKEN | TextPadToken | 129595 | 129595 | ok | |
| COARSE_SEMANTIC_PAD_TOKEN | CoarseSemanticPadToken | 12048 | 12048 | ok | |
| COARSE_INFER_TOKEN | CoarseInferToken | 12050 | 12051 | wrong | generation.py constant is 12050; our value is off by one. See https://github.com/suno-ai/bark/blob/main/bark/generation.py |
| generate_text_semantic temp | SemanticTemperature | 0.7 | 0.7 | ok | |
| generate_coarse temp | CoarseTemperature | 0.7 | 0.7 | ok | |
| generate_fine temp | FineTemperature | 0.5 | 0.5 | ok | |
| top_k default | TopK | None (disabled) | 0 (disabled) | ok | 0 is our disabled sentinel; equivalent to reference None |
| top_p default | TopP | None (disabled) | 1.0 (no-op) | ok | reference default disables top_p; 1.0 is a no-op equivalent |
| n_codes_given | (none) | 1 | n/a | missing | fine_acoustics_config.n_codes_given=1: first fine codebook is given (from coarse), remaining 7 predicted; no field exposes this |
| min_eos_p (semantic) | (none) | 0.2 | n/a | missing | generate_text_semantic early-stop EOS prob threshold; absence changes semantic termination/length |
| max_coarse_history | (none) | 630 | n/a | missing | generate_coarse sliding-window semantic-history bound |
| SEMANTIC_RATE_HZ | (none) | 49.9 | n/a | missing | used to size semantic-to-coarse ratio (COARSE_RATE_HZ=75 / SEMANTIC_RATE_HZ=49.9); if computed inline in pipeline this may be fine, flag for confirmation |

Action items:
- Correct CoarseInferToken from 12_051 to 12_050 (off-by-one vs upstream COARSE_INFER_TOKEN).
- Add a semantic min_eos_p field (default 0.2) and wire it into semantic-stage early stopping.
- Add max_coarse_history field (default 630) and use it to bound coarse-stage semantic history.
- Add n_codes_given (default 1) so the fine stage knows the first codebook is given vs predicted (or confirm it is hardcoded in BarkFine code).
- Confirm SEMANTIC_RATE_HZ (49.9) and COARSE_RATE_HZ (75) are represented somewhere in the coarse pipeline (token-rate ratio); add as config constants if currently magic numbers.
- No variants missing: Full and Small both covered.

<details><summary>Sources consulted</summary>

- https://huggingface.co/suno/bark/raw/main/config.json
- https://huggingface.co/suno/bark-small/raw/main/config.json
- https://raw.githubusercontent.com/suno-ai/bark/main/bark/generation.py
- https://github.com/suno-ai/bark/blob/main/bark/generation.py

</details>

---

## Whisper

Reference: openai/whisper HuggingFace configs (WhisperConfig / WhisperForConditionalGeneration) for tiny, large-v2, large-v3, large-v3-turbo. Our presets: Tiny, Base, Small, Medium, LargeV2, LargeV3, LargeV3Turbo, plus four distil variants (DistilLargeV2/V3, DistilMediumEn, DistilSmallEn). All shape parameters verified correct against the raw config.json of each fetched variant.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| vocab_size | VocabSize | 51865 (<=v2), 51866 (v3+) | 51865 / 51866 | ok | Matches per variant (tiny=51865, large-v3=51866). |
| num_mel_bins | NumMelBins | 80 (<=v2), 128 (v3+) | 80 / 128 | ok | Matches per variant. |
| max_source_positions | MaxAudioPositions | 1500 | 1500 | ok | Constant all variants. |
| max_target_positions | MaxTextPositions | 448 | 448 | ok | Constant all variants. |
| max_length | (= MaxTextPositions) | 448 | 448 | ok | Generation default equals target positions. |
| encoder_layers / num_hidden_layers | EncoderLayers | 4/6/12/24/32 | matches | ok | Per variant. |
| decoder_layers | DecoderLayers | 4/6/12/24/32, turbo=4 | matches | ok | Turbo=4 and distil=2 handled. |
| d_model | HiddenSize | 384/512/768/1024/1280 | matches | ok | Per variant. |
| encoder_attention_heads / decoder_attention_heads | NumHeads | 6/8/12/16/20 | matches | ok | Single field used for both (always equal in Whisper). |
| encoder_ffn_dim / decoder_ffn_dim | IntermediateSize | 1536/2048/3072/4096/5120 | matches | ok | Per variant, = 4*d_model. |
| layer_norm eps | LayerNormEps | 1e-5 (HF default) | 1e-5 | ok | Not in config.json; HF WhisperConfig default is 1e-5. |
| eos_token_id / bos_token_id | EndOfTextTokenId | 50257 | 50257 | ok | bos and eos both 50257. |
| decoder_start_token_id | StartOfTranscriptTokenId | 50258 | 50258 | ok | Start-of-transcript. |
| pad_token_id | EndOfTextTokenId (reused) | 50256 (large-v3) / 50257 (others) | 50257 | wrong | large-v3 config.json sets pad_token_id=50256, distinct from eos 50257; our config has no pad field and reuses 50257. Other variants are 50257 so only large-v3 mismatches. Source: large-v3 config.json. |
| scale_embedding | (none) | false | n/a | missing | No field; relies on model code never scaling. Confirm WhisperDecoder does not apply sqrt(d_model). |
| activation_function | (none) | "gelu" | n/a | missing | No field; relies on hardcoded GELU in FFN/conv stem. |
| begin_suppress_tokens | (none) | [220, 50257] | n/a | missing | Logit-suppression list not represented; needed for correct generation if suppression is applied. |
| suppress_tokens / forced_decoder_ids | (none) | per-variant lists | n/a | missing | Large suppress list and forced lang/task prefix ids not in config; must be sourced elsewhere for parity. |
| dropout / attention_dropout / activation_dropout | (none) | 0.0 | n/a | ok (excluded) | Training-only, inference-disabled. |
| layerdrop, init_std, mask_*, apply_spec_augment, median_filter_width, classifier_proj_size, use_weighted_layer_sum | (none) | various | n/a | ok (excluded) | Training/aux-head/word-timestamp params, not core inference tensor shapes. |
| use_cache | (none) | true | n/a | ok (excluded) | Inference behavior flag, not a tensor shape. |
| LanguageTokenStart 50259, Translate 50358, Transcribe 50359, NoSpeech 50362, NoTimestamps 50363, TimestampTokenStart 50364 | (those fields) | n/a in config.json | as listed | extra | Hardcoded special-token offsets not in config.json; consistent with the Whisper tokenizer special-token map (tiny suppress_tokens reference 50358/50359/50362), so values look correct but have no config.json basis. |

Action items:
- Add a dedicated PadTokenId field and set it to 50256 for the LargeV3 preset (and LargeV3-derived turbo if it inherits v3 pad; note turbo config.json actually uses 50257, so set turbo back to 50257 explicitly), keeping 50257 for all other variants.
- Add a ScaleEmbedding bool (default false) or assert in the decoder that no embedding scaling is applied, to make the false default explicit and guard against regressions.
- Confirm the FFN/conv activation is GELU in code; if any path is configurable, add an ActivationFunction field defaulting to "gelu".
- If decoding parity needs logit suppression / forced prefixes, surface begin_suppress_tokens, suppress_tokens, and forced_decoder_ids (or document where they live), since they are absent from the config.
- Optionally add explicit English-only presets (tiny.en/base.en/small.en/medium.en); shapes are covered by existing presets but distinct checkpoints differ in tokenizer/forced ids.

<details><summary>Sources consulted</summary>

- https://huggingface.co/openai/whisper-tiny/raw/main/config.json
- https://huggingface.co/openai/whisper-large-v2/raw/main/config.json
- https://huggingface.co/openai/whisper-large-v3/raw/main/config.json
- https://huggingface.co/openai/whisper-large-v3-turbo/raw/main/config.json

</details>

---

## Vocos

Reference: gemelo-ai/vocos (config.yaml on HuggingFace charactr/vocos-mel-24khz and charactr/vocos-encodec-24khz; dataclass/argparse defaults in vocos/models.py, heads.py, modules.py). Two published variants exist; our config only models the mel-24khz one.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| backbone.input_channels | InputChannels | 100 (mel) / 128 (encodec) | 100 | ok (mel) | encodec uses 128; see missing variant |
| backbone.dim | HiddenDim | 512 (mel) / 384 (encodec) | 512 | ok (mel) | encodec uses 384 |
| backbone.intermediate_dim | IntermediateDim | 1536 (mel) / 1152 (encodec) | 1536 | ok (mel) | encodec uses 1152 |
| backbone.num_layers | NumLayers | 8 | 8 | ok | same for both variants |
| ConvNeXtBlock dwconv kernel_size | DwConvKernel | 7 (padding=3) | 7 | ok | modules.py hardcodes kernel_size=7, padding=3 |
| LayerNorm eps | LayerNormEps | 1e-6 | 1e-6 | ok | models.py/modules.py both use eps=1e-6 |
| head.n_fft | NFft | 1024 (mel) / 1280 (encodec) | 1024 | ok (mel) | encodec uses 1280 |
| head.hop_length | HopLength | 256 (mel) / 320 (encodec) | 256 | ok (mel) | encodec uses 320 |
| feature_extractor.sample_rate | SampleRate | 24000 | 24000 | ok | both variants 24 kHz |
| backbone.layer_scale_init_value | (none) | None -> 1/num_layers (= 0.125 for 8 layers) | absent | missing | models.py default; sets per-block residual gamma. Source: github vocos/models.py |
| backbone.adanorm_num_embeddings | (none) | None (mel) / 4 (encodec) | absent | missing | When set, ConvNeXt uses AdaLayerNorm conditioned on bandwidth id. Architecture-changing. Source: encodec config.yaml + modules.py AdaLayerNorm |
| head.padding / feature_extractor.padding | (none) | center (mel) / same (encodec) | absent | missing | Governs STFT/iSTFT centering and length alignment. Source: both config.yaml; heads.py default padding="same" |
| feature_extractor (Encodec) encodec_model / bandwidths / train_codebooks | (none) | encodec_24khz, [1.5,3.0,6.0,12.0], false | absent | missing | Entire EncodecFeatures front-end for the encodec variant has no representation. Source: encodec config.yaml |
| head win_length | (implicit) | win_length = n_fft | n/a | ok | heads.py passes win_length=n_fft; our NFft covers it |
| (none) | OutputGain | not in reference (=1.0 effective) | 44.53 | extra | Empirical mel-extractor calibration gain, no reference basis; documented as a local workaround |

### Action items
- Add a static preset `Encodec24k` for charactr/vocos-encodec-24khz (InputChannels=128, HiddenDim=384, IntermediateDim=1152, NumLayers=8, NFft=1280, HopLength=320, plus AdaNorm and Encodec front-end fields below).
- Add `LayerScaleInitValue` field (float?, default null meaning 1/NumLayers) to match the learned per-block residual scale.
- Add `AdaNormNumEmbeddings` field (int?, default null for mel, 4 for encodec) and wire AdaLayerNorm in the ConvNeXt block path.
- Add a `Padding` field/enum (center vs same) covering both feature extractor and iSTFT head; mel-24khz=center, encodec-24khz=same.
- Add EncodecFeatures parameters for the encodec variant: encodec model id (encodec_24khz), bandwidths [1.5,3.0,6.0,12.0], train_codebooks=false (input is quantized codes, not mel).
- Review the extra `OutputGain=44.53`: keep as a local override but confirm it is not silently applied to checkpoints that do not need it (default it off or per-preset).

<details><summary>Sources consulted</summary>

- https://huggingface.co/charactr/vocos-mel-24khz/raw/main/config.yaml
- https://huggingface.co/charactr/vocos-encodec-24khz/raw/main/config.yaml
- https://raw.githubusercontent.com/gemelo-ai/vocos/main/vocos/models.py
- https://raw.githubusercontent.com/gemelo-ai/vocos/main/vocos/heads.py
- https://raw.githubusercontent.com/gemelo-ai/vocos/main/vocos/modules.py

</details>

---

## YuE

Reference: m-a-p/YuE (two-stage LLaMA-2 lyrics2song). Stage-1 = YuE-s1-7B-anneal-en-cot (LlamaForCausalLM, 32L/4096/32h GQA-4kv, vocab 83968). Stage-2 = YuE-s2-1B-general (LlamaForCausalLM, 32L/2048/16h MHA, vocab 83840). Our config exposes one V1 preset; 6 reference s1 variants are unpresetted.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| s1 hidden_size | Stage1.HiddenSize | 4096 | 4096 | ok | s1 config.json |
| s1 num_hidden_layers | Stage1.NumHiddenLayers | 32 | 32 | ok | s1 config.json |
| s1 num_attention_heads | Stage1.NumAttentionHeads | 32 | 32 | ok | s1 config.json |
| s1 num_key_value_heads | Stage1.NumKeyValueHeads | 4 | 4 | ok | GQA, s1 config.json |
| s1 intermediate_size | Stage1.IntermediateSize | 11008 | 11008 | ok | s1 config.json |
| s1 vocab_size | Stage1.VocabSize | 83968 | 83968 | ok | s1 config.json |
| s1 max_position_embeddings | Stage1.MaxPositionEmbeddings | 16384 | 16384 | ok | s1 config.json |
| s1 rope_theta | Stage1.RopeTheta | 10000 | 10000 | ok | s1 config.json |
| s1 rms_norm_eps | Stage1.RmsNormEps | 1e-5 | 1e-5 | ok | s1 config.json |
| s1 tie_word_embeddings | Stage1.TieWordEmbeddings | false | false | ok | s1 config.json |
| s1 attention_bias / mlp_bias | Stage1.AttentionBias | false | false | ok | s1 config.json |
| s1 hidden_act | (hardcoded silu in Qwen2) | silu | silu | ok | s1 config.json |
| s2 hidden_size | Stage2.HiddenSize | 2048 | 2048 | ok | s2 config.json |
| s2 num_hidden_layers | Stage2.NumHiddenLayers | 32 | 22 | wrong | s2 config.json says 32, not 22 |
| s2 num_attention_heads | Stage2.NumAttentionHeads | 16 | 16 | ok | s2 config.json |
| s2 num_key_value_heads | Stage2.NumKeyValueHeads | 16 | 16 | ok | s2 is MHA, config.json |
| s2 intermediate_size | Stage2.IntermediateSize | 5504 | 5632 | wrong | s2 config.json says 5504 |
| s2 vocab_size | Stage2.VocabSize | 83840 | 100000 | wrong | s2 config.json says 83840 (differs from s1 83968) |
| s2 max_position_embeddings | Stage2.MaxPositionEmbeddings | 8192 | 8192 | ok | s2 config.json |
| s2 rope_theta | Stage2.RopeTheta | 10000 | 10000 | ok | s2 config.json |
| s2 rms_norm_eps | Stage2.RmsNormEps | 1e-5 | 1e-5 | ok | s2 config.json |
| codebook count | NumCodebooks | 8 | 8 | ok | CodecManipulator("xcodec",0,8), infer.py |
| codebook size | CodebookSize | 1024 | 1024 | ok | xcodec 1024 entries (codec, not in LM config) |
| sample_rate | SampleRate | 16000 | 16000 | ok | xcodec 16kHz, infer.py |
| frame_rate | FrameRateHz | 50 | 50 | ok | xcodec 50Hz, infer.py |
| temperature | Temperature | 1.0 | 1.0 | ok | infer.py generate() default |
| top_p | TopP | 0.93 | 0.93 | ok | infer.py generate() default |
| top_k | TopK | (model default) | 50 | unverified | infer.py does not pass an explicit top_k; 50 is our assumption, not in reference |
| repetition_penalty | RepetitionPenalty | 1.1 | 1.1 | ok | infer.py argparse default |
| vocal cb0 token base | VocalTokenBase | dynamic | 45334 | unverified | mmtokenizer assigns codec/soa/eoa IDs dynamically from SentencePiece; no static constant to confirm |
| accomp cb0 token base | AccompTokenBase | dynamic | 46358 | unverified | same dynamic-assignment caveat |
| audio EOS (eoa) token | AudioEosToken | dynamic | 32002 | unverified | mmtokenizer.eoa assigned at runtime; cannot confirm 32002 from source |

Action items:
- Fix Stage2.NumHiddenLayers: 22 should be 32 (real YuE-s2-1B-general has 32 layers).
- Fix Stage2.IntermediateSize: 5632 should be 5504.
- Fix Stage2.VocabSize: 100000 should be 83840 (note this differs from the s1 vocab of 83968).
- Reconcile the audio-token base IDs (VocalTokenBase / AccompTokenBase / AudioEosToken) against the actual loaded YuE SentencePiece tokenizer, since they are assigned dynamically and currently cannot be ground-truthed; verify 45334 / 46358 / 32002 by dumping the tokenizer or load the soa/eoa/codec offsets at runtime instead of hardcoding.
- Either drop the explicit TopK=50 (reference passes no top_k) or document it as an engine choice.
- Add presets for the 6 missing s1 variants (en-icl, zh-cot, zh-icl, jp-kr-cot, jp-kr-icl, 0.5B); the 7B s1 variants share the en-cot architecture (so the same Stage1 shape with different weights/vocab where applicable), but the 0.5B needs its own dimensions.

<details><summary>Sources consulted</summary>

- https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot/raw/main/config.json
- https://huggingface.co/m-a-p/YuE-s2-1B-general/raw/main/config.json
- https://huggingface.co/m-a-p/YuE-s2-1B-general/raw/main/generation_config.json
- https://huggingface.co/collections/m-a-p/yue
- https://github.com/multimodal-art-projection/YuE
- https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/inference/infer.py
- https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/inference/mmtokenizer.py

</details>

---

## Bert (audio frontends)

Reference: standard HuggingFace BERT encoder used as a text-conditioning frontend by MeloTTS and Bert-VITS2 / GPT-SoVITS. Our two presets target `bert-base-uncased` (English prosody BERT, MeloTTS EN) and `hfl/chinese-roberta-wwm-ext-large` (GPT-SoVITS Chinese). Both preset values match their reference `config.json` exactly. Gaps are missing variants (Japanese tohoku BERT, base-size Chinese RoBERTa, DeBERTa-v2) and a few non-field hardcoded assumptions (activation, position-embedding type, pad id).

| Reference param | Our field | Reference value | Our value | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| hidden_size | Hidden | base 768 / large 1024 | base 768 / large 1024 | ok | bert-base-uncased=768; chinese-roberta-wwm-ext-large=1024 (both HF config.json) |
| num_hidden_layers | NumLayers | base 12 / large 24 | base 12 / large 24 | ok | matches both presets |
| num_attention_heads | NumHeads | base 12 / large 16 | base 12 / large 16 | ok | matches both presets |
| intermediate_size | Intermediate | base 3072 / large 4096 | base 3072 / large 4096 | ok | matches both presets |
| vocab_size | VocabSize | uncased 30522 / cn-large 21128 | 30522 / 21128 | ok | matches both presets |
| max_position_embeddings | MaxPositions | 512 | 512 | ok | both references = 512 |
| type_vocab_size | TypeVocab | 2 | 2 | ok | both references = 2 |
| layer_norm_eps | LayerNormEps | 1e-12 | 1e-12f | ok | both references = 1e-12 |
| hidden_act | (hardcoded) | "gelu" (exact erf GELU) | exact GELU (doc comment) | ok | No config field; assumption is correct for all listed checkpoints but not overridable. Sources: bert-base-uncased + hfl configs |
| position_embedding_type | (hardcoded) | "absolute" | absolute learned (doc comment) | ok | No config field; correct for bert-base-uncased and chinese-roberta-wwm-ext-large (both "absolute"). Cannot represent DeBERTa-v2 relative attention |
| pad_token_id | (none) | 0 | not stored | missing | All references pad_token_id=0. Only impacts inference if masking is derived from pad id rather than an explicit attention mask; verify caller passes a mask |
| Japanese tohoku variant | (no preset) | vocab 32768, 768/12/12/3072 | no preset | missing | cl-tohoku/bert-base-japanese-v3 (MeloTTS/Bert-VITS2 JP). Neither existing preset has vocab 32768 |
| Chinese RoBERTa base variant | (no preset) | vocab 21128, 768/12/12/3072 | no preset | missing | hfl/chinese-roberta-wwm-ext (base size). Only the large preset exists |

### Action items
- Add a `BertConfig` preset for `cl-tohoku/bert-base-japanese-v3` (Hidden 768, NumLayers 12, NumHeads 12, Intermediate 3072, VocabSize 32768) for the MeloTTS / Bert-VITS2 Japanese frontend; the current presets use the wrong vocab and would break the embedding shape.
- Add a base-size Chinese RoBERTa preset (`hfl/chinese-roberta-wwm-ext`: Hidden 768, NumLayers 12, NumHeads 12, Intermediate 3072, VocabSize 21128) since only the large variant is provided.
- Add an explicit `PadTokenId` field defaulting to 0, or document that callers must supply an attention mask (do not derive masking from pad id implicitly).
- Consider adding optional `HiddenAct` and `PositionEmbeddingType` fields so non-default frontends (for example a DeBERTa-v2 Chinese Bert-VITS2 v2.x frontend with relative attention) can be represented; the current record is structurally limited to absolute-position exact-GELU BERT.

<details><summary>Sources consulted</summary>

- https://huggingface.co/bert-base-uncased/raw/main/config.json
- https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/raw/main/config.json
- https://huggingface.co/hfl/chinese-roberta-wwm-ext/raw/main/config.json
- https://huggingface.co/cl-tohoku/bert-base-japanese-v3/raw/main/config.json
- https://huggingface.co/microsoft/deberta-v2-xlarge/raw/main/config.json

</details>

---

## Moonshine

Reference: UsefulSensors/moonshine-tiny and moonshine-base config.json plus HF transformers configuration_moonshine.py / modeling_moonshine.py. Two variants in the reference (tiny, base); we have presets for both. Conv stem params live in the modeling code (hardcoded), not config.json.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| hidden_size | HiddenSize | tiny 288 / base 416 | 288 / 416 | ok | Both checkpoints |
| encoder_num_hidden_layers | EncoderLayers | tiny 6 / base 8 | 6 / 8 | ok | |
| decoder_num_hidden_layers | DecoderLayers | tiny 6 / base 8 | 6 / 8 | ok | |
| encoder_num_attention_heads | NumHeads | 8 | 8 | ok | We use one NumHeads for both; reference enc/dec are both 8 |
| decoder_num_attention_heads | NumHeads | 8 | 8 | ok | Same value both sides |
| encoder_num_key_value_heads | (implicit MHA) | 8 (= heads, full MHA) | implicit 8 | ok | No GQA; our model assumes full MHA which matches |
| decoder_num_key_value_heads | (implicit MHA) | 8 (= heads) | implicit 8 | ok | |
| intermediate_size | IntermediateSize | tiny 1152 / base 1664 | 1152 / 1664 | ok | |
| vocab_size | VocabSize | 32768 | 32768 | ok | |
| max_position_embeddings | MaxTextPositions | 194 (both checkpoints) | 194 | ok | HF dataclass default is 512 but both real configs override to 194 |
| rope_theta | RopeTheta | 10000.0 | 10000 | ok | |
| partial_rotary_factor | PartialRotaryFactor | tiny 0.9 / base 0.62 | 0.9 / 0.62 | ok | RotaryDim computed as int(HeadDim*factor) & ~1 |
| encoder_hidden_act | EncoderUseSiluGated=false | gelu | gelu | ok | Encoder MLP is standard GELU |
| decoder_hidden_act | DecoderUseSiluGated=true | silu (SwiGLU gated) | SwiGLU | ok | Decoder MLP gated SiLU |
| layer_norm eps (GroupNorm + LN) | LayerNormEps | 1e-5 | 1e-5 | ok | GroupNorm(num_groups=1, eps=1e-5) after conv stem also uses 1e-5 |
| attention_bias | (implicit false) | false | implicit false | ok | We do not add attention bias |
| bos_token_id | BosTokenId | 1 | 1 | ok | |
| eos_token_id | EosTokenId | 2 | 2 | ok | |
| pad_token_id | PadTokenId | 2 (tiny) / 2 (base) | 2 | ok | Reference reuses 2 as pad |
| conv1 (k=127,s=64,bias=False,in=1,out=H) | Conv1Kernel/Stride/OutChannels | k127 s64 | k127 s64 | ok | Hardcoded in modeling code, matches |
| conv2 (k=7,s=3,in=H,out=2H) | Conv2Kernel/Stride/OutChannels | k7 s3 | k7 s3 | ok | |
| conv3 (k=3,s=2,in=2H,out=H) | Conv3Kernel/Stride/OutChannels | k3 s2 | k3 s2 | ok | |
| pad_head_dim_to_multiple_of | (none) | 8 | absent | missing | In both checkpoint configs. Pads attention-compute head_dim to multiple of 8 (base 52 to 56). Documented in our comment but no field; HeadDim is raw HiddenSize/NumHeads |
| tie_word_embeddings | (none) | true | absent | missing | Decoder embedding tied to lm_head; output projection must reuse input embedding |
| decoder_start_token_id | (none, BosTokenId reused) | 1 | absent (=Bos 1) | missing | Distinct from bos_token_id; coincidentally equal so generation is currently correct |

Action items:
- Add a PadHeadDimToMultipleOf field (default 8) and apply it in the attention head-dim computation, or confirm our attention path already pads, since base head_dim 52 is padded to 56 in the reference.
- Add a TieWordEmbeddings field (default true) and ensure the lm_head reuses the decoder token-embedding weights.
- Add a DecoderStartTokenId field (default 1) instead of implicitly reusing BosTokenId.
- No variant gaps: presets for tiny and base both match their real checkpoints; no missing variants.

<details><summary>Sources consulted</summary>

- https://huggingface.co/UsefulSensors/moonshine-tiny/raw/main/config.json
- https://huggingface.co/UsefulSensors/moonshine-base/raw/main/config.json
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/moonshine/configuration_moonshine.py
- https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/moonshine/modeling_moonshine.py

</details>

---

## VibeVoice

Reference: microsoft/VibeVoice on HuggingFace, three public checkpoints: VibeVoice-1.5B (Qwen2.5-1.5B), VibeVoice-Large / 7B (Qwen2.5-7B), and VibeVoice-Realtime-0.5B (Qwen2.5-0.5B split-LM streaming, model_type `vibevoice_streaming`). Our presets V15B / V7B / Streaming05B map one-to-one to these. Config is a composition of acoustic_tokenizer_config, semantic_tokenizer_config (absent on streaming), decoder_config (Qwen2), and diffusion_head_config.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| model_type | ModelType | vibevoice / vibevoice_streaming | same | ok | |
| acoustic_vae_dim | AcousticVaeDim | 64 | 64 | ok | |
| semantic_vae_dim | SemanticVaeDim | 128 | 128 | ok | |
| tts_backbone_num_hidden_layers | TtsBackboneNumHiddenLayers | 20 (streaming) | 20 | ok | only present in Realtime-0.5B config |
| acoustic_tokenizer.vae_dim | AcousticTokenizer.VaeDim | 64 | 64 | ok | |
| acoustic_tokenizer.fix_std | FixStd | 0.5 | 0.5 | ok | |
| acoustic_tokenizer.std_dist_type | StdDistType | gaussian | gaussian | ok | |
| acoustic/semantic.channels | Channels | 1 | 1 | ok | |
| acoustic/semantic.causal | Causal | true | true | ok | |
| encoder_n_filters | EncoderNFilters | 32 | 32 | ok | |
| decoder_n_filters | DecoderNFilters | 32 | 32 | ok | |
| encoder_ratios | EncoderRatios | [8,5,5,4,2,2] | [8,5,5,4,2,2] | ok | product 3200 |
| decoder_ratios | DecoderRatios | [8,5,5,4,2,2] (acoustic), null (semantic) | same | ok | |
| encoder_depths | EncoderDepths | "3-3-3-3-3-3-8" | same | ok | |
| decoder_depths | DecoderDepths | null | null | ok | |
| mixer_layer | MixerLayer | depthwise_conv | depthwise_conv | ok | |
| conv_norm | ConvNorm | none | none | ok | |
| pad_mode | PadMode | constant | constant | ok | |
| conv_bias | ConvBias | true | true | ok | |
| disable_last_norm | DisableLastNorm | true | true | ok | |
| layernorm | LayerNorm | RMSNorm | RMSNorm | ok | |
| layernorm_eps | LayerNormEps | 1e-05 | 1e-5 | ok | |
| layernorm_elementwise_affine | LayerNormElementwiseAffine | true | true | ok | |
| layer_scale_init_value | LayerScaleInitValue | 1e-06 | 1e-6 | ok | |
| weight_init_value | WeightInitValue | 0.01 | 1e-2 | ok | training-only, kept for audit |
| semantic_tokenizer.vae_dim | SemanticTokenizer.VaeDim | 128 | 128 | ok | |
| semantic_tokenizer.fix_std | SemanticTokenizer.FixStd | 0 | 0 | ok | |
| semantic_tokenizer.std_dist_type | SemanticTokenizer.StdDistType | none | none | ok | |
| corpus_normalize | (none) | 0.0 (both tokenizers, all variants) | absent | missing | no field in VibeVoiceTokenizerConfig; value is 0.0 (disabled) on all public checkpoints so inert, but a real reference param |
| decoder.hidden_size | Decoder.HiddenSize | 1536 / 3584 / 896 | 1536 / 3584 / 896 | ok | |
| decoder.num_hidden_layers | NumHiddenLayers | 28 / 28 / 24 | 28 / 28 / 24 | ok | |
| decoder.num_attention_heads | NumAttentionHeads | 12 / 28 / 14 | 12 / 28 / 14 | ok | |
| decoder.num_key_value_heads | NumKeyValueHeads | 2 / 4 / 2 | 2 / 4 / 2 | ok | |
| decoder.intermediate_size | IntermediateSize | 8960 / 18944 / 4864 | same | ok | |
| decoder.rope_theta | RopeTheta | 1000000.0 | 1e6 | ok | |
| decoder.vocab_size | VocabSize | 151936 / 152064 / 151936 | same | ok | |
| decoder.rms_norm_eps | RmsNormEps | 1e-06 | 1e-6 | ok | |
| decoder.hidden_act | (hardcoded) | silu | silu (Qwen2) | ok | not a config field; Qwen2 forward fixes silu |
| decoder.max_position_embeddings | MaxPositionEmbeddings | 65536 / 32768 / **8192** | 65536 / 32768 / **32768** | wrong | Streaming05B is 32768 but Realtime-0.5B config.json says 8192 |
| decoder.tie_word_embeddings | TieWordEmbeddings | true / false / **false** | true / false / **true** | wrong | Streaming05B (Qwen25_0_5B preset) sets true; Realtime-0.5B config.json says false |
| decoder.attention_dropout | (none) | 0.0 | absent | extra/ok | inference-disabled dropout, no shape impact, fine to omit |
| decoder.sliding_window / use_sliding_window | (none) | null / false | absent | ok | no shape impact (disabled) |
| decoder.rope_scaling | (none) | null | absent | ok | no shape impact (disabled) |
| diffusion_head.hidden_size | DiffusionHead.HiddenSize | 1536 / 3584 / 896 | 1536 / 3584 / 896 | ok | set via `Default with { HiddenSize = ... }` per variant |
| diffusion_head.head_layers | HeadLayers | 4 | 4 | ok | |
| diffusion_head.head_ffn_ratio | HeadFfnRatio | 3.0 | 3.0 | ok | |
| diffusion_head.latent_size | LatentSize | 64 | 64 | ok | |
| diffusion_head.speech_vae_dim | SpeechVaeDim | 64 | 64 | ok | |
| diffusion_head.prediction_type | PredictionType | v_prediction | v_prediction | ok | |
| diffusion_head.diffusion_type | DiffusionType | ddpm | ddpm | ok | |
| diffusion_head.ddpm_num_steps | DdpmNumSteps | 1000 | 1000 | ok | |
| diffusion_head.ddpm_num_inference_steps | DdpmNumInferenceSteps | 20 | 20 | ok | |
| diffusion_head.ddpm_beta_schedule | DdpmBetaSchedule | cosine | cosine | ok | 1.5B/0.5B raw config says "cosine"; a 7B mirror reports "squaredcos_cap_v2" which is the same cosine schedule under the diffusers alias |
| diffusion_head.ddpm_batch_mul | DdpmBatchMul | 4 | 4 | ok | training-only, kept for round-trip |
| diffusion_head.rms_norm_eps | RmsNormEps | 1e-05 | 1e-5 | ok | |

Action items:
- Fix Streaming05B: set Decoder (Qwen25_0_5B) `MaxPositionEmbeddings` to 8192 (Realtime-0.5B config.json), not 32768. If the 32768 value is also used by any non-streaming consumer of Qwen25_0_5B, override per-variant instead of editing the shared preset.
- Fix Streaming05B: set `TieWordEmbeddings = false` for the streaming decoder (Realtime-0.5B has a separate lm_head; tying true would load the wrong output projection).
- Add a `CorpusNormalize` field (default 0.0) to VibeVoiceTokenizerConfig to round-trip `corpus_normalize` and guard against a future checkpoint that enables it.
- Correct doc comments / any loader path keys: the streaming repo is `microsoft/VibeVoice-Realtime-0.5B` (not `vibevoice/VibeVoice-Streaming-0.5B`) and the multi-speaker repos are `microsoft/VibeVoice-1.5B` and `microsoft/VibeVoice-Large` (not `vibevoice/...`). No tensor impact, but path-based loaders will 404.
- No missing variants: all three public checkpoints have presets.

<details><summary>Sources consulted</summary>

- https://huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json
- https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/raw/main/config.json
- https://deepwiki.com/voicepowered-ai/VibeVoice-finetuning/5.2-7b-model-configuration
- https://huggingface.co/bezzam/VibeVoice-7B (config.json, decoder_config values)
- https://github.com/microsoft/VibeVoice

</details>

---

## Qwen2 LM (audio backbone)

Reference: Qwen2/Qwen2.5 decoder used as the VibeVoice TTS backbone. Ground truth taken from the nested `decoder_config` of microsoft/VibeVoice-1.5B and VibeVoice-Realtime-0.5B (the actual backbones), cross-checked against standalone Qwen/Qwen2.5-0.5B/1.5B/7B. Our three presets (1.5B, 7B, 0.5B) map to the VibeVoice variants. The 1.5B preset matches its VibeVoice decoder_config exactly; the 0.5B preset has two values copied from standalone Qwen instead of the realtime backbone.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| hidden_size (1.5B / 7B / 0.5B) | HiddenSize | 1536 / 3584 / 896 | 1536 / 3584 / 896 | ok | Matches VibeVoice decoder_config and standalone Qwen. |
| num_hidden_layers | NumHiddenLayers | 28 / 28 / 24 | 28 / 28 / 24 | ok | |
| num_attention_heads | NumAttentionHeads | 12 / 28 / 14 | 12 / 28 / 14 | ok | |
| num_key_value_heads | NumKeyValueHeads | 2 / 4 / 2 | 2 / 4 / 2 | ok | GQA ratios correct. |
| intermediate_size | IntermediateSize | 8960 / 18944 / 4864 | 8960 / 18944 / 4864 | ok | |
| vocab_size | VocabSize | 151936 / 152064 / 151936 | 151936 / 152064 / 151936 | ok | VibeVoice does not extend vocab. |
| rope_theta | RopeTheta | 1000000.0 | 1000000f | ok | Long-context theta across all variants. |
| rms_norm_eps | RmsNormEps | 1e-06 | 1e-6f | ok | |
| hidden_act | (hardcoded SwiGLU/silu) | silu | silu (SwiGLU) | ok | Hardcoded in model; reference always silu. |
| head_dim | HeadDim (derived) | hidden/heads = 128/128/64 | derived HiddenSize/NumAttentionHeads | ok | Reference has no explicit head_dim; derivation correct (note: 0.5B = 896/14 = 64, not 128 as the XML comment claims; comment is wrong but value is right). |
| attention_bias (qkv bias) | AttentionBias | true (Qwen2 qkv bias) | true | ok | Defining Qwen2 feature; correctly defaulted. |
| sliding_window / use_sliding_window / max_window_layers | (none) | use_sliding_window=false, sliding_window=null | not modeled | ok | SWA disabled in every Qwen2.5/VibeVoice config, so omission does not change inference. Not a real gap. |
| rope_scaling | RopeScaling | null | null | ok | VibeVoice decoder_config rope_scaling=null. |
| tie_word_embeddings (1.5B) | TieWordEmbeddings | true | true | ok | |
| tie_word_embeddings (7B) | TieWordEmbeddings | false | false | ok | |
| max_position_embeddings (1.5B) | MaxPositionEmbeddings | 65536 (VibeVoice-1.5B decoder_config) | 65536 | ok | Correct against the VibeVoice backbone (standalone Qwen2.5-1.5B is 131072, but the backbone is 65536). |
| max_position_embeddings (7B) | MaxPositionEmbeddings | 32768 (VibeVoice "qwen2.5_7b_32k" config) | 32768 | unverified | VibeVoice-Large raw config is gated (HTTP 401); value inferred from the repo config filename `qwen2.5_7b_32k.json` and DeepWiki. Standalone Qwen2.5-7B is 131072, so verify against the actual decoder_config when accessible. |
| max_position_embeddings (0.5B) | MaxPositionEmbeddings | 8192 (VibeVoice-Realtime-0.5B decoder_config) | 32768 | wrong | Our 32768 matches standalone Qwen2.5-0.5B, not the realtime backbone (8192). Caps prompt+AR length incorrectly. |
| tie_word_embeddings (0.5B) | TieWordEmbeddings | false (VibeVoice-Realtime-0.5B) | true | wrong | Realtime-0.5B has an untied lm_head; our preset and XML comment say tied. Affects weight loading (separate lm_head.weight). |
| bos_token_id | BosTokenId | 151643 | 151643 | ok | |
| eos_token_id | EosTokenId | 151643 (base/VibeVoice) ; 151645 (instruct) | 151645 | ok | Default is the instruct `<|im_end|>` value; fine for chat, mismatched for base AR loops. Informational only (AR loop owns sampling). |
| pad_token_id | PadTokenId | 151643 | 151643 | ok | |

Action items:
- Fix `Qwen25_0_5B.MaxPositionEmbeddings` from 32768 to 8192 to match the VibeVoice-Realtime-0.5B decoder_config (or rename the preset to make clear it targets standalone Qwen2.5-0.5B if 32768 is intended).
- Fix `Qwen25_0_5B.TieWordEmbeddings` from true to false (realtime-0.5B uses an untied lm_head); also correct the XML doc comment that says 0.5B is tied.
- Correct the XML comment on HeadDim: 0.5B head_dim is 64 (896/14), not 128; only the 1.5B (128) and 7B (128) are 128.
- Verify `Qwen25_7B.MaxPositionEmbeddings` (32768) against the gated VibeVoice-Large decoder_config once authenticated; the value is currently inferred from the repo config filename.
- Optional: if base (non-instruct) Qwen2 backbones are ever loaded, expose/override EosTokenId 151643 instead of the instruct default 151645.

<details><summary>Sources consulted</summary>

- https://huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json (decoder_config block)
- https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B/raw/main/config.json (decoder_config block)
- https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json
- VibeVoice repo config filename src/vibevoice/configs/qwen2.5_7b_32k.json (per DeepWiki VibeVoice 7B model configuration)

</details>

---

## HeartMuLa

Reference: HeartMuLa-oss-3B (heartlib, github.com/HeartMuLa/heartlib), a CSM/Sesame two-transformer music LM (torchtune llama3_2 backbone + depth decoder) plus HeartCodec. Released checkpoints: HeartMuLa-oss-3B and HeartMuLa-oss-3B-happy-new-year (identical architecture). Our single preset Oss3B matches the shipped architecture exactly; only two non-shape inference params are missing.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| backbone_flavor | Lm.Backbone (inline) | llama-3B | llama-3B body | ok | config.json |
| decoder_flavor | Lm.Decoder (inline) | llama-300M | llama-300M body | ok | config.json |
| backbone num_layers | Backbone.NumHiddenLayers | 28 | 28 | ok | modeling_heartmula.py llama3_2_3B |
| backbone embed_dim | Backbone.HiddenSize | 3072 | 3072 | ok | llama3_2_3B |
| backbone num_heads | Backbone.NumAttentionHeads | 24 | 24 | ok | head_dim 128 |
| backbone num_kv_heads | Backbone.NumKeyValueHeads | 8 | 8 | ok | llama3_2_3B |
| backbone intermediate_dim | Backbone.IntermediateSize | 8192 | 8192 | ok | llama3_2_3B |
| backbone max_seq_len | Backbone.MaxPositionEmbeddings | 8192 | 8192 | ok | llama3_2_3B |
| decoder num_layers | Decoder.NumHiddenLayers | 3 | 3 | ok | llama3_2_300M |
| decoder embed_dim | Decoder.HiddenSize | 3072 | 3072 | ok | same dim as backbone (not 1024) |
| decoder num_heads | Decoder.NumAttentionHeads | 8 | 8 | ok | head_dim 384 |
| decoder num_kv_heads | Decoder.NumKeyValueHeads | 4 | 4 | ok | llama3_2_300M |
| decoder intermediate_dim | Decoder.IntermediateSize | 8192 | 8192 | ok | llama3_2_300M |
| decoder max_seq_len | Decoder.MaxPositionEmbeddings | 2048 | 2048 | ok | llama3_2_300M |
| rope_base | RopeTheta (both) | 500000 | 500000 | ok | llama3_2_*  |
| scale_factor | RopeScaling.Factor | 32 | 32 | ok | Llama3 scaled RoPE |
| norm_eps | RmsNormEps (both) | 1e-5 | 1e-5 | ok | llama3_2_*  |
| audio_num_codebooks | Lm.NumCodebooks | 8 | 8 | ok | config.json |
| audio_vocab_size | Lm.AudioVocab | 8197 | 8197 | ok | config.json |
| text_vocab_size | Lm.TextVocab / VocabSize | 128256 | 128256 | ok | config.json |
| muq_dim | MuqDim | 512 | 512 | ok | config.json |
| sample_rate | Lm.SampleRate | 48000 | 48000 | ok | pipeline torchaudio.save(...,48000) |
| frame (80 ms) | Lm.FrameSamples | 80 ms => 3840 @ 48k | 3840 | ok | pipeline max_audio_frames = ms // 80 |
| audio_eos_id | Lm.AudioEosToken | 8193 | 8193 | ok | gen_config / pipeline |
| text_eos_id | (none) | 128001 | (absent) | missing | pipeline appends to tags and lyrics; HeartMuLaGenConfig.text_eos_id = 128001 |
| topk / temperature / cfg_scale | (none) | 50 / 1.0 / 1.5 | (absent) | missing | run_music_generation.py argparse defaults; runtime knobs, not tensor shapes |
| codec num_quantizers | CodecNumQuantizers | 8 | 8 | ok | HeartCodecConfig |
| codec codebook_size | CodecCodebookSize | 8192 | 8192 | ok | HeartCodecConfig |
| codec codebook_dim | CodecCodebookDim | 32 | 32 | ok | HeartCodecConfig |

Action items:
- Add a `TextEosToken` field defaulting to 128001 to HeartMulaConfig (or to the CsmConfig prompt-assembly layer) and append it to encoded tags and lyrics, mirroring music_generation.py. This is the one functional gap.
- Optionally carry reference inference defaults (Topk=50, Temperature=1.0, CfgScale=1.5) as named defaults so generation matches the reference CLI out of the box.
- No variant preset needs adding: the happy-new-year checkpoint shares the Oss3B architecture exactly. Only document that heartlib also defines unused llama-7B and llama-400M flavors if a future checkpoint ships them.

<details><summary>Sources consulted</summary>

- https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B/raw/main/config.json
- https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year/raw/main/config.json
- https://github.com/HeartMuLa/heartlib/blob/main/src/heartlib/heartmula/configuration_heartmula.py
- https://github.com/HeartMuLa/heartlib/blob/main/src/heartlib/heartmula/modeling_heartmula.py
- https://github.com/HeartMuLa/heartlib/blob/main/src/heartlib/pipelines/music_generation.py
- https://github.com/HeartMuLa/heartlib/blob/main/src/heartlib/heartcodec/configuration_heartcodec.py
- https://github.com/HeartMuLa/heartlib/blob/main/examples/run_music_generation.py

</details>

---

## Kokoro

Reference: hexgrad/Kokoro-82M, single config.json (one architecture variant, checkpoint kokoro-v1_0.pth). Our record exposes one preset (V1). Sources: HF raw config.json and the kokoro/model.py loader. No wrong values found; the only gaps are config.json keys that are metadata or tokenizer-owned, plus two plbert fields that are AlbertConfig defaults rather than explicit config.json keys.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| n_token | NumTokens | 178 | 178 | ok | matches config.json |
| n_mels | NumMels | 80 | 80 | ok | training-only acoustic head, but it sets a tensor shape so retained |
| style_dim | StyleDim | 128 | 128 | ok | voice pack is [N, 256] = 2*style_dim |
| dim_in | DimIn | 64 | 64 | ok | |
| hidden_dim | HiddenDim | 512 | 512 | ok | |
| max_conv_dim | MaxConvDim | 512 | 512 | ok | note: model.py never reads max_conv_dim, but value is correct |
| max_dur | MaxDuration | 50 | 50 | ok | |
| text_encoder_kernel_size | TextEncoderKernelSize | 5 | 5 | ok | |
| n_layer | TextEncoderNumLayers | 3 | 3 | ok | drives both TextEncoder and ProsodyPredictor depth in ref |
| istftnet.upsample_kernel_sizes | IStftNet.UpsampleKernelSizes | [20,12] | [20,12] | ok | |
| istftnet.upsample_rates | IStftNet.UpsampleRates | [10,6] | [10,6] | ok | |
| istftnet.gen_istft_hop_size | IStftNet.GenIstftHopSize | 5 | 5 | ok | |
| istftnet.gen_istft_n_fft | IStftNet.GenIstftNFft | 20 | 20 | ok | |
| istftnet.resblock_dilation_sizes | IStftNet.ResBlockDilationSizes | [[1,3,5]x3] | [[1,3,5]x3] | ok | |
| istftnet.resblock_kernel_sizes | IStftNet.ResBlockKernelSizes | [3,7,11] | [3,7,11] | ok | |
| istftnet.upsample_initial_channel | IStftNet.UpsampleInitialChannel | 512 | 512 | ok | |
| plbert.hidden_size | PlBert.HiddenSize | 768 | 768 | ok | |
| plbert.num_attention_heads | PlBert.NumAttentionHeads | 12 | 12 | ok | |
| plbert.intermediate_size | PlBert.IntermediateSize | 2048 | 2048 | ok | |
| plbert.max_position_embeddings | PlBert.MaxPositionEmbeddings | 512 | 512 | ok | |
| plbert.num_hidden_layers | PlBert.NumHiddenLayers | 12 | 12 | ok | |
| (plbert embedding_size, AlbertConfig default) | PlBert.EmbeddingSize | 128 (AlbertConfig default; absent from config.json) | 128 | unverified | not a config.json key; ref does AlbertConfig(**config['plbert']) so HF default embedding_size=128 applies; our value matches the default but is not config-sourced |
| (plbert layer_norm_eps, AlbertConfig default) | PlBert.LayerNormEps | 1e-12 (AlbertConfig default; absent from config.json) | 1e-12 | unverified | same as above, AlbertConfig default layer_norm_eps=1e-12 |
| (iSTFTNet output rate) | SampleRate | 24000 (not in config.json; implied by 24 kHz iSTFTNet) | 24000 | unverified | no sample_rate key in config.json; architecturally correct for the Kokoro/StyleTTS2 24 kHz iSTFT head |
| multispeaker | (none) | true | absent | missing | inference-irrelevant metadata; model.py does not read it; benign omission |
| vocab | (none, in tokenizer) | 178-slot sparse IPA->ID map | absent from config | missing | owned by the Kokoro tokenizer/frontend in our port, not KokoroConfig; verify tokenizer uses identical sparse IDs |
| dropout (top-level) | (none) | 0.2 | absent | excluded | training-only, disabled at inference |
| plbert.dropout | (none) | 0.1 | absent | excluded | training-only, disabled at inference |

Action items:
- No value corrections needed: all numeric architecture params match config.json exactly.
- Optional documentation fix: annotate PlBert.EmbeddingSize=128 and PlBert.LayerNormEps=1e-12 as HuggingFace AlbertConfig defaults (config.json omits them), so future forks that override plbert know these are not config-sourced.
- Optional: add a no-op or comment for multispeaker (metadata only) so a config diff against config.json does not flag it as an unhandled key.
- Verify (outside config) that the Kokoro tokenizer reproduces the exact 178-slot sparse vocab from config.json; this is the one functional risk since vocab is not mirrored in KokoroConfig.
- No missing variants: Kokoro-82M ships a single config and architecture, so the single V1 preset is complete.

<details><summary>Sources consulted</summary>

- https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json
- https://raw.githubusercontent.com/hexgrad/kokoro/main/kokoro/model.py
- https://huggingface.co/hexgrad/Kokoro-82M/tree/main

</details>

---

## GPT LM (audio backbone)

Reference: Suno Bark GPT-2-style pre-norm decoder (GitHub suno-ai/bark `GPTConfig` dataclass + HF `suno/bark` and `suno/bark-small` config.json, the semantic / coarse / fine sub-models). The same vanilla GPT-2 body is reused by MusicGen and Orpheus audio LMs. Both our presets (BarkFull 1024/24/16, BarkSmall 768/12/12) line up exactly with the HF semantic/coarse/fine sub-configs for the large and small checkpoints.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| n_embd / hidden_size | Hidden | 1024 (full), 768 (small) | required (presets 1024 / 768) | ok | Matches HF suno/bark and suno/bark-small. |
| n_layer / num_layers | NumLayers | 24 (full), 12 (small) | required (presets 24 / 12) | ok | Matches both checkpoints. |
| n_head / num_heads | NumHeads | 16 (full), 12 (small) | required (presets 16 / 12) | ok | Matches both checkpoints. |
| block_size | BlockSize | 1024 | 1024 (default) | ok | All Bark sub-models use 1024. |
| MLP intermediate dim | MlpDim (4*Hidden) | 4 * n_embd | 4 * Hidden | ok | model.py MLP is Linear(n_embd, 4*n_embd) GELU Linear, so 4x is correct. |
| bias | (none) | dataclass default True, but HF checkpoints set bias=false | not exposed | missing | suno/bark and suno/bark-small config.json both set "bias": false for semantic/coarse/fine. Real weights have NO bias on attn/MLP/LayerNorm. Add a Bias config field (default false to match shipped weights). If GptBackbone hardcodes bias=true it will fail to load real checkpoints; if it hardcodes false it is fine numerically but the value should still be surfaced/asserted. Source: HF config.json. |
| layer_norm eps | (none) | 1e-5 | not exposed | unverified | model.py uses F.layer_norm with eps 1e-5. Not a config field here; could not confirm what GptBackbone hardcodes without reading the backbone .cs. Verify GptBackbone uses 1e-5. |
| input_vocab_size / output_vocab_size | (none, by design) | semantic in 129600 / out 10048; coarse 12096 / 12096; fine 1056 / 1056 | owned per-model | ok | Intentionally excluded: doc comment states token vocab + output heads are owned by each model, not the shared body. |
| n_codes_total / n_codes_given | (none, by design) | fine: 8 / 1 | owned per-model | ok | Fine-acoustics-only; correctly out of scope for the shared GPT body. |
| dropout | (none) | 0.0 | n/a | ok | Inference-disabled (0.0 in all configs), correctly omitted. |
| head_dim | HeadDim (Hidden/NumHeads) | 64 (full), 64 (small) | derived | ok | Derived correctly (1024/16 = 64, 768/12 = 64). |

### Action items
- Add a `Bias` field to `GptConfig` (default `false`) to match the shipped suno/bark and suno/bark-small checkpoints (config.json bias=false), and confirm `GptBackbone` actually constructs attention/MLP/LayerNorm without bias. The upstream dataclass default of `true` is misleading: the real weights are bias-free.
- Verify `GptBackbone` hardcodes LayerNorm epsilon = 1e-5 (the upstream value); if not, fix or expose it.
- No new presets required: BarkFull and BarkSmall already cover both released Bark checkpoints. If MusicGen/Orpheus reuse this body with different width/depth/heads, add their presets when those models are wired (out of scope for the two Bark checkpoints audited here).

<details><summary>Sources consulted</summary>

- https://huggingface.co/suno/bark/raw/main/config.json
- https://huggingface.co/suno/bark-small/raw/main/config.json
- https://raw.githubusercontent.com/suno-ai/bark/main/bark/model.py

</details>

---

## Dia

Reference: nari-labs/Dia-1.6B, original config.json (version 0.1) plus the transformers-format nari-labs/Dia-1.6B-0626 config.json, cross-checked against the nari-labs GitHub generate() defaults. Single 1.6B checkpoint, no size/sample-rate variants. Our single preset DiaConfig.Dia1_6B covers it. Every architecture parameter matches; the only nit is the rope min-timescale field.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| encoder.n_layer / num_hidden_layers | EncoderLayers | 12 | 12 | ok | |
| encoder.n_embd / hidden_size | EncoderDim | 1024 | 1024 | ok | |
| encoder.n_head / num_attention_heads | EncoderHeads | 16 | 16 | ok | |
| encoder.num_key_value_heads | EncoderKvHeads | 16 | 16 | ok | full MHA |
| encoder.n_hidden / intermediate_size | EncoderFfn | 4096 | 4096 | ok | |
| encoder.head_dim | HeadDim | 128 | 128 | ok | shared HeadDim |
| decoder.n_layer / num_hidden_layers | DecoderLayers | 18 | 18 | ok | |
| decoder.n_embd / hidden_size | DecoderDim | 2048 | 2048 | ok | |
| decoder.gqa_query_heads / num_attention_heads | DecoderSelfQHeads | 16 | 16 | ok | |
| decoder.kv_heads / num_key_value_heads | DecoderSelfKvHeads | 4 | 4 | ok | GQA |
| decoder.cross_query_heads | DecoderCrossQHeads | 16 | 16 | ok | |
| decoder.cross_num_key_value_heads | DecoderCrossKvHeads | 16 | 16 | ok | |
| decoder.n_hidden / intermediate_size | DecoderFfn | 8192 | 8192 | ok | |
| decoder.gqa_head_dim / cross_head_dim | HeadDim | 128 | 128 | ok | both 128, shared field OK |
| decoder.cross_hidden_size | (encoder dim) | 1024 | EncoderDim=1024 | ok | cross-attn kv source dim equals EncoderDim, not a separate field but consistent |
| src_vocab_size | TextVocab | 256 | 256 | ok | UTF-8 bytes |
| tgt_vocab_size | AudioVocab | 1028 | 1028 | ok | |
| data.channels / num_channels | Channels | 9 | 9 | ok | DAC codebooks |
| data.text_pad_value | TextPad | 0 | 0 | ok | |
| data.audio_eos_value / eos_token_id | AudioEos | 1024 | 1024 | ok | |
| data.audio_pad_value / pad_token_id | AudioPad | 1025 | 1025 | ok | |
| data.audio_bos_value / bos_token_id | AudioBos | 1026 | 1026 | ok | |
| delay_pattern | DelayPattern | [0,8..15] | [0,8..15] | ok | MaxDelay=15 derived correctly |
| data.text_length / encoder max_position_embeddings | MaxText | 1024 | 1024 | ok | |
| data.audio_length / decoder max_position_embeddings | MaxAudio | 3072 | 3072 | ok | |
| rope_theta / rope_max_timescale | RopeTheta | 10000.0 | 10000 | ok | |
| rope_min_timescale | (none) | 1 | (hardcoded?) | missing | Original config.json sets rope_min_timescale=1. We expose no field. Likely fine if the rope kernel assumes min=1, but add a field or confirm the hardcode. https://huggingface.co/nari-labs/Dia-1.6B/raw/main/config.json |
| normalization_layer_epsilon / norm_eps | NormEps | 1e-5 | 1e-5 | ok | |
| weight_dtype / torch_dtype | (implicit F32) | float32 | F32 codec cast | ok | |
| generate() cfg_scale | CfgScale | 3.0 | 3.0 | ok | matches nari-labs model.py default |
| generate() temperature | Temperature | 1.2 | 1.2 | ok | matches nari-labs default; HF gen_config (0626) uses 1.8 |
| generate() top_p | TopP | 0.95 | 0.95 | ok | matches nari-labs default; HF gen_config uses 0.90 |
| generate() cfg_filter_top_k | TopK | 45 | 45 | ok | matches nari-labs default; HF gen_config uses 50 |
| initializer_range | (none) | 0.02 | n/a | extra/skip | training-only init, no inference shape impact |
| hidden_act | (hardcoded silu) | silu | n/a | ok | SwiGLU/silu assumed in code |

Extra fields in our config with no separate reference key (all benign): Codec (DacConfig.Dac44kHz with TransposeWeightNormDim0=true; this is the downstream DAC decoder choice, correct per the descript-native 44.1 kHz checkpoint).

Action items:
- Add an optional RopeMinTimescale field (default 1) or add a code comment confirming the rope implementation hardcodes min-timescale=1, to match the original config.json. This is the only reference param without an explicit equivalent.
- No value corrections needed: all architecture and the canonical generation defaults match exactly.
- Optionally note in a comment that the HF transformers Dia-1.6B-0626 generation_config.json ships different sampling presets (temperature 1.8, top_p 0.90, top_k 50) in case a user expects HF-default behavior; our values intentionally track the upstream nari-labs generate() defaults.
- No missing variants: there is only one Dia-1.6B checkpoint (the 0626 release is the same architecture re-exported into transformers format).

<details><summary>Sources consulted</summary>

- https://huggingface.co/nari-labs/Dia-1.6B/raw/main/config.json
- https://huggingface.co/nari-labs/Dia-1.6B-0626/raw/main/config.json
- https://huggingface.co/nari-labs/Dia-1.6B-0626/raw/main/generation_config.json
- https://raw.githubusercontent.com/nari-labs/dia/main/dia/model.py

</details>

---

## GPT-SoVITS

Reference: RVC-Boss/GPT-SoVITS stage-1 (s1) AR GPT `Text2SemanticDecoder`, configs `s1longer.yaml` (v1) and `s1longer-v2.yaml` (v2/v2Pro/v3/v4) plus `GPT_SoVITS/AR/models/t2s_model.py`. Variants differ only in `phoneme_vocab_size` (v1=512, v2+=732) and inference `top_k` (v1=5, v2=15); all other architecture dims are identical across versions.

| Reference param | Our field | Reference value | Our value | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| hidden_dim | Hidden | 512 | 512 | ok | s1longer*.yaml model section |
| embedding_dim | (== Hidden) | 512 | 512 (via Hidden) | ok | Reference embedding_dim equals hidden_dim; we fold both into Hidden. |
| n_layer | NumLayers | 24 | 24 | ok | s1longer*.yaml |
| head | NumHeads | 16 | 16 | ok | s1longer*.yaml |
| (hidden/head) | HeadDim | 32 | 32 | ok | Derived 512/16, not a reference key. |
| linear_units | FfnDim | 2048 | 2048 | ok | s1longer*.yaml linear_units; equals hidden*4. |
| vocab_size | SemanticVocab | 1025 | 1025 | ok | s1longer*.yaml vocab_size (1024 codes + EOS). |
| EOS | EosToken | 1024 | 1024 | ok | s1longer*.yaml EOS; also pad_val=1024. |
| phoneme_vocab_size | PhonemeVocab | 512 (v1) / 732 (v2/v2Pro/v3/v4) | 512 | wrong | s1longer.yaml=512, s1longer-v2.yaml=732. The preset is named V2 but carries the v1 value 512. Real v2+ ckpt ar_text_embedding is [732,512]; Russian fork confirms 732 base (extended to 753). Code comment claiming 512 is correct for v2 is wrong. |
| BERT proj input dim | BertDim | 1024 | 1024 | ok | Hardcoded 1024 in t2s_model.py (chinese-roberta-wwm-ext-large). |
| dropout / p_dropout | (none, inference-disabled) | 0 | n/a | ok | Inference-disabled; pos-emb dropout 0.1 is train-only, no shape effect. Correctly omitted. |
| norm_first | (post-norm, hardcoded) | False (post-norm) | post-norm (doc-stated) | ok | t2s_model.py default norm_first=False; our doc comment states post-LayerNorm. |
| top_k (infer) | TopK | 5 (v1 yaml) / 15 (v2 yaml) | 15 | wrong-ish | Matches v2 yaml (15); a V1 preset would need 5. infer_panel code default is -100 (disabled), the yaml provides the real runtime value. |
| temperature (infer) | Temperature | 1.0 | 1.0 | ok | infer_panel default. Not in yaml; from t2s_model.py. |
| repetition_penalty (infer) | RepetitionPenalty | 1.35 | 1.35 | ok | infer_panel default in t2s_model.py. |
| max gen tokens / early_stop | MaxNewTokens | 1500 | 1500 | ok | infer_panel caps at 1500 iterations. |
| norm eps | NormEps | not specified (PyTorch LayerNorm default 1e-5) | 1e-5 | extra | Reference uses default nn.LayerNorm eps (1e-5), so value matches but it is not an explicit reference key. |

Action items:
- Fix PhonemeVocab: the existing `V2` preset must use 732 (the real v2/v2Pro/v3/v4 phoneme_vocab_size), and correct the misleading code comment that asserts 512.
- Add a distinct `V1` factory: PhonemeVocab=512, TopK=5 (per s1longer.yaml).
- Keep V2 TopK=15 (matches s1longer-v2.yaml); ensure V1 uses TopK=5.
- Optionally expose embedding_dim separately or document that Hidden covers both embedding_dim and hidden_dim (they are equal in all released configs).

<details><summary>Sources consulted</summary>

- https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/configs/s1longer.yaml
- https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/configs/s1longer-v2.yaml
- https://raw.githubusercontent.com/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/AR/models/t2s_model.py
- https://huggingface.co/fuckSelf/GPT-SoVITS-Russian

</details>

---

## XCodec (YuE)

Reference: YuE xcodec_mini_infer (m-a-p/xcodec_mini_infer), the `SoundStream` in `models/soundstream_hubert_new.py` parameterized by `final_ckpt/config.yaml`, with the waveform decoder being a descript `dac2.Decoder`. Single 16 kHz checkpoint (ckpt_00360000.pth); no small/medium/large or alternate sample-rate variants exist for this YuE codec, so our single `XCodec16kHz` preset is complete.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| sample_rate | SampleRate | 16000 | 16000 | ok | config.yaml generator.config.sample_rate |
| D (acoustic latent) | AcousticDim | 256 | 256 | ok | config.yaml sets `D: 256` (overrides SoundStream default 128); fc_post2 out and dac2.Decoder input both = D |
| D+768 (RVQ dimension) | LatentDim (=AcousticDim+SemanticDim) | 1024 | 1024 | ok | quantizer dimension=D+768; e = cat([e_acoustic(256), e_semantic(768)]) |
| semantic dim | SemanticDim | 768 | 768 | ok | encoder_semantic encode_channels=768 (HuBERT hf_1_325000) |
| decoder_2 channels | DecoderDim | 1024 | 1024 | ok | dac2.Decoder(D, 1024, ratios) second arg = initial channels |
| ratios (upsample) | DecoderRates | [8,5,4,2] | [8,5,4,2] | ok | config.yaml ratios; hop=320 -> 50 Hz |
| bins (codebook_size) | CodebookSize | 1024 | 1024 | ok | config.yaml bins: 1024 |
| n_q | NCodebooks | 12 | 12 | ok | computed int(1000*6 // (ceil(16000/320)*10)) = 12; we hardcode 12 (decode uses cb0 only) |
| frame_rate / hop_length | FrameRate (derived), hop=prod(rates) | 50 Hz / 320 | 50 Hz / 320 | ok | math derived from rates in both |
| stem Conv1d kernel | StemKernelSize | 7 (padding 3) | 7 | ok | dac.py Decoder WNConv1d(...,kernel_size=7) |
| final Conv1d kernel | DecoderFinalKernelSize | 7 (padding 3) | 7 | ok | dac.py final WNConv1d kernel_size=7 |
| final nn.Tanh | DecoderFinalTanh | commented out (disabled) | false | ok | dac.py `# nn.Tanh()` is commented |
| ResidualUnit dilations | ResidualDilations | [1,3,9] | [1,3,9] | ok | dac.py DecoderBlock ResidualUnit dilations |
| codebook EMA decay / epsilon | (none) | EMA decay 0.99, eps 1e-5 (training EMA) | n/a | unverified | EuclideanCodebook EMA params not surfaced in config.yaml; training-only, decode reads frozen `embed` table directly so no inference-shape impact |
| normalize / causal | (none, fixed) | normalize=False, causal=False | implicit False | unverified | SoundStream.__init__ defaults; not in config.yaml so not over-ridden. causal=False => standard symmetric padding, which our DacDecoder uses; no field needed but worth a code comment |
| channels (Conv1d output) | Channels | 1 | 1 | ok | mono 16 kHz |

### Action items
- None required for parity: all decode-path architecture parameters match the reference checkpoint config.
- Optional clarity: add a short comment on `AcousticDim` noting that 256 comes from `final_ckpt/config.yaml` (`D: 256`) and intentionally overrides the SoundStream dataclass default of 128, so a future reader does not "fix" it to 128.
- Optional clarity: document that `normalize=False`/`causal=False` (SoundStream defaults, not set in config.yaml) are why the reused `DacDecoder` uses standard symmetric padding.
- Optional robustness: if you ever support deriving `NCodebooks` from `target_bandwidths`, add a `TargetBandwidths` field; currently hardcoding 12 is correct and decode only consumes cb0, so this is non-blocking.

<details><summary>Sources consulted</summary>

- https://huggingface.co/m-a-p/xcodec_mini_infer/raw/main/final_ckpt/config.yaml
- https://huggingface.co/m-a-p/xcodec_mini_infer/raw/main/models/soundstream_hubert_new.py
- https://huggingface.co/m-a-p/xcodec_mini_infer/raw/main/descriptaudiocodec/dac/model/dac.py

</details>

---

## DAC codec

Reference: descript-audio-codec (Kumar et al. 2023), HF checkpoints descript/dac_44khz, dac_24khz, dac_16khz, plus the upstream DAC dataclass in descriptinc/descript-audio-codec dac/model/dac.py. We expose all three official variants (Dac44kHz, Dac24kHz, Dac16kHz) and they all match the reference. No missing or wrong values were found.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| encoder_hidden_size / encoder_dim | EncoderDim | 64 | 64 | ok | Same across all 3 checkpoints (HF configs + dac.py default). |
| decoder_hidden_size / decoder_dim | DecoderDim | 1536 | 1536 | ok | Same across all 3 checkpoints. |
| downsampling_ratios / encoder_rates | EncoderRates | [2,4,8,8] (44k); [2,4,5,8] (24k,16k) | [2,4,8,8]; [2,4,5,8] | ok | Per-variant presets correct (HF configs). |
| upsampling_ratios / decoder_rates | DecoderRates | [8,8,4,2] (44k); [8,5,4,2] (24k,16k) | [8,8,4,2]; [8,5,4,2] | ok | Mirror of ratios, correct per variant. |
| n_codebooks | NCodebooks | 9 (44k); 32 (24k); 12 (16k) | 9; 32; 12 | ok | All three correct. |
| codebook_size | CodebookSize | 1024 | 1024 | ok | Constant across variants. |
| codebook_dim | CodebookDim | 8 | 8 | ok | Constant across variants. |
| sampling_rate / sample_rate | SampleRate | 44100 / 24000 / 16000 | 44100 / 24000 / 16000 | ok | Correct per variant. |
| hidden_size / latent_dim | LatentDim (computed) | 1024 | 1024 (64 * 2^4) | ok | Reference HF hidden_size=1024; dac.py derives latent_dim = encoder_dim * 2^len(encoder_rates). We compute the same value; correct but derived, not stored. |
| hop_length | FrameRate / implicit | 512 (= product of ratios) | product of EncoderRates (512) | ok | Not a stored field; equals product of EncoderRates (2*4*8*8=512, 2*4*5*8=320 for 24k/16k). HF lists 512 for all three even though 24k/16k product is 320; this is an HF config artifact, the true hop is the ratio product, which we compute correctly. |
| ResidualUnit dilations | ResidualDilations | [1,3,9] | [1,3,9] | ok | From dac.py ResidualUnit. |
| residual conv kernel | ResidualKernelSize | 7 | 7 | ok | dac.py kernel 7 (dilated) + 1. |
| stem/init conv kernel | StemKernelSize | 7 | 7 | ok | dac.py initial conv kernel 7, pad 3. |
| quantizer_dropout | (none) | False / 0.0 | n/a | ok (excluded) | Training-only, inference-disabled, no tensor-shape effect. Correctly omitted. |
| commitment/codebook_loss_weight | (none) | 0.25 / 1.0 | n/a | ok (excluded) | Training-loss params, no inference effect. Correctly omitted. |
| (none) | DecoderKernelSizes | n/a | null (DAC) | extra | For Spark-TTS BiCodec [16,11,8,4]; null keeps DAC's 2*stride convention. |
| (none) | TransposeWeightNormDim0 | n/a | false | extra | Selects weight_norm composition (DAC default actually norms dim 0); see action items. |
| (none) | DecoderFinalTanh | n/a | true | extra | DAC default true; YuE x-codec dac2 sets false. |
| (none) | EncoderProjKernelSize / DecoderFinalKernelSize | n/a | 3 / 7 | extra | Matches DAC (encoder proj kernel 3, decoder final conv kernel 7); legitimately exposed. |

Action items:
- No required fixes for DAC parity: all reference architecture params are present with matching values across all three variants.
- (Doc nit) The XML comment on TransposeWeightNormDim0 states the descript-audio-codec default norms over dim 0, yet the field default is false. Confirm the actual DAC weight loader passes TransposeWeightNormDim0=true for descript checkpoints (the default false would only be correct for EnCodec/SeaNet-style decoders), otherwise flip the default or document why the loader overrides it.
- (Optional robustness) LatentDim is derived assuming len(EncoderRates)==4; this holds for all official DAC checkpoints but would break for any non-4-stage variant. Fine to leave as is given DAC only ships 4-stage configs.
- No missing variants: 44.1/24/16 kHz all covered.

<details><summary>Sources consulted</summary>

- https://huggingface.co/descript/dac_44khz/raw/main/config.json
- https://huggingface.co/descript/dac_24khz/raw/main/config.json
- https://huggingface.co/descript/dac_16khz/raw/main/config.json
- https://raw.githubusercontent.com/descriptinc/descript-audio-codec/main/dac/model/dac.py

</details>

---

## Oobleck VAE

Reference: diffusers `AutoencoderOobleck` (Stability AI Stable Audio autoencoder). Variants we expose: StableAudioOpen (44.1 kHz, hop 2048, ratios 2,4,4,8,8) and AceStep15 (48 kHz, hop 1920, ratios 2,4,4,6,10). Both match their reference configs exactly; no missing or wrong params were found.

| Reference param | Our field | Reference value | Our value | Status | Notes |
|---|---|---|---|---|---|
| encoder_hidden_size | EncoderHiddenSize | 128 | 128 | ok | diffusers `__init__` default (autoencoder_oobleck.py) |
| downsampling_ratios | DownsamplingRatios | [2,4,4,8,8] (SAO); [2,4,4,6,10] (ACE 1.5) | [2,4,4,8,8] / [2,4,4,6,10] | ok | SAO = stable_audio_2_0_vae.json strides; ACE = Ace-Step1.5/vae/config.json |
| channel_multiples | ChannelMultiples | [1,2,4,8,16] | [1,2,4,8,16] | ok | matches both reference configs |
| decoder_channels | DecoderChannels | 128 | 128 | ok | diffusers default; stable_audio_2_0 decoder channels=128 |
| decoder_input_channels | DecoderInputChannels | 64 | 64 | ok | = latent_dim 64 in both refs |
| audio_channels | AudioChannels | 2 | 2 | ok | stereo in both refs |
| sampling_rate | SamplingRate | 44100 (SAO); 48000 (ACE 1.5) | 44100 / 48000 | ok | ACE confirmed 48000 in vae/config.json |
| (hop, derived) | HopLength | 2048 (SAO); 1920 (ACE) | 2048 / 1920 | ok | product of ratios; equals downsampling_ratio in refs |
| use_snake | (hardcoded) | true | always-on Snake1d | ok | diffusers hardcodes Snake1d; original stable_audio_2_0_vae.json sets use_snake=true |
| final_tanh | (hardcoded) | false | no output tanh | ok | diffusers has no final tanh; original config final_tanh=false |
| residual dilations | (hardcoded) | [1,3,9] | n/a (model code) | ok | hardcoded in OobleckResidualUnit per diffusers source |
| ChannelMultiples leading 1 | ChannelMultiples (full list) | implicit 1 prepended | already includes leading 1 | extra/ok | our list already starts with 1; doc comment says "implicit 1 prepended at build time", verify build code does not double-prepend |

### Action items
- No parameter fixes required: every reference architecture/inference param is present with a matching value, and both variants match their reference configs.
- Minor: double check the build/model code does not prepend an extra leading 1 to `ChannelMultiples`, since the stored list already starts with 1 while the doc comment says a leading 1 is prepended at build time (potential off-by-one in stage count if both happen).
- Optional documentation: note in the record that `use_snake=true` and `final_tanh=false` are fixed by the diffusers Oobleck architecture (hardcoded), so no config field is needed.

<details><summary>Sources consulted</summary>

- https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/autoencoders/autoencoder_oobleck.py
- https://huggingface.co/ACE-Step/Ace-Step1.5/raw/main/vae/config.json
- https://raw.githubusercontent.com/Stability-AI/stable-audio-tools/main/stable_audio_tools/configs/model_configs/autoencoders/stable_audio_2_0_vae.json
- https://raw.githubusercontent.com/Stability-AI/stable-audio-tools/main/stable_audio_tools/configs/model_configs/autoencoders/stable_audio_1_0_vae.json

</details>

---

## OpenVoice

Reference: myshell-ai OpenVoice V1 and V2 tone-color converter, ReferenceEncoder (speaker/tone encoder) in `openvoice/models.py`, parameterized by the converter `config.json`. Our `OpenVoiceSpeakerConfig` is scoped only to this ReferenceEncoder (not the flow or HiFiGAN decoder). V1 and V2 converter configs are byte-for-byte identical for every encoder-relevant field, so a single Default preset covers both.

| Reference param | Our field | Reference value | Our value | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| data.sampling_rate | SampleRate | 22050 | 22050 | ok | Same in V1 and V2 (HF converter config.json). |
| data.filter_length (n_fft) -> spec_channels = filter_length//2+1 | SpecChannels | 1024 -> 513 | 513 | ok | filter_length 1024 in both V1/V2 configs; 1024//2+1 = 513. LayerNorm is over spec_channels (models.py). |
| conv stack channels (1->32->32->64->64->128->128) | Channels | [32,32,64,64,128,128] | [32,32,64,64,128,128] | ok | Six weight_norm Conv2d layers, ReLU after each (models.py ReferenceEncoder). |
| conv kernel_size | KernelSize | (3,3) | 3 | ok | All six layers 3x3 (models.py). |
| conv stride | Stride | (2,2) | 2 | ok | All six layers stride 2x2 (models.py). |
| conv padding | Padding | (1,1) | 1 | ok | All six layers pad 1 (models.py). |
| GRU hidden_size = 256 // 2 | GruHidden | 128 | 128 | ok | Hardcoded 256//2 in reference (single layer, batch_first). Our field defaults 128. See note below. |
| model.gin_channels (proj out_features) | Gin | 256 | 256 | ok | proj = nn.Linear(128, gin_channels) with gin_channels 256 in both V1/V2. |
| layernorm flag | (hardcoded true) | True | applied | ok | Reference default layernorm=True; our encoder applies the input LayerNorm unconditionally, matching the V2 converter usage. |

Action items:
- No required fixes: every ReferenceEncoder parameter is present with the correct value, and the single Default preset correctly covers both OpenVoice V1 and V2 (their converter configs are identical for all encoder fields).
- Optional doc clarity: the comment on GruHidden says "gin / 2", but in the reference the GRU hidden size is the literal 256 // 2 = 128 and the proj is nn.Linear(128, gin_channels). It is not coupled to Gin in code. Consider rewording the comment to "fixed 128 (256 // 2 upstream)" so changing Gin alone does not imply GruHidden auto-adjusts.
- No missing variants: there is no separate small/medium/large or alternate sample-rate converter checkpoint; V1 and V2 share these encoder hyperparameters.

<details><summary>Sources consulted</summary>

- https://huggingface.co/myshell-ai/OpenVoiceV2/raw/main/converter/config.json
- https://huggingface.co/myshell-ai/OpenVoice/raw/main/checkpoints/converter/config.json
- https://raw.githubusercontent.com/myshell-ai/OpenVoice/main/openvoice/models.py

</details>

---
