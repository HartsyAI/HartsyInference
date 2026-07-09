# TTS Model Campaign Scorecard

Goal: every TTS model correct (Python ref vs our engine, STT-verified), documented for time+output, then per-model optimization to match/beat Python. GPU-only (RAM OOMs). Test sentence: "The speech synthesizer is now working correctly." Target STT = exact.

| Model | Python: time / RTF / RMS / STT | Ours: time / RMS / STT | Correctness | Perf vs Python | Notes |
|---|---|---|---|---|---|
| **F5-TTS** | 3.47s / RTF 0.729 / 0.153 / ✅ | ~opt / 0.109 / ✅ | ✅ FIXED | attn 12x (SDPA), text-cache; host-conv bottleneck remains | resampler+Vocos-gain fixed 2026-07-09 |
| **Kokoro-82M** | 0.104s / RTF 0.030 / 0.048 / ✅ | 1.49s / 0.041 / ✅ (minor slur) | ✅ | **14x slower** (RTF 0.458 vs 0.030) | conv-heavy decoder → host-conv bottleneck. Perf target. |
| **MeloTTS** | melo lib NOT installed | — | ✅ (prior) | — | need melo install for ref |
| **Dia-1.6B** | — | 328s / RTF 16 / 0.097 / **intelligible but LOOPS** + never-EOS (20s cap) | 🔧 **MAJOR progress** (silent→coherent), loop remains | slow | Revert cond→guided FIXED the silence: STT now reads the prompt correctly ("The speech synthesizer is now working correctly, and I am...") then repeats. Needs short-prompt warning heeded (used long dialogue text). Remaining: repetition-loop + never-EOS — likely the unpadded-text-to-encoder issue (ref pads to 1024) or sampling. |
| **Qwen3-TTS 0.6B** | — | silent RMS0 | ✗ **DRIVER bug** (fix known, not installed in AudioLab) | — | NOT among AudioLab's 17 engines → can't verify via SwarmUI. Fix: CustomVoice needs speaker id (3061=Ryan/"Ryan"); default -1→voice-design→no speaker→EOS@2frames→silent. Verify via in-process test QWEN3TTS_SPEAKER=3061 (RAM-risky) or install the engine. BuildPrefill already correct. |
| **NeuTTS-air** | — | 66s / RTF 5.3 / 0.119 / garbled babble (default voice) | 🔧 **decoder FIXED**, needs encoder | slow | NeuCodecDecoder fully rewritten to real HF X-Codec2 checkpoint (acoustic_decoder + FSQ + iSTFT) — now decodes real audio (was crash). But coherent output needs a REAL reference voice via the X-Codec2 ENCODER (CodecEnc.*/semantic+acoustic encoders) which isn't mapped; default-voice path is off-distribution → rambles (12s babble). Rep-penalty=1.0 applied. Remaining: implement the encode path. |
| **FishSpeech-1.5** | (fish_speech lib not installed) | 6.44s / RTF 1.95 / 0.095 / ✅ | ✅ **FIXED** | heavy dual-AR (RTF ~2) | FSQ dequant convention fix worked: RMS too-quiet→0.095, STT perfect. |
| **VibeVoice-1.5B** | — | 65s / RTF 15 / 0.046 / near-complete ("The speech and society is now working correctly.") | 🔧🔧 **MUCH improved** (noise + CFG) | slow | Noise fix + diffusion CFG (cfg 1.3, dual-stream, greedy): edges now correct ("The speech...is now working correctly"), only "synthesizer"→"and society" slur remains (short-prompt / neg-stream simplification). Was "Each synthesizer...for EGLE" before CFG. |
| **Bark** | — | ? | ? | — | installed, untested here |
| **Chatterbox** | — | ? | ? | — | installed, untested here |

## Shared engine perf levers (help many models)
1. **GPU grouped/depthwise Conv1D kernel + input-embed concat kernel — THE bottleneck.** F5 end-to-end 171s→161s AFTER SDPA(12x)+text-cache = barely moved, because attention was NOT the bottleneck. The dominant per-forward cost is host-loop code that reads `.DataPointer` (forces a device sync every call): F5InputEmbed concat loop (per-step, reads noisyMel/condMel/text on host) + F5ConvPosEmbed grouped Conv1D + F5ConvNeXtV2 DepthwiseConv1D + Vocos DepthwiseConv1D. Hits F5, Kokoro (14x slow), Vocos, Melo. This kernel is the real perf win.
2. DiT/transformer attention → ScaledDotProductAttention (TF32 GEMM), not monolithic FlashAttention. 12x on F5 attention kernel (but attention wasn't the e2e bottleneck — see #1). Applies to any DiT.

## RAM/GPU rules
- One GPU (RTX 3060 12GB). Gens SERIAL. RAM 31GB, VSCode eats ~24GB → big-model in-process = OOM.
- Prefer SwarmUI API (ProcessTTS/GenerateText2Image) for our engine — separate process, AudioLab unloads other providers under memory pressure.
- Python refs: standalone scripts, GPU, freed between runs.
