# Inworld Voice Isolator

Source snapshot 2026-11-24. Not a status page: see [MODEL_STATUS_AUDIO.md](../Checklists/MODEL_STATUS_AUDIO.md) for what runs and [PARITY_VERIFICATION.md](../Checklists/PARITY_VERIFICATION.md) for evidence.

## Pins

- Weights: `inworld/voice-isolator` @ `9b0285b3fe8403aa8e2f1dbb2165a80a4294c970` (ungated, `license:apache-2.0`).
- Reference code: `inworld-ai/voice-isolator` @ `c73b3fab81a870410cbac50d2f05f25edc4d10aa`: `src/inworld/voice_isolation/{model,streaming,scan,isolator,registry}.py`, golden fixtures in `tests/fixtures/`.

| Model | File | Bytes | sha256 | Parameters |
|---|---|---|---|---|
| offline (`-m voice-isolator`) | `denoiser.safetensors` | 256,260,128 | `647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53` | 64,047,112 |
| realtime (`-m voice-isolator-realtime`) | `denoiser_realtime.safetensors` | 232,780,800 | `76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f` | 58,178,056 |

Both are F32. The architecture is in each file's `__metadata__["config"]` and the loader refuses fields it does not implement (`dual_mode`, `mode_film`, `offline_passes` above 1, `exclusive_axes` other than `time`).

## Licence

Code and weights are Apache-2.0. The upstream NOTICE: weights trained on Inworld first-party data (some bought from vendors under private terms) and VocalSet (CC BY 4.0), IndicVoices-R (CC BY 4.0), VCTK (ODC-By 1.0). The golden fixtures carry a modified LibriSpeech clip under CC BY 4.0. No territory or commercial restriction found.

## Network

STFT (n_fft 2048, hop 480, periodic Hann, centred) at 48 kHz mono, cut into 46 bands (bin ranges from `band_plan`, rounded half to even), one width-512 token per band and frame. Seven blocks `T M T M T M T`:

- `T`: time transformer within each band (pre-norm, fused qkv, rotate-half RoPE base 10000 over head dim 64, SDPA, exclusive attention `o -= (o.v / max(v.v, 1e-6)) v` per head, erf-GELU FFN 4x), then a band transformer across the 46 bands of each frame (`nn.MultiheadAttention` packed projections, no exclusive step).
- `M`: Mamba along time then across bands, each `LayerNorm(output_proj(cat(fwd + x, bwd + x)))`; the realtime time Mamba is forward only. Layer: in_proj, depthwise causal conv (4 taps) + SiLU, x_proj to (dt 32, B 16, C 16), softplus step, selective scan, SiLU gate.
- Decoder per band: FiLM with the constant controls `[1, 0]`, LayerNorm, Linear, erf-GELU, Linear to `4 * bins`; output `x + iSTFT(spec * 2tanh(mask) + 0.5 |spec| tanh(res) / max(1, |tanh(res)|))`.
- Offline renders 8 s windows with a 2 s linear crossfade (`enhance`: every window full, the last ends at the clip end). Realtime: attention window 1200 frames, look-ahead 0, 20 or 40 ms read from the decoder, latency `n_fft + hop * lookahead` samples (62.7 ms at 2 frames); the stream zero-pads the clip edges where the offline forward reflects them.
- Windowed stream (the reference's 600-frame window, 200-frame crossfade): a second lane starts 200 frames before each boundary with the Mamba states cloned and attention empty; the raw decoder outputs blend linearly, then lane two carries on.

## Engine surface

`hartsy fx separate in.wav -m voice-isolator[-realtime]` (or `IFxService.SeparateAsync`) returns a `voice` and a `background` stem, mono 48 kHz, `background = input - voice`. It runs on the engine's backend with the STFT on the host; `IBackend.SelectiveScan` has a host reference (no CUDA kernel yet). 16-bit PCM output clamps peaks; the reference scales a peak above 1.0 down instead.

## Reuse map

`IBackend` Linear, LayerNorm, SDPA with additive mask and Add; `Fft`/`FftPlan` for the host STFT; `ModelAssets` safetensors loader; `FxService.SeparateAsync` and `StemsResult` from the Kim_Vocal_2 separator. New: `IBackend.SelectiveScan` (Mamba-1). LLM `MambaModel` still carries its own host scan.

## Reference runs

`tests/python-reference/voice_isolator/dump_tiny_reference.py` builds tiny random-weight checkpoints with the upstream PyTorch network and dumps whole-clip, offline-render and stream outputs; `VoiceIsolatorTinyReferenceTests` compares them. The real-weight check is `VoiceIsolatorGoldenParityTests` against the upstream fixtures (`tests/fixtures/{offline,realtime_stream}_{3s,11s}`, fp32 CPU, tolerance 1e-4).

## Not verified

GPUs below Ampere (the card lists A100, H100, B200, B300 and CPU only); third-party ports (GGUF, MLX, WebGPU, ExecuTorch) were not read.
