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

`hartsy fx separate in.wav -m voice-isolator[-realtime]` (or `IFxService.SeparateAsync`) returns a `voice` and a `background` stem, mono 48 kHz, `background = input - voice`. Where it runs (`FxService.IsolateVoiceAsync`): **both models** on the engine's backend when that is CUDA or Vulkan, with a CPU rerun when the card is full. The CLI passes `-b` through for both isolators (Demucs still forces CPU). The STFT is on the host on every backend (`IBackend.Stft` has no GPU path). The Mamba causal-conv runs on CUDA and Vulkan backend ops (no host round trip per block); on the CPU it keeps the host loop, which is faster there. The network runs under `ExactGemmScope`, so CUDA matches the CPU to the same level as Vulkan.

Measured on the 3060 against the CPU on the same golden clips (fp32, wall time of the render/stream call):

| Case | CPU | CUDA | Vulkan |
|---|---|---|---|
| offline 3 s | 25.8 s | 4.8 s | 6.6 s |
| offline 11 s | 184 s | 21.0 s | 30.5 s |
| realtime 3 s | 38.0 s | 7.6 s | 18.7 s |
| realtime 11 s | 213 s | 32.4 s | 68.9 s |

On the final head the device runs beat the CPU in every case, so both models route to the engine backend. Times are the log time of the render or stream call, measured 2026-11-26 on commit f9b2a2178 (the device conv and exact-GEMM code; the later commits change docs and routing only, not the goldens' path). The CPU column is from the baseline run at 5bf639317 (the CPU path is unchanged since; CPU reruns on the shared machine took 30 to 396 s). Every device case passes the 1e-4 tolerance, and the device errors are at the CPU level (CUDA 1.2e-7 to 1.8e-7, Vulkan 1.7e-7 to 5.7e-7) because the network runs under exact GEMMs. The earlier CUDA error (2.6e-5 to 8.5e-5) was TF32 GEMMs.

16-bit PCM output clamps peaks; the reference scales a peak above 1.0 down instead.

## Reuse map

`IBackend` Linear, LayerNorm, SDPA with additive mask and Add; `Fft`/`FftPlan` for the host STFT; `ModelAssets` safetensors loader; `FxService.SeparateAsync` and `StemsResult` from the Kim_Vocal_2 separator. New: `IBackend.SelectiveScan` (Mamba-1), with CUDA (`ssm_selective_scan_f32`) and Vulkan (`ssm_selective_scan`) kernels, verified against `SelectiveScanReference`. The stream attention cache runs on backend ops (`CopyStrided`, `Rotary`). The LLM `MambaModel` keeps its own host scan until the item C change (#1191) moves it onto this op.

## Reference runs

`tests/python-reference/voice_isolator/dump_tiny_reference.py` builds tiny random-weight checkpoints with the upstream PyTorch network and dumps whole-clip, offline-render and stream outputs; `VoiceIsolatorTinyReferenceTests` compares them. The real-weight check is `VoiceIsolatorGoldenParityTests` against the upstream fixtures (`tests/fixtures/{offline,realtime_stream}_{3s,11s}`, fp32 CPU, tolerance 1e-4). It passed 4/4 on real weights on 2026-11-26 (worst max abs 5.7e-7); figures in [PARITY_VERIFICATION.md](../Checklists/PARITY_VERIFICATION.md).

## Not verified

GPUs below Ampere (the card lists A100, H100, B200, B300 and CPU only); third-party ports (GGUF, MLX, WebGPU, ExecuTorch) were not read. The `fx separate sample_14.wav` CLI runs for both models passed on the 3060 (offline `-b cuda` 25 s; realtime `-b cuda` 37 s, on CUDA; 2026-11-26, commit f9b2a2178). The STFT stays on the host; the Mamba conv runs on the device (CUDA and Vulkan) and on the host on the CPU (see Engine surface).
