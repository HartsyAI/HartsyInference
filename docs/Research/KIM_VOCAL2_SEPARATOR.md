# Kim_Vocal_2 vocal separator (UVR MDX-Net)

The C# port of the vocal separator that LongCat-Video-Avatar runs before its audio encoder. Code: `src/HartsyInference.Audio/Models/Separation/` (network, STFT loop),
`src/HartsyInference.ModelAssets/CheckpointConverters/KimVocal2OnnxConverter.cs` (graph binder), `src/HartsyInference.Engine/Audio/Fx/` (runner, catalog).
Status and evidence: [MODEL_STATUS_AUDIO.md](../Checklists/MODEL_STATUS_AUDIO.md).

## Pins

| Item | Value |
|---|---|
| File | `vocal_separator/Kim_Vocal_2.onnx` in `meituan-longcat/LongCat-Video-Avatar-1.6` (Avatar 1.5 ships the same file), 66,759,214 bytes |
| sha256 | `ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b` (the LFS oid) |
| MDX row | md5 of the last 10,240,000 bytes `970b3f9492014d18fefeedfe4773cb42` (checked) gives n_fft 7680, dim_f 3072, dim_t 2^8, compensate 1.009 in audio-separator's `mdx_model_data.json` |
| Reference | LongCat-Video `longcat_video/audio_process/vocal_separator.py` @ `58eb336edc5c8c63389838727e40b8c18434e54e` over `audio-separator==0.30.2` (`mdx_separator.py`, `uvr_lib_v5/stft.py`, `spec_utils.normalize`) |
| Licence | Not stated by the file, by the LongCat repo (its MIT covers the Avatar weights), or by the UVR repositories read for this record (the UVR GUI code is MIT; nothing says the weights are; `TRvlvr/model_repo` has no LICENSE). Recorded as unknown; the identity row says `other`. |

## Graph (read from the file, not from the plan)

178 nodes, 220 initializers, input `input` and output `output` both `[B, 4, 3072, 256]` (channels: left real, left imaginary, right real, right imaginary; then frequency, then time).
Ops: Conv 40, Relu 66, MatMul 22, BatchNormalization 27, Add 11, Mul 5, ConvTranspose 5, Transpose 2. Conv, MatMul and ConvTranspose weights have anonymous numeric names;
only BatchNorm parameters keep their module names. Binding is by walking the node order (`KimVocal2OnnxConverter`), checking every op, attribute and data-flow edge.

```
first:      Conv 1x1 (4 → 48) → Relu → Transpose [0,1,3,2]            # layout becomes [C, T, F]
encoder i:  3 × (Conv 3x3 pad 1 → Relu) → TDF → Add(residual)         # i = 0..4, widths 48 96 144 192 240
            Conv 2x2 stride 2 → Relu                                   # halves T and F
bottleneck: 3 × (Conv 3x3 → Relu) → TDF → Add                          # width 288, F 96, T 8
decoder i:  ConvTranspose 2x2 stride 2 → BatchNorm → Relu → Mul(skip)  # skip = encoder (4 − i) block output; multiplicative, not additive
            3 × (Conv 3x3 → Relu) → TDF → Add
last:       Transpose [0,1,3,2] → Conv 1x1 (48 → 4)                    # no activation
TDF:        MatMul over F (F → F/8) → BatchNorm → Relu → MatMul (F/8 → F) → BatchNorm → Relu
```

F per level 3072, 1536, 768, 384, 192, 96 (the MatMul weights are `[in, out]`, no bias). 16,667,524 conv and linear parameters plus BatchNorm vectors.
Findings that differ from the earlier plan: the decoder skips are `Mul`; the exporter already folded the BatchNorms that follow convs (the conv weights are
anonymous and carry biases), so what remains is the 22 TDF BatchNorms (per-channel affine after a MatMul shared across channels, which cannot fold into it) and the
5 ConvTranspose BatchNorms (folded here along weight axis 1, `[Cin, Cout, kh, kw]`). Epsilon is read from each node (`epsilon`, default 1e-5).

Ops the engine did not have: none. `IBackend.Stft` is radix-2 only (n_fft 7680 = 2⁹·3·5 is not a power of two) and the CPU `ConvTranspose2d` is the scalar default, so the STFT uses `FftPlan`
(mixed radix 2/3/5, the same plan Rnnoise uses) and `IStft.Apply` (Bluestein for 7680); the per-channel affine runs as a 1x1 depthwise conv and `Linear` takes the MatMul.

## Pipeline (matches LongCat's `separate_vocals`)

1. Mix: 44.1 kHz stereo (a mono source duplicated), peak-normalized to 0.9 across both channels when it exceeds 0.9 (`spec_utils.normalize`; the `min_peak` branch never fires).
2. Optional crop to `min(ceil(duration · 44100) + chunk, N)` samples. LongCat passes `generate_duration + 1 s`.
3. chunk = 1024 · 255 = 261,120; trim = 3840; gen = chunk − 2 · trim = 253,440; pad = gen + trim − N mod gen; mixture = zeros(trim) ‖ mix ‖ zeros(pad); step = ⌊0.75 · chunk⌋ = 195,840.
4. Per chunk (zero-padded to chunk): `torch.stft(n_fft 7680, hop 1024, periodic Hann, center, reflect)`, first 3072 bins, bins 0–2 zeroed, network, zero-pad bins to 3841, `torch.istft`.
5. Blend with `numpy.hanning(actual)` (symmetric), divide by the window sum, trim, × compensate 1.009, normalize to 0.9 again, mean of the two channels, resample 44.1 → 16 kHz, crop to `ceil(target · 16000 / 44100)`.

`audio-separator`'s own `demix` and LongCat's `_demix_batched` do the same arithmetic; LongCat batches two chunks per model call, which changes nothing numerically. This port follows LongCat's function.
The port runs one chunk per call.

Not reproduced bit for bit: librosa's default `soxr_hq` resampler. The port uses the engine's `SincResampler.ResampleSoxrHqStandIn`, a Kaiser-windowed sinc fitted on 2026-11-26 to the response measured from python-soxr 1.1.0 (-6 dB at 0.956872 of the lower rate's Nyquist, beta 13.04, 97 samples of the lower rate on each side). Fitted per ratio, this form matches soxr's response to under 1e-6 for 24 kHz to 16 kHz and 44.1 kHz to 22.05 kHz, with half-widths of 96.8 and 97.3; the one set shipped uses 97, which leaves 7.2e-5 on white noise at 24 to 16 kHz. For ratios with many phases (44.1 kHz to 16 kHz, 24 kHz to 44.1 kHz) soxr's measured response droops by about 1e-3 below the cut-off, and at 44.1 to 16 kHz its sine gain in the transition band differs from the shipped filter by up to 6e-3 (the first stand-in: 6e-2); neither is reproduced. C# against `soxr_hq`, relRMS, with the first stand-in (64-crossing squared-cosine sinc, roll-off 0.955) and the current one:

| Signal, rates | First | Current |
|---|---|---|
| Separated vocals, second clip, 44.1 to 16 kHz | 2.5e-3 | 2.3e-4 |
| Separated vocals, 8 s test mix (stand-in reference stem), 44.1 to 16 kHz | 2.7e-4 | 2.5e-5 |
| LongCat `man.mp3` (82 s), 44.1 to 16 / 22.05 kHz | 1.8e-3 / 1.4e-3 | 1.7e-4 / 1.6e-5 |
| The same speech at 24 kHz to 16 / 22.05 / 44.1 kHz | 1.8e-3 / 1.4e-3 / 8.1e-4 | 1.4e-5 / 1.6e-4 / 8.6e-5 |
| The same speech at 48 kHz to 16 / 44.1 kHz | 1.8e-3 / 3.5e-6 | 1.4e-5 / 3.3e-6 |
| White noise, four rate pairs | 9.5e-3 to 9.9e-3 | 7.2e-5 to 1.1e-3 |

The 24 and 48 kHz inputs were made from `man.mp3` with `soxr_vhq`. `KimVocal2ParityTests.SixteenKilohertzVocalsMatchSoxrHq` holds the first two rows below 1e-3 on the reference's own stem; `SoxrHqStandInTests` pins the 24 to 16 kHz and 44.1 to 16 kHz transition bands without Python. Kernel rows are about 20% longer than before at 44.1 to 16 kHz.
Loudness normalization is not part of the separator: LongCat applies it downstream (`-23 LUFS`, off for Avatar 1.6, on for 1.5 and 1.0).

## Engine surface

- `hartsy fx separate <wav> -m kim-vocal-2` returns one `vocals` stem (44.1 kHz stereo, normalized to 0.9). Demucs-only options (`Shifts`, `Segment`) are refused; `Overlap` is honoured (default 0.25).
- `VideoRequest.SeparateDrivingVocals` / `hartsy video --driving-audio <wav> --separate-vocals`: `VideoService` replaces the driving audio with the vocals stem (44.1 kHz stereo, 16-bit PCM WAV) before the recipe runs; the recipe decodes it to the rate and channels it needs. Alongside that clip it hands the recipe 16 kHz mono taken from the float stem (`MdxSeparator.ToMono`: channel mean, SoXR-HQ stand-in), the reference route; LongCat-Avatar encodes that and cuts it to the generated clip length plus one second, as `separate_vocals` does. The separator input is resampled to 44.1 kHz with the same stand-in. Measured on the 8 s test mix with the first stand-in (CPU, 2026-11-26), the earlier route (16-bit stereo WAV, decode, mean, the engine's Kaiser polyphase `Resampler`) differs from this one by relRMS 3.5e-3, max 3.1e-3: the resampler accounts for 3.5e-3 and the 16-bit round trip for 8.3e-5; on a 24 kHz speech excerpt the two input resamplers differ by 1.0e-3. Against `soxr_hq` itself (librosa 0.11.0) that route was 2.7e-4 and the earlier one 3.4e-3 on the same vocals. The reference pads short speech with zeros up to that duration, while the engine recipe repeats the last audio window (unchanged here). The pre-step separates only the clip length of audio (frames / fps, plus one second) when the frame total and fps are known, and the whole track otherwise; it runs before generation and is logged, not reported through step progress. A request without driving
  audio is refused at planning (`video.separate_vocals.no_driving_audio`); a recipe without driving-audio support refuses the audio itself. Any soundtrack a recipe attaches from the driving clip is then the separated vocals;
  a caller who wants the original mix heard passes it as `VideoAudioInput`.
- `fx separate` runs the network on the engine's backend when that is CUDA or Vulkan, and on the CPU otherwise; if the card has no room (a resident image or video model) it logs a warning and separates on the CPU. The driving-audio pre-step of a video request stays on the CPU: it runs just before a model that needs the card, and no such request has run on a GPU. The STFT is host code, and Vulkan has its own 2D transposed convolution since 2026-11-26 (before that it took the host default). `KimVocal2Network.Forward` runs under `ExactGemmScope` (see GPU runs below). One chunk is 261,120 samples and the window advances 195,840 (4.44 s); a clip of N seconds takes about ⌈(N · 44100 + 3840 + pad) / 195,840⌉ chunks (3 for the 8 s test clip). The network alone took 37 s per chunk on 8 logical cores in a Release build on random input, peak working set 1.8 GB. The parity test passes on the real weights on CUDA and Vulkan (see Evidence).
- Weights are read from the `.onnx` at load (about a tenth of a second); the Hartsy-hosted safetensors repack of decision D1 is not made, and nothing was uploaded.

## Known limits

- `ExactGemmScope` flips one backend-wide `HighPrecisionGemm` property with no synchronisation, and `fx separate` uses it on the engine's own backend. Audio, image and video generations on one device are serialised by the audio lock and `DeviceGate`, so nothing else runs during a `Forward`. With `vram.sameGpuConcurrent` on, or for host work that uses `ComputeBackend` without the gate, concurrent work also gets exact GEMMs (BF16 weights are then upcast on every call) and overlapping scopes can restore each other's value, as with the other models that set the flag.

## Evidence (2026-11-23, CPU F32, Debug build of the CLI)

| Check | Result |
|---|---|
| Graph binds | the real file binds 156 tensors (5 encoder levels 48/96/144/192/240, bottleneck 288, 3072 bins); `KimVocal2OnnxConverterTests` pin the walk on a small graph with the same op order |
| Network vs plain-loop reference | relRMS < 1e-5, random small config (`KimVocal2NetworkTests`); this checks wiring and layout, not the real weights |
| STFT front end vs numpy | chunk-0 spectrogram of the test mix against a float64 `numpy.fft.rfft` implementation of `torch.stft`: relRMS 1.6e-7 |
| Overlap-add | identity network returns the input at relRMS < 2e-5 over several overlapping chunks (`MdxSeparatorTests`) |
| Constructed mixed clip | `hartsy fx separate mix.wav -m kim-vocal-2`: 8 s stereo mix constructed from known stems = clean speech (a Haru reply) + synthetic backing (chord pad, bass, kick, hat). SI-SDR of the vocals stem against the clean speech 17.10 dB (the mix itself: 0.51 dB), correlation 0.990, against the backing 0.01. Backing is synthetic and the speech is generated, so this shows the net and the loop separate; it is not a benchmark |
| Cost | 5 m 58 s for the 8 s clip (3 chunks) through a Debug CLI build; the network alone was 37 s per chunk in a Release build on random input; peak working set 1.8 GB |
| Against the reference (2026-11-26, CPU F32) | `KimVocal2ParityTests` on the same mix against `dump_kim_vocal2_reference.py` (`audio-separator` 0.30.2, onnxruntime 1.32.0, librosa 0.11.0): STFT 1.8e-7, 11 block outputs at most 1.2e-6, network output 1.5e-6, normalized stem 8.1e-7 / 8.0e-7, 16 kHz vocals 2.7e-4 with the first stand-in. The current stand-in was not run against this reference: on the stem of the stand-in reference for the same mix it is 2.5e-5 from python-soxr 1.1.0. The stand-in reference (`standin_kim_vocal2_reference.py`) gave the same network figures. Row in [PARITY_VERIFICATION.md](../Checklists/PARITY_VERIFICATION.md) |
| Network on a GPU, synthetic weights (2026-11-26) | `KimVocal2NetworkTests` on an RTX 4090 at `dfb769638` (#1277, merged as `5ff49e539`): 6 of 6 passed in 51 s, four CPU tests plus one forward test each on CUDA and Vulkan. A tiny random-weight network against the plain-loop reference, relRMS < 1e-5, CUDA with high-precision GEMM (not TF32); on Vulkan the transposed convolutions take the host fallback. As reported by the GPU test session; its log is not in the repository |
| Real weights on a GPU (2026-11-26) | Three runs at `dfb769638` on an RTX 4090, all failed the test as it then stood; figures in [GPU runs](#gpu-runs-with-the-real-weights) |
| Second clip, CPU F32 (2026-11-26) | First 7.44 s of LongCat-Video `assets/avatar/single/man.mp3` (at `58eb336e`, sha256 `6ddd02fda20d883d602fd59bcaf75ad1e17c6f6c5b8fb35e382a8bc792b03f18`), cut to 16-bit WAV with soundfile 0.14.0 (sha256 `0c04ffada9f6a71c667fba53f1cf9fe9206e4782d2348994bd4a645758a38767`), stand-in reference with python-soxr 1.1.0: STFT 1.8e-7, block outputs at most 1.0e-6, network 5.4e-6, normalized stem 6.2e-7; 16 kHz vocals 2.5e-3 with the first stand-in, over the limit, and 2.3e-4 with the current one. Chunks finished at 65.5 / 126.7 / 171.5 s (Release, 8 logical cores) |
| Real weights on a server RTX 3060, CUDA (2026-11-26) | Commit `03be72744` (#1298, merged as `575d2faa7`), exact GEMMs, the stand-in reference of the 4090 runs: `SeparatorMatchesTheReference` 1 of 1 in 15 s: STFT 1.79e-7, worst block 1.58e-6, network 2.95e-6, stems 1.66e-6 / 1.64e-6; chunk lines at 1.3 / 1.9 / 2.5 s; peak GPU-wide memory 2887 MiB. `KimVocal2NetworkTests` 6 of 6 (13 s, peak 445 MiB; its Vulkan test probably ran on that host's 4090, as the Vulkan device was not pinned). As reported by the GPU test session; logs on that host, not in the repository |
| Real weights on a server RTX 4090, Vulkan (2026-11-26) | Commit `03be72744`, the Vulkan device not pinnable: `SeparatorMatchesTheReference` 1 of 1 in 80 s: largest relRMS 2.94e-6 (network output), stems 1.662e-6 / 1.638e-6; chunk lines at 23.3 / 46.1 / 67.9 s; peak GPU-wide memory 2561 MiB. As reported by the GPU test session |
| Real weights on an RTX 3060, CUDA and Vulkan (2026-11-26) | Commit `12902f63b` (a second PC; its only GPU), second-clip stand-in reference. `KimVocal2NetworkTests` 6 of 6. `SeparatorMatchesTheReference` passed on both: CUDA network 5.5e-6, stems 6.3e-7, chunk lines at 3.5 / 4.8 / 6.1 s, test 13 s, peak GPU-wide 4000 MiB; Vulkan network 5.5e-6, stems 6.1e-7, chunk lines at 31.5 / 66.9 / 108.2 s, test 1 m 56 s, peak GPU-wide 2988 MiB (about 1300 MiB of each peak was in use before the run) |
| `fx separate` through the CLI, CUDA against the CPU (2026-11-26) | Server RTX 3060: `main` `d0e76642a` on the CPU against `8a0ed543c` with `-b cuda`, the 7.44 s clip, two alternating rounds, all exit 0. Each arm bit-identical between rounds (PCM sha256 `68da241a755d2f37` CPU, `95f5f09d577ee936` CUDA). CUDA stem against CPU stem: relRMS 2.3e-5, at most 1 16-bit step, 0.05% of samples differ. Wall time 143 and 139 s on the CPU, 3 and 3 s on CUDA; GPU memory 290 MiB before, 2887 MiB peak; `Separated 7.4s on cuda:0`, no CPU fall back. As reported by the GPU test session; logs on that host (`/home/hartsy/evidence/peer-job-ai-kim-ab-183305`) |
| **Not run** | the `--separate-vocals` video path end to end (no driving-audio video model was run); `fx separate` on Vulkan; the out-of-memory fall back on a card; any precision other than F32 |

## GPU runs with the real weights

2026-11-26, RTX 4090 (driver 595.99.02), commit `dfb769638`, `KimVocal2ParityTests.SeparatorMatchesOnnxRuntime` (since renamed `SeparatorMatchesTheReference`). Reference: the stand-in script of that commit (torch 2.15.0 CPU, onnxruntime 1.32.0,
scipy `resample_poly`; no audio-separator, librosa or soxr) on a 7.44 s clip from a LongCat speech file; weight sha256 verified. Reported by the GPU test session; its logs are on that host, not in the repository.

| relRMS | CUDA, TF32 GEMM | CUDA, exact GEMM | Vulkan |
|---|---|---|---|
| STFT, chunk 0 | 1.79e-7 | 1.79e-7 | 1.79e-7 |
| Worst block (limit 1e-4) | **1.70e-3** | ~1.55e-6 | ~1.66e-6 |
| Network output, chunk 0 (limit 1e-4) | **2.25e-3** | 2.94e-6 | 2.94e-6 |
| Normalized stem (limit 1e-3) | **1.25e-3** | 1.67e-6 | 1.66e-6 |
| 16 kHz vocals (limit then 1e-2) | **2.571e-2** | **2.570e-2** | **2.570e-2** |
| Wall time | 14 s | 14 s | 78 s |
| Peak GPU-wide memory | 3633 MiB | 3679 MiB | 2511 MiB |

- **16 kHz:** at that commit the test compared the C# SoXR-HQ stand-in with scipy `resample_poly`, two different resamplers, and asserted it below 1e-2 whatever the reference. The stem feeding it matched within 1.7e-6 on the exact lanes.
  The current test no longer compares those two; it compares the 16 kHz output only against a `soxr_hq` reference, in its own test.
- **TF32:** it misses the block, network and stem limits, with roughly 750 times the exact lane's network and stem error, and the two CUDA runs took the same 14 s for the whole test. Their chunk lines are at 1.0 / 1.3 / 1.7 s (TF32) and
  1.0 / 1.3 / 1.6 s (exact), so exact GEMMs cost no measurable time there. `Forward` therefore runs under `ExactGemmScope`. No TF32 limit was added.
- **Vulkan, about 22 s per chunk (as measured, before the shader):** Vulkan had no 2D transposed convolution then, so the five decoder up-convolutions ran in the scalar `IBackend.ConvTranspose2d` default on the host, with a download before and an upload after.
  On a different PC (8 logical cores, random input without zeros) those five calls alone take 18.6 s per chunk in a Release build and 56 to 71 s in Debug, the largest 7.8 s in Release. The loop skips zero inputs, so real activations cost less. That still makes the host loop the likely
  cost, but the split was not profiled on the GPU host, whose build configuration was not reported. On the RTX 3060 above a Vulkan chunk takes 31 to 41 s, about what a CPU chunk takes on that PC (38 to 39 s), against about 1.3 s on CUDA after the first. The Vulkan kernel for the op was added later on 2026-11-26; with it a chunk on that RTX 3060 takes about 1 s ([VULKAN_STATUS](../Checklists/VULKAN_STATUS.md)).

