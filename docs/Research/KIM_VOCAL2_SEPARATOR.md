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

Not reproduced bit for bit: librosa's default `soxr_hq` resampler. The port uses the engine's `SincResampler.ResampleSoxrHqStandIn` (64-crossing windowed sinc, roll-off 0.955), so the 16 kHz output is compared as audio.
Loudness normalization is not part of the separator: LongCat applies it downstream (`-23 LUFS`, off for Avatar 1.6, on for 1.5 and 1.0).

## Engine surface

- `hartsy fx separate <wav> -m kim-vocal-2` returns one `vocals` stem (44.1 kHz stereo, normalized to 0.9). Demucs-only options (`Shifts`, `Segment`) are refused; `Overlap` is honoured (default 0.25).
- `VideoRequest.SeparateDrivingVocals` / `hartsy video --driving-audio <wav> --separate-vocals`: `VideoService` replaces the driving audio with the vocals stem (44.1 kHz stereo, 16-bit PCM WAV) before the recipe runs; the recipe decodes it to the rate and channels it needs. Alongside that clip it hands the recipe 16 kHz mono taken from the float stem (`MdxSeparator.ToMono`: channel mean, SoXR-HQ stand-in), the reference route; LongCat-Avatar encodes that and cuts it to the generated clip length plus one second, as `separate_vocals` does. The separator input is resampled to 44.1 kHz with the same stand-in. Measured on the 8 s test mix (CPU, 2026-11-26), the earlier route (16-bit stereo WAV, decode, mean, the engine's Kaiser polyphase `Resampler`) differs from this one by relRMS 3.5e-3, max 3.1e-3: the resampler accounts for 3.5e-3 and the 16-bit round trip for 8.3e-5; on a 24 kHz speech excerpt the two input resamplers differ by 1.0e-3. Neither is compared with `soxr_hq` itself. The reference pads short speech with zeros up to that duration, while the engine recipe repeats the last audio window (unchanged here). The pre-step separates only the clip length of audio (frames / fps, plus one second) when the frame total and fps are known, and the whole track otherwise; it runs before generation and is logged, not reported through step progress. A request without driving
  audio is refused at planning (`video.separate_vocals.no_driving_audio`); a recipe without driving-audio support refuses the audio itself. Any soundtrack a recipe attaches from the driving clip is then the separated vocals;
  a caller who wants the original mix heard passes it as `VideoAudioInput`.
- The network runs on the CPU backend like Demucs (the transpose writes through reshaped views, which is only safe on CPU). One chunk is 261,120 samples and the window advances 195,840 (4.44 s); a clip of N seconds takes about ⌈(N · 44100 + 3840 + pad) / 195,840⌉ chunks (3 for the 8 s test clip). The network alone took 37 s per chunk on 8 logical cores in a Release build on random input, peak working set 1.8 GB. No GPU run is claimed.
- Weights are read from the `.onnx` at load (about a tenth of a second); the Hartsy-hosted safetensors repack of decision D1 is not made, and nothing was uploaded.

## Evidence (2026-11-23, CPU F32, Debug build of the CLI)

| Check | Result |
|---|---|
| Graph binds | the real file binds 156 tensors (5 encoder levels 48/96/144/192/240, bottleneck 288, 3072 bins); `KimVocal2OnnxConverterTests` pin the walk on a small graph with the same op order |
| Network vs plain-loop reference | relRMS < 1e-5, random small config (`KimVocal2NetworkTests`); this checks wiring and layout, not the real weights |
| STFT front end vs numpy | chunk-0 spectrogram of the test mix against a float64 `numpy.fft.rfft` implementation of `torch.stft`: relRMS 1.6e-7 |
| Overlap-add | identity network returns the input at relRMS < 2e-5 over several overlapping chunks (`MdxSeparatorTests`) |
| Constructed mixed clip | `hartsy fx separate mix.wav -m kim-vocal-2`: 8 s stereo mix constructed from known stems = clean speech (a Haru reply) + synthetic backing (chord pad, bass, kick, hat). SI-SDR of the vocals stem against the clean speech 17.10 dB (the mix itself: 0.51 dB), correlation 0.990, against the backing 0.01. Backing is synthetic and the speech is generated, so this shows the net and the loop separate; it is not a benchmark |
| Cost | 5 m 58 s for the 8 s clip (3 chunks) through a Debug CLI build; the network alone was 37 s per chunk in a Release build on random input; peak working set 1.8 GB |
| Against onnxruntime (2026-11-26, CPU F32) | `KimVocal2ParityTests` on the same mix, reference from `standin_kim_vocal2_reference.py` in `hartsy-ref:cpu` (onnxruntime 1.32.0): STFT 1.8e-7, 11 block outputs ≤ 1.2e-6, network output 1.5e-6, normalized stem 8.1e-7 / 8.0e-7. The stand-in output was bit-identical across two runs. Row in [PARITY_VERIFICATION.md](../Checklists/PARITY_VERIFICATION.md) |
| **Not run** | `dump_kim_vocal2_reference.py` (needs `audio-separator==0.30.2` and librosa), so the real MDX wrapper and the `soxr_hq` 16 kHz resample are uncompared (the test's 16 kHz figure, 3.5e-3, is against scipy `resample_poly`); the `--separate-vocals` video path end to end (no driving-audio video model was run); CUDA, Vulkan and any precision other than F32 (the service runs the network on the CPU backend, and the transpose writes through reshaped views) |
