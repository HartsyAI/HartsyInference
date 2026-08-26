using System.Diagnostics;
using HartsyInference.Audio.Models.Codecs.Vocos;
using HartsyInference.Audio.Models.Codecs.XCodec;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>YuE full-song music pipeline. Stage-1 LLaMA-2 emits the interleaved vocal+accompaniment
/// codebook-0 streams; Stage-2 (<see cref="YueStage2Lm"/>, optional) upsamples each track to the full 8
/// residual codebooks; X-Codec decodes to 16 kHz. Token-IDs-in: the caller tokenizes lyrics + genre tags
/// (with `[verse]/[chorus]` structure markers) via the YuE tokenizer. Reuses `Qwen2Model` (both stages) +
/// the built `XCodec` decoder.
///
/// <para>When <c>stage2</c> is null the pipeline decodes vocal cb0 only (low-quality single-codebook);
/// when supplied it runs the full two-stage path and mixes the vocal + accompaniment stems (equal-gain
/// sum, upstream <c>infer.py</c>).</para></summary>
public sealed unsafe class YuePipeline : IDisposable
{
    private readonly YueConfig _cfg;
    private readonly YueStage1Lm _stage1;
    private readonly YueStage2Lm? _stage2;
    private readonly XCodec _xcodec;
    private readonly VocosDecoder? _vocalVocoder;   // YuE's real 44.1 kHz output path (per-stem Vocos)
    private readonly VocosDecoder? _instVocoder;
    private int _disposed;

    public YuePipeline(YueConfig cfg, YueStage1Lm stage1, XCodec xcodec, YueStage2Lm? stage2 = null,
        VocosDecoder? vocalVocoder = null, VocosDecoder? instVocoder = null)
    {
        _cfg = cfg;
        _stage1 = stage1;
        _stage2 = stage2;
        _xcodec = xcodec;
        _vocalVocoder = vocalVocoder;
        _instVocoder = instVocoder;
    }

    /// <summary>True when both per-stem Vocos vocoders are loaded — output is 44.1 kHz, else the 16 kHz x-codec draft.</summary>
    private bool HasVocoders => _vocalVocoder is not null && _instVocoder is not null;

    /// <summary>Output sample rate: 44.1 kHz when the Vocos vocoders are present, else the x-codec's 16 kHz.</summary>
    public int OutputSampleRate => HasVocoders ? VocosDecoder.SampleRate : _cfg.SampleRate;

    /// <summary>Synthesizes a 16 kHz song from the tokenized lyric+genre prompt. Returns the vocal track
    /// audio (accompaniment mixing + the Stage-2 residual upsampler are deferred — see checklist). YuE's
    /// generation params are exposed per-call (default to config): <paramref name="temperature"/>,
    /// <paramref name="topK"/>, <paramref name="topP"/>, <paramref name="repetitionPenalty"/>, and classifier-free
    /// guidance (<paramref name="guidanceScale"/>) when a negative prompt <paramref name="uncondTokenIds"/> is given.</summary>
    public float[] Synthesize(IBackend backend, int[] promptTokenIds, int maxFrames = 3000, int seed = 0,
        float? temperature = null, int? topK = null, float? topP = null, float? repetitionPenalty = null,
        float? guidanceScale = null, int[]? uncondTokenIds = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        // Pin the Stage-1 weights resident up front. The audio stack otherwise relies on lazy auto-promote,
        // whose headroom gate loses to the transient-upload pool for a 7B — the weights never promote and stream
        // from host EVERY token (~0.6 s/token). A one-shot preload makes decode read resident weights via the
        // fused quant GEMV (or the dense GEMV when un-quantized). Freed at the Stage-1→Stage-2 boundary below.
        PreloadStage1(backend);
        // YuE generates with classifier-free guidance by default (infer.py: guidance_scale=1.5). The HF-faithful
        // unconditional context is the prompt's LAST token (the <xcodec> sep) — the negative branch runs with no
        // genre/lyrics conditioning. Auto-build it when the caller supplies none so CFG actually engages (without
        // it the vocals drift/gibber). Disable by passing guidanceScale ≤ 1.
        float cfg = guidanceScale ?? _cfg.GuidanceScale;
        int[] uncond = uncondTokenIds
            ?? (cfg > 1f && promptTokenIds.Length > 0 ? [promptTokenIds[^1]] : []);
        (List<int> vocal, List<int> accomp) = _stage1.GenerateCb0(backend, promptTokenIds, maxFrames, seed,
            temperature, topK, topP, repetitionPenalty, cfg, uncond);
        Logs.Info($"YuE S1: {vocal.Count} frames (vocal+accomp cb0) in {sw.ElapsedMilliseconds}ms.");
        if (vocal.Count == 0) return [];

        // Stage-1 (7B) is done — free its resident weights so Stage-2 (1B) + x-codec get full VRAM instead of
        // streaming against the 7B's cache (same phase-boundary free as HeartMuLa's LM→codec handoff).
        FreeStage1(backend);
        return DecodeAndMix(backend, vocal, accomp, sw);
    }

    /// <summary>Segment-by-segment Stage-1 synthesis (infer.py's authoritative loop). Each segment is generated
    /// conditioned on the accumulated context (head + prior segments' prompts + their generated cb0 + &lt;EOA&gt;),
    /// injecting THAT segment's lyrics — the fix for gibberish vocals (single-shot conditioned only on segment 1).
    /// <paramref name="headIds"/> = the head prompt ids (prepended once); <paramref name="segmentPrompts"/> = the
    /// per-segment prompt id arrays (segment[0] starts with [start_of_segment]; later ones with
    /// [end_of_segment][start_of_segment]). Each segment is capped at <paramref name="maxFramesPerSegment"/> and
    /// stops at its own &lt;EOA&gt;; the song is their concatenation. Then Stage-2 + vocoders as usual.</summary>
    public float[] Synthesize(IBackend backend, int[] headIds, IReadOnlyList<int[]> segmentPrompts,
        int maxFramesPerSegment = 1500, int seed = 0, float? temperature = null, int? topK = null, float? topP = null,
        float? repetitionPenalty = null, float? guidanceScale = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        if (segmentPrompts.Count == 0) return [];
        PreloadStage1(backend);

        float cfg = guidanceScale ?? _cfg.GuidanceScale;
        int maxContext = _cfg.Stage1.MaxPositionEmbeddings;
        List<int> context = [];
        List<int> allVocal = [], allAccomp = [];

        for (int s = 0; s < segmentPrompts.Count; s++)
        {
            int[] seg = segmentPrompts[s];
            // input = (s==0 ? head : accumulated context) ++ this segment's prompt.
            List<int> input = new((s == 0 ? headIds.Length : context.Count) + seg.Length);
            input.AddRange(s == 0 ? headIds : context);
            input.AddRange(seg);
            // Cap to the model window (official: input_ids[:, -max_context:]) leaving room for this segment's generation.
            int budget = maxContext - maxFramesPerSegment * 2 - 8;
            if (budget > 0 && input.Count > budget) input.RemoveRange(0, input.Count - budget);

            int[] inputArr = [.. input];
            // Per-segment CFG (infer.py: 1.5 for the first segment, 1.2 after); uncond = the prompt's last token
            // (<xcodec> sep), same mechanism as the single-shot path. Disabled when the base scale is ≤ 1.
            float segCfg = s == 0 ? cfg : (cfg > 1f ? 1.2f : cfg);
            int[] uncond = segCfg > 1f && inputArr.Length > 0 ? [inputArr[^1]] : [];
            (List<int> v, List<int> a, List<int> rawTokens) = _stage1.GenerateSegment(backend, inputArr,
                maxFramesPerSegment, seed + s, temperature, topK, topP, repetitionPenalty, segCfg, uncond);
            allVocal.AddRange(v);
            allAccomp.AddRange(a);
            // Next segment's context = this segment's input + its generated tokens + trailing <EOA> (infer.py raw_output).
            context = new List<int>(inputArr.Length + rawTokens.Count + 1);
            context.AddRange(inputArr);
            context.AddRange(rawTokens);
            context.Add(_cfg.AudioEosToken);
            Logs.Info($"YuE S1 seg {s + 1}/{segmentPrompts.Count}: {v.Count} frames ({sw.ElapsedMilliseconds}ms).");
        }

        Logs.Info($"YuE S1: {allVocal.Count} frames total (vocal+accomp cb0) over {segmentPrompts.Count} segments in {sw.ElapsedMilliseconds}ms.");
        if (allVocal.Count == 0) return [];
        FreeStage1(backend);
        return DecodeAndMix(backend, allVocal, allAccomp, sw);
    }

    /// <summary>Preloads Stage-1 resident: per-stage asymmetric sets when a layer-split placement pools VRAM
    /// (each backend gets ONLY its layer range — preloading the full set on every stage would replicate
    /// instead of pool), else the whole set on <paramref name="backend"/>.</summary>
    private void PreloadStage1(IBackend backend)
    {
        if (_stage1.Placement is { IsSingle: false } placement)
        {
            for (int s = 0; s < placement.Stages.Count; s++)
            {
                LlmStage stage = placement.Stages[s];
                // Canonical set only (no redundant fused/split views): the un-quantized 7B is the tight-fit
                // case this path exists for, and the duplicates would cost ~40% more resident VRAM.
                stage.Backend.PreloadWeights(_stage1.EnumerateStageWeights(
                    stage.StartLayer, stage.EndLayer, s == 0, s == placement.Stages.Count - 1,
                    includeRedundantSplits: false));
            }
            return;
        }
        backend.PreloadWeights(_stage1.EnumerateWeights());
    }

    /// <summary>Frees Stage-1's resident weights from every backend that preloaded them (mirror of
    /// <see cref="PreloadStage1"/>).</summary>
    private void FreeStage1(IBackend backend)
    {
        if (_stage1.Placement is { IsSingle: false } placement)
        {
            for (int s = 0; s < placement.Stages.Count; s++)
            {
                LlmStage stage = placement.Stages[s];
                // Full set (redundant views included): anything auto-promoted during decode must go too.
                stage.Backend.FreeWeights(_stage1.EnumerateStageWeights(
                    stage.StartLayer, stage.EndLayer, s == 0, s == placement.Stages.Count - 1));
            }
            return;
        }
        backend.FreeWeights(_stage1.EnumerateWeights());
    }

    /// <summary>Stage-1 cb0 → Stage-2 residual upsample → x-codec/Vocos decode → stem mix. Shared by both
    /// <c>Synthesize</c> overloads (post-Stage-1 path is identical).</summary>
    private float[] DecodeAndMix(IBackend backend, List<int> vocal, List<int> accomp, Stopwatch sw)
    {
        float[] audio;
        if (_stage2 is not null)
        {
            // Same rationale: pin Stage-2's weights resident so its 8×/frame ×2-track decode stays on-device.
            backend.PreloadWeights(_stage2.EnumerateWeights());
            // Full path: Stage-2 upsamples each track's cb0 to 8 codebooks, x-codec decodes each, mix (sum).
            System.Span<int> vSpan = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(vocal);
            System.Span<int> aSpan = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(accomp);
            // Batch both tracks (B=2) through Stage-2 in one pass — same codes as two sequential Upsample calls, ~half the wall-clock.
            (int[][] vocalCodes, int[][] accompCodes) = _stage2.UpsampleBoth(backend, vSpan, aSpan);
            Logs.Info($"YuE S2: upsampled {vocal.Count} frames x2 tracks to 8 codebooks in {sw.ElapsedMilliseconds}ms.");
            backend.FreeWeights(_stage2.EnumerateWeights());   // Stage-2 done — free before decode
            // YuE's REAL output path: each stem's 8-codebook grid → x-codec 1024-d quantizer latent → its own Vocos
            // vocoder → 44.1 kHz, then equal-gain sum (upstream infer.py `vocoder` path). Falls back to the 16 kHz
            // x-codec draft (`DecodeFull`) when the vocoders aren't loaded. The plain draft under-reconstructs the
            // vocal stem (quiet/rough); the Vocos vocal decoder fixes both the level and the fidelity.
            float[] vocalWav = DecodeStem(backend, vocalCodes, _vocalVocoder);
            float[] accompWav = DecodeStem(backend, accompCodes, _instVocoder);
            // Per-stem levels before the sum: the only way to tell a mix-stage headroom problem (stems fine, sum
            // clips) from a decode gain bug (a stem is already blown out).
            Logs.Info($"YuE stems pre-mix: vocal peak={Peak(vocalWav):F3} rms={Rms(vocalWav):F3}, "
                + $"inst peak={Peak(accompWav):F3} rms={Rms(accompWav):F3}.");
            audio = MixSum(vocalWav, accompWav);
            if (HasVocoders)
            {
                // The Vocos mix is upstream's INTERMEDIATE, not its output: infer.py ends in
                // post_process_audio.replace_low_freq_with_energy_matched, which keeps only the vocoder's HIGH band
                // and takes everything below 5.5 kHz from the 16 kHz x-codec reconstruction. Vocos is a
                // super-resolution decoder — its own low band is smeared (audibly "echoey"), and the energy match
                // also pulls the level back down, which is what stops the sum from slamming into the clamp.
                float[] vocalDraft = DecodeFull(backend, vocalCodes);
                float[] instDraft = DecodeFull(backend, accompCodes);
                Logs.Info($"YuE x-codec 16k stems: vocal peak={Peak(vocalDraft):F3} rms={Rms(vocalDraft):F3}, "
                    + $"inst peak={Peak(instDraft):F3} rms={Rms(instDraft):F3}.");
                float[] draftMix = MixSum(vocalDraft, instDraft);
                float[] vocoderMix = audio;
                _draftStems = (vocalDraft, instDraft);
                audio = ReplaceLowFreqEnergyMatched(draftMix, _cfg.SampleRate, vocoderMix, VocosDecoder.SampleRate);
                Logs.Info($"YuE post-process: low band < {CrossoverHz} Hz from the x-codec recon, energy-matched.");
                DumpParityArtifacts(backend, vocalCodes, accompCodes, vocalWav, accompWav, vocoderMix, draftMix, audio);
            }
        }
        else
        {
            // Stage-1-only fallback: decode vocal cb0 through x-codec with n_q=1 (CodecManipulator("xcodec",0,1)).
            // Index 0 is a valid codeword (not silence) — decode exactly cb0, do NOT zero-fill residuals.
            Tensor audioT = _xcodec.DecodeCb0(backend, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(vocal));
            int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
            audio = new float[n];
            Buffer.MemoryCopy((void*)audioT.DataPointer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
            audioT.Dispose();
        }

        sw.Stop();
        Logs.Info($"YuE synthesis complete: {audio.Length} samples ({audio.Length / (double)OutputSampleRate:F1}s @ {OutputSampleRate} Hz) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    /// <summary>Decodes a full <c>[8][T]</c> codebook grid to a 16 kHz mono waveform. Builds the
    /// <c>[n_q=8, B=1, T]</c> I32 grid x-codec expects (upstream <c>codec_model.decode</c> permute layout).</summary>
    private float[] DecodeFull(IBackend backend, int[][] codes)
    {
        int nq = codes.Length;
        int t = codes[0].Length;
        Tensor grid = new(new TensorShape(nq, 1, t), DType.I32);
        int* gp = (int*)grid.DataPointer;
        for (int q = 0; q < nq; q++)
        {
            int[] row = codes[q];
            for (int i = 0; i < t; i++) gp[q * t + i] = row[i];
        }
        Tensor pcm = _xcodec.Decode(backend, grid, batch: 1, tFrames: t);
        grid.Dispose();
        int n = (int)pcm.Shape[pcm.Shape.Rank - 1];
        float[] wav = new float[n];
        Buffer.MemoryCopy((void*)pcm.DataPointer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref wav[0]), n * 4, n * 4);
        pcm.Dispose();
        return wav;
    }

    /// <summary>Decodes one stem's <c>[8][T]</c> codebook grid to a waveform: through the stem's Vocos vocoder
    /// (x-codec 1024-d quantizer latent → 44.1 kHz) when supplied, else the 16 kHz x-codec draft.</summary>
    private float[] DecodeStem(IBackend backend, int[][] codes, VocosDecoder? vocoder)
    {
        if (vocoder is null) return DecodeFull(backend, codes);

        int nq = codes.Length;
        int t = codes[0].Length;
        Tensor grid = new(new TensorShape(nq, 1, t), DType.I32);
        int* gp = (int*)grid.DataPointer;
        for (int q = 0; q < nq; q++)
        {
            int[] row = codes[q];
            for (int i = 0; i < t; i++) gp[q * t + i] = row[i];
        }
        Tensor latent = _xcodec.DecodeToLatent(backend, grid, batch: 1, tFrames: t);   // [1, 1024, T]
        grid.Dispose();
        Tensor pcm = vocoder.Forward(backend, latent, batch: 1, tFrames: t);            // [1, 1, T·882] @ 44.1 kHz
        latent.Dispose();
        int n = (int)pcm.Shape[pcm.Shape.Rank - 1];
        float[] wav = new float[n];
        Buffer.MemoryCopy((void*)pcm.DataPointer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref wav[0]), n * 4, n * 4);
        pcm.Dispose();
        return wav;
    }

    /// <summary>Crossover between the x-codec recon and the Vocos output, per upstream's post-process.</summary>
    private const double CrossoverHz = 5500d;

    /// <summary>The 16 kHz x-codec stem decodes, held only so the parity dump can record them.</summary>
    private (float[] Vocal, float[] Inst)? _draftStems;

    /// <summary>Peak absolute sample.</summary>
    private static float Peak(float[] x)
    {
        float peak = 0f;
        for (int i = 0; i < x.Length; i++) peak = Math.Max(peak, Math.Abs(x[i]));
        return peak;
    }

    /// <summary>Writes every decode-stage intermediate to <c>$HARTSY_YUE_DUMP/yue_dump.safetensors</c> so the Python
    /// reference can be run on the SAME codes. Opt-in and off by default — this is a diagnostic, not a product path.</summary>
    private void DumpParityArtifacts(IBackend backend, int[][] vocalCodes, int[][] accompCodes,
        float[] vocalWav, float[] instWav, float[] vocoderMix, float[] draftMix, float[] final)
    {
        string? directory = EngineKnobs.YueDump.Value;
        if (string.IsNullOrWhiteSpace(directory)) return;
        try
        {
            Directory.CreateDirectory(directory);
            Dictionary<string, Tensor> dump = new(StringComparer.Ordinal)
            {
                ["vocal_codes"] = CodesTensor(vocalCodes),
                ["accomp_codes"] = CodesTensor(accompCodes),
                ["vocal_44k"] = WaveTensor(vocalWav),
                ["inst_44k"] = WaveTensor(instWav),
                ["vocoder_mix"] = WaveTensor(vocoderMix),
                ["draft_16k"] = WaveTensor(draftMix),
                ["final_44k"] = WaveTensor(final),
            };
            if (_draftStems is { } stems)
            {
                dump["vocal_16k"] = WaveTensor(stems.Vocal);
                dump["inst_16k"] = WaveTensor(stems.Inst);
            }
            // The Vocos input, so the vocoder can be checked in isolation from the x-codec RVQ that feeds it.
            dump["vocal_latent"] = LatentTensor(backend, vocalCodes);
            dump["accomp_latent"] = LatentTensor(backend, accompCodes);
            string path = Path.Combine(directory, "yue_dump.safetensors");
            SafeTensorsWriter.Save(path, dump);
            foreach (Tensor tensor in dump.Values) tensor.Dispose();
            Logs.Info($"[Audio][YuE] Wrote parity dump to '{path}'.");
        }
        catch (Exception ex)
        {
            Logs.Error("[Audio][YuE] Parity dump failed — generation is unaffected", ex);
        }
    }

    private static Tensor CodesTensor(int[][] codes)
    {
        int nq = codes.Length, t = codes[0].Length;
        Tensor tensor = new(new TensorShape(nq, t), DType.I32);
        int* p = (int*)tensor.DataPointer;
        for (int q = 0; q < nq; q++)
        {
            for (int i = 0; i < t; i++) p[q * t + i] = codes[q][i];
        }
        return tensor;
    }

    private static Tensor WaveTensor(float[] wave)
    {
        Tensor tensor = new(new TensorShape(wave.Length), DType.F32);
        float* p = (float*)tensor.DataPointer;
        for (int i = 0; i < wave.Length; i++) p[i] = wave[i];
        return tensor;
    }

    private Tensor LatentTensor(IBackend backend, int[][] codes)
    {
        int nq = codes.Length, t = codes[0].Length;
        Tensor grid = new(new TensorShape(nq, 1, t), DType.I32);
        int* gp = (int*)grid.DataPointer;
        for (int q = 0; q < nq; q++)
        {
            for (int i = 0; i < t; i++) gp[q * t + i] = codes[q][i];
        }
        Tensor latent = _xcodec.DecodeToLatent(backend, grid, batch: 1, tFrames: t);
        grid.Dispose();
        return latent;
    }

    /// <summary>Upstream <c>replace_low_freq_with_energy_matched</c>: resample the 16 kHz x-codec mix up to the
    /// vocoder rate, low-pass both at <see cref="CrossoverHz"/>, scale the recon's low band so its RMS matches the
    /// vocoder's low band, and add it to the vocoder's high-passed band. Filters are single-pass RBJ biquads at
    /// Q = 1/√2, matching torchaudio's <c>lowpass_biquad</c>/<c>highpass_biquad</c> + causal <c>lfilter</c>.</summary>
    private static float[] ReplaceLowFreqEnergyMatched(float[] recon, int reconRate, float[] vocoded, int vocodedRate)
    {
        float[] reconUp = reconRate == vocodedRate ? recon : Resampler.Create(reconRate, vocodedRate).Resample(recon);
        float[] reconLow = LowPass(reconUp, vocodedRate, CrossoverHz);
        float[] vocodedLow = LowPass(vocoded, vocodedRate, CrossoverHz);
        const double eps = 1e-10;
        double scale = (Rms(vocodedLow) + eps) / (Rms(reconLow) + eps);
        float[] vocodedHigh = HighPass(vocoded, vocodedRate, CrossoverHz);
        // Upstream truncates to the shorter of the two rather than padding.
        int n = Math.Min(reconLow.Length, vocodedHigh.Length);
        float[] mix = new float[n];
        for (int i = 0; i < n; i++)
        {
            float v = (float)(reconLow[i] * scale) + vocodedHigh[i];
            mix[i] = v > 0.99f ? 0.99f : (v < -0.99f ? -0.99f : v);
        }
        return mix;
    }

    /// <summary>Root-mean-square over the whole buffer (upstream's global, not per-channel, RMS).</summary>
    private static double Rms(float[] x)
    {
        if (x.Length == 0) return 0d;
        double sum = 0d;
        for (int i = 0; i < x.Length; i++) sum += (double)x[i] * x[i];
        return Math.Sqrt(sum / x.Length);
    }

    /// <summary>RBJ low-pass biquad (Q = 1/√2), applied causally like torchaudio's <c>lfilter</c>.</summary>
    private static float[] LowPass(float[] x, int sampleRate, double cutoff)
    {
        (double w0, double alpha, double cosW0) = BiquadTerms(sampleRate, cutoff);
        double b1 = 1d - cosW0;
        return Biquad(x, b1 / 2d, b1, b1 / 2d, 1d + alpha, -2d * cosW0, 1d - alpha);
    }

    /// <summary>RBJ high-pass biquad (Q = 1/√2), applied causally like torchaudio's <c>lfilter</c>.</summary>
    private static float[] HighPass(float[] x, int sampleRate, double cutoff)
    {
        (double w0, double alpha, double cosW0) = BiquadTerms(sampleRate, cutoff);
        double b0 = (1d + cosW0) / 2d;
        return Biquad(x, b0, -(1d + cosW0), b0, 1d + alpha, -2d * cosW0, 1d - alpha);
    }

    private static (double W0, double Alpha, double CosW0) BiquadTerms(int sampleRate, double cutoff)
    {
        const double q = 0.707106781186547524401d;   // 1/√2, torchaudio's default Q
        double w0 = 2d * Math.PI * cutoff / sampleRate;
        return (w0, Math.Sin(w0) / (2d * q), Math.Cos(w0));
    }

    /// <summary>Direct-form-I biquad with zero initial state.</summary>
    private static float[] Biquad(float[] x, double b0, double b1, double b2, double a0, double a1, double a2)
    {
        b0 /= a0; b1 /= a0; b2 /= a0; a1 /= a0; a2 /= a0;
        float[] y = new float[x.Length];
        double x1 = 0d, x2 = 0d, y1 = 0d, y2 = 0d;
        for (int i = 0; i < x.Length; i++)
        {
            double xi = x[i];
            double yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1; x1 = xi; y2 = y1; y1 = yi;
            y[i] = (float)yi;
        }
        return y;
    }

    /// <summary>Mixes two stems by equal-gain sum (upstream <c>infer.py</c>: <c>(vocal + inst) / 1</c>),
    /// clamped to ±0.99 to match the reference <c>save_audio</c> (no rescale).</summary>
    private static float[] MixSum(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        float[] mix = new float[n];
        for (int i = 0; i < n; i++)
        {
            float v = a[i] + b[i];
            mix[i] = v > 0.99f ? 0.99f : (v < -0.99f ? -0.99f : v);
        }
        return mix;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _stage1.Dispose();
        _stage2?.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(YuePipeline));
    }
}
