using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>CosyVoice 2's HiFTNet vocoder (mel → 24 kHz waveform). Mirrors
/// <c>cosyvoice/hifigan/generator.py:HiFTGenerator</c>: an internal F0 predictor drives an NSF
/// harmonic-plus-noise source; the source STFT is injected at every upsample level; the backbone is a
/// HiFiGAN transposed-conv stack with plain (non-style-conditioned) Snake MRF resblocks; the output is
/// produced by a magnitude/phase iSTFT head rather than a final waveform conv.
///
/// <para>This shares the NSF-source + forward-STFT + iSTFT-head <i>pattern</i> with the Kokoro
/// iSTFTNet decoder, but differs in: mel (not prosody) input via <c>conv_pre</c>; an internal F0
/// predictor (Kokoro takes F0 from the prosody predictor); plain Snake resblocks (no AdaIN); and the
/// CosyVoice upsample schedule. <b>Checkpoint-validation pending</b> — weight keys + the source-inject
/// downsample params follow the FunAudioLLM layout and need reconciling against the real
/// <c>hift.pt</c>.</para></summary>
public sealed unsafe class HiFTNetVocoder : IDisposable
{
    private const float LeakySlope = 0.1f;
    private const int Harmonics = 9;             // fundamental + 8
    private const float SineAmp = 0.1f;
    private const float NoiseStd = 0.003f;
    private const float VoicedThreshold = 10f;

    private readonly CosyVoiceHiftConfig _cfg;
    private readonly int _numUp;
    private readonly int[] _levelChannels;       // base/2^(i+1) per stage
    private readonly int[] _srcDownStride;       // prod(upRates[i+1:]) per stage
    private int _disposed;

    private Tensor? _convPreW, _convPreB;
    private Tensor? _convPostW, _convPostB;
    private Tensor? _mSourceW, _mSourceB;        // l_linear [1, 9]
    private Tensor?[] _upsW;
    private Tensor?[] _upsB;
    private Tensor?[] _srcDownW;
    private Tensor?[] _srcDownB;
    private SnakeResBlock[] _srcResBlocks;
    private SnakeResBlock[] _resBlocks;       // numUp * 3
    private readonly F0Predictor _f0;

    public HiFTNetVocoder(CosyVoiceHiftConfig cfg)
    {
        _cfg = cfg;
        _numUp = cfg.UpsampleRates.Length;
        _levelChannels = new int[_numUp];
        _srcDownStride = new int[_numUp];
        for (int i = 0; i < _numUp; i++)
        {
            _levelChannels[i] = cfg.UpsampleInitialChannel >> (i + 1);
            int stride = 1;
            for (int j = i + 1; j < _numUp; j++) stride *= cfg.UpsampleRates[j];
            _srcDownStride[i] = stride;
        }
        _upsW = new Tensor[_numUp];
        _upsB = new Tensor[_numUp];
        _srcDownW = new Tensor[_numUp];
        _srcDownB = new Tensor[_numUp];
        _srcResBlocks = new SnakeResBlock[_numUp];
        int[] srcK = SourceResBlockKernels(_numUp);
        for (int i = 0; i < _numUp; i++)
            _srcResBlocks[i] = new SnakeResBlock(_levelChannels[i], srcK[i], [1, 3, 5]);
        _resBlocks = new SnakeResBlock[_numUp * cfg.ResBlockKernelSizes.Length];
        for (int i = 0; i < _numUp; i++)
            for (int j = 0; j < cfg.ResBlockKernelSizes.Length; j++)
                _resBlocks[i * cfg.ResBlockKernelSizes.Length + j] =
                    new SnakeResBlock(_levelChannels[i], cfg.ResBlockKernelSizes[j], cfg.ResBlockDilationSizes[j]);
        _f0 = new F0Predictor(cfg.MelBins);
    }

    private static int[] SourceResBlockKernels(int n) => n switch
    {
        2 => [7, 11],
        3 => [7, 7, 11],
        _ => Enumerable.Repeat(7, n).ToArray(),
    };

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _convPreW = WeightNorm.Compose(w, $"{p}conv_pre");
        _convPreB = WhisperOps.EnsureF32(w[$"{p}conv_pre.bias"]);
        _convPostW = WeightNorm.Compose(w, $"{p}conv_post");
        _convPostB = WhisperOps.EnsureF32(w[$"{p}conv_post.bias"]);
        _mSourceW = WhisperOps.EnsureF32(w[$"{p}m_source.l_linear.weight"]);
        _mSourceB = WhisperOps.EnsureF32(w[$"{p}m_source.l_linear.bias"]);
        for (int i = 0; i < _numUp; i++)
        {
            _upsW[i] = WeightNorm.Compose(w, $"{p}ups.{i}");
            _upsB[i] = WhisperOps.EnsureF32(w[$"{p}ups.{i}.bias"]);
            _srcDownW[i] = WhisperOps.EnsureF32(w[$"{p}source_downs.{i}.weight"]);
            _srcDownB[i] = WhisperOps.EnsureF32(w[$"{p}source_downs.{i}.bias"]);
            _srcResBlocks[i].LoadWeights(w, $"{p}source_resblocks.{i}");
        }
        for (int i = 0; i < _resBlocks.Length; i++) _resBlocks[i].LoadWeights(w, $"{p}resblocks.{i}");
        _f0.LoadWeights(w, $"{p}f0_predictor");
    }

    /// <summary>Synthesizes a waveform from an <c>[1, 80, T_mel]</c> log-mel.</summary>
    // TODO(gpu-residency): the NSF source generation, forward STFT, source injection, and the iSTFT head
    // (NsfVocoderDsp.*) run as host C# DSP over `(float*)DataPointer`, so this vocoder syncs device→host and
    // breaks GPU residency on CUDA. Port the harmonic-source / STFT / iSTFT to PTX kernels (or keep DSP on host
    // but batch the syncs) to keep the mel→wav path on-device.
    public float[] Forward(IBackend backend, Tensor mel)
    {
        int nFft = _cfg.IstftNFft;
        int hop = _cfg.IstftHopSize;
        int upProd = 1;
        foreach (int u in _cfg.UpsampleRates) upProd *= u;

        // 1. F0 → NSF harmonic source → forward STFT.
        Tensor f0 = _f0.Forward(backend, mel);                       // [1, 1, T_mel] Hz
        // Deterministic (no-noise) NSF source for parity validation; production stays stochastic.
        int noiseSeed = Environment.GetEnvironmentVariable("HIFT_DETERMINISTIC") == "1" ? -1 : 0;
        float[] harSource = NsfVocoderDsp.GenerateHarmonicSource(f0, upProd * hop, _cfg.SampleRate, Harmonics, _mSourceW!, _mSourceB!, SineAmp, NoiseStd, VoicedThreshold, noiseSeed);
        f0.Dispose();
        Tensor sStft = NsfVocoderDsp.ForwardStftRealImag(harSource, nFft, hop);    // [1, n_fft+2, frames_src] = cat([Re, Im])

        // 2. conv_pre.
        int tMel = (int)mel.Shape[2];
        Tensor x = new(new TensorShape(1, _cfg.UpsampleInitialChannel, tMel), DType.F32);
        backend.Conv1d(x, mel, _convPreW!, _convPreB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);

        int numKernels = _cfg.ResBlockKernelSizes.Length;
        for (int i = 0; i < _numUp; i++)
        {
            backend.LeakyRelu(x, x, LeakySlope);
            Tensor xUp = UpsampleConvT(backend, x, i);
            x.Dispose();
            x = xUp;
            if (i == _numUp - 1)
            {
                Tensor xp = NsfVocoderDsp.ReflectionPadLeft1(x);
                x.Dispose();
                x = xp;
            }

            // Source injection: downsample the source STFT to this level + plain Snake resblock.
            Tensor si = SourceDown(backend, sStft, i);
            Tensor siRes = _srcResBlocks[i].Forward(backend, si);
            si.Dispose();
            NsfVocoderDsp.AddInPlaceCropped(x, siRes);
            siRes.Dispose();

            // MRF: mean of the kernel resblocks.
            Tensor acc = _resBlocks[i * numKernels].Forward(backend, x);
            for (int j = 1; j < numKernels; j++)
            {
                Tensor rb = _resBlocks[i * numKernels + j].Forward(backend, x);
                NsfVocoderDsp.AddInPlaceCropped(acc, rb);
                rb.Dispose();
            }
            x.Dispose();
            NsfVocoderDsp.ScaleInPlace(acc, 1f / numKernels);
            x = acc;
        }
        sStft.Dispose();

        // Post-loop activation uses LeakyReLU's DEFAULT slope (0.01), not the in-loop 0.1; and there is NO
        // extra reflection pad here — the only reflection pad is inside the loop on the last upsample.
        backend.LeakyRelu(x, x, 0.01f);

        int frames = (int)x.Shape[2];
        Tensor post = new(new TensorShape(1, nFft + 2, frames), DType.F32);
        backend.Conv1d(post, x, _convPostW!, _convPostB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        x.Dispose();

        float[] audio = NsfVocoderDsp.IstftHead(post, nFft, hop);
        post.Dispose();
        float limit = 0.99f;
        for (int i = 0; i < audio.Length; i++) audio[i] = Math.Clamp(audio[i], -limit, limit);
        return audio;
    }

    // Upper bound on HIFT's total receptive field translated to 24 kHz audio samples (F0 predictor mel-domain
    // margin + per-level MRF/source-resblock dilated convs [1,3,5] compounding sequentially within each
    // SnakeResBlock, propagated back up through 3 upsample stages [8,5,3] and the final iSTFT hop=4). Only
    // needs to be AT LEAST the true receptive field for emitted samples to be exact (see HiFTStreamState's doc
    // comment) — empirically calibrated against real `hift.pt` weights via HiftStreamParityTests: the true
    // requirement is ~38-42k samples (maxAbs crosses below 1e-3 there), but that boundary is noisy — jitters
    // non-monotonically with margin size, consistent with cuDNN conv-algorithm selection varying by tensor
    // shape rather than genuine unsettled content — so 48k is picked with real margin below that noise floor
    // (measured maxAbs 1.6-2.9e-4 across chunk sizes 1/2/6/12, an order of magnitude under the 1e-3 gate).
    // HARTSY_HIFT_STREAM_MARGIN overrides it for recalibrating without a rebuild.
    private static readonly int StreamMarginSamples = ReadMarginOverride() ?? 48_000;

    private static int? ReadMarginOverride() =>
        int.TryParse(Environment.GetEnvironmentVariable("HARTSY_HIFT_STREAM_MARGIN"), out int v) ? v : null;

    /// <summary>Streaming counterpart to <see cref="Forward"/>: feed successive mel chunks of one utterance
    /// (via <paramref name="state"/>), get back only the audio newly settled by this chunk. Set
    /// <paramref name="isFinal"/> on the last chunk of an utterance to flush the remaining
    /// <see cref="StreamMarginSamples"/>-sample tail unconditionally (mirrors <see cref="Forward"/>'s own
    /// natural end-of-utterance behavior — there's no more future mel coming, so nothing is held back).
    /// See <see cref="HiFTStreamState"/> for why this reuses <see cref="Forward"/> verbatim instead of a
    /// separate causal implementation.
    ///
    /// <para><b>NOT PRODUCTION-READY — do not wire into <see cref="Pipelines.CosyVoicePipeline"/> or deploy
    /// yet.</b> The recompute-with-margin design is exact for the local/convolutional parts of the network
    /// but NOT for the NSF harmonic source's unbounded phase accumulator (see
    /// <see cref="Dsp.NsfVocoderDsp.GenerateHarmonicSource"/>'s <c>cum</c> array): tiny per-call numeric
    /// differences at otherwise-settled positions integrate through that running sum and, inside a
    /// <c>sin()</c>, grow with utterance length instead of decaying with distance from the current edge.
    /// Verified 2026-08-10 (see <c>HiftStreamParityTests</c>'s doc comment for the exact measurements): fine
    /// for short clips, relL2≈0.6-0.7 by ~10s regardless of chunk size or margin generosity, matching the
    /// original plan's own risk call that this component needs real carried phase state, not recompute.</para>
    /// </summary>
    public float[] ForwardStreaming(IBackend backend, Tensor melChunk, HiFTStreamState state, bool isFinal = false)
    {
        Tensor melSoFar = state.MelHistory is null ? CopyMel(melChunk) : ConcatMelTime(state.MelHistory, melChunk);
        state.MelHistory?.Dispose();
        state.MelHistory = melSoFar;

        // Forward() runs melSoFar through a GPU backend, which can leave a device-side cache/binding on the
        // Tensor object; a later plain host read of that SAME tensor's DataPointer (next call's ConcatMelTime,
        // reading state.MelHistory) can then sync back stale/reused device memory instead of the real host
        // content. So state.MelHistory itself must never be handed to a backend op — feed Forward() a
        // throwaway copy instead, which is discarded (never read again) right after this call returns.
        Tensor forwardInput = CopyMel(melSoFar);
        float[] audio = Forward(backend, forwardInput);
        forwardInput.Dispose();
        int emitEnd = isFinal ? audio.Length : Math.Max(state.EmittedSamples, audio.Length - StreamMarginSamples);
        emitEnd = Math.Min(emitEnd, audio.Length);
        int start = Math.Min(state.EmittedSamples, emitEnd);
        float[] chunk = audio[start..emitEnd];
        state.EmittedSamples = emitEnd;
        return chunk;
    }

    private static unsafe Tensor CopyMel(Tensor mel)
    {
        Tensor copy = new(mel.Shape, DType.F32);
        Buffer.MemoryCopy((void*)mel.DataPointer, (void*)copy.DataPointer, mel.ElementCount * 4, mel.ElementCount * 4);
        return copy;
    }

    /// <summary>Concatenates two channels-first <c>[1, C, T]</c> mel tensors along the time axis.</summary>
    private static unsafe Tensor ConcatMelTime(Tensor prev, Tensor next)
    {
        int c = (int)prev.Shape[1];
        int tPrev = (int)prev.Shape[2];
        int tNext = (int)next.Shape[2];
        Tensor outT = new(new TensorShape(1, c, tPrev + tNext), DType.F32);
        float* pp = (float*)prev.DataPointer;
        float* np = (float*)next.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int cc = 0; cc < c; cc++)
        {
            long dst = (long)cc * (tPrev + tNext);
            Buffer.MemoryCopy(pp + (long)cc * tPrev, op + dst, (long)tPrev * 4, (long)tPrev * 4);
            Buffer.MemoryCopy(np + (long)cc * tNext, op + dst + tPrev, (long)tNext * 4, (long)tNext * 4);
        }
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_convPreW, _convPreB, _convPostW, _convPostB, _mSourceW, _mSourceB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        for (int i = 0; i < _numUp; i++)
        {
            if (_upsW[i] is not null) yield return _upsW[i]!;
            if (_upsB[i] is not null) yield return _upsB[i]!;
            if (_srcDownW[i] is not null) yield return _srcDownW[i]!;
            if (_srcDownB[i] is not null) yield return _srcDownB[i]!;
            foreach (Tensor t in _srcResBlocks[i].EnumerateWeights()) yield return t;
        }
        foreach (SnakeResBlock r in _resBlocks) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        foreach (Tensor t in _f0.EnumerateWeights()) yield return t;
    }

    private Tensor UpsampleConvT(IBackend backend, Tensor x, int i)
    {
        Tensor wgt = _upsW[i]!;
        int outCh = (int)wgt.Shape[1];
        int kernel = (int)wgt.Shape[2];
        int stride = _cfg.UpsampleRates[i];
        int pad = (kernel - stride) / 2;
        int inLen = (int)x.Shape[2];
        int outLen = (inLen - 1) * stride + (kernel - 1) + 1 - 2 * pad;
        Tensor outT = new(new TensorShape(1, outCh, outLen), DType.F32);
        backend.ConvTranspose1d(outT, x, wgt, _upsB[i], stride: stride, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
        return outT;
    }

    private Tensor SourceDown(IBackend backend, Tensor sStft, int i)
    {
        Tensor wgt = _srcDownW[i]!;
        int outCh = (int)wgt.Shape[0];
        int kernel = (int)wgt.Shape[2];
        int stride = _srcDownStride[i];
        int pad = stride == 1 ? 0 : stride / 2;
        int inLen = (int)sStft.Shape[2];
        int outLen = (inLen + 2 * pad - (kernel - 1) - 1) / stride + 1;
        Tensor outT = new(new TensorShape(1, outCh, outLen), DType.F32);
        backend.Conv1d(outT, sStft, wgt, _srcDownB[i], stride: stride, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
        return outT;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

/// <summary>ConvRNNF0Predictor: 5 × (Conv1d k=3 + ELU) condnet → Linear(→1) → abs, predicting F0 in Hz
/// from the input mel. Output is channels-first <c>[1, 1, T]</c>.</summary>
internal sealed unsafe class F0Predictor
{
    private const int CondLayers = 5;
    private const int CondChannels = 512;
    private readonly int _melBins;
    private readonly Tensor?[] _condW = new Tensor[CondLayers];
    private readonly Tensor?[] _condB = new Tensor[CondLayers];
    private Tensor? _classW, _classB;

    public F0Predictor(int melBins) => _melBins = melBins;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // condnet is a Sequential of Conv1d/ELU; the conv layers sit at even indices 0,2,4,6,8.
        for (int i = 0; i < CondLayers; i++)
        {
            int idx = i * 2;
            _condW[i] = WeightNorm.Compose(w, $"{prefix}.condnet.{idx}");
            _condB[i] = WhisperOps.EnsureF32(w[$"{prefix}.condnet.{idx}.bias"]);
        }
        _classW = WhisperOps.EnsureF32(w[$"{prefix}.classifier.weight"]);
        _classB = WhisperOps.EnsureF32(w[$"{prefix}.classifier.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor mel)
    {
        int t = (int)mel.Shape[2];
        Tensor x = new(new TensorShape(1, CondChannels, t), DType.F32);
        backend.Conv1d(x, mel, _condW[0]!, _condB[0], 1, 1, 1, 1, 1);
        backend.Elu(x, x, 1f);
        for (int i = 1; i < CondLayers; i++)
        {
            Tensor nx = new(new TensorShape(1, CondChannels, t), DType.F32);
            backend.Conv1d(nx, x, _condW[i]!, _condB[i], 1, 1, 1, 1, 1);
            x.Dispose();
            x = nx;
            backend.Elu(x, x, 1f);
        }
        // classifier: Linear over the channel dim per time step → [1, T, 1].
        Tensor xt = new(new TensorShape(1, t, CondChannels), DType.F32);   // [1, T, 512]
        backend.Transpose2D(xt, x, CondChannels, t);
        x.Dispose();
        Tensor f0t = WhisperOps.ProjectLinear(backend, xt, _classW!, _classB, 1, t, CondChannels, 1);
        xt.Dispose();
        // abs → [1, 1, T].
        Tensor f0 = new(new TensorShape(1, 1, t), DType.F32);
        float* sp = (float*)f0t.DataPointer;
        float* dp = (float*)f0.DataPointer;
        for (int i = 0; i < t; i++) dp[i] = MathF.Abs(sp[i]);
        f0t.Dispose();
        return f0;
    }


    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < CondLayers; i++)
        {
            if (_condW[i] is not null) yield return _condW[i]!;
            if (_condB[i] is not null) yield return _condB[i]!;
        }
        if (_classW is not null) yield return _classW;
        if (_classB is not null) yield return _classB;
    }
}
