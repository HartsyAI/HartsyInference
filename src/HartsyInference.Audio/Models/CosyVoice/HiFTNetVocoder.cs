using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>CosyVoice 2's HiFTNet vocoder (mel → 24 kHz waveform).</summary>
/// <remarks>Mirrors
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
/// <c>hift.pt</c>.</para></remarks>
public sealed unsafe class HiFTNetVocoder : IDisposable
{
    private const float LeakySlope = 0.1f;
    internal const int Harmonics = 9;             // fundamental + 8 — HiFTStreamState sizes its carried phase array off this
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
        int upProd = 1;
        foreach (int u in _cfg.UpsampleRates) upProd *= u;

        // F0 → NSF harmonic source, computed fresh from t=0 (this is the non-streaming path — no phase state
        // to carry). Deterministic (no-noise) NSF source for parity validation; production stays stochastic.
        Tensor f0 = _f0.Forward(backend, mel);                       // [1, 1, T_mel] Hz
        int noiseSeed = Environment.GetEnvironmentVariable("HIFT_DETERMINISTIC") == "1" ? -1 : 0;
        float[] harSource = NsfVocoderDsp.GenerateHarmonicSource(f0, upProd * _cfg.IstftHopSize, _cfg.SampleRate, Harmonics, _mSourceW!, _mSourceB!, SineAmp, NoiseStd, VoicedThreshold, noiseSeed);
        f0.Dispose();
        return ForwardCore(backend, mel, harSource);
    }

    /// <summary>Monolithic variant of <see cref="Forward"/> that computes F0 via <see cref="F0Predictor.ForwardHostCpu"/> instead of the GPU <see cref="F0Predictor.Forward"/> — the SAME F0 computation <see cref="ForwardStreaming"/> uses internally. Numerically close to but NOT bit-identical with <see cref="Forward"/> (cuDNN vs. a naive host conv give slightly different F32 roundings for the same math — see <see cref="HiFTStreamState"/>'s doc comment). Exists as the correct reference for streaming self-parity tests (comparing <see cref="ForwardStreaming"/> against <see cref="Forward"/> conflates "is chunking exact" with "does host-CPU F0 match GPU F0", two different questions) and as a non-chunked entry point when exact consistency with a streamed run of the same utterance matters more than matching the plain <see cref="Forward"/> path.</summary>
    public float[] ForwardHostCpuF0(IBackend backend, Tensor mel)
    {
        int upProd = 1;
        foreach (int u in _cfg.UpsampleRates) upProd *= u;
        int melBins = (int)mel.Shape[1];
        int tMel = (int)mel.Shape[2];
        float[] melArray = new float[melBins * tMel];
        unsafe
        {
            float* mp = (float*)mel.DataPointer;
            for (int i = 0; i < melArray.Length; i++) melArray[i] = mp[i];
        }
        float[] f0Array = _f0.ForwardHostCpu(melArray, melBins, tMel);
        int noiseSeed = Environment.GetEnvironmentVariable("HIFT_DETERMINISTIC") == "1" ? -1 : 0;
        double[] phase = new double[Harmonics];
        uint rng = noiseSeed == 0 ? 0x9E3779B9u : DeterministicRng.Seed(Math.Abs(noiseSeed));
        float[] harSource = NsfVocoderDsp.GenerateHarmonicSourceChunk(f0Array, phase, ref rng,
            upProd * _cfg.IstftHopSize, _cfg.SampleRate, Harmonics, _mSourceW!, _mSourceB!, SineAmp, NoiseStd,
            VoicedThreshold, addNoise: noiseSeed >= 0);
        return ForwardCore(backend, mel, harSource);
    }

    /// <summary>Shared tail of <see cref="Forward"/>: everything from the harmonic source's forward STFT onward (source injection, conv_pre/upsample/MRF loop, conv_post, iSTFT). <paramref name="harSource"/> is the audio-domain NSF signal (<c>mel.Shape[2] · upsampleProd · hop</c> samples) — <see cref="Forward"/> builds it fresh each call; <see cref="ForwardStreaming"/> builds it from carried phase state instead (see <see cref="HiFTStreamState"/>) so this method itself has no streaming-specific logic at all.</summary>
    public float[] ForwardCore(IBackend backend, Tensor mel, float[] harSource)
    {
        int nFft = _cfg.IstftNFft;
        int hop = _cfg.IstftHopSize;
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

    // Upper bound on HIFT's LOCAL (non-accumulating) receptive field translated to 24 kHz audio samples: F0
    // predictor mel-domain margin + per-level MRF/source-resblock dilated convs [1,3,5] compounding
    // sequentially within each SnakeResBlock, propagated back up through 3 upsample stages [8,5,3] and the
    // final iSTFT hop=4. This does NOT need to (and must not be relied on to) hide the NSF harmonic source's
    // phase drift — that's handled separately by carrying real phase state (see HiFTStreamState's doc
    // comment) — so this margin only has to cover genuine local unsettled-boundary content, not the
    // utterance-length-scaling failure the carried-phase fix eliminates. Once that fix landed, error stopped
    // growing with utterance length and plateaued around relL2≈2e-3 (maxAbs bounces non-monotonically with
    // margin size, 48k-96k) — a bounded, non-growing GPU-shape-dependent cuDNN algorithm-selection noise
    // floor (same class as F0Predictor's, just far smaller since it doesn't feed an accumulator), not a
    // correctness bug; see HiftStreamParityTests for the real-generation listen-test verification this rests
    // on. Recalibrate via HARTSY_HIFT_STREAM_MARGIN whenever this network's conv/resblock shapes change.
    private static readonly int StreamMarginSamples = ReadMarginOverride() ?? 96_000;

    private static int? ReadMarginOverride() =>
        int.TryParse(Environment.GetEnvironmentVariable("HARTSY_HIFT_STREAM_MARGIN"), out int v) ? v : null;

    /// <summary>Streaming counterpart to <see cref="Forward"/>: feed successive mel chunks of one utterance and get back only the audio newly settled by this chunk.</summary>
    /// <remarks>Consumed via <paramref name="state"/>. Set
    /// <paramref name="isFinal"/> on the last chunk of an utterance to flush the remaining
    /// <see cref="StreamMarginSamples"/>-sample tail unconditionally (mirrors <see cref="Forward"/>'s own
    /// natural end-of-utterance behavior — there's no more future mel coming, so nothing is held back).
    ///
    /// <para>Two different mechanisms cover the two different pieces of this network (see
    /// <see cref="HiFTStreamState"/>'s doc comment for the full reasoning):</para>
    /// <list type="bullet">
    ///   <item><b>conv_pre/upsample/MRF/conv_post + iSTFT</b> (purely local, no cross-frame accumulation):
    ///         recompute-with-margin, via <see cref="ForwardCore"/> over the growing mel-so-far — safe because
    ///         nothing here integrates error across time, only local receptive field matters.</item>
    ///   <item><b>NSF harmonic source</b> (an unbounded phase integral — recompute-with-margin is NOT safe
    ///         here, see <see cref="HiFTStreamState"/>): F0 for each historical mel frame is derived exactly
    ///         once, the moment it first crosses the settlement margin, and immediately consumed into
    ///         <see cref="HiFTStreamState.HarmonicPhase"/>/<see cref="HiFTStreamState.NoiseRngState"/> — never
    ///         re-derived. The still-unsettled tail gets a THROWAWAY continuation (a copy of the carried
    ///         phase/RNG, advanced but not persisted) purely so <see cref="ForwardCore"/> has something to
    ///         inject at those positions this call; its output there is discarded by the margin anyway.</item>
    /// </list></remarks>
    public float[] ForwardStreaming(IBackend backend, Tensor melChunk, HiFTStreamState state, bool isFinal = false)
    {
        Tensor melSoFar = state.MelHistory is null ? CopyMel(melChunk) : ConcatMelTime(state.MelHistory, melChunk);
        state.MelHistory?.Dispose();
        state.MelHistory = melSoFar;

        // Forward()/ForwardCore() run melSoFar through a GPU backend, which can leave a device-side
        // cache/binding on the Tensor object; a later plain host read of that SAME tensor's DataPointer (next
        // call's ConcatMelTime, reading state.MelHistory) can then sync back stale/reused device memory
        // instead of the real host content. So state.MelHistory itself must never be handed to a backend op —
        // feed everything below a throwaway copy instead, discarded (never read again) right after this call.
        Tensor forwardInput = CopyMel(melSoFar);
        int tMel = (int)melSoFar.Shape[2];

        int upProd = 1;
        foreach (int u in _cfg.UpsampleRates) upProd *= u;
        int scale = upProd * _cfg.IstftHopSize;
        int marginMelFrames = Math.Max(1, StreamMarginSamples / scale);
        int settledMelFrames = isFinal ? tMel
            : Math.Min(tMel, Math.Max(state.HarmonicSettledMelFrames, tMel - marginMelFrames));

        // F0 for the harmonic source MUST come from a genuinely position-local computation — see
        // F0Predictor.ForwardHostCpu's doc comment: backend.Conv1d's cuDNN output for a given position
        // varies with the TOTAL tensor length, not just local content, which silently breaks the "settled
        // by margin" assumption the phase accumulator's correctness depends on. Bound the window fed to it
        // (not the full growing melSoFar — cheap host-CPU cost per call instead of growing over the whole
        // utterance) — but crucially, leave REAL left context (f0LeftMargin) before the point we're about to
        // consume: F0Predictor's own conv stack zero-pads the window's OWN left edge exactly like it zero-pads
        // a true utterance start, so consuming right at that edge would (and did, first attempt) read the
        // single most UNSETTLED position in the whole window, not a settled one. F0Predictor's true receptive
        // field is tiny (5 layers × k=3 ≈ ±5 frames); this margin is generously larger.
        int melBins = (int)melSoFar.Shape[1];
        int f0LeftMargin = Math.Min(marginMelFrames, 64);
        int f0WindowStart = Math.Max(0, state.HarmonicSettledMelFrames - f0LeftMargin);
        int f0WindowLen = tMel - f0WindowStart;
        float[] melWindow = new float[melBins * f0WindowLen];
        unsafe
        {
            float* mp = (float*)melSoFar.DataPointer;
            for (int c = 0; c < melBins; c++)
                for (int i = 0; i < f0WindowLen; i++)
                    melWindow[c * f0WindowLen + i] = mp[c * tMel + (f0WindowStart + i)];
        }
        float[] f0Window = _f0.ForwardHostCpu(melWindow, melBins, f0WindowLen);   // covers [f0WindowStart, tMel)

        int noiseSeed = Environment.GetEnvironmentVariable("HIFT_DETERMINISTIC") == "1" ? -1 : 0;
        bool addNoise = noiseSeed >= 0;

        // Consume exactly the NEWLY settled F0 range into the carried phase/RNG — permanent, never redone.
        if (settledMelFrames > state.HarmonicSettledMelFrames)
        {
            float[] newF0 = f0Window[(state.HarmonicSettledMelFrames - f0WindowStart)..(settledMelFrames - f0WindowStart)];
            float[] newAudio = NsfVocoderDsp.GenerateHarmonicSourceChunk(newF0, state.HarmonicPhase,
                ref state.NoiseRngState, scale, _cfg.SampleRate, Harmonics, _mSourceW!, _mSourceB!, SineAmp,
                NoiseStd, VoicedThreshold, addNoise);
            state.HarmonicAudioSoFar.AddRange(newAudio);
            state.HarmonicSettledMelFrames = settledMelFrames;
        }

        // Throwaway continuation for the still-unsettled tail (this call's ForwardCore needs SOMETHING there,
        // but its output in that region is cropped away below, never emitted) — a copy of the carried state,
        // never written back.
        float[] harmonicAudioForThisCall;
        if (settledMelFrames < tMel)
        {
            float[] tailF0 = f0Window[(settledMelFrames - f0WindowStart)..(tMel - f0WindowStart)];
            double[] tailPhase = (double[])state.HarmonicPhase.Clone();
            uint tailRng = state.NoiseRngState;
            float[] tailAudio = NsfVocoderDsp.GenerateHarmonicSourceChunk(tailF0, tailPhase, ref tailRng, scale,
                _cfg.SampleRate, Harmonics, _mSourceW!, _mSourceB!, SineAmp, NoiseStd, VoicedThreshold, addNoise);
            harmonicAudioForThisCall = new float[state.HarmonicAudioSoFar.Count + tailAudio.Length];
            state.HarmonicAudioSoFar.CopyTo(harmonicAudioForThisCall);
            Array.Copy(tailAudio, 0, harmonicAudioForThisCall, state.HarmonicAudioSoFar.Count, tailAudio.Length);
        }
        else
        {
            harmonicAudioForThisCall = state.HarmonicAudioSoFar.ToArray();
        }

        float[] audio = ForwardCore(backend, forwardInput, harmonicAudioForThisCall);
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

/// <summary>ConvRNNF0Predictor: 5 × (Conv1d k=3 + ELU) condnet → Linear(→1) → abs, predicting F0 in Hz from the input mel. Output is channels-first <c>[1, 1, T]</c>.</summary>
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

    /// <summary>Pure host C# reimplementation of <see cref="Forward"/> — no <see cref="IBackend"/> call at all. Exists ONLY for <see cref="HiFTNetVocoder.ForwardStreaming"/>, where F0 feeds an unbounded phase accumulator: confirmed empirically (2026-08-11) that <see cref="Forward"/>'s <c>backend.Conv1d</c> output for a given INTERIOR position (far from either edge, well past any plausible local receptive field) still varies by up to ~0.16 Hz depending on the TOTAL input length — cuDNN's algorithm selection depends on overall tensor shape, not just local content, so "settled by margin" does not hold for it the way it does for the rest of this vocoder (see <see cref="HiFTStreamState"/>'s doc comment). A naive, un-tiled host loop has no such shape-dependent code path — a given output position depends only on its own literal kernel window, computed the same way regardless of how much more input exists beyond it. F0Predictor is tiny (5×512-channel k=3 convs + 1 linear), so the O(T) host cost here is negligible next to the GPU work the rest of the vocoder still does. <paramref name="mel"/> is channel-major <c>[melBins, T]</c> (row-major, T fastest-varying, matching <see cref="Tensor"/>'s own layout); returns F0 in Hz, length <c>T</c>.</summary>
    internal unsafe float[] ForwardHostCpu(float[] mel, int melBins, int t)
    {
        float[] cur = mel;
        int curCh = melBins;
        for (int layer = 0; layer < CondLayers; layer++)
        {
            float* w = (float*)_condW[layer]!.DataPointer;
            float* b = (float*)_condB[layer]!.DataPointer;
            int kernel = (int)_condW[layer]!.Shape[2];
            int pad = (kernel - 1) / 2;
            float[] next = new float[CondChannels * t];
            for (int oc = 0; oc < CondChannels; oc++)
            {
                long ocBase = (long)oc * curCh * kernel;
                for (int ti = 0; ti < t; ti++)
                {
                    float acc = b[oc];
                    for (int ic = 0; ic < curCh; ic++)
                    {
                        long wBase = ocBase + (long)ic * kernel;
                        long inBase = (long)ic * t;
                        for (int k = 0; k < kernel; k++)
                        {
                            int inPos = ti - pad + k;
                            if ((uint)inPos < (uint)t) acc += w[wBase + k] * cur[inBase + inPos];
                        }
                    }
                    // ELU, alpha=1 (matches backend.Elu(x, x, 1f)).
                    next[(long)oc * t + ti] = acc > 0f ? acc : MathF.Exp(acc) - 1f;
                }
            }
            cur = next;
            curCh = CondChannels;
        }

        // classifier: Linear(CondChannels -> 1) per timestep, then abs.
        float* cw = (float*)_classW!.DataPointer;
        float cb = ((float*)_classB!.DataPointer)[0];
        float[] f0 = new float[t];
        for (int ti = 0; ti < t; ti++)
        {
            float acc = cb;
            for (int c = 0; c < curCh; c++) acc += cw[c] * cur[(long)c * t + ti];
            f0[ti] = MathF.Abs(acc);
        }
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
