using SharpInference.Audio.Layers;
using SharpInference.Audio.Models.Vocoders;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Preprocessing;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.CosyVoice;

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
    private HiftSnakeResBlock[] _srcResBlocks;
    private HiftSnakeResBlock[] _resBlocks;       // numUp * 3
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
        _srcResBlocks = new HiftSnakeResBlock[_numUp];
        int[] srcK = SourceResBlockKernels(_numUp);
        for (int i = 0; i < _numUp; i++)
            _srcResBlocks[i] = new HiftSnakeResBlock(_levelChannels[i], srcK[i], [1, 3, 5]);
        _resBlocks = new HiftSnakeResBlock[_numUp * cfg.ResBlockKernelSizes.Length];
        for (int i = 0; i < _numUp; i++)
            for (int j = 0; j < cfg.ResBlockKernelSizes.Length; j++)
                _resBlocks[i * cfg.ResBlockKernelSizes.Length + j] =
                    new HiftSnakeResBlock(_levelChannels[i], cfg.ResBlockKernelSizes[j], cfg.ResBlockDilationSizes[j]);
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
    public float[] Forward(IBackend backend, Tensor mel)
    {
        int nFft = _cfg.IstftNFft;
        int hop = _cfg.IstftHopSize;
        int upProd = 1;
        foreach (int u in _cfg.UpsampleRates) upProd *= u;

        // 1. F0 → NSF harmonic source → forward STFT.
        Tensor f0 = _f0.Forward(backend, mel);                       // [1, 1, T_mel] Hz
        float[] harSource = GenerateHarmonicSource(f0, upProd * hop, _cfg.SampleRate);
        f0.Dispose();
        Tensor sStft = ForwardStftMagPhase(harSource, nFft, hop);    // [1, n_fft+2, frames_src]

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
                Tensor xp = ReflectionPadLeft1(x);
                x.Dispose();
                x = xp;
            }

            // Source injection: downsample the source STFT to this level + plain Snake resblock.
            Tensor si = SourceDown(backend, sStft, i);
            Tensor siRes = _srcResBlocks[i].Forward(backend, si);
            si.Dispose();
            AddInPlaceCropped(x, siRes);
            siRes.Dispose();

            // MRF: mean of the kernel resblocks.
            Tensor acc = _resBlocks[i * numKernels].Forward(backend, x);
            for (int j = 1; j < numKernels; j++)
            {
                Tensor rb = _resBlocks[i * numKernels + j].Forward(backend, x);
                AddInPlaceCropped(acc, rb);
                rb.Dispose();
            }
            x.Dispose();
            ScaleInPlace(acc, 1f / numKernels);
            x = acc;
        }
        sStft.Dispose();

        backend.LeakyRelu(x, x, LeakySlope);
        Tensor xpad = ReflectionPadLeft1(x);
        x.Dispose();
        x = xpad;

        int frames = (int)x.Shape[2];
        Tensor post = new(new TensorShape(1, nFft + 2, frames), DType.F32);
        backend.Conv1d(post, x, _convPostW!, _convPostB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        x.Dispose();

        float[] audio = IstftHead(post, nFft, hop);
        post.Dispose();
        float limit = 0.99f;
        for (int i = 0; i < audio.Length; i++) audio[i] = Math.Clamp(audio[i], -limit, limit);
        return audio;
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
        foreach (HiftSnakeResBlock r in _resBlocks) foreach (Tensor t in r.EnumerateWeights()) yield return t;
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
        backend.ConvTranspose1d(outT, x, wgt, _upsB[i], stride: stride, padLeft: pad, padRight: pad, dilation: 1);
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

    // ── NSF source / STFT / iSTFT helpers (same pattern as the Kokoro iSTFTNet decoder) ──

    private float[] GenerateHarmonicSource(Tensor f0, int scale, int sampleRate)
    {
        int t0 = (int)f0.Shape[2];
        int audioLen = t0 * scale;
        float* fp = (float*)f0.DataPointer;
        double[] cum = new double[Harmonics];
        float* mW = (float*)_mSourceW!.DataPointer;
        float mB = ((float*)_mSourceB!.DataPointer)[0];
        float[] merged = new float[audioLen];
        uint rng = 0x9E3779B9u;
        for (int i = 0; i < t0; i++)
        {
            float hz = fp[i];
            float uv = hz > VoicedThreshold ? 1f : 0f;
            float noiseAmp = uv * NoiseStd + (1f - uv) * (SineAmp / 3f);
            for (int rep = 0; rep < scale; rep++)
            {
                float lin = mB;
                for (int h = 0; h < Harmonics; h++)
                {
                    cum[h] += (double)hz * (h + 1) / sampleRate;
                    cum[h] -= Math.Floor(cum[h]);
                    float sine = (float)Math.Sin(2.0 * Math.PI * cum[h]) * SineAmp;
                    float noise = noiseAmp * NextGaussian(ref rng);
                    lin += mW[h] * (sine * uv + noise);
                }
                merged[i * scale + rep] = MathF.Tanh(lin);
            }
        }
        return merged;
    }

    private static Tensor ForwardStftMagPhase(float[] signal, int nFft, int hop)
    {
        int half = nFft / 2;
        int numBins = half + 1;
        int pad = half;
        int paddedLen = signal.Length + 2 * pad;
        float[] padded = new float[paddedLen];
        for (int i = 0; i < pad; i++)
        {
            padded[i] = signal[Math.Min(pad - i, signal.Length - 1)];
            padded[paddedLen - 1 - i] = signal[Math.Max(signal.Length - 2 - i, 0)];
        }
        Array.Copy(signal, 0, padded, pad, signal.Length);
        int frames = 1 + (paddedLen - nFft) / hop;
        if (frames < 1) frames = 1;
        float[] window = HannWindow.Get(nFft);
        Tensor outT = new(new TensorShape(1, nFft + 2, frames), DType.F32);
        float* op = (float*)outT.DataPointer;
        float[] frame = new float[nFft];
        float[] re = new float[numBins];
        float[] im = new float[numBins];
        for (int f = 0; f < frames; f++)
        {
            int start = f * hop;
            for (int k = 0; k < nFft; k++) frame[k] = padded[start + k] * window[k];
            Fft.RealTransform(frame, re, im, nFft);
            for (int b = 0; b < numBins; b++)
            {
                op[b * frames + f] = MathF.Sqrt(re[b] * re[b] + im[b] * im[b]);
                op[(numBins + b) * frames + f] = MathF.Atan2(im[b], re[b]);
            }
        }
        return outT;
    }

    private static float[] IstftHead(Tensor post, int nFft, int hop)
    {
        int numBins = nFft / 2 + 1;
        int frames = (int)post.Shape[2];
        float* pp = (float*)post.DataPointer;
        float[] real = new float[frames * numBins];
        float[] imag = new float[frames * numBins];
        for (int f = 0; f < frames; f++)
            for (int b = 0; b < numBins; b++)
            {
                float mag = MathF.Exp(pp[b * frames + f]);
                float ang = MathF.Sin(pp[(numBins + b) * frames + f]);
                real[f * numBins + b] = mag * MathF.Cos(ang);
                imag[f * numBins + b] = mag * MathF.Sin(ang);
            }
        return IStft.Apply(real, imag, frames, nFft, hop);
    }

    private static Tensor ReflectionPadLeft1(Tensor x)
    {
        int c = (int)x.Shape[1];
        int t = (int)x.Shape[2];
        Tensor outT = new(new TensorShape(1, c, t + 1), DType.F32);
        float* ip = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int cc = 0; cc < c; cc++)
        {
            long src = (long)cc * t;
            long dst = (long)cc * (t + 1);
            op[dst] = ip[src + Math.Min(1, t - 1)];
            for (int j = 0; j < t; j++) op[dst + 1 + j] = ip[src + j];
        }
        return outT;
    }

    private static void AddInPlaceCropped(Tensor dst, Tensor src)
    {
        int dc = (int)dst.Shape[1], sc = (int)src.Shape[1];
        int c = Math.Min(dc, sc);
        int td = (int)dst.Shape[2], ts = (int)src.Shape[2];
        int t = Math.Min(td, ts);
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        for (int cc = 0; cc < c; cc++)
        {
            long db = (long)cc * td, sb = (long)cc * ts;
            for (int j = 0; j < t; j++) dp[db + j] += sp[sb + j];
        }
    }

    private static void ScaleInPlace(Tensor x, float factor)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) p[i] *= factor;
    }

    private static float NextGaussian(ref uint state)
    {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        float u1 = (state & 0xFFFFFF) / 16777216f;
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        float u2 = (state & 0xFFFFFF) / 16777216f;
        if (u1 < 1e-7f) u1 = 1e-7f;
        return MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

/// <summary>HiFiGAN-style ResBlock1 with plain (non-style) Snake activations — three sequential
/// dilated branches, each <c>Snake → dilated Conv1d → Snake → Conv1d</c> added back as a residual.
/// Used for both the generator MRF and the source-injection resblocks.</summary>
internal sealed unsafe class HiftSnakeResBlock
{
    private readonly int _channels;
    private readonly int _kernel;
    private readonly int[] _dilations;
    private readonly Tensor?[] _convs1W;
    private readonly Tensor?[] _convs1B;
    private readonly Tensor?[] _convs2W;
    private readonly Tensor?[] _convs2B;
    private readonly Tensor?[] _alpha1;
    private readonly Tensor?[] _alpha2;

    public HiftSnakeResBlock(int channels, int kernel, int[] dilations)
    {
        _channels = channels;
        _kernel = kernel;
        _dilations = dilations;
        int n = dilations.Length;
        _convs1W = new Tensor[n];
        _convs1B = new Tensor[n];
        _convs2W = new Tensor[n];
        _convs2B = new Tensor[n];
        _alpha1 = new Tensor[n];
        _alpha2 = new Tensor[n];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < _dilations.Length; i++)
        {
            _convs1W[i] = WeightNorm.Compose(w, $"{prefix}.convs1.{i}");
            _convs1B[i] = WhisperOps.EnsureF32(w[$"{prefix}.convs1.{i}.bias"]);
            _convs2W[i] = WeightNorm.Compose(w, $"{prefix}.convs2.{i}");
            _convs2B[i] = WhisperOps.EnsureF32(w[$"{prefix}.convs2.{i}.bias"]);
            _alpha1[i] = WhisperOps.EnsureF32(w[$"{prefix}.activations1.{i}.alpha"]);
            _alpha2[i] = WhisperOps.EnsureF32(w[$"{prefix}.activations2.{i}.alpha"]);
        }
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int t = (int)x.Shape[2];
        Tensor cur = new(x.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)cur.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
        for (int i = 0; i < _dilations.Length; i++)
        {
            int pad1 = (_kernel - 1) * _dilations[i] / 2;
            int pad2 = (_kernel - 1) / 2;
            Tensor a1 = new(cur.Shape, DType.F32);
            backend.Snake(a1, cur, _alpha1[i]!, null);
            Tensor c1 = new(new TensorShape(1, _channels, t), DType.F32);
            backend.Conv1d(c1, a1, _convs1W[i]!, _convs1B[i], 1, pad1, pad1, _dilations[i], 1);
            a1.Dispose();
            Tensor a2 = new(c1.Shape, DType.F32);
            backend.Snake(a2, c1, _alpha2[i]!, null);
            c1.Dispose();
            Tensor c2 = new(new TensorShape(1, _channels, t), DType.F32);
            backend.Conv1d(c2, a2, _convs2W[i]!, _convs2B[i], 1, pad2, pad2, 1, 1);
            a2.Dispose();
            float* cp = (float*)cur.DataPointer;
            float* c2p = (float*)c2.DataPointer;
            long n = cur.ElementCount;
            for (long k = 0; k < n; k++) cp[k] += c2p[k];
            c2.Dispose();
        }
        return cur;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _dilations.Length; i++)
        {
            Tensor?[] all = [_convs1W[i], _convs1B[i], _convs2W[i], _convs2B[i], _alpha1[i], _alpha2[i]];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
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
        Tensor xt = Transpose1d(x);            // [1, T, 512]
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

    private static Tensor Transpose1d(Tensor x)
    {
        int c = (int)x.Shape[1];
        int t = (int)x.Shape[2];
        Tensor outT = new(new TensorShape(1, t, c), DType.F32);
        float* ip = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int cc = 0; cc < c; cc++)
            for (int j = 0; j < t; j++)
                op[(long)j * c + cc] = ip[(long)cc * t + j];
        return outT;
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
