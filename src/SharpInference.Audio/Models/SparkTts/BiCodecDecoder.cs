using SharpInference.Audio.Layers;
using SharpInference.Audio.Models.Codecs;
using SharpInference.Audio.Models.Vocoders;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.SparkTts;

/// <summary>BiCodec decode side (Spark-TTS) — reconstructs a 16 kHz waveform from the LM's global +
/// semantic tokens. The semantic VQ stream provides time-varying content; the 32 global FSQ tokens
/// provide a time-invariant speaker d-vector. A DAC-style HiFi-GAN wave generator (Snake1d + transposed
/// convs at rates [8,5,4,2]) conditioned on the d-vector produces the audio.
///
/// <para><b>Reuse:</b> global dequant is the shared <see cref="Fsq.Dequantize"/>; the MRF blocks are the
/// shared <see cref="SnakeResBlock"/>; upsampling/activation route through the backend (`ConvTranspose1d`,
/// `Snake`, `Conv1d`). <b>Checkpoint-reconciliation pending:</b> exact BiCodec key names + the
/// Vocos-ConvNeXt prenet detail need the 626 MB <c>BiCodec/model.safetensors</c>; the speaker
/// conditioning here is a FiLM-lite add (the full AdaLayerNorm prenet is the deferred piece). The token
/// dequant + wave-gen structure are exercisable once weights load.</para></summary>
public sealed unsafe class BiCodecDecoder : IDisposable
{
    private readonly SparkBiCodecConfig _cfg;
    private readonly int _numUp;
    private readonly int[] _levelChannels;
    private int _disposed;

    private Tensor? _semanticCodebook;   // [SemanticVocab, semDim] VQ embedding (factorized → projected)
    private Tensor? _globalProjW, _globalProjB;   // FSQ d-vector → globalDim
    private Tensor? _convPreW, _convPreB;
    private Tensor? _condProjW, _condProjB;        // d-vector → base channels (FiLM-lite bias)
    private Tensor?[] _upsW;
    private Tensor?[] _upsB;
    private SnakeResBlock[] _resBlocks;
    private Tensor? _convPostW, _convPostB;

    public BiCodecDecoder(SparkBiCodecConfig cfg)
    {
        _cfg = cfg;
        _numUp = cfg.UpsampleRates.Length;
        _levelChannels = new int[_numUp];
        for (int i = 0; i < _numUp; i++) _levelChannels[i] = cfg.UpsampleInitialChannel >> (i + 1);
        _upsW = new Tensor[_numUp];
        _upsB = new Tensor[_numUp];
        _resBlocks = new SnakeResBlock[_numUp * cfg.ResBlockKernelSizes.Length];
        for (int i = 0; i < _numUp; i++)
            for (int j = 0; j < cfg.ResBlockKernelSizes.Length; j++)
                _resBlocks[i * cfg.ResBlockKernelSizes.Length + j] =
                    new SnakeResBlock(_levelChannels[i], cfg.ResBlockKernelSizes[j], cfg.ResBlockDilationSizes[j]);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _semanticCodebook = WhisperOps.EnsureF32(w[$"{p}quantizer.codebook.weight"]);
        _globalProjW = WhisperOps.EnsureF32(w[$"{p}speaker_proj.weight"]);
        _globalProjB = w.TryGetValue($"{p}speaker_proj.bias", out Tensor? gb) ? WhisperOps.EnsureF32(gb) : null;
        _convPreW = WeightNorm.Compose(w, $"{p}generator.conv_pre");
        _convPreB = WhisperOps.EnsureF32(w[$"{p}generator.conv_pre.bias"]);
        _condProjW = WhisperOps.EnsureF32(w[$"{p}generator.cond_proj.weight"]);
        _condProjB = w.TryGetValue($"{p}generator.cond_proj.bias", out Tensor? cb) ? WhisperOps.EnsureF32(cb) : null;
        for (int i = 0; i < _numUp; i++)
        {
            _upsW[i] = WeightNorm.Compose(w, $"{p}generator.ups.{i}");
            _upsB[i] = WhisperOps.EnsureF32(w[$"{p}generator.ups.{i}.bias"]);
        }
        for (int i = 0; i < _resBlocks.Length; i++) _resBlocks[i].LoadWeights(w, $"{p}generator.resblocks.{i}");
        _convPostW = WeightNorm.Compose(w, $"{p}generator.conv_post");
        _convPostB = WhisperOps.EnsureF32(w[$"{p}generator.conv_post.bias"]);
    }

    /// <summary>Decodes global + semantic token streams to a 16 kHz waveform.</summary>
    public float[] Decode(IBackend backend, IReadOnlyList<int> globalTokens, IReadOnlyList<int> semanticTokens)
    {
        if (_semanticCodebook is null) throw new InvalidOperationException("BiCodecDecoder weights not loaded.");

        // 1. Speaker d-vector from the 32 global FSQ tokens.
        Tensor dVector = GlobalDVector(backend, globalTokens);

        // 2. Semantic VQ embeddings → [1, semDim, T].
        Tensor feat = SemanticFeatures(semanticTokens);

        // 3. conv_pre → base channels, + FiLM-lite speaker conditioning.
        int t = (int)feat.Shape[2];
        Tensor x = new(new TensorShape(1, _cfg.UpsampleInitialChannel, t), DType.F32);
        backend.Conv1d(x, feat, _convPreW!, _convPreB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        feat.Dispose();
        AddSpeakerBias(backend, x, dVector);
        dVector.Dispose();

        // 4. Upsample stages: ConvTranspose1d + Snake-MRF average.
        int numKernels = _cfg.ResBlockKernelSizes.Length;
        for (int i = 0; i < _numUp; i++)
        {
            backend.LeakyRelu(x, x, 0.1f);
            Tensor xUp = UpsampleConvT(backend, x, i);
            x.Dispose();
            x = xUp;
            Tensor acc = _resBlocks[i * numKernels].Forward(backend, x);
            for (int j = 1; j < numKernels; j++)
            {
                Tensor rb = _resBlocks[i * numKernels + j].Forward(backend, x);
                AddInPlace(acc, rb);
                rb.Dispose();
            }
            x.Dispose();
            Scale(acc, 1f / numKernels);
            x = acc;
        }

        // 5. conv_post → 1-ch waveform → tanh.
        backend.LeakyRelu(x, x, 0.01f);
        int outLen = (int)x.Shape[2];
        Tensor wave = new(new TensorShape(1, 1, outLen), DType.F32);
        backend.Conv1d(wave, x, _convPostW!, _convPostB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        x.Dispose();

        float[] audio = new float[outLen];
        float* wp = (float*)wave.DataPointer;
        for (int i = 0; i < outLen; i++) audio[i] = MathF.Tanh(wp[i]);
        wave.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_semanticCodebook, _globalProjW, _globalProjB, _convPreW, _convPreB, _condProjW, _condProjB, _convPostW, _convPostB];
        foreach (Tensor? x in core) if (x is not null) yield return x;
        for (int i = 0; i < _numUp; i++)
        {
            if (_upsW[i] is not null) yield return _upsW[i]!;
            if (_upsB[i] is not null) yield return _upsB[i]!;
        }
        foreach (SnakeResBlock r in _resBlocks) foreach (Tensor t in r.EnumerateWeights()) yield return t;
    }

    /// <summary>32 global FSQ codes → dequant → mean-pool → linear → speaker d-vector <c>[1, globalDim]</c>.</summary>
    private Tensor GlobalDVector(IBackend backend, IReadOnlyList<int> globalTokens)
    {
        int n = globalTokens.Count;
        int d = _cfg.FsqLevels.Length;
        Tensor codes = new(new TensorShape(1, n), DType.F32);
        float* cp = (float*)codes.DataPointer;
        for (int i = 0; i < n; i++) cp[i] = globalTokens[i];
        Tensor zHat = new(new TensorShape(1, n, d), DType.F32);
        Fsq.Dequantize(zHat, codes, _cfg.FsqLevels);
        codes.Dispose();

        // Mean-pool over the 32 tokens → [1, 1, d], then project to globalDim.
        Tensor pooled = new(new TensorShape(1, 1, d), DType.F32);
        float* zp = (float*)zHat.DataPointer;
        float* pp = (float*)pooled.DataPointer;
        for (int j = 0; j < d; j++)
        {
            double s = 0;
            for (int i = 0; i < n; i++) s += zp[i * d + j];
            pp[j] = (float)(s / Math.Max(1, n));
        }
        zHat.Dispose();
        Tensor dVec = WhisperOps.ProjectLinear(backend, pooled, _globalProjW!, _globalProjB, 1, 1, d, _cfg.GlobalDim);
        pooled.Dispose();
        return dVec.Reshape(new TensorShape(1, _cfg.GlobalDim));
    }

    /// <summary>Semantic VQ codes → codebook lookup → channels-first <c>[1, semDim, T]</c>.</summary>
    private Tensor SemanticFeatures(IReadOnlyList<int> semanticTokens)
    {
        int t = Math.Max(1, semanticTokens.Count);
        int dim = (int)_semanticCodebook!.Shape[1];
        Tensor feat = new(new TensorShape(1, dim, t), DType.F32);
        float* fp = (float*)feat.DataPointer;
        float* cb = (float*)_semanticCodebook.DataPointer;
        int vocab = (int)_semanticCodebook.Shape[0];
        for (int i = 0; i < semanticTokens.Count; i++)
        {
            int code = semanticTokens[i];
            if ((uint)code >= (uint)vocab) code = 0;
            float* row = cb + (long)code * dim;
            for (int c = 0; c < dim; c++) fp[(long)c * t + i] = row[c];   // channels-first
        }
        return feat;
    }

    private void AddSpeakerBias(IBackend backend, Tensor x, Tensor dVector)
    {
        int baseCh = (int)x.Shape[1];
        int t = (int)x.Shape[2];
        Tensor dv3 = dVector.Reshape(new TensorShape(1, 1, _cfg.GlobalDim));
        Tensor bias = WhisperOps.ProjectLinear(backend, dv3, _condProjW!, _condProjB, 1, 1, _cfg.GlobalDim, baseCh);
        float* xp = (float*)x.DataPointer;
        float* bp = (float*)bias.DataPointer;
        for (int c = 0; c < baseCh; c++)
        {
            float add = bp[c];
            long off = (long)c * t;
            for (int j = 0; j < t; j++) xp[off + j] += add;
        }
        bias.Dispose();
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

    private static void AddInPlace(Tensor dst, Tensor src)
    {
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        long n = Math.Min(dst.ElementCount, src.ElementCount);
        for (long i = 0; i < n; i++) dp[i] += sp[i];
    }

    private static void Scale(Tensor x, float f)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) p[i] *= f;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
