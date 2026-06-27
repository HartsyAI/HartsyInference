using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.SparkTts;

/// <summary>BiCodec decode side (Spark-TTS) — reconstructs a 16 kHz waveform from the LM's global +
/// semantic tokens, faithfully mirroring <c>BiCodec.detokenize</c>:
/// <code>
///   z_q      = quantizer.detokenize(semantic)        # codebook[idx] (8-D) → out_project WNConv1d(8→1024)
///   d_vector = speaker_encoder.detokenize(global)    # FSQ idx→6-D → project_out(6→128) → flatten 32·128 → project(→1024)
///   x        = prenet(z_q, d_vector)                 # linear_pre → 2× downsample VocosBackbone → AdaLN VocosBackbone(×12) → linear
///   x        = x + d_vector[..., None]
///   wav      = decoder(x)                            # DAC-style HiFi-GAN WaveGenerator
/// </code>
/// PostNet is training-only (not on the decode path). The wave generator is the descript-audio-codec
/// decoder, reused verbatim via <see cref="DacDecoder"/>; the PreNet (Vocos ConvNeXt backbone with
/// AdaLayerNorm conditioning) is Spark-specific. Semantic VQ is <b>factorized</b> (codebook is 8-D,
/// projected up), and the global d-vector flattens the 32 FSQ codes (not mean-pooling).</summary>
public sealed unsafe class BiCodecDecoder : IDisposable
{
    private readonly SparkBiCodecConfig _cfg;
    private readonly DacDecoder _waveGen;
    private int _disposed;

    // Semantic factorized VQ.
    private Tensor? _codebook;        // [codebook_size, codebook_dim]  (8-D)
    private Tensor? _vqOutW, _vqOutB; // out_project WNConv1d(codebook_dim → 1024, k=1) — composed weight [1024, 8, 1]

    // Global FSQ d-vector path.
    private Tensor? _fsqProjOutW, _fsqProjOutB;   // ResidualFSQ.project_out Linear(6 → latent_dim=128)
    private Tensor? _spkProjW, _spkProjB;         // SpeakerEncoder.project Linear(latent*token_num=4096 → 1024)

    // PreNet.
    private Tensor? _linPreW, _linPreB;           // Linear(1024 → vocos_dim=384)
    private VocosBackbone[] _downsample = [];     // 2 plain-LN backbones (2 ConvNeXt each), each preceded by SamplingBlock(×3)
    private VocosBackbone? _mainBackbone;         // 12-layer AdaLN backbone conditioned on the d-vector
    private Tensor? _linW, _linB;                 // Linear(vocos_dim=384 → 1024)

    public BiCodecDecoder(SparkBiCodecConfig cfg)
    {
        _cfg = cfg;
        DacConfig dac = new()
        {
            DecoderDim = cfg.WaveGenChannels,
            DecoderRates = cfg.UpsampleRates,
            DecoderKernelSizes = cfg.UpsampleKernelSizes,
            ResidualKernelSize = 7,
            StemKernelSize = 7,
            DecoderFinalKernelSize = 7,
            ResidualDilations = [1, 3, 9],
            Channels = 1,
            TransposeWeightNormDim0 = true,   // descript/Spark WNConvTranspose1d norms over dim 0 (C_in)
        };
        _waveGen = new DacDecoder(dac, "decoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";

        _codebook = WhisperOps.EnsureF32(w[$"{p}quantizer.codebook.weight"]);
        _vqOutW = WeightNorm.Compose(w, $"{p}quantizer.out_project");
        _vqOutB = WhisperOps.EnsureF32(w[$"{p}quantizer.out_project.bias"]);

        _fsqProjOutW = WhisperOps.EnsureF32(w[$"{p}speaker_encoder.quantizer.project_out.weight"]);
        _fsqProjOutB = WhisperOps.EnsureF32(w[$"{p}speaker_encoder.quantizer.project_out.bias"]);
        _spkProjW = WhisperOps.EnsureF32(w[$"{p}speaker_encoder.project.weight"]);
        _spkProjB = WhisperOps.EnsureF32(w[$"{p}speaker_encoder.project.bias"]);

        _linPreW = WhisperOps.EnsureF32(w[$"{p}prenet.linear_pre.weight"]);
        _linPreB = WhisperOps.EnsureF32(w[$"{p}prenet.linear_pre.bias"]);
        _downsample = new VocosBackbone[_cfg.DownsampleStages];
        for (int i = 0; i < _cfg.DownsampleStages; i++)
        {
            _downsample[i] = new VocosBackbone(_cfg.VocosDim, _cfg.VocosIntermediate, numLayers: 2, conditionDim: 0);
            _downsample[i].LoadWeights(w, $"{p}prenet.downsample.{i}.1");
        }
        _mainBackbone = new VocosBackbone(_cfg.VocosDim, _cfg.VocosIntermediate, _cfg.PrenetLayers, conditionDim: _cfg.GlobalDim);
        _mainBackbone.LoadWeights(w, $"{p}prenet.vocos_backbone");
        _linW = WhisperOps.EnsureF32(w[$"{p}prenet.linear.weight"]);
        _linB = WhisperOps.EnsureF32(w[$"{p}prenet.linear.bias"]);

        _waveGen.LoadWeights(ToDac(w, p));
    }

    /// <summary>Decodes global + semantic token streams to a 16 kHz waveform.</summary>
    public float[] Decode(IBackend backend, IReadOnlyList<int> globalTokens, IReadOnlyList<int> semanticTokens)
    {
        if (_codebook is null) throw new InvalidOperationException("BiCodecDecoder weights not loaded.");
        int tFrames = Math.Max(1, semanticTokens.Count);

        using Tensor zq = SemanticZq(backend, semanticTokens, tFrames);   // [1, 1024, T]
        using Tensor dVector = SpeakerDVector(backend, globalTokens);     // [1, GlobalDim]

        using Tensor preOut = Prenet(backend, zq, dVector, tFrames);      // [1, 1024, T]
        AddBroadcastChannel(preOut, dVector);                            // x + d_vector[..., None]

        using Tensor wav = _waveGen.Forward(backend, preOut, batch: 1, tFrames: tFrames);  // [1, 1, T*hop], tanh'd
        int n = (int)wav.Shape[2];
        float[] audio = new float[n];
        float* wp = (float*)wav.DataPointer;
        for (int i = 0; i < n; i++) audio[i] = wp[i];
        return audio;
    }

    /// <summary>Diagnostics — returns the decode-path intermediates (z_q, d-vector, prenet output before the
    /// d-vector add) for layer-by-layer parity checks. Caller disposes.</summary>
    public (Tensor Zq, Tensor DVec, Tensor PrenetOut) DecodeIntermediates(IBackend backend,
        IReadOnlyList<int> globalTokens, IReadOnlyList<int> semanticTokens)
    {
        int t = Math.Max(1, semanticTokens.Count);
        Tensor zq = SemanticZq(backend, semanticTokens, t);
        Tensor dVec = SpeakerDVector(backend, globalTokens);
        Tensor pre = Prenet(backend, zq, dVec, t);
        return (zq, dVec, pre);
    }

    // ── Semantic VQ: codebook[idx] (8-D) → out_project WNConv1d(8→1024, k=1). ──
    private Tensor SemanticZq(IBackend backend, IReadOnlyList<int> tokens, int t)
    {
        int cdim = (int)_codebook!.Shape[1];               // codebook_dim = 8
        int vocab = (int)_codebook.Shape[0];
        Tensor codes = new(new TensorShape(1, cdim, t), DType.F32);   // channels-first [1, 8, T]
        float* cp = (float*)codes.DataPointer;
        float* cb = (float*)_codebook.DataPointer;
        for (int i = 0; i < t; i++)
        {
            int code = i < tokens.Count ? tokens[i] : 0;
            if ((uint)code >= (uint)vocab) code = 0;
            float* row = cb + (long)code * cdim;
            for (int c = 0; c < cdim; c++) cp[(long)c * t + i] = row[c];
        }
        int outDim = (int)_vqOutW!.Shape[0];
        Tensor zq = new(new TensorShape(1, outDim, t), DType.F32);
        backend.Conv1d(zq, codes, _vqOutW!, _vqOutB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        codes.Dispose();
        return zq;
    }

    // ── Global FSQ → d-vector. Each token: idx → 6-D code (level k → (k - L/2)/(L/2)) →
    //    project_out(6→128); flatten the 32×128 codes → project(4096→1024). ──
    private Tensor SpeakerDVector(IBackend backend, IReadOnlyList<int> tokens)
    {
        int[] levels = _cfg.FsqLevels;
        int dim = levels.Length;                  // 6
        int tokenNum = _cfg.TokenNum;             // 32
        int latent = _cfg.LatentDim;              // 128

        // FSQ index → centered code per axis (basis = cumprod([1] + levels[:-1])).
        Tensor codes = new(new TensorShape(1, tokenNum, dim), DType.F32);
        float* cdp = (float*)codes.DataPointer;
        for (int n = 0; n < tokenNum; n++)
        {
            int idx = n < tokens.Count ? tokens[n] : 0;
            int basis = 1;
            for (int d = 0; d < dim; d++)
            {
                int L = levels[d];
                int level = (idx / basis) % L;
                int half = L / 2;
                cdp[(long)n * dim + d] = (level - half) / (float)half;
                basis *= L;
            }
        }
        // project_out: Linear(6 → latent) applied per token → [1, 32, 128] (token-major).
        Tensor proj = WhisperOps.ProjectLinear(backend, codes, _fsqProjOutW!, _fsqProjOutB, 1, tokenNum, dim, latent);
        codes.Dispose();
        // Reference flattens AFTER transpose to [128, 32] (channel-major): flat[c*token_num + n] = proj[n, c].
        Tensor flat = new(new TensorShape(1, 1, tokenNum * latent), DType.F32);
        float* pp = (float*)proj.DataPointer;
        float* fp = (float*)flat.DataPointer;
        for (int n = 0; n < tokenNum; n++)
            for (int c = 0; c < latent; c++) fp[(long)c * tokenNum + n] = pp[(long)n * latent + c];
        proj.Dispose();
        Tensor dVec = WhisperOps.ProjectLinear(backend, flat, _spkProjW!, _spkProjB, 1, 1, tokenNum * latent, _cfg.GlobalDim);
        flat.Dispose();
        return dVec.Reshape(new TensorShape(1, _cfg.GlobalDim));
    }

    // ── PreNet: linear_pre → 2× (SamplingBlock×3 + VocosBackbone) → AdaLN VocosBackbone(×12) → linear. ──
    private Tensor Prenet(IBackend backend, Tensor zq, Tensor dVector, int t)
    {
        int vd = _cfg.VocosDim;
        // linear_pre over channels-last: [1,1024,T] → [1,T,1024] → Linear(1024→384) → [1,T,384].
        Tensor zqTC = Transpose(zq, 1024, t);
        Tensor x = WhisperOps.ProjectLinear(backend, zqTC, _linPreW!, _linPreB, 1, t, (int)_linPreW!.Shape[1], vd); // [1,T,384]
        zqTC.Dispose();

        // 2 downsample stages. SamplingBlock(ratio=1) transposes [1,T,C]→[1,C,T] and scales ×3.
        foreach (VocosBackbone bb in _downsample)
        {
            Tensor cf = Transpose(x, t, vd);     // [1,384,T]
            x.Dispose();
            Scale(cf, 3f);
            Tensor outTc = bb.Run(backend, cf, t, null);  // → [1,T,384]
            cf.Dispose();
            x = outTc;
        }

        // Main AdaLN backbone (input channels-first).
        Tensor xCf = Transpose(x, t, vd);        // [1,384,T]
        x.Dispose();
        Tensor mainTc = _mainBackbone!.Run(backend, xCf, t, dVector);  // [1,T,384]
        xCf.Dispose();

        // linear(384→1024) → transpose → [1,1024,T].
        Tensor lin = WhisperOps.ProjectLinear(backend, mainTc, _linW!, _linB, 1, t, vd, (int)_linW!.Shape[0]);  // [1,T,1024]
        mainTc.Dispose();
        Tensor outCf = Transpose(lin, t, (int)_linW!.Shape[0]);  // [1,1024,T]
        lin.Dispose();
        return outCf;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_codebook, _vqOutW, _vqOutB, _fsqProjOutW, _fsqProjOutB, _spkProjW, _spkProjB,
            _linPreW, _linPreB, _linW, _linB];
        foreach (Tensor? x in own) if (x is not null) yield return x;
        foreach (VocosBackbone bb in _downsample) foreach (Tensor t in bb.EnumerateWeights()) yield return t;
        if (_mainBackbone is not null) foreach (Tensor t in _mainBackbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _waveGen.EnumerateWeights()) yield return t;
    }

    // Re-key the BiCodec "decoder.*" tensors into the bare names DacDecoder expects ("decoder.model.*" stay as-is).
    private static Dictionary<string, Tensor> ToDac(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        if (p.Length == 0) return w as Dictionary<string, Tensor> ?? new Dictionary<string, Tensor>(w);
        Dictionary<string, Tensor> outW = new(w.Count);
        foreach ((string k, Tensor v) in w)
            if (k.StartsWith($"{p}decoder.", StringComparison.Ordinal)) outW[k[p.Length..]] = v;
        return outW;
    }

    // x [1, C, T] += bias [1, C] broadcast over T.
    private static void AddBroadcastChannel(Tensor x, Tensor bias)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2];
        float* xp = (float*)x.DataPointer;
        float* bp = (float*)bias.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            float add = bp[ci];
            long off = (long)ci * t;
            for (int j = 0; j < t; j++) xp[off + j] += add;
        }
    }

    // [1, A, B] → [1, B, A] contiguous.
    internal static Tensor Transpose(Tensor x, int a, int b)
    {
        Tensor o = new(new TensorShape(1, b, a), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int i = 0; i < a; i++)
            for (int j = 0; j < b; j++)
                op[(long)j * a + i] = xp[(long)i * b + j];
        return o;
    }

    internal static void Scale(Tensor x, float f)
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
