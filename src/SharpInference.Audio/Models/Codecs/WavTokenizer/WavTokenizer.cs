using SharpInference.Audio.Models.Codecs.Dac;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Preprocessing;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Codecs.WavTokenizer;

/// <summary>WavTokenizer — single-codebook neural audio codec with iSTFT-based
/// decoder. The encoder is SEANet-style (reuses <see cref="Dac.DacResidualUnit"/>
/// directly via DAC types); the quantizer is a degenerate <see cref="DacResidualVectorQuantizer"/>
/// with <c>nCodebooks=1</c>; the decoder is a Vocos-style ConvNeXt + iSTFT head.
///
/// <para>State-dict roots:</para>
/// <list type="bullet">
///   <item><c>feature_extractor.encoder.*</c></item>
///   <item><c>feature_extractor.quantizer.quantizers.0.*</c></item>
///   <item><c>head.*</c></item>
/// </list></summary>
public sealed unsafe class WavTokenizer
{
    public WavTokenizerConfig Config { get; }
    public int LatentDim => Config.LatentDim;
    public int SampleRate => Config.SampleRate;
    public int FrameRate => Config.FrameRate;

    private readonly WavTokenizerEncoder _encoder;
    private readonly DacResidualVectorQuantizer _quantizer;
    private readonly WavTokenizerHead _head;

    public WavTokenizer(WavTokenizerConfig config)
    {
        Config = config;
        _encoder = new WavTokenizerEncoder(config, "feature_extractor.encoder");
        _quantizer = new DacResidualVectorQuantizer(
            nCodebooks: 1,
            codebookSize: config.CodebookSize,
            codebookDim: config.CodebookDim,
            latentDim: config.LatentDim);
        _head = new WavTokenizerHead(config, "head");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _encoder.LoadWeights(w);
        _quantizer.LoadWeights(w, "feature_extractor.quantizer");
        _head.LoadWeights(w);
    }

    public Tensor Encode(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        Tensor latent = _encoder.Forward(backend, pcm, batch, tPcm);
        int tFrames = (int)latent.Shape[2];
        Tensor codes = _quantizer.Encode(backend, latent, batch, tFrames);
        latent.Dispose();
        return codes;
    }

    public Tensor Decode(IBackend backend, Tensor codes, int batch, int tFrames)
    {
        Tensor latent = _quantizer.Decode(backend, codes, batch, tFrames);
        Tensor pcm = _head.Forward(backend, latent, batch, tFrames);
        latent.Dispose();
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _quantizer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _head.EnumerateWeights()) yield return t;
    }
}

/// <summary>WavTokenizer SEANet-style encoder. Same structure as DAC encoder but
/// projects to <see cref="WavTokenizerConfig.LatentDim"/> at the bottleneck.</summary>
internal sealed unsafe class WavTokenizerEncoder
{
    private readonly WavTokenizerConfig _cfg;
    private readonly string _prefix;
    private readonly int _nStages;
    private readonly int[] _strides;

    private Tensor? _stemW;
    private Tensor? _stemB;
    private readonly DacResidualUnit[][] _stageUnits;
    private readonly Tensor?[] _downsampleSnakeAlpha;
    private readonly Tensor?[] _downsampleW;
    private readonly Tensor?[] _downsampleB;
    private Tensor? _finalSnakeAlpha;
    private Tensor? _finalProjW;
    private Tensor? _finalProjB;

    public WavTokenizerEncoder(WavTokenizerConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _nStages = cfg.EncoderRates.Count;
        _strides = [.. cfg.EncoderRates];
        _stageUnits = new DacResidualUnit[_nStages][];
        _downsampleSnakeAlpha = new Tensor?[_nStages];
        _downsampleW = new Tensor?[_nStages];
        _downsampleB = new Tensor?[_nStages];

        int dim = cfg.EncoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            dim *= 2;
            int innerDim = dim / 2;
            _stageUnits[i] = new DacResidualUnit[cfg.ResidualDilations.Count];
            for (int j = 0; j < cfg.ResidualDilations.Count; j++)
            {
                _stageUnits[i][j] = new DacResidualUnit(
                    prefix: $"{prefix}.block.{i + 1}.block.{j}",
                    dim: innerDim,
                    kernel: cfg.ResidualKernelSize,
                    dilation: cfg.ResidualDilations[j]);
            }
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _stemW = LoadFusedWeight(w, $"{_prefix}.block.0");
        _stemB = WhisperOps.EnsureF32(w[$"{_prefix}.block.0.bias"]);

        int dim = _cfg.EncoderDim;
        for (int i = 0; i < _nStages; i++)
        {
            dim *= 2;
            int innerDim = dim / 2;
            for (int j = 0; j < _cfg.ResidualDilations.Count; j++) _stageUnits[i][j].LoadWeights(w);

            int snakeIdx = _cfg.ResidualDilations.Count;
            int convIdx = snakeIdx + 1;
            _downsampleSnakeAlpha[i] = WhisperOps.EnsureF32(w[$"{_prefix}.block.{i + 1}.block.{snakeIdx}.alpha"]).Reshape(new TensorShape(innerDim));
            _downsampleW[i] = LoadFusedWeight(w, $"{_prefix}.block.{i + 1}.block.{convIdx}");
            _downsampleB[i] = WhisperOps.EnsureF32(w[$"{_prefix}.block.{i + 1}.block.{convIdx}.bias"]);
        }

        int finalDim = _cfg.EncoderDim * (1 << _nStages);
        _finalSnakeAlpha = WhisperOps.EnsureF32(w[$"{_prefix}.block.{_nStages + 1}.alpha"]).Reshape(new TensorShape(finalDim));
        _finalProjW = LoadFusedWeight(w, $"{_prefix}.block.{_nStages + 2}");
        _finalProjB = WhisperOps.EnsureF32(w[$"{_prefix}.block.{_nStages + 2}.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor pcm, int batch, int tPcm)
    {
        int stemPad = _cfg.StemKernelSize / 2;
        int tStem = tPcm + 2 * stemPad - (_cfg.StemKernelSize - 1);
        Tensor x = new(new TensorShape(batch, _cfg.EncoderDim, tStem), DType.F32);
        backend.Conv1d(x, pcm, _stemW!, _stemB,
            stride: 1, padLeft: stemPad, padRight: stemPad, dilation: 1, groups: 1);

        int t = tStem;
        int dim = _cfg.EncoderDim;

        for (int i = 0; i < _nStages; i++)
        {
            int innerDim = dim;
            foreach (DacResidualUnit unit in _stageUnits[i])
            {
                Tensor next = unit.Forward(backend, x, batch, t);
                x.Dispose(); x = next; t = (int)x.Shape[2];
            }

            Tensor snk = new(x.Shape, DType.F32);
            backend.Snake(snk, x, _downsampleSnakeAlpha[i]!, null);
            x.Dispose(); x = snk;

            int stride = _strides[i];
            int kDown = 2 * stride;
            int padDown = (stride + 1) / 2;
            int dimOut = innerDim * 2;
            int tDown = (t + 2 * padDown - (kDown - 1) - 1) / stride + 1;
            Tensor down = new(new TensorShape(batch, dimOut, tDown), DType.F32);
            backend.Conv1d(down, x, _downsampleW[i]!, _downsampleB[i],
                stride: stride, padLeft: padDown, padRight: padDown, dilation: 1, groups: 1);
            x.Dispose(); x = down; t = tDown; dim = dimOut;
        }

        Tensor preProj = new(x.Shape, DType.F32);
        backend.Snake(preProj, x, _finalSnakeAlpha!, null);
        x.Dispose();

        int finalPad = _cfg.StemKernelSize / 2;
        int tFinal = t + 2 * finalPad - (_cfg.StemKernelSize - 1);
        Tensor latent = new(new TensorShape(batch, _cfg.LatentDim, tFinal), DType.F32);
        backend.Conv1d(latent, preProj, _finalProjW!, _finalProjB,
            stride: 1, padLeft: finalPad, padRight: finalPad, dilation: 1, groups: 1);
        preProj.Dispose();
        return latent;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_stemW is not null) yield return _stemW;
        if (_stemB is not null) yield return _stemB;
        for (int i = 0; i < _nStages; i++)
        {
            foreach (DacResidualUnit unit in _stageUnits[i])
                foreach (Tensor t in unit.EnumerateWeights()) yield return t;
            if (_downsampleSnakeAlpha[i] is not null) yield return _downsampleSnakeAlpha[i]!;
            if (_downsampleW[i] is not null) yield return _downsampleW[i]!;
            if (_downsampleB[i] is not null) yield return _downsampleB[i]!;
        }
        if (_finalSnakeAlpha is not null) yield return _finalSnakeAlpha;
        if (_finalProjW is not null) yield return _finalProjW;
        if (_finalProjB is not null) yield return _finalProjB;
    }

    private static Tensor LoadFusedWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }
}

/// <summary>WavTokenizer iSTFT head. ConvNeXt-style block stack + linear projection
/// to log-magnitude + phase + iSTFT overlap-add. Operates in frequency domain so
/// reconstruction is fast even at high sample rates.</summary>
internal sealed unsafe class WavTokenizerHead
{
    private readonly WavTokenizerConfig _cfg;
    private readonly string _prefix;
    private readonly int _nBins;

    // ConvNeXt blocks (Vocos-style): depthwise Conv1d + LayerNorm + Linear + GELU + Linear + γ
    private Tensor? _stemW;
    private Tensor? _stemB;
    private readonly Tensor?[] _blockDwW;
    private readonly Tensor?[] _blockDwB;
    private readonly Tensor?[] _blockNormW;
    private readonly Tensor?[] _blockNormB;
    private readonly Tensor?[] _blockFc1W;
    private readonly Tensor?[] _blockFc1B;
    private readonly Tensor?[] _blockFc2W;
    private readonly Tensor?[] _blockFc2B;
    private readonly Tensor?[] _blockGamma;
    private Tensor? _finalNormW;
    private Tensor? _finalNormB;
    private Tensor? _outProjW;       // [n_bins * 2, head_dim] (mag + phase)
    private Tensor? _outProjB;

    public WavTokenizerHead(WavTokenizerConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _nBins = cfg.NFft / 2 + 1;
        int n = cfg.HeadConvNeXtBlocks;
        _blockDwW = new Tensor?[n];
        _blockDwB = new Tensor?[n];
        _blockNormW = new Tensor?[n];
        _blockNormB = new Tensor?[n];
        _blockFc1W = new Tensor?[n];
        _blockFc1B = new Tensor?[n];
        _blockFc2W = new Tensor?[n];
        _blockFc2B = new Tensor?[n];
        _blockGamma = new Tensor?[n];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _stemW = WhisperOps.EnsureF32(w[$"{_prefix}.input_proj.weight"]);
        if (w.TryGetValue($"{_prefix}.input_proj.bias", out Tensor? sb)) _stemB = WhisperOps.EnsureF32(sb);

        for (int i = 0; i < _blockDwW.Length; i++)
        {
            string p = $"{_prefix}.convnext.{i}";
            _blockDwW[i] = WhisperOps.EnsureF32(w[$"{p}.dwconv.weight"]);
            if (w.TryGetValue($"{p}.dwconv.bias", out Tensor? db)) _blockDwB[i] = WhisperOps.EnsureF32(db);
            _blockNormW[i] = WhisperOps.EnsureF32(w[$"{p}.norm.weight"]);
            _blockNormB[i] = WhisperOps.EnsureF32(w[$"{p}.norm.bias"]);
            _blockFc1W[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv1.weight"]);
            _blockFc1B[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv1.bias"]);
            _blockFc2W[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv2.weight"]);
            _blockFc2B[i] = WhisperOps.EnsureF32(w[$"{p}.pwconv2.bias"]);
            _blockGamma[i] = WhisperOps.EnsureF32(w[$"{p}.gamma"]);
        }

        _finalNormW = WhisperOps.EnsureF32(w[$"{_prefix}.final_norm.weight"]);
        _finalNormB = WhisperOps.EnsureF32(w[$"{_prefix}.final_norm.bias"]);
        _outProjW = WhisperOps.EnsureF32(w[$"{_prefix}.out.weight"]);
        _outProjB = WhisperOps.EnsureF32(w[$"{_prefix}.out.bias"]);
    }

    /// <summary>Forward — <paramref name="latent"/> channels-first <c>[B, latent_dim, T_frames]</c>
    /// → PCM <c>[B, 1, T_pcm]</c> via spectrogram prediction + iSTFT.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tFrames)
    {
        // Project latent_dim → head_dim via 1×1 conv.
        Tensor x = new(new TensorShape(batch, _cfg.HeadDim, tFrames), DType.F32);
        backend.Conv1d(x, latent, _stemW!, _stemB,
            stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

        // ConvNeXt blocks (operating channels-last with periodic re-transpose).
        for (int i = 0; i < _blockDwW.Length; i++)
        {
            // Depthwise Conv1d (kernel=7, padding=3, groups=head_dim).
            int padDw = 3;
            int tDw = tFrames + 2 * padDw - 6;
            Tensor dw = new(new TensorShape(batch, _cfg.HeadDim, tDw), DType.F32);
            backend.Conv1d(dw, x, _blockDwW[i]!, _blockDwB[i],
                stride: 1, padLeft: padDw, padRight: padDw, dilation: 1, groups: _cfg.HeadDim);

            // Permute to channels-last [B, T, head_dim].
            Tensor cl = new(new TensorShape(batch, tDw, _cfg.HeadDim), DType.F32);
            backend.Transpose2D(cl, dw, _cfg.HeadDim, tDw);
            dw.Dispose();

            // LayerNorm.
            Tensor normed = new(cl.Shape, DType.F32);
            backend.LayerNorm(normed, cl, _blockNormW[i]!, _blockNormB[i]!, 1e-6f);
            cl.Dispose();

            // Linear up: head_dim → head_dim * head_ffn_ratio (typically 3x).
            int ffnDim = _cfg.HeadDim * _cfg.HeadFfnRatio;
            Tensor up = WhisperOps.ProjectLinear(backend, normed, _blockFc1W[i]!, _blockFc1B[i], batch, tDw, _cfg.HeadDim, ffnDim);
            normed.Dispose();

            // GELU.
            Tensor act = new(up.Shape, DType.F32);
            backend.Gelu(act, up);
            up.Dispose();

            // Linear down: ffn_dim → head_dim.
            Tensor down = WhisperOps.ProjectLinear(backend, act, _blockFc2W[i]!, _blockFc2B[i], batch, tDw, ffnDim, _cfg.HeadDim);
            act.Dispose();

            // Apply per-channel layer-scale gamma.
            float* dp = (float*)down.DataPointer;
            float* gp = (float*)_blockGamma[i]!.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < tDw; ti++)
                {
                    int rowBase = (b * tDw + ti) * _cfg.HeadDim;
                    for (int d = 0; d < _cfg.HeadDim; d++) dp[rowBase + d] *= gp[d];
                }
            }

            // Permute back to channels-first.
            Tensor cf = new(new TensorShape(batch, _cfg.HeadDim, tDw), DType.F32);
            backend.Transpose2D(cf, down, tDw, _cfg.HeadDim);
            down.Dispose();

            // Residual add (size-matched).
            Tensor result = new(x.Shape, DType.F32);
            if (tDw == tFrames)
            {
                backend.Add(result, x, cf);
            }
            else
            {
                int trim = (tDw - tFrames) / 2;
                float* sp = (float*)cf.DataPointer;
                float* xp = (float*)x.DataPointer;
                float* rp = (float*)result.DataPointer;
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < _cfg.HeadDim; c++)
                    {
                        int srcBase = (b * _cfg.HeadDim + c) * tDw + trim;
                        int xBase = (b * _cfg.HeadDim + c) * tFrames;
                        int dstBase = (b * _cfg.HeadDim + c) * tFrames;
                        for (int j = 0; j < tFrames; j++) rp[dstBase + j] = xp[xBase + j] + sp[srcBase + j];
                    }
            }
            cf.Dispose();
            x.Dispose();
            x = result;
        }

        // Final LayerNorm (channels-last).
        Tensor cl2 = new(new TensorShape(batch, tFrames, _cfg.HeadDim), DType.F32);
        backend.Transpose2D(cl2, x, _cfg.HeadDim, tFrames);
        x.Dispose();
        Tensor finalNormed = new(cl2.Shape, DType.F32);
        backend.LayerNorm(finalNormed, cl2, _finalNormW!, _finalNormB!, 1e-6f);
        cl2.Dispose();

        // Output projection to mag + phase: head_dim → n_bins * 2.
        Tensor magPhase = WhisperOps.ProjectLinear(backend, finalNormed, _outProjW!, _outProjB, batch, tFrames, _cfg.HeadDim, _nBins * 2);
        finalNormed.Dispose();

        // Decode to mag + phase float arrays, then iSTFT to PCM. For batch=1 path
        // we use the existing scalar iSTFT helper. Multi-batch loops over samples.
        Tensor pcm = ApplyIStft(magPhase, batch, tFrames);
        magPhase.Dispose();
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_stemW is not null) yield return _stemW;
        if (_stemB is not null) yield return _stemB;
        for (int i = 0; i < _blockDwW.Length; i++)
        {
            Tensor?[] all = [_blockDwW[i], _blockDwB[i], _blockNormW[i], _blockNormB[i],
                _blockFc1W[i], _blockFc1B[i], _blockFc2W[i], _blockFc2B[i], _blockGamma[i]];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
        if (_finalNormW is not null) yield return _finalNormW;
        if (_finalNormB is not null) yield return _finalNormB;
        if (_outProjW is not null) yield return _outProjW;
        if (_outProjB is not null) yield return _outProjB;
    }

    /// <summary>Splits the network output into (log-magnitude, phase), exponentiates the
    /// magnitude, builds a complex spectrogram, runs iSTFT with overlap-add. Per
    /// Vocos's convention: <c>mag = exp(network_out[:, :, 0:n_bins])</c> and
    /// <c>phase = network_out[:, :, n_bins:2*n_bins]</c> (already in radians).</summary>
    private Tensor ApplyIStft(Tensor magPhase, int batch, int tFrames)
    {
        int outLen = (tFrames - 1) * _cfg.HopLength;
        Tensor pcm = new(new TensorShape(batch, 1, outLen), DType.F32);
        float* dp = (float*)pcm.DataPointer;
        float* mp = (float*)magPhase.DataPointer;

        // iSTFT only supports batch processing per-sample (single-channel mono PCM).
        for (int b = 0; b < batch; b++)
        {
            float[] specReal = new float[tFrames * _nBins];
            float[] specImag = new float[tFrames * _nBins];
            for (int t = 0; t < tFrames; t++)
            {
                int srcBase = (b * tFrames + t) * (_nBins * 2);
                for (int k = 0; k < _nBins; k++)
                {
                    float mag = MathF.Exp(MathF.Min(mp[srcBase + k], 30f));     // numerical safety
                    float phase = mp[srcBase + _nBins + k];
                    specReal[t * _nBins + k] = mag * MathF.Cos(phase);
                    specImag[t * _nBins + k] = mag * MathF.Sin(phase);
                }
            }
            float[] sampleAudio = IStftRun(specReal, specImag, tFrames);
            int copy = Math.Min(sampleAudio.Length, outLen);
            for (int j = 0; j < copy; j++) dp[b * outLen + j] = sampleAudio[j];
            for (int j = copy; j < outLen; j++) dp[b * outLen + j] = 0f;
        }
        return pcm;
    }

    private float[] IStftRun(float[] specReal, float[] specImag, int frames)
    {
        return SharpInference.Audio.Models.Vocoders.IStft.Apply(specReal, specImag, frames, _cfg.NFft, _cfg.HopLength);
    }
}
