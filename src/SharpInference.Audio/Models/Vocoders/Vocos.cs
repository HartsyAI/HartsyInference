using SharpInference.Audio.Models.Whisper;  // WhisperOps for EnsureF32 + ProjectLinear
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Vocoders;

/// <summary>Vocos vocoder. Takes a log-mel spectrogram and synthesizes a 24 kHz audio
/// waveform via 8 ConvNeXt blocks + an iSTFT head. Significantly faster than HiFi-GAN
/// (no transposed convolutions, no temporal upsampling at all — all the time-domain
/// expansion happens in the final iSTFT).
///
/// <para>Architecture (matches <c>gemelo-ai/vocos</c> v1.0 exactly):
/// <code>
/// mel [B, 100, T]
///   → embed: Conv1D(100→512, k=7, p=3)            [B, 512, T]
///   → transpose to [B, T, 512]
///   → LayerNorm(512, eps=1e-6)
///   → 8 × ConvNeXt block (channels-first dwconv with groups=512; channels-last MLP)
///   → final_layer_norm(512)
///   → head.out: Linear(512 → 1026)               [B, T, 1026]
///   → split: [magnitude_log (513), phase_raw (513)]
///   → magnitude = clip(exp(magnitude_log), 100)
///   → S = magnitude * (cos(phase_raw) + 1j sin(phase_raw))
///   → iSTFT(S, n_fft=1024, hop=256, center=True)
///   → audio [B, ~T*256]
/// </code></para>
///
/// <para><b>Per-op layout notes:</b>
/// <list type="bullet">
///   <item>The depthwise convolution in each ConvNeXt block has <c>groups=dim</c> which
///         <see cref="IBackend.Conv2D"/> doesn't directly support. We dispatch it via a
///         private channel-wise inline path that walks <c>[C]</c> independent 1D
///         filters — same effect, no kernel API expansion needed for one vocoder family.</item>
///   <item>LayerNorm applies to the last dim, so we transpose to channels-last around it
///         (matching the upstream code's <c>x.transpose(1, 2)</c> calls).</item>
///   <item>The "gamma" layer-scale is just an element-wise multiply by a learned <c>[dim]</c>
///         vector.</item>
/// </list></para></summary>
public sealed unsafe class Vocos : IDisposable
{
    private readonly VocosConfig _cfg;
    private readonly VocosConvNeXtBlock[] _blocks;

    // Stem
    private Tensor? _embedWeight;       // [dim, in_channels, 1, 7] (4-D reshape for Conv2D)
    private Tensor? _embedBias;
    private Tensor? _normWeight;        // LayerNorm gain
    private Tensor? _normBias;

    // Tail
    private Tensor? _finalNormWeight;
    private Tensor? _finalNormBias;
    private Tensor? _headWeight;        // [n_fft + 2, dim]
    private Tensor? _headBias;          // [n_fft + 2]

    private bool _loaded;
    private int _disposed;

    public VocosConfig Config => _cfg;

    public Vocos(VocosConfig cfg)
    {
        _cfg = cfg;
        _blocks = new VocosConvNeXtBlock[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) _blocks[i] = new VocosConvNeXtBlock(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        Tensor embedRaw = WhisperOps.EnsureF32(weights["backbone.embed.weight"]);
        _embedWeight = embedRaw.Reshape(new TensorShape(_cfg.HiddenDim, _cfg.InputChannels, 1, 7));
        _embedBias = WhisperOps.EnsureF32(weights["backbone.embed.bias"]);

        _normWeight = WhisperOps.EnsureF32(weights["backbone.norm.weight"]);
        _normBias = WhisperOps.EnsureF32(weights["backbone.norm.bias"]);

        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"backbone.convnext.{i}");

        _finalNormWeight = WhisperOps.EnsureF32(weights["backbone.final_layer_norm.weight"]);
        _finalNormBias = WhisperOps.EnsureF32(weights["backbone.final_layer_norm.bias"]);

        _headWeight = WhisperOps.EnsureF32(weights["head.out.weight"]);
        _headBias = WhisperOps.EnsureF32(weights["head.out.bias"]);
        _loaded = true;
    }

    /// <summary>Forward pass. <paramref name="mel"/> is <c>[B, 100, T]</c> log-mel spectrogram
    /// (natural-log scale, NOT Whisper's log10/normalized variant). Returns a float[] of
    /// length <c>(T - 1) * 256</c> audio samples at 24 kHz.</summary>
    public float[] Forward(IBackend backend, Tensor mel)
    {
        ThrowIfDisposed();
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before Forward.");
        if (mel.Shape.Rank != 3) throw new ArgumentException("mel must be 3-D [B, n_mels, T]");
        int batch = (int)mel.Shape[0];
        if (batch != 1) throw new NotSupportedException("Vocos.Forward supports batch=1");
        int t = (int)mel.Shape[2];
        int dim = _cfg.HiddenDim;
        int pad = _cfg.DwConvKernel / 2;

        // Embed: Conv1D(100→dim, k=7, p=3). Dispatch via Conv2D with H=1.
        Tensor mel4d = mel.Reshape(new TensorShape(1, _cfg.InputChannels, 1, t));
        Tensor embedded4d = new(new TensorShape(1, dim, 1, t), DType.F32);
        backend.Conv2D(embedded4d, mel4d, _embedWeight!, _embedBias, strideH: 1, strideW: 1, padH: 0, padW: pad);
        Tensor embeddedCF = embedded4d.Reshape(new TensorShape(1, dim, t));  // drop H=1

        // Transpose to channels-last [1, T, dim] for LayerNorm and ConvNeXt MLP path.
        Tensor x = new(new TensorShape(1, t, dim), DType.F32);
        backend.Transpose2D(x, embeddedCF, dim, t);
        embedded4d.Dispose();

        // backbone.norm: LayerNorm(dim, eps=1e-6) on the last dim.
        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, _normWeight!, _normBias!, _cfg.LayerNormEps);
        x.Dispose();
        x = normed;

        // 8 ConvNeXt blocks. Each block transposes channels-first ⇄ channels-last internally;
        // we keep the running tensor as channels-last between blocks (matches upstream order).
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, x, t, dim);
            x.Dispose();
            x = next;
        }

        // Final LayerNorm.
        Tensor finalNorm = new(x.Shape, DType.F32);
        backend.LayerNorm(finalNorm, x, _finalNormWeight!, _finalNormBias!, _cfg.LayerNormEps);
        x.Dispose();

        // head.out: Linear(dim → n_fft + 2). [1, T, dim] @ [n_fft+2, dim].T + bias → [1, T, n_fft+2].
        int outDim = _cfg.NFft + 2;
        Tensor headOut = WhisperOps.ProjectLinear(backend, finalNorm, _headWeight!, _headBias, 1, t, dim, outDim);
        finalNorm.Dispose();

        // Split + iSTFT.
        // Layout convention: upstream Vocos does `x.out(x).transpose(1, 2)` then chunk(2, dim=1).
        // After transpose: shape [1, n_fft+2, T]. chunk(2, dim=1) splits along channels into:
        //   mag = log-magnitude [1, n_fft/2+1, T]
        //   phase = raw phase   [1, n_fft/2+1, T]
        // Note: our headOut is [1, T, n_fft+2] (channels-last). The mathematically-equivalent
        // split is along the LAST dim into two halves. We then walk through with explicit
        // transposition during the spectral conversion step.
        int half = (_cfg.NFft / 2) + 1;
        float* hp = (float*)headOut.DataPointer;

        // Build real/imag spectrograms of shape [T, half] (frame-major, matching IStft.Apply contract).
        float[] specRe = new float[(long)t * half];
        float[] specIm = new float[(long)t * half];
        for (int frame = 0; frame < t; frame++)
        {
            int srcBase = frame * outDim;
            int dstBase = frame * half;
            for (int k = 0; k < half; k++)
            {
                float magLog = hp[srcBase + k];
                float phase = hp[srcBase + half + k];
                float mag = MathF.Exp(magLog);
                if (mag > 100f) mag = 100f;  // upstream safeguard
                specRe[dstBase + k] = mag * MathF.Cos(phase);
                specIm[dstBase + k] = mag * MathF.Sin(phase);
            }
        }
        headOut.Dispose();

        return IStft.Apply(specRe, specIm, t, _cfg.NFft, _cfg.HopLength);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        if (_embedBias is not null) yield return _embedBias;
        if (_normWeight is not null) yield return _normWeight;
        if (_normBias is not null) yield return _normBias;
        foreach (VocosConvNeXtBlock b in _blocks)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_finalNormWeight is not null) yield return _finalNormWeight;
        if (_finalNormBias is not null) yield return _finalNormBias;
        if (_headWeight is not null) yield return _headWeight;
        if (_headBias is not null) yield return _headBias;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(Vocos));
    }

    public void Dispose() { Interlocked.Exchange(ref _disposed, 1); }
}

/// <summary>Single Vocos ConvNeXt block. Channels-first depthwise Conv1D + channels-last
/// LayerNorm + 2-layer GELU MLP + learnable per-channel gamma + residual.</summary>
internal sealed unsafe class VocosConvNeXtBlock
{
    private readonly VocosConfig _cfg;

    private Tensor? _dwConvWeight;  // [dim, 1, k] — depthwise (groups=dim)
    private Tensor? _dwConvBias;
    private Tensor? _normWeight;
    private Tensor? _normBias;
    private Tensor? _pwConv1Weight; // [intermediate, dim] (Linear)
    private Tensor? _pwConv1Bias;
    private Tensor? _pwConv2Weight; // [dim, intermediate]
    private Tensor? _pwConv2Bias;
    private Tensor? _gamma;         // [dim]

    public VocosConvNeXtBlock(VocosConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _dwConvWeight = WhisperOps.EnsureF32(weights[$"{prefix}.dwconv.weight"]);
        _dwConvBias = WhisperOps.EnsureF32(weights[$"{prefix}.dwconv.bias"]);
        _normWeight = WhisperOps.EnsureF32(weights[$"{prefix}.norm.weight"]);
        _normBias = WhisperOps.EnsureF32(weights[$"{prefix}.norm.bias"]);
        _pwConv1Weight = WhisperOps.EnsureF32(weights[$"{prefix}.pwconv1.weight"]);
        _pwConv1Bias = WhisperOps.EnsureF32(weights[$"{prefix}.pwconv1.bias"]);
        _pwConv2Weight = WhisperOps.EnsureF32(weights[$"{prefix}.pwconv2.weight"]);
        _pwConv2Bias = WhisperOps.EnsureF32(weights[$"{prefix}.pwconv2.bias"]);
        _gamma = WhisperOps.EnsureF32(weights[$"{prefix}.gamma"]);
    }

    /// <summary>Forward: input <paramref name="x"/> is <c>[1, T, dim]</c> (channels-last).
    /// We compute the residual in channels-last as well — the only transposition happens
    /// inside the depthwise-conv path where we need channels-first <c>[1, dim, T]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t, int dim)
    {
        int kernel = _cfg.DwConvKernel;
        int pad = kernel / 2;

        // Transpose [1, T, dim] -> [1, dim, T] for the depthwise Conv1D.
        Tensor xCF = new(new TensorShape(1, dim, t), DType.F32);
        backend.Transpose2D(xCF, x, t, dim);

        // Depthwise Conv1D: each of `dim` channels is convolved with its own [1, k] kernel.
        // No groups support in IBackend.Conv2D, so we walk the channels inline. Output
        // shape matches input: [1, dim, T] (kernel=7, padding=3 keeps length).
        Tensor dwOut = new(xCF.Shape, DType.F32);
        DepthwiseConv1D(dwOut, xCF, _dwConvWeight!, _dwConvBias!, dim, t, kernel, pad);
        xCF.Dispose();

        // Transpose back to channels-last for LayerNorm + MLP.
        Tensor dwOutCL = new(new TensorShape(1, t, dim), DType.F32);
        backend.Transpose2D(dwOutCL, dwOut, dim, t);
        dwOut.Dispose();

        // LayerNorm(dim, eps=1e-6) on the last dim.
        Tensor normed = new(dwOutCL.Shape, DType.F32);
        backend.LayerNorm(normed, dwOutCL, _normWeight!, _normBias!, _cfg.LayerNormEps);
        dwOutCL.Dispose();

        // pwconv1: Linear(dim → intermediate)
        Tensor h1 = WhisperOps.ProjectLinear(backend, normed, _pwConv1Weight!, _pwConv1Bias, 1, t, dim, _cfg.IntermediateDim);
        normed.Dispose();
        Tensor act = new(h1.Shape, DType.F32);
        backend.Gelu(act, h1);
        h1.Dispose();

        // pwconv2: Linear(intermediate → dim)
        Tensor h2 = WhisperOps.ProjectLinear(backend, act, _pwConv2Weight!, _pwConv2Bias, 1, t, _cfg.IntermediateDim, dim);
        act.Dispose();

        // Layer-scale: multiply by per-channel gamma. Broadcast over [1, T, dim].
        ApplyLayerScale(h2, _gamma!, t, dim);

        // Residual add (still in channels-last).
        Tensor outX = new(x.Shape, DType.F32);
        backend.Add(outX, x, h2);
        h2.Dispose();
        return outX;
    }

    /// <summary>Channel-wise 1D conv: each of <c>channels</c> output channels reads from
    /// the same-indexed input channel only. Kernel is <c>[channels, 1, k]</c> — one filter
    /// of length k per channel.</summary>
    private static void DepthwiseConv1D(Tensor output, Tensor input, Tensor weight, Tensor bias, int channels, int t, int kernel, int pad)
    {
        float* outPtr = (float*)output.DataPointer;
        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;

        for (int c = 0; c < channels; c++)
        {
            int inBase = c * t;        // [1, c, :]
            int outBase = c * t;
            int wBase = c * kernel;    // [c, 0, :] flattened
            float b = bPtr[c];

            for (int j = 0; j < t; j++)
            {
                float acc = b;
                for (int k = 0; k < kernel; k++)
                {
                    int srcIdx = j + k - pad;
                    if ((uint)srcIdx < (uint)t) acc += inPtr[inBase + srcIdx] * wPtr[wBase + k];
                }
                outPtr[outBase + j] = acc;
            }
        }
    }

    /// <summary>Multiply every <c>[1, T, dim]</c> row by the per-channel gamma vector.</summary>
    private static void ApplyLayerScale(Tensor x, Tensor gamma, int t, int dim)
    {
        float* xp = (float*)x.DataPointer;
        float* gp = (float*)gamma.DataPointer;
        for (int s = 0; s < t; s++)
        {
            int off = s * dim;
            for (int d = 0; d < dim; d++) xp[off + d] *= gp[d];
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_dwConvWeight, _dwConvBias, _normWeight, _normBias,
                         _pwConv1Weight, _pwConv1Bias, _pwConv2Weight, _pwConv2Bias, _gamma];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
