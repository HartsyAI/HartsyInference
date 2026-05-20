using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.F5Tts;

/// <summary>F5-TTS text embedding: token IDs → 512-dim text features.
///
/// <para>Pipeline:
/// <list type="number">
///   <item>Token embedding lookup (vocab+1 → text_dim, ID 0 reserved as filler).</item>
///   <item>Sinusoidal positional embedding added in-place (precomputed table of shape
///         <c>[max_pos, text_dim]</c> with format <c>cat([cos_half], [sin_half])</c>).</item>
///   <item>4× ConvNeXt-V2 blocks at <c>text_dim=512</c> with <c>intermediate=1024</c>,
///         <see cref="F5Ops.Grn"/> instead of layer-scale.</item>
/// </list></para>
///
/// <para>Per the upstream impl, padding positions are masked to zero AROUND each ConvNeXt
/// block — for our v1 zero-shot voice clone path the text input is always dense to the
/// padded length (every position is valid), so we skip masking. This is a simplification
/// that matches every reference inference invocation in the F5-TTS Python repo.</para></summary>
internal sealed unsafe class F5TextEmbedding
{
    private readonly F5TtsConfig _cfg;

    private Tensor? _embedWeight;       // [vocab+1, text_dim]
    private Tensor? _freqsCis;          // [max_pos, text_dim]  cos cat sin
    private readonly F5ConvNeXtV2Block[] _blocks;

    public F5TextEmbedding(F5TtsConfig cfg)
    {
        _cfg = cfg;
        _blocks = new F5ConvNeXtV2Block[cfg.TextConvLayers];
        for (int i = 0; i < cfg.TextConvLayers; i++)
            _blocks[i] = new F5ConvNeXtV2Block(cfg.TextDim, cfg.TextDim * cfg.TextConvMult);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _embedWeight = WhisperOps.EnsureF32(w[$"{prefix}.text_embed.weight"]);
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(w, $"{prefix}.text_blocks.{i}");

        // Precompute the sinusoidal freqs table (the persistent buffer is non-persistent
        // upstream, so it's not in the safetensors). Same formula as precompute_freqs_cis.
        _freqsCis = BuildFreqsCis(_cfg.TextDim, _cfg.TextMaxPos, theta: 10_000f);
    }

    /// <summary>Forward: token IDs <c>[1, T_text]</c>, target sequence length <c>seq_len</c>
    /// (= audio mel frame count). Returns text features <c>[1, seq_len, text_dim]</c> —
    /// the text is padded/truncated to <paramref name="seqLen"/> with the filler token (ID 0)
    /// for positions beyond the input text length.</summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> textIds, int seqLen, bool dropText = false)
    {
        if (_embedWeight is null || _freqsCis is null)
            throw new InvalidOperationException("Call LoadWeights first.");

        int textDim = _cfg.TextDim;
        Tensor textHidden = new(new TensorShape(1, seqLen, textDim), DType.F32);
        float* hp = (float*)textHidden.DataPointer;
        float* ep = (float*)_embedWeight.DataPointer;

        // 1. Token embed lookup. Upstream adds +1 to every token id (use 0 as filler).
        //    Positions beyond textIds.Length stay 0.
        int validLen = Math.Min(textIds.Length, seqLen);
        for (int t = 0; t < seqLen; t++)
        {
            int rowOff = t * textDim;
            int id = (t < validLen) ? textIds[t] + 1 : 0;
            if (dropText) id = 0;
            if (id == 0)
            {
                for (int d = 0; d < textDim; d++) hp[rowOff + d] = 0f;
            }
            else
            {
                int eOff = id * textDim;
                for (int d = 0; d < textDim; d++) hp[rowOff + d] = ep[eOff + d];
            }
        }

        // 2. Add sinusoidal positional embedding from the precomputed freqs table.
        float* fp = (float*)_freqsCis.DataPointer;
        for (int t = 0; t < seqLen; t++)
        {
            int rowOff = t * textDim;
            int freqOff = t * textDim;
            for (int d = 0; d < textDim; d++) hp[rowOff + d] += fp[freqOff + d];
        }

        // 3. 4 ConvNeXt-V2 blocks at text_dim=512.
        bool dump = Environment.GetEnvironmentVariable("F5_DUMP") == "1";
        Tensor running = textHidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            if (dump) Console.Error.WriteLine($"    text block {i}");
            Tensor next = _blocks[i].Forward(backend, running, seqLen, textDim);
            running.Dispose();
            running = next;
        }

        return running;
    }

    /// <summary>Computes the same sinusoidal table that <c>precompute_freqs_cis</c> returns
    /// in the upstream Python code: <c>[end, dim]</c> with the first half cosines and the
    /// second half sines, all using <c>theta=10000</c> per the F5-TTS default.</summary>
    private static Tensor BuildFreqsCis(int dim, int end, float theta)
    {
        int half = dim / 2;
        Tensor t = new(new TensorShape(end, dim), DType.F32);
        float* p = (float*)t.DataPointer;

        double[] invFreq = new double[half];
        for (int i = 0; i < half; i++)
            invFreq[i] = 1.0 / Math.Pow(theta, (double)(2 * i) / dim);

        for (int pos = 0; pos < end; pos++)
        {
            int rowOff = pos * dim;
            for (int i = 0; i < half; i++)
            {
                double angle = pos * invFreq[i];
                p[rowOff + i] = (float)Math.Cos(angle);
                p[rowOff + half + i] = (float)Math.Sin(angle);
            }
        }
        return t;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embedWeight is not null) yield return _embedWeight;
        foreach (F5ConvNeXtV2Block b in _blocks)
            foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }
}

/// <summary>F5-TTS ConvNeXt-V2 block. Same shape as the original ConvNeXt block but
/// with GRN (Global Response Normalization) in place of layer scale + no skip mod.
/// All operations channels-last <c>[B, T, dim]</c> except the depthwise conv which
/// transposes to channels-first <c>[B, dim, T]</c>.</summary>
internal sealed unsafe class F5ConvNeXtV2Block
{
    private readonly int _dim;
    private readonly int _intermediate;

    private Tensor? _dwConvW, _dwConvB;
    private Tensor? _normW, _normB;
    private Tensor? _pwConv1W, _pwConv1B;
    private Tensor? _pwConv2W, _pwConv2B;
    private Tensor? _grnGamma, _grnBeta;

    public F5ConvNeXtV2Block(int dim, int intermediate)
    {
        _dim = dim;
        _intermediate = intermediate;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _dwConvW = WhisperOps.EnsureF32(w[$"{prefix}.dwconv.weight"]);
        _dwConvB = WhisperOps.EnsureF32(w[$"{prefix}.dwconv.bias"]);
        _normW = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        _normB = WhisperOps.EnsureF32(w[$"{prefix}.norm.bias"]);
        _pwConv1W = WhisperOps.EnsureF32(w[$"{prefix}.pwconv1.weight"]);
        _pwConv1B = WhisperOps.EnsureF32(w[$"{prefix}.pwconv1.bias"]);
        _pwConv2W = WhisperOps.EnsureF32(w[$"{prefix}.pwconv2.weight"]);
        _pwConv2B = WhisperOps.EnsureF32(w[$"{prefix}.pwconv2.bias"]);
        _grnGamma = WhisperOps.EnsureF32(w[$"{prefix}.grn.gamma"]);
        _grnBeta = WhisperOps.EnsureF32(w[$"{prefix}.grn.beta"]);
    }

    /// <summary>Forward: x in channels-last <c>[1, T, dim]</c>. Returns same shape.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int t, int dim)
    {
        const int kernel = 7;
        int pad = kernel / 2;

        // Transpose to channels-first [1, dim, T] for the depthwise Conv1D.
        Tensor xCF = new(new TensorShape(1, dim, t), DType.F32);
        backend.Transpose2D(xCF, x, t, dim);

        // Depthwise Conv1D (groups=dim, each channel has its own [1, kernel] filter).
        Tensor dwOut = new(xCF.Shape, DType.F32);
        DepthwiseConv1D(dwOut, xCF, _dwConvW!, _dwConvB!, dim, t, kernel, pad);
        xCF.Dispose();

        // Back to channels-last [1, T, dim] for LayerNorm + MLP path.
        Tensor dwCL = new(new TensorShape(1, t, dim), DType.F32);
        backend.Transpose2D(dwCL, dwOut, dim, t);
        dwOut.Dispose();

        // LayerNorm (eps=1e-6).
        Tensor normed = new(dwCL.Shape, DType.F32);
        backend.LayerNorm(normed, dwCL, _normW!, _normB!, 1e-6f);
        dwCL.Dispose();

        // pwconv1 (dim → intermediate)
        Tensor h1 = WhisperOps.ProjectLinear(backend, normed, _pwConv1W!, _pwConv1B, 1, t, dim, _intermediate);
        normed.Dispose();

        // GELU activation.
        Tensor act = new(h1.Shape, DType.F32);
        backend.Gelu(act, h1);
        h1.Dispose();

        // GRN (Global Response Normalization) — operates in place on [1, T, intermediate].
        F5Ops.Grn(act, _grnGamma!, _grnBeta!, 1, t, _intermediate);

        // pwconv2 (intermediate → dim)
        Tensor h2 = WhisperOps.ProjectLinear(backend, act, _pwConv2W!, _pwConv2B, 1, t, _intermediate, dim);
        act.Dispose();

        // Residual add (channels-last).
        Tensor outX = new(x.Shape, DType.F32);
        backend.Add(outX, x, h2);
        h2.Dispose();
        return outX;
    }

    /// <summary>Depthwise Conv1D (groups=channels). Each output channel reads only from
    /// the same-indexed input channel; weight shape is <c>[channels, 1, kernel]</c>.</summary>
    private static void DepthwiseConv1D(Tensor output, Tensor input, Tensor weight, Tensor bias, int channels, int t, int kernel, int pad)
    {
        float* o = (float*)output.DataPointer;
        float* ip = (float*)input.DataPointer;
        float* w = (float*)weight.DataPointer;
        float* b = (float*)bias.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            int inBase = c * t;
            int outBase = c * t;
            int wBase = c * kernel;
            float bv = b[c];
            for (int j = 0; j < t; j++)
            {
                float acc = bv;
                for (int k = 0; k < kernel; k++)
                {
                    int src = j + k - pad;
                    if ((uint)src < (uint)t) acc += ip[inBase + src] * w[wBase + k];
                }
                o[outBase + j] = acc;
            }
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_dwConvW, _dwConvB, _normW, _normB, _pwConv1W, _pwConv1B, _pwConv2W, _pwConv2B, _grnGamma, _grnBeta];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
