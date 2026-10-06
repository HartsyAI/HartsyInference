using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>IndexTTS-1.5's <c>gpt.conditioning_encoder</c> (ESPnet <c>ConformerEncoder</c>, <c>input_layer="conv2d2"</c>): a single
/// 3×3/stride-2 Conv2d subsampling front end over the mel "image" (halving the time axis), a linear projection
/// back to <see cref="IndexTtsConformerConfig.OutputSize"/>, 6 <see cref="IndexTtsConformerLayer"/> blocks, and a
/// final LayerNorm.</summary>
/// <remarks>The subsampling conv and the position table are built host-side: both run once per reference
/// clip (not per generated token), so a device round trip would only add latency without changing output.</remarks>
internal sealed unsafe class IndexTtsConformerEncoder : IDisposable
{
    private readonly IndexTtsConformerConfig _cfg;
    private readonly IndexTtsConformerLayer[] _layers;
    private int _disposed;

    private Tensor? _subConvW, _subConvB;    // embed.conv.0: Conv2d [odim, 1, 3, 3]
    private Tensor? _subOutW, _subOutB;      // embed.out.0: Linear [odim, odim * freqOut]
    private Tensor? _afterNormW, _afterNormB;

    public IndexTtsConformerEncoder(IndexTtsConformerConfig cfg)
    {
        _cfg = cfg;
        _layers = new IndexTtsConformerLayer[cfg.NumBlocks];
        for (int i = 0; i < cfg.NumBlocks; i++) _layers[i] = new IndexTtsConformerLayer(cfg);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _subConvW = WhisperOps.EnsureF32(w[$"{prefix}.embed.conv.0.weight"]);
        _subConvB = WhisperOps.EnsureF32(w[$"{prefix}.embed.conv.0.bias"]);
        _subOutW = WhisperOps.EnsureF32(w[$"{prefix}.embed.out.0.weight"]);
        _subOutB = WhisperOps.EnsureF32(w[$"{prefix}.embed.out.0.bias"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.encoders.{i}");
        _afterNormW = WhisperOps.EnsureF32(w[$"{prefix}.after_norm.weight"]);
        _afterNormB = WhisperOps.EnsureF32(w[$"{prefix}.after_norm.bias"]);
    }

    /// <summary>Encodes a mel spectrogram <c>[1, T, nMels]</c> (channels-last, time-major) into conditioning features
    /// <c>[1, T', outputSize]</c> with <c>T' ≈ T/2</c> (<c>Conv2dSubsampling2</c>, valid padding).</summary>
    public Tensor Forward(IBackend backend, Tensor mel, int t)
    {
        if (_subConvW is null) throw new InvalidOperationException("IndexTtsConformerEncoder weights not loaded.");
        int nMels = _cfg.InputSize, odim = _cfg.OutputSize;

        (Tensor subsampled, int tOut) = Subsample(mel, t, nMels, odim);
        float scale = MathF.Sqrt(odim);
        float* sp = (float*)subsampled.DataPointer;
        long n = subsampled.ElementCount;
        for (long i = 0; i < n; i++) sp[i] *= scale;

        Tensor posEmb = BuildPos(tOut, odim);
        Tensor x = subsampled;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, x, posEmb, tOut);
            x.Dispose();
            x = next;
        }
        posEmb.Dispose();

        Tensor outT = new(x.Shape, DType.F32);
        backend.LayerNorm(outT, x, _afterNormW!, _afterNormB!, _cfg.LayerNormEps);
        x.Dispose();
        return outT;
    }

    /// <summary>Conv2dSubsampling2: unsqueeze to a single-channel "image" <c>[1,1,T,F]</c>, 3×3/stride-2 valid conv
    /// + ReLU → <c>[1,odim,T',F']</c>, transpose+flatten to <c>[1,T',odim·F']</c>, project to <c>[1,T',odim]</c>.
    /// Host loops: this runs once per reference clip, not per token.</summary>
    private (Tensor, int) Subsample(Tensor mel, int t, int nMels, int odim)
    {
        const int kernel = 3, stride = 2;
        int tOut = (t - kernel) / stride + 1;
        int fOut = (nMels - kernel) / stride + 1;
        if (tOut < 1 || fOut < 1) throw new ArgumentException($"Reference mel too short for conv2d subsampling: T={t}, nMels={nMels}.");

        float* mp = (float*)mel.DataPointer;         // [1, T, nMels] channels-last
        float* wp = (float*)_subConvW!.DataPointer;   // [odim, 1, 3, 3]
        float* bp = (float*)_subConvB!.DataPointer;
        Tensor conv = new(new TensorShape(1, odim, tOut, fOut), DType.F32);
        float* cp = (float*)conv.DataPointer;

        for (int oc = 0; oc < odim; oc++)
        {
            float* wocBase = wp + (long)oc * kernel * kernel;
            for (int ti = 0; ti < tOut; ti++)
            {
                int tBase = ti * stride;
                for (int fi = 0; fi < fOut; fi++)
                {
                    int fBase = fi * stride;
                    float acc = bp[oc];
                    for (int kt = 0; kt < kernel; kt++)
                    {
                        float* mRow = mp + (long)(tBase + kt) * nMels + fBase;
                        float* wRow = wocBase + kt * kernel;
                        for (int kf = 0; kf < kernel; kf++) acc += mRow[kf] * wRow[kf];
                    }
                    cp[((long)oc * tOut + ti) * fOut + fi] = MathF.Max(0f, acc);   // ReLU
                }
            }
        }

        // Transpose [1,odim,T',F'] -> [1,T',odim,F'] -> flatten to [1,T',odim*F'], then project to [1,T',odim].
        Tensor flat = new(new TensorShape(1, tOut, odim * fOut), DType.F32);
        float* fp = (float*)flat.DataPointer;
        for (int ti = 0; ti < tOut; ti++)
        {
            float* dst = fp + (long)ti * odim * fOut;
            for (int oc = 0; oc < odim; oc++)
            {
                float* src = cp + ((long)oc * tOut + ti) * fOut;
                for (int fi = 0; fi < fOut; fi++) dst[oc * fOut + fi] = src[fi];
            }
        }
        conv.Dispose();

        Tensor projected = new(new TensorShape(1, tOut, odim), DType.F32);
        float* pp = (float*)projected.DataPointer;
        float* projW = (float*)_subOutW!.DataPointer;   // [odim, odim*fOut]
        float* projB = (float*)_subOutB!.DataPointer;
        int inDim = odim * fOut;
        for (int ti = 0; ti < tOut; ti++)
        {
            float* row = fp + (long)ti * inDim;
            float* outRow = pp + (long)ti * odim;
            for (int oc = 0; oc < odim; oc++)
            {
                float acc = projB[oc];
                float* wRow = projW + (long)oc * inDim;
                for (int e = 0; e < inDim; e++) acc += row[e] * wRow[e];
                outRow[oc] = acc;
            }
        }
        flat.Dispose();
        return (projected, tOut);
    }

    /// <summary>The position table <c>[1, T, d]</c> the reference actually feeds its "relative" attention: IndexTTS's
    /// <c>RelPositionalEncoding</c> inherits <c>PositionalEncoding.__init__</c> (which ignores <c>reverse</c>), so
    /// <c>position_encoding(0, T)</c> is simply <c>pe[:, 0:T]</c> — the ordinary forward sinusoid for positions
    /// <c>0..T-1</c>, even dims sin / odd dims cos at <c>1/10000^(2k/d)</c> — and its attention has the
    /// <c>rel_shift</c> call commented out. The score's position term is therefore <c>(q_i + v)·linear_pos(pe[j])</c>
    /// for key <c>j</c>, NOT the Transformer-XL/ESPnet <c>2T-1</c> table indexed by <c>i-j</c> that CosyVoice's
    /// conformer uses; the released checkpoints were trained with this variant.</summary>
    private static Tensor BuildPos(int t, int d)
    {
        Tensor pe = new(new TensorShape(1, t, d), DType.F32);
        float* p = (float*)pe.DataPointer;
        int half = d / 2;
        for (int pos = 0; pos < t; pos++)
        {
            float* row = p + (long)pos * d;
            for (int k = 0; k < half; k++)
            {
                double inv = Math.Exp(-(2.0 * k / d) * Math.Log(10000.0));
                double angle = pos * inv;
                row[2 * k] = (float)Math.Sin(angle);
                row[2 * k + 1] = (float)Math.Cos(angle);
            }
        }
        return pe;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_subConvW, _subConvB, _subOutW, _subOutB, _afterNormW, _afterNormB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (IndexTtsConformerLayer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
