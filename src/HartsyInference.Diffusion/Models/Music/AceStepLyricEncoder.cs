using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>ACE-Step's Conformer lyric encoder (wenet lineage, <c>lyrics_utils/lyric_encoder.py</c>): linear input
/// embed (+ LayerNorm) with ESPnet relative positional encoding, then N pre-norm Conformer layers — optional macaron
/// FFN (×0.5), <b>relative-position</b> multi-head self-attention (Transformer-XL style with <c>pos_bias_u/v</c> and
/// a learned position projection), a convolution module (pointwise→GLU→depthwise→BatchNorm→SiLU→pointwise), the main
/// FFN, and a final LayerNorm — closing with <c>after_norm</c>. Consumes lyric embeddings <c>[T, hidden]</c>
/// (the embedding table lives on the DiT). BatchNorm running stats are fused at load. First Conformer in the
/// codebase — reusable for future ASR work. Numerics validation-pending.</summary>
public sealed unsafe class AceStepLyricEncoder
{
    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _numLayers;

    private Tensor? _embedW, _embedB, _embedNormW, _embedNormB;
    private Layer[] _layers = [];
    private Tensor? _afterNormW, _afterNormB;

    public AceStepLyricEncoder(int dim = 1024, int heads = 16, int numLayers = 8)
    {
        _dim = dim;
        _heads = heads;
        _headDim = dim / heads;
        _numLayers = numLayers;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _embedW = w[$"{prefix}.embed.out.0.weight"]; w.TryGetValue($"{prefix}.embed.out.0.bias", out _embedB);
        _embedNormW = GetF32(w, $"{prefix}.embed.out.1.weight"); w.TryGetValue($"{prefix}.embed.out.1.bias", out _embedNormB);
        _layers = new Layer[_numLayers];
        for (int i = 0; i < _numLayers; i++)
        {
            _layers[i] = new Layer(this);
            _layers[i].LoadWeights(w, $"{prefix}.encoders.{i}");
        }
        _afterNormW = GetF32(w, $"{prefix}.after_norm.weight"); w.TryGetValue($"{prefix}.after_norm.bias", out _afterNormB);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _embedW, _embedB, _embedNormW, _embedNormB, _afterNormW, _afterNormB })
            if (t is not null) yield return t;
        foreach (Layer l in _layers)
            foreach (Tensor t in l.EnumerateWeights()) yield return t;
    }

    /// <summary>Encodes lyric embeddings <c>[T, dim]</c> → contextual states <c>[T, dim]</c> (full attention).</summary>
    public Tensor Forward(IBackend backend, Tensor lyricEmbeds)
    {
        int t = (int)lyricEmbeds.Shape[0];
        Tensor x = new Tensor(new TensorShape(t, _dim), DType.F32);
        backend.Linear(x, lyricEmbeds, _embedW!, _embedB);
        LayerNormRows(x, _embedNormW!, _embedNormB, t, _dim);
        Scale(x, MathF.Sqrt(_dim));   // xscale convention of the ESPnet positional encoding

        Tensor posProjBase = BuildRelPositions(t);   // raw sinusoid [2T−1, dim]
        foreach (Layer layer in _layers)
        {
            Tensor next = layer.Forward(backend, x, posProjBase, t);
            x.Dispose();
            x = next;
        }
        posProjBase.Dispose();
        LayerNormRows(x, _afterNormW!, _afterNormB, t, _dim);
        return x;
    }

    /// <summary>ESPnet relative sinusoidal table for positions <c>T−1 … −(T−1)</c> (index 0 = most positive).</summary>
    private Tensor BuildRelPositions(int t)
    {
        int len = 2 * t - 1;
        Tensor pe = new Tensor(new TensorShape(len, _dim), DType.F32);
        float* pp = (float*)pe.DataPointer;
        for (int i = 0; i < len; i++)
        {
            int pos = t - 1 - i;
            for (int d = 0; d < _dim / 2; d++)
            {
                double freq = Math.Exp(-(2.0 * d / _dim) * Math.Log(10000.0));
                pp[(long)i * _dim + 2 * d] = (float)Math.Sin(pos * freq);
                pp[(long)i * _dim + 2 * d + 1] = (float)Math.Cos(pos * freq);
            }
        }
        return pe;
    }

    internal static void LayerNormRows(Tensor x, Tensor weight, Tensor? bias, int rows, int dim, float eps = 1e-5f)
    {
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            long off = (long)r * dim;
            double mean = 0;
            for (int d = 0; d < dim; d++) mean += xp[off + d];
            mean /= dim;
            double var = 0;
            for (int d = 0; d < dim; d++) { double dd = xp[off + d] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / dim) + eps);
            for (int d = 0; d < dim; d++)
            {
                float n = (float)((xp[off + d] - mean) * inv);
                xp[off + d] = n * wp[d] + (bp is null ? 0f : bp[d]);
            }
        }
    }

    private static void Scale(Tensor x, float s)
    {
        float* p = (float*)x.DataPointer;
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] *= s;
    }

    private static Tensor? GetF32(IReadOnlyDictionary<string, Tensor> w, string k) =>
        w.TryGetValue(k, out Tensor? t) ? (t.DType == DType.F32 ? t : t.CastTo(DType.F32)) : null;

    private sealed class Layer
    {
        private readonly AceStepLyricEncoder _enc;
        private Tensor? _ffMacaron1W, _ffMacaron1B, _ffMacaron2W, _ffMacaron2B, _normFfMacaronW, _normFfMacaronB;
        private Tensor? _normMhaW, _normMhaB;
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _outW, _outB, _posW, _biasU, _biasV;
        private Tensor? _normConvW, _normConvB;
        private Tensor? _pw1W, _pw1B, _dwW, _dwB, _pw2W, _pw2B;
        private float[]? _bnScale, _bnShift;       // fused BatchNorm (or LayerNorm fallback)
        private Tensor? _convLnW, _convLnB;
        private Tensor? _normFfW, _normFfB, _ff1W, _ff1B, _ff2W, _ff2B;
        private Tensor? _normFinalW, _normFinalB;

        public Layer(AceStepLyricEncoder enc) => _enc = enc;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            if (w.ContainsKey($"{p}.feed_forward_macaron.w_1.weight"))
            {
                _ffMacaron1W = w[$"{p}.feed_forward_macaron.w_1.weight"]; w.TryGetValue($"{p}.feed_forward_macaron.w_1.bias", out _ffMacaron1B);
                _ffMacaron2W = w[$"{p}.feed_forward_macaron.w_2.weight"]; w.TryGetValue($"{p}.feed_forward_macaron.w_2.bias", out _ffMacaron2B);
                _normFfMacaronW = GetF32(w, $"{p}.norm_ff_macaron.weight"); w.TryGetValue($"{p}.norm_ff_macaron.bias", out _normFfMacaronB);
            }
            _normMhaW = GetF32(w, $"{p}.norm_mha.weight"); w.TryGetValue($"{p}.norm_mha.bias", out _normMhaB);
            _qW = w[$"{p}.self_attn.linear_q.weight"]; w.TryGetValue($"{p}.self_attn.linear_q.bias", out _qB);
            _kW = w[$"{p}.self_attn.linear_k.weight"]; w.TryGetValue($"{p}.self_attn.linear_k.bias", out _kB);
            _vW = w[$"{p}.self_attn.linear_v.weight"]; w.TryGetValue($"{p}.self_attn.linear_v.bias", out _vB);
            _outW = w[$"{p}.self_attn.linear_out.weight"]; w.TryGetValue($"{p}.self_attn.linear_out.bias", out _outB);
            _posW = w[$"{p}.self_attn.linear_pos.weight"];
            _biasU = GetF32(w, $"{p}.self_attn.pos_bias_u");
            _biasV = GetF32(w, $"{p}.self_attn.pos_bias_v");

            if (w.ContainsKey($"{p}.conv_module.pointwise_conv1.weight"))
            {
                _normConvW = GetF32(w, $"{p}.norm_conv.weight"); w.TryGetValue($"{p}.norm_conv.bias", out _normConvB);
                _pw1W = w[$"{p}.conv_module.pointwise_conv1.weight"]; w.TryGetValue($"{p}.conv_module.pointwise_conv1.bias", out _pw1B);
                _dwW = w[$"{p}.conv_module.depthwise_conv.weight"]; w.TryGetValue($"{p}.conv_module.depthwise_conv.bias", out _dwB);
                _pw2W = w[$"{p}.conv_module.pointwise_conv2.weight"]; w.TryGetValue($"{p}.conv_module.pointwise_conv2.bias", out _pw2B);
                FuseConvNorm(w, p);
                _normFinalW = GetF32(w, $"{p}.norm_final.weight"); w.TryGetValue($"{p}.norm_final.bias", out _normFinalB);
            }

            _normFfW = GetF32(w, $"{p}.norm_ff.weight"); w.TryGetValue($"{p}.norm_ff.bias", out _normFfB);
            _ff1W = w[$"{p}.feed_forward.w_1.weight"]; w.TryGetValue($"{p}.feed_forward.w_1.bias", out _ff1B);
            _ff2W = w[$"{p}.feed_forward.w_2.weight"]; w.TryGetValue($"{p}.feed_forward.w_2.bias", out _ff2B);
        }

        /// <summary>BatchNorm at inference is an affine transform — fuse running stats; LayerNorm variant kept as-is.</summary>
        private void FuseConvNorm(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            if (w.TryGetValue($"{p}.conv_module.norm.running_mean", out Tensor? meanT))
            {
                Tensor varT = w[$"{p}.conv_module.norm.running_var"];
                Tensor? gw = GetF32(w, $"{p}.conv_module.norm.weight");
                w.TryGetValue($"{p}.conv_module.norm.bias", out Tensor? gb);
                int c = (int)meanT.Shape[0];
                _bnScale = new float[c];
                _bnShift = new float[c];
                float* mp = (float*)meanT.CastTo(DType.F32).DataPointer;
                float* vp = (float*)varT.CastTo(DType.F32).DataPointer;
                float* gwp = gw is null ? null : (float*)gw.DataPointer;
                float* gbp = gb is null ? null : (float*)gb.DataPointer;
                for (int i = 0; i < c; i++)
                {
                    float scale = (gwp is null ? 1f : gwp[i]) / MathF.Sqrt(vp[i] + 1e-5f);
                    _bnScale[i] = scale;
                    _bnShift[i] = (gbp is null ? 0f : gbp[i]) - mp[i] * scale;
                }
            }
            else if (w.TryGetValue($"{p}.conv_module.norm.weight", out Tensor? lnW))
            {
                _convLnW = lnW.DType == DType.F32 ? lnW : lnW.CastTo(DType.F32);
                w.TryGetValue($"{p}.conv_module.norm.bias", out _convLnB);
            }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _ffMacaron1W, _ffMacaron1B, _ffMacaron2W, _ffMacaron2B, _normFfMacaronW, _normFfMacaronB,
                _normMhaW, _normMhaB, _qW, _qB, _kW, _kB, _vW, _vB, _outW, _outB, _posW, _biasU, _biasV,
                _normConvW, _normConvB, _pw1W, _pw1B, _dwW, _dwB, _pw2W, _pw2B, _convLnW, _convLnB,
                _normFfW, _normFfB, _ff1W, _ff1B, _ff2W, _ff2B, _normFinalW, _normFinalB })
                if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor relPos, int t)
        {
            int dim = _enc._dim;
            Tensor cur = Clone(x);

            if (_ffMacaron1W is not null)
                FfnResidual(backend, cur, _normFfMacaronW!, _normFfMacaronB, _ffMacaron1W, _ffMacaron1B, _ffMacaron2W!, _ffMacaron2B, t, 0.5f);

            // Relative-position MHSA.
            Tensor n = Clone(cur);
            LayerNormRows(n, _normMhaW!, _normMhaB, t, dim);
            Tensor attn = RelPosAttention(backend, n, relPos, t);
            n.Dispose();
            AddInPlace(cur, attn, 1f);
            attn.Dispose();

            if (_pw1W is not null)
            {
                Tensor nc = Clone(cur);
                LayerNormRows(nc, _normConvW!, _normConvB, t, dim);
                Tensor conv = ConvModule(backend, nc, t);
                nc.Dispose();
                AddInPlace(cur, conv, 1f);
                conv.Dispose();
            }

            FfnResidual(backend, cur, _normFfW!, _normFfB, _ff1W!, _ff1B, _ff2W!, _ff2B, t,
                _ffMacaron1W is not null ? 0.5f : 1f);

            if (_normFinalW is not null)
                LayerNormRows(cur, _normFinalW, _normFinalB, t, dim);
            return cur;
        }

        private void FfnResidual(IBackend backend, Tensor cur, Tensor normW, Tensor? normB,
            Tensor w1, Tensor? b1, Tensor w2, Tensor? b2, int t, float scale)
        {
            int dim = _enc._dim, hidden = (int)w1.Shape[0];
            Tensor n = Clone(cur);
            LayerNormRows(n, normW, normB, t, dim);
            Tensor h1 = new Tensor(new TensorShape(t, hidden), DType.F32);
            backend.Linear(h1, n, w1, b1);
            n.Dispose();
            backend.Silu(h1, h1);   // wenet PositionwiseFeedForward default activation
            Tensor h2 = new Tensor(new TensorShape(t, dim), DType.F32);
            backend.Linear(h2, h1, w2, b2);
            h1.Dispose();
            AddInPlace(cur, h2, scale);
            h2.Dispose();
        }

        /// <summary>Transformer-XL relative attention: <c>score = ((q+u)·kᵀ + (q+v)·pᵀ_rel)/√d</c>, where the p term
        /// is indexed directly by relative distance (no shift trick needed).</summary>
        private Tensor RelPosAttention(IBackend backend, Tensor n, Tensor relPos, int t)
        {
            int dim = _enc._dim, heads = _enc._heads, hd = _enc._headDim;
            Tensor q = Proj(backend, n, _qW!, _qB, t, dim);
            Tensor k = Proj(backend, n, _kW!, _kB, t, dim);
            Tensor v = Proj(backend, n, _vW!, _vB, t, dim);
            int posLen = (int)relPos.Shape[0];
            Tensor p = Proj(backend, relPos, _posW!, null, posLen, dim);

            Tensor outT = new Tensor(new TensorShape(t, dim), DType.F32);
            float* qp = (float*)q.DataPointer;
            float* kp = (float*)k.DataPointer;
            float* vp = (float*)v.DataPointer;
            float* pp = (float*)p.DataPointer;
            float* up = (float*)_biasU!.DataPointer;
            float* vbp = (float*)_biasV!.DataPointer;
            float* op = (float*)outT.DataPointer;
            float invSqrt = 1f / MathF.Sqrt(hd);
            float[] scores = new float[t];

            for (int h = 0; h < heads; h++)
            {
                int hOff = h * hd;
                for (int i = 0; i < t; i++)
                {
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < t; j++)
                    {
                        // relative index for distance (i−j): row (t−1) − (i−j) in the T−1…−(T−1) table.
                        int rel = t - 1 - i + j;
                        double ac = 0, bd = 0;
                        for (int d = 0; d < hd; d++)
                        {
                            float qd = qp[(long)i * dim + hOff + d];
                            ac += (qd + up[hOff + d]) * kp[(long)j * dim + hOff + d];
                            bd += (qd + vbp[hOff + d]) * pp[(long)rel * dim + hOff + d];
                        }
                        float s = (float)(ac + bd) * invSqrt;
                        scores[j] = s;
                        if (s > max) max = s;
                    }
                    double denom = 0;
                    for (int j = 0; j < t; j++) { scores[j] = MathF.Exp(scores[j] - max); denom += scores[j]; }
                    float inv = (float)(1.0 / denom);
                    for (int d = 0; d < hd; d++)
                    {
                        double acc = 0;
                        for (int j = 0; j < t; j++) acc += scores[j] * vp[(long)j * dim + hOff + d];
                        op[(long)i * dim + hOff + d] = (float)(acc * inv);
                    }
                }
            }
            q.Dispose(); k.Dispose(); v.Dispose(); p.Dispose();

            Tensor projected = new Tensor(new TensorShape(t, dim), DType.F32);
            backend.Linear(projected, outT, _outW!, _outB);
            outT.Dispose();
            return projected;
        }

        /// <summary>Conformer conv module: pointwise ×2 → GLU → depthwise → fused norm → SiLU → pointwise.</summary>
        private Tensor ConvModule(IBackend backend, Tensor n, int t)
        {
            int dim = _enc._dim;
            // Token rows → channel-first [1, dim, T].
            Tensor cf = RowsToChannels(n, dim, t);
            Tensor pw1 = new Tensor(new TensorShape(1, 2 * dim, t), DType.F32);
            backend.Conv1d(pw1, cf, _pw1W!, _pw1B, 1, 0, 0, 1, 1);
            cf.Dispose();

            // GLU: first half · sigmoid(second half).
            Tensor glu = new Tensor(new TensorShape(1, dim, t), DType.F32);
            float* gp = (float*)glu.DataPointer;
            float* pp = (float*)pw1.DataPointer;
            for (int c = 0; c < dim; c++)
                for (int i = 0; i < t; i++)
                {
                    float a = pp[(long)c * t + i];
                    float b = pp[(long)(dim + c) * t + i];
                    gp[(long)c * t + i] = a * (1f / (1f + MathF.Exp(-b)));
                }
            pw1.Dispose();

            int k = (int)_dwW!.Shape[2];
            Tensor dw = new Tensor(new TensorShape(1, dim, t), DType.F32);
            backend.Conv1d(dw, glu, _dwW, _dwB, 1, k / 2, k / 2, 1, dim);
            glu.Dispose();

            float* dp = (float*)dw.DataPointer;
            if (_bnScale is not null)
            {
                for (int c = 0; c < dim; c++)
                {
                    float sc = _bnScale[c], sh = _bnShift![c];
                    for (int i = 0; i < t; i++) dp[(long)c * t + i] = dp[(long)c * t + i] * sc + sh;
                }
            }
            else if (_convLnW is not null)
            {
                Tensor rows = ChannelsToRows(dw, dim, t);
                LayerNormRows(rows, _convLnW, _convLnB, t, dim);
                Tensor back = RowsToChannels(rows, dim, t);
                rows.Dispose();
                dw.Dispose();
                dw = back;
                dp = (float*)dw.DataPointer;
            }
            backend.Silu(dw, dw);

            Tensor pw2 = new Tensor(new TensorShape(1, dim, t), DType.F32);
            backend.Conv1d(pw2, dw, _pw2W!, _pw2B, 1, 0, 0, 1, 1);
            dw.Dispose();
            Tensor rowsOut = ChannelsToRows(pw2, dim, t);
            pw2.Dispose();
            return rowsOut;
        }

        private static Tensor Proj(IBackend backend, Tensor x, Tensor w, Tensor? b, int rows, int dim)
        {
            Tensor o = new Tensor(new TensorShape(rows, dim), DType.F32);
            backend.Linear(o, x, w, b);
            return o;
        }

        private static Tensor RowsToChannels(Tensor rows, int c, int t)
        {
            Tensor o = new Tensor(new TensorShape(1, c, t), DType.F32);
            float* rp = (float*)rows.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int ci = 0; ci < c; ci++)
                for (int i = 0; i < t; i++) op[(long)ci * t + i] = rp[(long)i * c + ci];
            return o;
        }

        private static Tensor ChannelsToRows(Tensor cf, int c, int t)
        {
            Tensor o = new Tensor(new TensorShape(t, c), DType.F32);
            float* sp = (float*)cf.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int ci = 0; ci < c; ci++)
                for (int i = 0; i < t; i++) op[(long)i * c + ci] = sp[(long)ci * t + i];
            return o;
        }

        private static void AddInPlace(Tensor target, Tensor value, float scale)
        {
            long count = target.Shape.ElementCount;
            float* tp = (float*)target.DataPointer;
            float* vp = (float*)value.DataPointer;
            for (long i = 0; i < count; i++) tp[i] += scale * vp[i];
        }

        private static Tensor Clone(Tensor x)
        {
            Tensor o = new Tensor(x.Shape, DType.F32);
            long count = x.Shape.ElementCount;
            Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, count * 4, count * 4);
            return o;
        }

        private static Tensor? GetF32(IReadOnlyDictionary<string, Tensor> w, string k) =>
            w.TryGetValue(k, out Tensor? t) ? (t.DType == DType.F32 ? t : t.CastTo(DType.F32)) : null;
    }
}
