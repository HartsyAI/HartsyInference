using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Demucs;

/// <summary>HTDemucs cross-domain transformer: two parallel token streams (spectrogram tokens flattened
/// <c>(t1·fr)</c> and time tokens <c>t2</c>) processed by alternating <b>self-attention</b> (even layers) and
/// <b>cross-attention</b> (odd layers — each stream attends to the other), with sinusoidal positions, a
/// learnable position scale, and gated LayerScale residuals. Reuses <c>ScaledDotProductAttention</c> +
/// <c>DiaHeads</c> + <c>WhisperOps.ProjectLinear</c>.</summary>
public sealed unsafe class DemucsCrossTransformer
{
    private readonly HtDemucsConfig _cfg;
    private readonly int _dim, _heads, _hd;
    private Tensor? _normInW, _normInB, _normInTW, _normInTB;
    private float _posScale = 1f;
    private readonly Layer[] _spec;     // self/cross layers for the spec stream
    private readonly Layer[] _time;     // for the time stream

    public DemucsCrossTransformer(HtDemucsConfig cfg)
    {
        _cfg = cfg; _dim = cfg.BottomChannels; _heads = cfg.THeads; _hd = cfg.TransformerHeadDim;
        _spec = new Layer[cfg.TLayers]; _time = new Layer[cfg.TLayers];
        for (int i = 0; i < cfg.TLayers; i++) { _spec[i] = new Layer(cfg, i % 2 == 1); _time[i] = new Layer(cfg, i % 2 == 1); }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "crosstransformer")
    {
        _normInW = WhisperOps.EnsureF32(w[$"{prefix}.norm_in.weight"]); _normInB = WhisperOps.EnsureF32(w[$"{prefix}.norm_in.bias"]);
        _normInTW = WhisperOps.EnsureF32(w[$"{prefix}.norm_in_t.weight"]); _normInTB = WhisperOps.EnsureF32(w[$"{prefix}.norm_in_t.bias"]);
        if (w.TryGetValue($"{prefix}.weight_pos_embed", out Tensor? wp)) _posScale = ((float*)WhisperOps.EnsureF32(wp).DataPointer)[0];
        for (int i = 0; i < _cfg.TLayers; i++) { _spec[i].LoadWeights(w, $"{prefix}.layers.{i}"); _time[i].LoadWeights(w, $"{prefix}.layers_t.{i}"); }
    }

    /// <summary>Mixes spec tokens <c>[1, dim, nSpec]</c> and time tokens <c>[1, dim, nTime]</c> (channels-first),
    /// returning the updated streams in the same layout.</summary>
    public (Tensor Spec, Tensor Time) Forward(IBackend backend, Tensor spec, int nSpec, Tensor time, int nTime)
    {
        // To channels-last tokens [1, n, dim] + norm_in + sin positions.
        Tensor x = ToTokens(backend, spec, nSpec);
        Tensor xn = new(x.Shape, DType.F32); backend.LayerNorm(xn, x, _normInW!, _normInB!, _cfg.NormEps); x.Dispose();
        AddSin(xn, nSpec, _dim, _posScale);
        Tensor xt = ToTokens(backend, time, nTime);
        Tensor xtn = new(xt.Shape, DType.F32); backend.LayerNorm(xtn, xt, _normInTW!, _normInTB!, _cfg.NormEps); xt.Dispose();
        AddSin(xtn, nTime, _dim, _posScale);

        Tensor s = xn, t = xtn;
        for (int i = 0; i < _cfg.TLayers; i++)
        {
            if (i % 2 == 0)   // self-attention
            {
                Tensor ns = _spec[i].Forward(backend, s, s, nSpec, nSpec);
                Tensor nt = _time[i].Forward(backend, t, t, nTime, nTime);
                s.Dispose(); t.Dispose(); s = ns; t = nt;
            }
            else              // cross-attention (each attends to the other's pre-update state)
            {
                Tensor oldS = s;
                Tensor ns = _spec[i].Forward(backend, s, t, nSpec, nTime);
                Tensor nt = _time[i].Forward(backend, t, oldS, nTime, nSpec);
                s.Dispose(); t.Dispose(); s = ns; t = nt;
            }
        }
        Tensor specOut = FromTokens(backend, s, nSpec); s.Dispose();
        Tensor timeOut = FromTokens(backend, t, nTime); t.Dispose();
        return (specOut, timeOut);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_normInW, _normInB, _normInTW, _normInTB];
        foreach (Tensor? x in own) if (x is not null) yield return x;
        for (int i = 0; i < _cfg.TLayers; i++) { foreach (Tensor x in _spec[i].EnumerateWeights()) yield return x; foreach (Tensor x in _time[i].EnumerateWeights()) yield return x; }
    }

    private Tensor ToTokens(IBackend backend, Tensor cf, int n)
    {
        Tensor cl = new(new TensorShape(1, n, _dim), DType.F32);
        backend.Transpose2D(cl, cf, _dim, n);
        return cl;
    }

    private Tensor FromTokens(IBackend backend, Tensor cl, int n)
    {
        Tensor cf = new(new TensorShape(1, _dim, n), DType.F32);
        backend.Transpose2D(cf, cl, n, _dim);
        return cf;
    }

    private static void AddSin(Tensor x, int n, int dim, float scale)
    {
        float* p = (float*)x.DataPointer;
        int half = dim / 2;
        for (int j = 0; j < n; j++)
            for (int c = 0; c < half; c++)
            {
                float div = MathF.Exp(-(2f * c / dim) * MathF.Log(10000f));
                p[(long)j * dim + c] += scale * MathF.Sin(j * div);
                p[(long)j * dim + half + c] += scale * MathF.Cos(j * div);
            }
    }

    /// <summary>One transformer layer (pre-norm): attention (self or cross) with gated LayerScale + FFN.</summary>
    private sealed class Layer
    {
        private readonly HtDemucsConfig _cfg;
        private readonly bool _cross;
        private readonly int _dim, _heads, _hd, _ffn;
        private Tensor? _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _n1W, _n1B, _n2W, _n2B, _n3W, _n3B, _f1W, _f1B, _f2W, _f2B, _g1, _g2;

        public Layer(HtDemucsConfig cfg, bool cross)
        {
            _cfg = cfg; _cross = cross; _dim = cfg.BottomChannels; _heads = cfg.THeads; _hd = cfg.TransformerHeadDim; _ffn = cfg.TransformerFfn;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            string attn = _cross ? "cross_attn" : "self_attn";
            _qW = WhisperOps.EnsureF32(w[$"{p}.{attn}.q_proj.weight"]); _qB = Bias(w, $"{p}.{attn}.q_proj.bias");
            _kW = WhisperOps.EnsureF32(w[$"{p}.{attn}.k_proj.weight"]); _kB = Bias(w, $"{p}.{attn}.k_proj.bias");
            _vW = WhisperOps.EnsureF32(w[$"{p}.{attn}.v_proj.weight"]); _vB = Bias(w, $"{p}.{attn}.v_proj.bias");
            _oW = WhisperOps.EnsureF32(w[$"{p}.{attn}.out_proj.weight"]); _oB = Bias(w, $"{p}.{attn}.out_proj.bias");
            _n1W = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]); _n1B = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
            _n2W = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]); _n2B = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
            _n3W = WhisperOps.EnsureF32(w[$"{p}.norm3.weight"]); _n3B = WhisperOps.EnsureF32(w[$"{p}.norm3.bias"]);
            _f1W = WhisperOps.EnsureF32(w[$"{p}.linear1.weight"]); _f1B = WhisperOps.EnsureF32(w[$"{p}.linear1.bias"]);
            _f2W = WhisperOps.EnsureF32(w[$"{p}.linear2.weight"]); _f2B = WhisperOps.EnsureF32(w[$"{p}.linear2.bias"]);
            _g1 = WhisperOps.EnsureF32(w[$"{p}.gamma_1.scale"]); _g2 = WhisperOps.EnsureF32(w[$"{p}.gamma_2.scale"]);
        }

        public Tensor Forward(IBackend backend, Tensor q, Tensor kv, int nq, int nkv)
        {
            // Pre-norm attention with LayerScale gamma_1 residual.
            Tensor qn = new(q.Shape, DType.F32); backend.LayerNorm(qn, q, _n1W!, _n1B!, _cfg.NormEps);
            Tensor kvn = new(kv.Shape, DType.F32); backend.LayerNorm(kvn, kv, _n2W!, _n2B!, _cfg.NormEps);
            Tensor attn = Attend(backend, qn, kvn, nq, nkv); qn.Dispose(); kvn.Dispose();
            ScaleInPlace(attn, _g1!, nq);
            Tensor afterAttn = new(q.Shape, DType.F32); backend.Add(afterAttn, q, attn); attn.Dispose();

            // Pre-norm FFN with LayerScale gamma_2 residual.
            Tensor fn = new(q.Shape, DType.F32); backend.LayerNorm(fn, afterAttn, _n3W!, _n3B!, _cfg.NormEps);
            Tensor h1 = WhisperOps.ProjectLinear(backend, fn, _f1W!, _f1B, 1, nq, _dim, _ffn); fn.Dispose();
            Tensor act = new(h1.Shape, DType.F32); backend.Gelu(act, h1); h1.Dispose();
            Tensor h2 = WhisperOps.ProjectLinear(backend, act, _f2W!, _f2B, 1, nq, _ffn, _dim); act.Dispose();
            ScaleInPlace(h2, _g2!, nq);
            Tensor outT = new(q.Shape, DType.F32); backend.Add(outT, afterAttn, h2); afterAttn.Dispose(); h2.Dispose();
            return outT;
        }

        private Tensor Attend(IBackend backend, Tensor q, Tensor kv, int nq, int nkv)
        {
            Tensor qp = WhisperOps.ProjectLinear(backend, q, _qW!, _qB, 1, nq, _dim, _dim);
            Tensor kp = WhisperOps.ProjectLinear(backend, kv, _kW!, _kB, 1, nkv, _dim, _dim);
            Tensor vp = WhisperOps.ProjectLinear(backend, kv, _vW!, _vB, 1, nkv, _dim, _dim);
            Tensor qM = new(new TensorShape(1, _heads, nq, _hd), DType.F32);
            Tensor kM = new(new TensorShape(1, _heads, nkv, _hd), DType.F32);
            Tensor vM = new(new TensorShape(1, _heads, nkv, _hd), DType.F32);
            DiaHeads.FlatToHeads(qM, qp, nq, _heads, _hd); qp.Dispose();
            DiaHeads.FlatToHeads(kM, kp, nkv, _heads, _hd); kp.Dispose();
            DiaHeads.FlatToHeads(vM, vp, nkv, _heads, _hd); vp.Dispose();
            Tensor attn = new(new TensorShape(1, _heads, nq, _hd), DType.F32);
            backend.ScaledDotProductAttention(attn, qM, kM, vM, null, 1f / MathF.Sqrt(_hd));
            qM.Dispose(); kM.Dispose(); vM.Dispose();
            Tensor flat = new(new TensorShape(1, nq, _dim), DType.F32);
            DiaHeads.HeadsToFlat(flat, attn, nq, _heads, _hd); attn.Dispose();
            Tensor outT = WhisperOps.ProjectLinear(backend, flat, _oW!, _oB, 1, nq, _dim, _dim); flat.Dispose();
            return outT;
        }

        private void ScaleInPlace(Tensor x, Tensor gamma, int n)
        {
            float* xp = (float*)x.DataPointer; float* gp = (float*)gamma.DataPointer;
            for (int j = 0; j < n; j++) for (int c = 0; c < _dim; c++) xp[(long)j * _dim + c] *= gp[c];
        }

        private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key)
            => w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [_qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _n1W, _n1B, _n2W, _n2B, _n3W, _n3B, _f1W, _f1B, _f2W, _f2B, _g1, _g2];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
    }
}
