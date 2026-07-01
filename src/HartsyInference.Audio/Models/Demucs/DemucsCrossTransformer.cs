using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Demucs;

/// <summary>HTDemucs cross-domain transformer (demucs <c>CrossTransformerEncoder</c>). Two token streams, the
/// spectrogram (flattened <c>(t1·fr)</c>, t1 outer) and the time branch (<c>t2</c>), each get a sinusoidal
/// positional embedding (2D for the spec, 1D for time) added after a <c>norm_in</c> LayerNorm. Layers alternate
/// (<c>classic_parity=0</c>): even indices are <b>self</b>-attention (<c>MyTransformerEncoderLayer</c>), odd are
/// <b>cross</b>-attention (each stream attends to the other's pre-update state). Every layer is pre-norm with
/// LayerScale (<c>gamma_1/2</c>), a GELU FFN, and a final <c>norm_out</c>. Attention uses a fused
/// <c>in_proj_weight</c> [3·dim, dim] (nn.MultiheadAttention), 8 heads, <c>1/sqrt(head_dim)</c>.</summary>
public sealed unsafe class DemucsCrossTransformer
{
    private readonly HtDemucsConfig _cfg;
    private readonly int _dim, _heads, _hd, _ffn, _layers;
    private Tensor? _normInW, _normInB, _normInTW, _normInTB;
    private readonly Layer[] _spec;
    private readonly Layer[] _time;
    /// <summary>Parity-debug only. Not used in production.</summary>
    internal static Action<string, Tensor>? Probe;
    /// <summary>Parity-debug only: one-shot probe fired with the next layer's raw attention output. Not used in production.</summary>
    internal static Action<Tensor>? AttnProbe;
    /// <summary>Parity-debug only: one-shot probe fired with the next layer's post-attention residual. Not used in production.</summary>
    internal static Action<Tensor>? AfterAttnProbe;
    /// <summary>Parity-debug only: one-shot probe fired with the next layer's post-FFN output (pre norm_out).</summary>
    internal static Action<Tensor>? PreNormOutProbe;

    public DemucsCrossTransformer(HtDemucsConfig cfg)
    {
        _cfg = cfg;
        _dim = cfg.BottomChannels; _heads = cfg.THeads; _hd = cfg.TransformerHeadDim; _ffn = cfg.TransformerFfn;
        _layers = cfg.TLayers;
        _spec = new Layer[_layers];
        _time = new Layer[_layers];
        for (int i = 0; i < _layers; i++)
        {
            bool cross = (i % 2) != 0;       // classic_parity=0 → even self, odd cross
            _spec[i] = new Layer(cfg, cross);
            _time[i] = new Layer(cfg, cross);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "crosstransformer")
    {
        _normInW = WhisperOps.EnsureF32(w[$"{prefix}.norm_in.weight"]); _normInB = WhisperOps.EnsureF32(w[$"{prefix}.norm_in.bias"]);
        _normInTW = WhisperOps.EnsureF32(w[$"{prefix}.norm_in_t.weight"]); _normInTB = WhisperOps.EnsureF32(w[$"{prefix}.norm_in_t.bias"]);
        for (int i = 0; i < _layers; i++) { _spec[i].LoadWeights(w, $"{prefix}.layers.{i}"); _time[i].LoadWeights(w, $"{prefix}.layers_t.{i}"); }
    }

    /// <summary>Forward over the spec stream <paramref name="x"/> <c>[1, C, Fr, T1]</c> and the time stream
    /// <paramref name="xt"/> <c>[1, C, T2]</c>; returns the updated streams in the same layouts.</summary>
    public (Tensor X, Tensor Xt) Forward(IBackend backend, Tensor x, int c, int fr, int t1, Tensor xt, int t2)
    {
        // Spec → tokens [1, t1*fr, C] (t1 outer, fr inner), norm_in, + 2D sin pos.
        int nSpec = t1 * fr;
        Tensor s = new(new TensorShape(1, nSpec, c), DType.F32);
        float* sp = (float*)s.DataPointer; float* xp = (float*)x.DataPointer;
        for (int it = 0; it < t1; it++)
            for (int ifr = 0; ifr < fr; ifr++)
            {
                long tok = (long)it * fr + ifr;
                for (int ch = 0; ch < c; ch++) sp[tok * c + ch] = xp[(((long)ch * fr + ifr) * t1) + it];
            }
        Tensor sn = new(s.Shape, DType.F32); backend.LayerNorm(sn, s, _normInW!, _normInB!, _cfg.NormEps); s.Dispose();
        Add2dSinPos(sn, c, fr, t1, _cfg.TMaxPeriod);

        // Time → tokens [1, t2, C], norm_in_t, + 1D sin pos.
        Tensor tt = new(new TensorShape(1, t2, c), DType.F32);
        float* ttp = (float*)tt.DataPointer; float* xtp = (float*)xt.DataPointer;
        for (int j = 0; j < t2; j++) for (int ch = 0; ch < c; ch++) ttp[(long)j * c + ch] = xtp[(long)ch * t2 + j];
        Tensor tn = new(tt.Shape, DType.F32); backend.LayerNorm(tn, tt, _normInTW!, _normInTB!, _cfg.NormEps); tt.Dispose();
        Add1dSinPos(tn, c, t2, _cfg.TMaxPeriod);

        Probe?.Invoke("ct_pos_x", sn); Probe?.Invoke("ct_pos_xt", tn);
        Tensor sCur = sn, tCur = tn;
        for (int i = 0; i < _layers; i++)
        {
            if (i % 2 == 0)
            {
                Tensor ns = _spec[i].Forward(backend, sCur, sCur, nSpec, nSpec);
                Tensor nt = _time[i].Forward(backend, tCur, tCur, t2, t2);
                sCur.Dispose(); tCur.Dispose(); sCur = ns; tCur = nt;
            }
            else
            {
                Tensor oldS = sCur;
                Tensor ns = _spec[i].Forward(backend, sCur, tCur, nSpec, t2);
                Tensor nt = _time[i].Forward(backend, tCur, oldS, t2, nSpec);
                sCur.Dispose(); tCur.Dispose(); sCur = ns; tCur = nt;
            }
            if (i == 0) { Probe?.Invoke("ct_l0_x", sCur); Probe?.Invoke("ct_l0_xt", tCur); }
        }

        // Untokenize.
        Tensor outX = new(new TensorShape(1, c, fr, t1), DType.F32);
        float* oxp = (float*)outX.DataPointer; float* scp = (float*)sCur.DataPointer;
        for (int it = 0; it < t1; it++)
            for (int ifr = 0; ifr < fr; ifr++)
            {
                long tok = (long)it * fr + ifr;
                for (int ch = 0; ch < c; ch++) oxp[(((long)ch * fr + ifr) * t1) + it] = scp[tok * c + ch];
            }
        sCur.Dispose();
        Tensor outXt = new(new TensorShape(1, c, t2), DType.F32);
        float* oxtp = (float*)outXt.DataPointer; float* tcp = (float*)tCur.DataPointer;
        for (int j = 0; j < t2; j++) for (int ch = 0; ch < c; ch++) oxtp[(long)ch * t2 + j] = tcp[(long)j * c + ch];
        tCur.Dispose();
        return (outX, outXt);
    }

    /// <summary>Adds the demucs 1D sinusoidal position embedding to channels-last tokens [1, n, dim].</summary>
    private static void Add1dSinPos(Tensor x, int dim, int n, int maxPeriod)
    {
        int half = dim / 2;
        float* xp = (float*)x.DataPointer;
        for (int t = 0; t < n; t++)
            for (int k = 0; k < half; k++)
            {
                float phase = t / MathF.Pow(maxPeriod, (float)k / (half - 1));
                xp[(long)t * dim + k] += MathF.Cos(phase);
                xp[(long)t * dim + half + k] += MathF.Sin(phase);
            }
    }

    /// <summary>Adds the demucs 2D sinusoidal position embedding to spec tokens [1, t1*fr, dim] (token=t1·fr+fr).</summary>
    private static void Add2dSinPos(Tensor x, int dim, int fr, int t1, int maxPeriod)
    {
        int d = dim / 2;                                  // half for width(time), half for height(freq)
        float* xp = (float*)x.DataPointer;
        for (int it = 0; it < t1; it++)
            for (int ifr = 0; ifr < fr; ifr++)
            {
                long tok = (long)it * fr + ifr;
                long baseI = tok * dim;
                for (int j = 0; j < d; j += 2)
                {
                    float div = MathF.Exp(j * -(MathF.Log(maxPeriod) / d));
                    xp[baseI + j] += MathF.Sin(it * div);          // pe[0:d:2] width=time
                    xp[baseI + j + 1] += MathF.Cos(it * div);      // pe[1:d:2]
                    xp[baseI + d + j] += MathF.Sin(ifr * div);     // pe[d::2] height=freq
                    xp[baseI + d + j + 1] += MathF.Cos(ifr * div); // pe[d+1::2]
                }
            }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_normInW, _normInB, _normInTW, _normInTB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        for (int i = 0; i < _layers; i++)
        {
            foreach (Tensor t in _spec[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _time[i].EnumerateWeights()) yield return t;
        }
    }

    /// <summary>One transformer layer. Self (even): <c>x = x + g1·sa(norm1(x)); x = x + g2·ff(norm2(x)); x = norm_out(x)</c>.
    /// Cross (odd): <c>x = q + g1·ca(norm1(q), norm2(k)); x = x + g2·ff(norm3(x)); x = norm_out(x)</c>. Fused QKV.</summary>
    private sealed class Layer
    {
        private readonly HtDemucsConfig _cfg;
        private readonly bool _cross;
        private readonly int _dim, _heads, _hd, _ffn;
        private Tensor? _inW, _inB, _oW, _oB;            // fused in_proj [3dim,dim], out_proj
        private Tensor? _n1W, _n1B, _n2W, _n2B, _n3W, _n3B, _noW, _noB;
        private Tensor? _f1W, _f1B, _f2W, _f2B, _g1, _g2;

        public Layer(HtDemucsConfig cfg, bool cross)
        {
            _cfg = cfg; _cross = cross; _dim = cfg.BottomChannels; _heads = cfg.THeads; _hd = cfg.TransformerHeadDim; _ffn = cfg.TransformerFfn;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            string attn = _cross ? "cross_attn" : "self_attn";
            _inW = WhisperOps.EnsureF32(w[$"{p}.{attn}.in_proj_weight"]); _inB = Bias(w, $"{p}.{attn}.in_proj_bias");
            _oW = WhisperOps.EnsureF32(w[$"{p}.{attn}.out_proj.weight"]); _oB = Bias(w, $"{p}.{attn}.out_proj.bias");
            _n1W = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]); _n1B = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
            _n2W = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]); _n2B = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
            if (_cross) { _n3W = WhisperOps.EnsureF32(w[$"{p}.norm3.weight"]); _n3B = WhisperOps.EnsureF32(w[$"{p}.norm3.bias"]); }
            _noW = WhisperOps.EnsureF32(w[$"{p}.norm_out.weight"]); _noB = WhisperOps.EnsureF32(w[$"{p}.norm_out.bias"]);
            _f1W = WhisperOps.EnsureF32(w[$"{p}.linear1.weight"]); _f1B = Bias(w, $"{p}.linear1.bias");
            _f2W = WhisperOps.EnsureF32(w[$"{p}.linear2.weight"]); _f2B = Bias(w, $"{p}.linear2.bias");
            _g1 = WhisperOps.EnsureF32(w[$"{p}.gamma_1.scale"]); _g2 = WhisperOps.EnsureF32(w[$"{p}.gamma_2.scale"]);
        }

        public Tensor Forward(IBackend backend, Tensor q, Tensor kv, int nq, int nkv)
        {
            // Attention sub-block: norm1(q) [+ norm2(kv) for cross] → MHA → gamma_1 → residual on q.
            Tensor qn = new(q.Shape, DType.F32); backend.LayerNorm(qn, q, _n1W!, _n1B!, _cfg.NormEps);
            Tensor kvn;
            if (_cross) { kvn = new(kv.Shape, DType.F32); backend.LayerNorm(kvn, kv, _n2W!, _n2B!, _cfg.NormEps); }
            else kvn = qn;       // self-attention: keys/values are the same normed tensor
            Tensor attn = Attend(backend, qn, kvn, nq, nkv);
            qn.Dispose(); if (_cross) kvn.Dispose();
            if (AttnProbe is not null) { Action<Tensor> p = AttnProbe; AttnProbe = null; p(attn); }   // one-shot: raw attn out
            ScaleInPlace(attn, _g1!, nq);
            Tensor afterAttn = new(q.Shape, DType.F32); backend.Add(afterAttn, q, attn); attn.Dispose();
            if (AfterAttnProbe is not null) { Action<Tensor> p = AfterAttnProbe; AfterAttnProbe = null; p(afterAttn); }   // one-shot

            // FFN sub-block: ffn-norm (norm2 for self, norm3 for cross) → linear→gelu→linear → gamma_2 → residual.
            Tensor ffW = _cross ? _n3W! : _n2W!, ffB = _cross ? _n3B! : _n2B!;
            Tensor fn = new(q.Shape, DType.F32); backend.LayerNorm(fn, afterAttn, ffW, ffB, _cfg.NormEps);
            Tensor h1 = WhisperOps.ProjectLinear(backend, fn, _f1W!, _f1B, 1, nq, _dim, _ffn); fn.Dispose();
            HartsyInference.Audio.Layers.Activations.ErfGelu(h1);   // demucs F.gelu = exact erf
            Tensor h2 = WhisperOps.ProjectLinear(backend, h1, _f2W!, _f2B, 1, nq, _ffn, _dim); h1.Dispose();
            ScaleInPlace(h2, _g2!, nq);
            Tensor outT = new(q.Shape, DType.F32); backend.Add(outT, afterAttn, h2); afterAttn.Dispose(); h2.Dispose();
            if (PreNormOutProbe is not null) { Action<Tensor> p = PreNormOutProbe; PreNormOutProbe = null; p(outT); }   // one-shot

            // norm_out is a MyGroupNorm(num_groups=1): normalize over ALL tokens×channels jointly (NOT per-token
            // like LayerNorm), then per-channel affine. (demucs transposes to (B,C,T) and runs GroupNorm(1,C).)
            GroupNormOut(outT, nq, _noW!, _noB!);
            return outT;
        }

        private Tensor Attend(IBackend backend, Tensor q, Tensor kv, int nq, int nkv)
        {
            // Fused in_proj: rows [0:dim]=Wq, [dim:2dim]=Wk, [2dim:3dim]=Wv.
            Tensor qp = LinearSlice(backend, q, nq, 0);
            Tensor kp = LinearSlice(backend, kv, nkv, _dim);
            Tensor vp = LinearSlice(backend, kv, nkv, 2 * _dim);
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

        /// <summary>Linear with a row-slice [rowOffset : rowOffset+dim) of the fused in_proj weight/bias.</summary>
        private Tensor LinearSlice(IBackend backend, Tensor x, int n, int rowOffset)
        {
            Tensor outT = new(new TensorShape(1, n, _dim), DType.F32);
            float* xp = (float*)x.DataPointer; float* op = (float*)outT.DataPointer;
            float* wp = (float*)_inW!.DataPointer + (long)rowOffset * _dim;
            float* bp = _inB is null ? null : (float*)_inB.DataPointer + rowOffset;
            for (int t = 0; t < n; t++)
            {
                float* row = xp + (long)t * _dim;
                float* dst = op + (long)t * _dim;
                for (int o = 0; o < _dim; o++)
                {
                    float acc = bp is null ? 0f : bp[o];
                    float* wrow = wp + (long)o * _dim;
                    for (int k = 0; k < _dim; k++) acc += wrow[k] * row[k];
                    dst[o] = acc;
                }
            }
            return outT;
        }

        /// <summary>MyGroupNorm(num_groups=1) over tokens [1, n, dim] in place: one mean/var over all n·dim
        /// (biased, eps 1e-5), then per-channel affine. Matches demucs transpose→GroupNorm(1,C)→transpose.</summary>
        private void GroupNormOut(Tensor x, int n, Tensor weight, Tensor bias)
        {
            float* xp = (float*)x.DataPointer; float* wp = (float*)weight.DataPointer; float* bp = (float*)bias.DataPointer;
            long total = (long)n * _dim;
            double sum = 0, sumSq = 0;
            for (long i = 0; i < total; i++) { float v = xp[i]; sum += v; sumSq += (double)v * v; }
            double mean = sum / total;
            double var = sumSq / total - mean * mean;
            float invStd = (float)(1.0 / Math.Sqrt(var + _cfg.NormEps));
            for (int t = 0; t < n; t++)
                for (int c = 0; c < _dim; c++)
                {
                    long idx = (long)t * _dim + c;
                    xp[idx] = (float)((xp[idx] - mean) * invStd) * wp[c] + bp[c];
                }
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
            Tensor?[] all = [_inW, _inB, _oW, _oB, _n1W, _n1B, _n2W, _n2B, _n3W, _n3B, _noW, _noB, _f1W, _f1B, _f2W, _f2B, _g1, _g2];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
    }
}
