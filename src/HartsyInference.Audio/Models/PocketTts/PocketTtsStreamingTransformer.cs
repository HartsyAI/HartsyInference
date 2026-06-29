using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.PocketTts;

/// <summary>The moshi/pocket-tts <c>StreamingTransformer</c> (pre-LayerNorm, fused-QKV attention with
/// interleaved RoPE and optional sliding-window <c>context</c>, optional LayerScale, bias-free exact-GELU MLP).
/// Shared by the FlowLM backbone (1024 dim / 16 heads / 6 layers, no LayerScale, full causal) and the Mimi
/// decoder transformer (512 / 8 / 2, LayerScale 0.01, context 250). Full-sequence (fresh state) causal mask.</summary>
internal sealed unsafe class PocketTtsStreamingTransformer
{
    private readonly string _p;
    private readonly int _dim, _heads, _headDim, _layers, _ffn;
    private readonly int? _context;
    private readonly bool _layerScale;
    private readonly float _maxPeriod;
    private const float LnEps = 1e-5f;

    private readonly Tensor?[] _n1W, _n1B, _n2W, _n2B, _inProjW, _outProjW, _lin1W, _lin2W, _ls1, _ls2;

    public PocketTtsStreamingTransformer(string prefix, int dim, int heads, int layers, int ffn,
        int? context, bool layerScale, float maxPeriod = 10_000f)
    {
        _p = prefix; _dim = dim; _heads = heads; _headDim = dim / heads; _layers = layers; _ffn = ffn;
        _context = context; _layerScale = layerScale; _maxPeriod = maxPeriod;
        _n1W = new Tensor?[layers]; _n1B = new Tensor?[layers]; _n2W = new Tensor?[layers]; _n2B = new Tensor?[layers];
        _inProjW = new Tensor?[layers]; _outProjW = new Tensor?[layers]; _lin1W = new Tensor?[layers]; _lin2W = new Tensor?[layers];
        _ls1 = new Tensor?[layers]; _ls2 = new Tensor?[layers];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        for (int i = 0; i < _layers; i++)
        {
            string p = $"{_p}.layers.{i}";
            _n1W[i] = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]);
            _n1B[i] = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
            _n2W[i] = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]);
            _n2B[i] = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
            _inProjW[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.in_proj.weight"]);
            _outProjW[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.out_proj.weight"]);
            _lin1W[i] = WhisperOps.EnsureF32(w[$"{p}.linear1.weight"]);
            _lin2W[i] = WhisperOps.EnsureF32(w[$"{p}.linear2.weight"]);
            if (_layerScale)
            {
                _ls1[i] = WhisperOps.EnsureF32(w[$"{p}.layer_scale_1.scale"]);
                _ls2[i] = WhisperOps.EnsureF32(w[$"{p}.layer_scale_2.scale"]);
            }
        }
    }

    /// <summary>x channels-last <c>[B,T,dim]</c> -> <c>[B,T,dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        Tensor cur = new(x.Shape, DType.F32);
        long n = x.Shape.ElementCount;
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)cur.DataPointer, n * 4, n * 4);

        Tensor mask = BuildMask(t, _context);
        for (int i = 0; i < _layers; i++)
        {
            Tensor normed = new(cur.Shape, DType.F32);
            backend.LayerNorm(normed, cur, _n1W[i]!, _n1B[i]!, LnEps);
            Tensor qkv = WhisperOps.ProjectLinear(backend, normed, _inProjW[i]!, null, batch, t, _dim, 3 * _dim);
            normed.Dispose();

            Tensor q = new(new TensorShape(batch, _heads, t, _headDim), DType.F32);
            Tensor k = new(new TensorShape(batch, _heads, t, _headDim), DType.F32);
            Tensor v = new(new TensorShape(batch, _heads, t, _headDim), DType.F32);
            SplitQkv(qkv, q, k, v, batch, t);
            qkv.Dispose();
            Rope(q, batch, t); Rope(k, batch, t);

            Tensor attn = new(q.Shape, DType.F32);
            backend.ScaledDotProductAttention(attn, q, k, v, mask, 1f / MathF.Sqrt(_headDim));
            q.Dispose(); k.Dispose(); v.Dispose();
            Tensor flat = new(new TensorShape(batch, t, _dim), DType.F32);
            Merge(attn, flat, batch, t);
            attn.Dispose();
            Tensor outp = WhisperOps.ProjectLinear(backend, flat, _outProjW[i]!, null, batch, t, _dim, _dim);
            flat.Dispose();
            AddUpdate(cur, outp, _ls1[i], batch, t);
            outp.Dispose();

            Tensor normed2 = new(cur.Shape, DType.F32);
            backend.LayerNorm(normed2, cur, _n2W[i]!, _n2B[i]!, LnEps);
            Tensor h = WhisperOps.ProjectLinear(backend, normed2, _lin1W[i]!, null, batch, t, _dim, _ffn);
            normed2.Dispose();
            Activations.ErfGelu(h);
            Tensor ff = WhisperOps.ProjectLinear(backend, h, _lin2W[i]!, null, batch, t, _ffn, _dim);
            h.Dispose();
            AddUpdate(cur, ff, _ls2[i], batch, t);
            ff.Dispose();
        }
        mask.Dispose();
        return cur;
    }

    private void SplitQkv(Tensor qkv, Tensor q, Tensor k, Tensor v, int batch, int t)
    {
        float* s = (float*)qkv.DataPointer;
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int ti = 0; ti < t; ti++)
            {
                long rb = ((long)b * t + ti) * 3 * _dim;
                for (int h = 0; h < _heads; h++)
                    for (int dd = 0; dd < _headDim; dd++)
                    {
                        long dst = (((long)b * _heads + h) * t + ti) * _headDim + dd;
                        int c = h * _headDim + dd;
                        qp[dst] = s[rb + c]; kp[dst] = s[rb + _dim + c]; vp[dst] = s[rb + 2 * _dim + c];
                    }
            }
    }

    private void Merge(Tensor attn, Tensor flat, int batch, int t)
    {
        float* ap = (float*)attn.DataPointer, fp = (float*)flat.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int ti = 0; ti < t; ti++)
                for (int h = 0; h < _heads; h++)
                    for (int dd = 0; dd < _headDim; dd++)
                        fp[((long)b * t + ti) * _dim + h * _headDim + dd]
                            = ap[(((long)b * _heads + h) * t + ti) * _headDim + dd];
    }

    private void AddUpdate(Tensor cur, Tensor upd, Tensor? scale, int batch, int t)
    {
        float* cp = (float*)cur.DataPointer, up = (float*)upd.DataPointer;
        float* sp = scale is null ? null : (float*)scale.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int ti = 0; ti < t; ti++)
                for (int c = 0; c < _dim; c++)
                {
                    long idx = ((long)b * t + ti) * _dim + c;
                    cp[idx] += (sp is null ? 1f : sp[c]) * up[idx];
                }
    }

    private void Rope(Tensor x, int batch, int t)
    {
        float* xp = (float*)x.DataPointer;
        int half = _headDim / 2;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < _heads; h++)
                for (int ti = 0; ti < t; ti++)
                {
                    long rb = (((long)b * _heads + h) * t + ti) * _headDim;
                    for (int kk = 0; kk < half; kk++)
                    {
                        float freq = MathF.Exp(kk * (-MathF.Log(_maxPeriod) * 2f / _headDim));
                        float ang = ti * freq, cs = MathF.Cos(ang), sn = MathF.Sin(ang);
                        float r = xp[rb + 2 * kk], im = xp[rb + 2 * kk + 1];
                        xp[rb + 2 * kk] = r * cs - im * sn;
                        xp[rb + 2 * kk + 1] = r * sn + im * cs;
                    }
                }
    }

    private static Tensor BuildMask(int t, int? context)
    {
        Tensor mask = new(new TensorShape(t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int q = 0; q < t; q++)
            for (int kk = 0; kk < t; kk++)
            {
                int delta = q - kk;
                bool ok = delta >= 0 && (context is null || delta < context.Value);
                mp[(long)q * t + kk] = ok ? 0f : float.NegativeInfinity;
            }
        return mask;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _layers; i++)
        {
            Tensor?[] a = [_n1W[i], _n1B[i], _n2W[i], _n2B[i], _inProjW[i], _outProjW[i], _lin1W[i], _lin2W[i], _ls1[i], _ls2[i]];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}
