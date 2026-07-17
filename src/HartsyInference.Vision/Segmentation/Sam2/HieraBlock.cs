using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>One Hiera <c>MultiScaleBlock</c>: <c>LayerNorm → (windowed) MultiScaleAttention → residual</c>
/// then <c>LayerNorm → MLP(4×, GELU) → residual</c>, operating on channels-last <c>[B,H,W,C]</c> tokens.
/// When the block starts a new stage it projects the residual to <see cref="DimOut"/> and pools the query
/// (stride-2) so spatial resolution halves and the embed dim doubles. Windowed attention partitions the token
/// grid into <c>window×window</c> non-overlapping windows before attending and stitches them back after.</summary>
public sealed unsafe class HieraBlock
{
    private readonly int _dimIn;
    private readonly int _dimOut;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _windowSize;
    private readonly bool _qPool;
    private readonly int _mlpHidden;
    private readonly float _eps;

    private Tensor? _norm1W, _norm1B;
    private Tensor? _qkvW, _qkvB;     // qkv: [3*dimOut, dimIn]
    private Tensor? _attnProjW, _attnProjB; // [dimOut, dimOut]
    private Tensor? _norm2W, _norm2B;
    private Tensor? _mlpW0, _mlpB0;   // [4*dimOut, dimOut]
    private Tensor? _mlpW1, _mlpB1;   // [dimOut, 4*dimOut]
    private Tensor? _projW, _projB;   // residual projection [dimOut, dimIn], present only when dimIn != dimOut

    /// <summary>Output channel count (== input when the block stays within a stage).</summary>
    public int DimOut => _dimOut;

    /// <summary>Creates a block from its resolved schedule metadata; weights load via <see cref="LoadWeights"/>.</summary>
    public HieraBlock(int dimIn, int dimOut, int heads, int windowSize, bool qPool, int mlpRatio, float eps)
    {
        if (dimOut % heads != 0)
            throw new ArgumentException($"dimOut {dimOut} must be divisible by heads {heads}.", nameof(heads));
        _dimIn = dimIn;
        _dimOut = dimOut;
        _heads = heads;
        _headDim = dimOut / heads;
        _windowSize = windowSize;
        _qPool = qPool;
        _mlpHidden = dimOut * mlpRatio;
        _eps = eps;
    }

    /// <summary>Loads the block's norm/attn/mlp (and optional residual-proj) weights under <paramref name="prefix"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1W = F32(weights, $"{prefix}norm1.weight");
        _norm1B = F32(weights, $"{prefix}norm1.bias");
        _qkvW = F32(weights, $"{prefix}attn.qkv.weight");
        _qkvB = F32(weights, $"{prefix}attn.qkv.bias");
        _attnProjW = F32(weights, $"{prefix}attn.proj.weight");
        _attnProjB = F32(weights, $"{prefix}attn.proj.bias");
        _norm2W = F32(weights, $"{prefix}norm2.weight");
        _norm2B = F32(weights, $"{prefix}norm2.bias");
        _mlpW0 = F32(weights, $"{prefix}mlp.layers.0.weight");
        _mlpB0 = F32(weights, $"{prefix}mlp.layers.0.bias");
        _mlpW1 = F32(weights, $"{prefix}mlp.layers.1.weight");
        _mlpB1 = F32(weights, $"{prefix}mlp.layers.1.bias");
        if (_dimIn != _dimOut)
        {
            _projW = F32(weights, $"{prefix}proj.weight");
            _projB = F32(weights, $"{prefix}proj.bias");
        }
    }

    /// <summary>Yields every loaded weight tensor.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _norm1W, _norm1B, _qkvW, _qkvB, _attnProjW, _attnProjB,
                                      _norm2W, _norm2B, _mlpW0, _mlpB0, _mlpW1, _mlpB1, _projW, _projB })
        {
            if (t is not null) yield return t;
        }
    }

    /// <summary>Runs the block on <paramref name="x"/> <c>[1,H,W,dimIn]</c>; returns the output tokens and their
    /// (possibly halved) spatial size.</summary>
    public (Tensor y, int outH, int outW) Forward(IBackend backend, Tensor x, int h, int w)
    {
        if (_norm1W is null)
            throw new InvalidOperationException("HieraBlock weights not loaded.");

        int b = (int)x.Shape[0];

        Tensor xn = new Tensor(x.Shape, DType.F32);
        backend.LayerNorm(xn, x, _norm1W!, _norm1B!, _eps);

        // Residual path: project + pool when entering a new stage so it matches the attention output.
        Tensor shortcut = x;
        int shH = h, shW = w;
        if (_dimIn != _dimOut)
        {
            shortcut = LinearNhwc(backend, xn, _projW!, _projB!, h, w, _dimOut);
        }
        if (_qPool)
        {
            shortcut = MaxPoolNhwc(backend, shortcut, _dimOut, ref shH, ref shW);
        }

        // Attention (optionally windowed).
        Tensor attnIn = xn;
        int attnH = h, attnW = w;
        int ws = _windowSize;
        if (ws > 0)
        {
            attnIn = WindowPartition(attnIn, b, h, w, ws, _dimIn);
            attnH = ws;
            attnW = ws;
        }

        (Tensor attnOut, int aH, int aW) = Attention(backend, attnIn, attnH, attnW);

        int outH = shH, outW = shW;
        if (ws > 0)
        {
            int unWs = _qPool ? ws / 2 : ws; // a pooled query shrinks each window by 2
            int padHp = outH + (unWs - outH % unWs) % unWs;
            int padWp = outW + (unWs - outW % unWs) % unWs;
            attnOut = WindowUnpartition(attnOut, b, unWs, padHp, padWp, outH, outW, _dimOut);
        }
        else
        {
            outH = aH;
            outW = aW;
        }

        Tensor added = new Tensor(new TensorShape(b, outH, outW, _dimOut), DType.F32);
        backend.Add(added, shortcut, attnOut);

        Tensor n2 = new Tensor(added.Shape, DType.F32);
        backend.LayerNorm(n2, added, _norm2W!, _norm2B!, _eps);
        Tensor hidden = LinearNhwc(backend, n2, _mlpW0!, _mlpB0!, outH, outW, _mlpHidden);
        backend.Gelu(hidden, hidden);
        Tensor mlpOut = LinearNhwc(backend, hidden, _mlpW1!, _mlpB1!, outH, outW, _dimOut);

        Tensor y = new Tensor(added.Shape, DType.F32);
        backend.Add(y, added, mlpOut);
        return (y, outH, outW);
    }

    /// <summary>MultiScaleAttention on <c>[B,H,W,dimIn]</c> (B is the window-batch when called windowed):
    /// projects q/k/v, optionally pools the query spatially, runs multi-head SDPA, projects out. Returns
    /// <c>[B,H',W',dimOut]</c> where H'/W' are halved iff this block pools.</summary>
    private (Tensor outT, int outH, int outW) Attention(IBackend backend, Tensor x, int h, int w)
    {
        int b = (int)x.Shape[0];
        int n = h * w;

        Tensor xf = x.Reshape(new TensorShape(b, n, _dimIn));
        Tensor qkv = new Tensor(new TensorShape(b, n, 3 * _dimOut), DType.F32);
        backend.Linear(qkv, xf, _qkvW!, _qkvB);

        // Split the packed qkv [B,N,3,dimOut] along the "3" axis into q/k/v.
        Tensor qkv4 = qkv.Reshape(new TensorShape(b, n, 3, _dimOut));
        Tensor q3 = new Tensor(new TensorShape(b, n, 1, _dimOut), DType.F32);
        Tensor k3 = new Tensor(new TensorShape(b, n, 1, _dimOut), DType.F32);
        Tensor v3 = new Tensor(new TensorShape(b, n, 1, _dimOut), DType.F32);
        Span<Tensor> parts = [q3, k3, v3];
        backend.Split(parts, qkv4, 2);

        Tensor qFlat = q3.Reshape(new TensorShape(b, n, _dimOut));
        Tensor kFlat = k3.Reshape(new TensorShape(b, n, _dimOut));
        Tensor vFlat = v3.Reshape(new TensorShape(b, n, _dimOut));

        int outH = h, outW = w;
        int nq = n;
        if (_qPool)
        {
            Tensor qNhwc = qFlat.Reshape(new TensorShape(b, h, w, _dimOut));
            Tensor qPooled = MaxPoolNhwc(backend, qNhwc, _dimOut, ref outH, ref outW);
            nq = outH * outW;
            qFlat = qPooled.Reshape(new TensorShape(b, nq, _dimOut));
        }

        Tensor qHeads = ToHeads(backend, qFlat, b, nq);
        Tensor kHeads = ToHeads(backend, kFlat, b, n);
        Tensor vHeads = ToHeads(backend, vFlat, b, n);

        Tensor attn = new Tensor(new TensorShape(b, _heads, nq, _headDim), DType.F32);
        float scale = 1f / MathF.Sqrt(_headDim);
        backend.ScaledDotProductAttention(attn, qHeads, kHeads, vHeads, null, scale);

        // [B,heads,Nq,hd] → [B,Nq,heads,hd] → [B,Nq,dimOut]
        Tensor merged = new Tensor(new TensorShape(b, nq, _heads, _headDim), DType.F32);
        backend.Permute0213(merged, attn, _heads, nq, _headDim);
        Tensor mergedFlat = merged.Reshape(new TensorShape(b, nq, _dimOut));

        Tensor proj = new Tensor(new TensorShape(b, nq, _dimOut), DType.F32);
        backend.Linear(proj, mergedFlat, _attnProjW!, _attnProjB);
        return (proj.Reshape(new TensorShape(b, outH, outW, _dimOut)), outH, outW);
    }

    /// <summary>[B,N,dimOut] → [B,heads,N,headDim] (reshape to heads, then swap the seq/head axes).</summary>
    private Tensor ToHeads(IBackend backend, Tensor x, int b, int n)
    {
        Tensor reshaped = x.Reshape(new TensorShape(b, n, _heads, _headDim));
        Tensor outT = new Tensor(new TensorShape(b, _heads, n, _headDim), DType.F32);
        backend.Permute0213(outT, reshaped, n, _heads, _headDim);
        return outT;
    }

    private static Tensor LinearNhwc(IBackend backend, Tensor x, Tensor weight, Tensor bias, int h, int w, int outDim)
    {
        int b = (int)x.Shape[0];
        int inDim = (int)x.Shape[3];
        Tensor v = x.Reshape(new TensorShape(b, h * w, inDim));
        Tensor o = new Tensor(new TensorShape(b, h * w, outDim), DType.F32);
        backend.Linear(o, v, weight, bias);
        return o.Reshape(new TensorShape(b, h, w, outDim));
    }

    /// <summary>Stride-2 max-pool of a channels-last <c>[B,H,W,C]</c> tensor; updates <paramref name="h"/>/<paramref name="w"/>.</summary>
    private static Tensor MaxPoolNhwc(IBackend backend, Tensor x, int c, ref int h, ref int w)
    {
        int b = (int)x.Shape[0];
        Tensor nchw = HieraImageEncoder.NhwcToNchw(backend, x, b, h, w, c);
        int oH = (h - 2) / 2 + 1;
        int oW = (w - 2) / 2 + 1;
        Tensor pooled = new Tensor(new TensorShape(b, c, oH, oW), DType.F32);
        backend.MaxPool2D(pooled, nchw, 2, 2, 2, 2, 0, 0);
        h = oH;
        w = oW;
        return HieraImageEncoder.NchwToNhwc(backend, pooled, b, c, oH, oW);
    }

    /// <summary>Partitions <c>[B,H,W,C]</c> into <c>[B·nH·nW, ws, ws, C]</c> windows, zero-padding the bottom/right
    /// edges to a multiple of <paramref name="ws"/>. Host layout shuffle (no math).</summary>
    private static Tensor WindowPartition(Tensor x, int b, int h, int w, int ws, int c)
    {
        int padH = (ws - h % ws) % ws;
        int padW = (ws - w % ws) % ws;
        int hp = h + padH;
        int wp = w + padW;
        int nH = hp / ws;
        int nW = wp / ws;
        Tensor outT = new Tensor(new TensorShape(b * nH * nW, ws, ws, c), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ph = 0; ph < hp; ph++)
            {
                for (int pw = 0; pw < wp; pw++)
                {
                    int winIdx = bi * nH * nW + (ph / ws) * nW + (pw / ws);
                    long dOff = (((long)winIdx * ws + ph % ws) * ws + pw % ws) * c;
                    if (ph < h && pw < w)
                    {
                        long sOff = (((long)bi * h + ph) * w + pw) * c;
                        for (int ch = 0; ch < c; ch++) dst[dOff + ch] = src[sOff + ch];
                    }
                    else
                    {
                        for (int ch = 0; ch < c; ch++) dst[dOff + ch] = 0f;
                    }
                }
            }
        }
        return outT;
    }

    /// <summary>Inverse of <see cref="WindowPartition"/>: gathers <c>[B·nH·nW, ws, ws, C]</c> windows back into
    /// <c>[B,H,W,C]</c>, cropping the padding.</summary>
    private static Tensor WindowUnpartition(Tensor windows, int b, int ws, int hp, int wp, int h, int w, int c)
    {
        int nH = hp / ws;
        int nW = wp / ws;
        Tensor outT = new Tensor(new TensorShape(b, h, w, c), DType.F32);
        float* src = (float*)windows.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int ih = 0; ih < h; ih++)
            {
                for (int iw = 0; iw < w; iw++)
                {
                    int winIdx = bi * nH * nW + (ih / ws) * nW + (iw / ws);
                    long sOff = (((long)winIdx * ws + ih % ws) * ws + iw % ws) * c;
                    long dOff = (((long)bi * h + ih) * w + iw) * c;
                    for (int ch = 0; ch < c; ch++) dst[dOff + ch] = src[sOff + ch];
                }
            }
        }
        return outT;
    }

    private static Tensor F32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
            throw new KeyNotFoundException($"HieraBlock: missing weight '{key}'.");
        return t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
    }
}
