using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Video.Tokenizers;

/// <summary>Shared building blocks of the Cosmos-Tokenize1-DV8x16x16 conv autoencoder — used by both
/// <see cref="CosmosDvEncoder"/> and <see cref="CosmosDvDecoder"/>. Holds the F32 weight dict and provides the
/// factorized causal convolutions (spatial <c>1×3×3</c> same-pad + temporal <c>3×1×1</c> causal replicate-2, via
/// <see cref="CausalConv3d"/> / <see cref="IBackend.Conv2D"/>), GroupNorm(num_groups=1, per frame), residual blocks,
/// and the spatial / causal-temporal self-attention. Norm and attention run host-side in F32 (the tokenizer is a
/// once-per-clip path — determinism is preferred). B=1.</summary>
internal abstract unsafe class CosmosDvModule : IDisposable
{
    protected const float Wav = 0.70703125f;      // BF16(1/√2) cast to F32 — the exact value in the .jit
    protected const float GnEps = 1e-6f;
    protected readonly Dictionary<string, Tensor> W;
    private readonly Dictionary<string, CausalConv3d> _convs = new();
    private int _disposed;

    protected CosmosDvModule(Dictionary<string, Tensor> weightsF32) => W = weightsF32;

    // ── conv builders (cached) ─────────────────────────────────────────────
    protected CausalConv3d Conv(string name, int sT, int sH, int sW, int padT, int padH, int padW, bool replicate)
    {
        string key = $"{name}|{sT},{sH},{sW}|{padT},{padH},{padW}|{replicate}";
        if (_convs.TryGetValue(key, out CausalConv3d? c)) return c;
        Tensor w = W[name + ".weight"];
        W.TryGetValue(name + ".bias", out Tensor? b);
        c = new CausalConv3d(w, b, sT, sH, sW, padT, padH, padW, replicateFirstPad: replicate, causal: true);
        _convs[key] = c;
        return c;
    }

    protected Tensor SpatialConv(IBackend backend, Tensor x, string name)   // 1×3×3 same-pad
        => Conv(name, 1, 1, 1, 0, 1, 1, false).Forward(backend, x);
    protected Tensor TemporalConv(IBackend backend, Tensor x, string name)  // 3×1×1 causal replicate-2
        => Conv(name, 1, 1, 1, 1, 0, 0, true).Forward(backend, x);
    protected Tensor OneXOne(IBackend backend, Tensor x, string name)       // 1×1×1
        => Conv(name, 1, 1, 1, 0, 0, 0, false).Forward(backend, x);

    protected Tensor FactConv(IBackend backend, Tensor x, string baseName)
    {
        Tensor s = SpatialConv(backend, x, baseName + ".0.conv3d");
        Tensor t = TemporalConv(backend, s, baseName + ".1.conv3d");
        s.Dispose();
        return t;
    }

    protected static (int B, int C, int T, int H, int W) Dims(Tensor t)
        => ((int)t.Shape[0], (int)t.Shape[1], (int)t.Shape[2], (int)t.Shape[3], (int)t.Shape[4]);

    /// <summary>GroupNorm(num_groups=1) applied per frame (time folded into batch), optional SiLU. Two-pass
    /// mean/variance, matching <c>F.group_norm</c>.</summary>
    protected Tensor GroupNorm(Tensor x, string normName, bool silu)
    {
        (int b, int c, int t, int h, int w) = Dims(x);
        float* wt = (float*)W[normName + ".norm.weight"].DataPointer;
        float* bs = (float*)W[normName + ".norm.bias"].DataPointer;
        Tensor outT = new(x.Shape, DType.F32);
        float* sp = (float*)x.DataPointer, op = (float*)outT.DataPointer;
        long hw = (long)h * w, n = (long)c * hw;
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
            {
                double mean = 0;
                for (int ci = 0; ci < c; ci++)
                {
                    long baseOff = (((long)bi * c + ci) * t + ti) * hw;
                    for (long p = 0; p < hw; p++) mean += sp[baseOff + p];
                }
                mean /= n;
                double var = 0;
                for (int ci = 0; ci < c; ci++)
                {
                    long baseOff = (((long)bi * c + ci) * t + ti) * hw;
                    for (long p = 0; p < hw; p++) { double d = sp[baseOff + p] - mean; var += d * d; }
                }
                var /= n;
                float inv = (float)(1.0 / Math.Sqrt(var + GnEps));
                float fmean = (float)mean;
                for (int ci = 0; ci < c; ci++)
                {
                    long baseOff = (((long)bi * c + ci) * t + ti) * hw;
                    float g = wt[ci], be = bs[ci];
                    for (long p = 0; p < hw; p++)
                    {
                        float v = (sp[baseOff + p] - fmean) * inv * g + be;
                        op[baseOff + p] = silu ? v * (1f / (1f + MathF.Exp(-v))) : v;
                    }
                }
            }
        return outT;
    }

    protected Tensor ResBlock(IBackend backend, Tensor x, string baseName)
    {
        Tensor h = GroupNorm(x, baseName + ".norm1", silu: true);
        Tensor c1 = FactConv(backend, h, baseName + ".conv1"); h.Dispose();
        Tensor h2 = GroupNorm(c1, baseName + ".norm2", silu: true); c1.Dispose();
        Tensor c2 = FactConv(backend, h2, baseName + ".conv2"); h2.Dispose();
        Tensor shortcut = W.ContainsKey(baseName + ".nin_shortcut.conv3d.weight")
            ? OneXOne(backend, x, baseName + ".nin_shortcut.conv3d") : x.To(x.Device);
        AddInPlace(shortcut, c2); c2.Dispose();
        return shortcut;
    }

    /// <summary>Spatial self-attention (per frame, over H·W), scale <c>C^-0.5</c>. Host F32.</summary>
    protected Tensor SpatialAttn(IBackend backend, Tensor x, string baseName)
    {
        Tensor hn = GroupNorm(x, baseName + ".norm", silu: false);
        Tensor q = OneXOne(backend, hn, baseName + ".q.conv3d");
        Tensor k = OneXOne(backend, hn, baseName + ".k.conv3d");
        Tensor v = OneXOne(backend, hn, baseName + ".v.conv3d");
        hn.Dispose();
        (int b, int c, int t, int h, int w) = Dims(q);
        int hwn = h * w;
        float scale = 1f / MathF.Sqrt(c);
        Tensor outAttn = new(q.Shape, DType.F32);
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer, op = (float*)outAttn.DataPointer;
        float[] scores = new float[hwn];
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
                for (int i = 0; i < hwn; i++)
                {
                    float mx = float.NegativeInfinity;
                    for (int j = 0; j < hwn; j++)
                    {
                        float s = 0f;
                        for (int ci = 0; ci < c; ci++)
                        {
                            long off = (((long)bi * c + ci) * t + ti) * hwn;
                            s += qp[off + i] * kp[off + j];
                        }
                        s *= scale; scores[j] = s; if (s > mx) mx = s;
                    }
                    float sum = 0f;
                    for (int j = 0; j < hwn; j++) { float e = MathF.Exp(scores[j] - mx); scores[j] = e; sum += e; }
                    float invSum = 1f / sum;
                    for (int ci = 0; ci < c; ci++)
                    {
                        long off = (((long)bi * c + ci) * t + ti) * hwn;
                        float acc = 0f;
                        for (int j = 0; j < hwn; j++) acc += vp[off + j] * scores[j];
                        op[off + i] = acc * invSum;
                    }
                }
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor proj = OneXOne(backend, outAttn, baseName + ".proj_out.conv3d");
        outAttn.Dispose();
        AddInPlace(proj, x);
        return proj;
    }

    /// <summary>Causal temporal self-attention (per spatial location, over T with a lower-triangular mask),
    /// scale <c>C^-0.5</c>. Host F32.</summary>
    protected Tensor TemporalAttn(IBackend backend, Tensor x, string baseName)
    {
        Tensor hn = GroupNorm(x, baseName + ".norm", silu: false);
        Tensor q = OneXOne(backend, hn, baseName + ".q.conv3d");
        Tensor k = OneXOne(backend, hn, baseName + ".k.conv3d");
        Tensor v = OneXOne(backend, hn, baseName + ".v.conv3d");
        hn.Dispose();
        (int b, int c, int t, int h, int w) = Dims(q);
        int hwn = h * w;
        float scale = 1f / MathF.Sqrt(c);
        Tensor outAttn = new(q.Shape, DType.F32);
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer, op = (float*)outAttn.DataPointer;
        float[] scores = new float[t];
        for (int bi = 0; bi < b; bi++)
            for (int p = 0; p < hwn; p++)
                for (int i = 0; i < t; i++)
                {
                    float mx = float.NegativeInfinity;
                    for (int j = 0; j <= i; j++)
                    {
                        float s = 0f;
                        for (int ci = 0; ci < c; ci++)
                        {
                            long off = (((long)bi * c + ci) * t) * hwn + p;
                            s += qp[off + (long)i * hwn] * kp[off + (long)j * hwn];
                        }
                        s *= scale; scores[j] = s; if (s > mx) mx = s;
                    }
                    float sum = 0f;
                    for (int j = 0; j <= i; j++) { float e = MathF.Exp(scores[j] - mx); scores[j] = e; sum += e; }
                    float invSum = 1f / sum;
                    for (int ci = 0; ci < c; ci++)
                    {
                        long off = (((long)bi * c + ci) * t) * hwn + p;
                        float acc = 0f;
                        for (int j = 0; j <= i; j++) acc += vp[off + (long)j * hwn] * scores[j];
                        op[off + (long)i * hwn] = acc * invSum;
                    }
                }
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor proj = OneXOne(backend, outAttn, baseName + ".proj_out.conv3d");
        outAttn.Dispose();
        AddInPlace(proj, x);
        return proj;
    }

    // ── small tensor utilities ─────────────────────────────────────────────
    protected static void AddInPlace(Tensor acc, Tensor add)
    {
        long n = acc.Shape.ElementCount;
        float* a = (float*)acc.DataPointer, b = (float*)add.DataPointer;
        for (long i = 0; i < n; i++) a[i] += b[i];
    }

    protected static float Sgn(int band, int i) => band == 0 ? 1f : (i == 0 ? 1f : -1f);   // lo=[+,+], hi=[+,-]

    public virtual void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _convs.Clear();
    }
}
