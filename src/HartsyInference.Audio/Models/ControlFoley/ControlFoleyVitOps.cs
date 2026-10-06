using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Pre-norm ViT primitives shared by the Synchformer and CAV-MAE-ST ports, on F32 <c>[B, N, D]</c> tensors.</summary>
internal static unsafe class ControlFoleyVitOps
{
    internal static Tensor LayerNorm(IBackend backend, Tensor x, Tensor weight, Tensor bias, float eps)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.LayerNorm(o, x, weight, bias, eps);
        return o;
    }

    /// <summary><c>fc2(gelu(fc1(x)))</c> with the exact erf GELU of <c>nn.GELU()</c>.</summary>
    internal static Tensor Mlp(IBackend backend, Tensor x, Tensor fc1W, Tensor fc1B, Tensor fc2W, Tensor fc2B)
    {
        using Tensor hidden = ControlFoleyOps.Linear(backend, x, fc1W, fc1B);
        using Tensor activated = new(hidden.Shape, DType.F32);
        backend.GeluErf(activated, hidden);
        return ControlFoleyOps.Linear(backend, activated, fc2W, fc2B);
    }

    /// <summary><c>a += b</c> for same-shaped tensors.</summary>
    internal static void AddInPlace(Tensor a, Tensor b)
    {
        if (a.ElementCount != b.ElementCount)
        {
            throw new ArgumentException($"Cannot add {b.Shape} to {a.Shape}.");
        }

        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer;
        for (long i = 0; i < a.ElementCount; i++)
        {
            ap[i] += bp[i];
        }
    }

    /// <summary>Splits a packed <c>[B, N, 3 * D]</c> projection (q, k, v blocks, head-major inside each) into three
    /// <c>[B, heads, N, headDim]</c> tensors.</summary>
    internal static (Tensor Q, Tensor K, Tensor V) SplitQkv(Tensor qkv, int heads)
    {
        int b = (int)qkv.Shape[0], n = (int)qkv.Shape[1], d = (int)qkv.Shape[2] / 3, hd = d / heads;
        TensorShape shape = new(b, heads, n, hd);
        Tensor q = new(shape, DType.F32), k = new(shape, DType.F32), v = new(shape, DType.F32);
        float* src = (float*)qkv.DataPointer;
        float*[] dst = [(float*)q.DataPointer, (float*)k.DataPointer, (float*)v.DataPointer];
        for (int bi = 0; bi < b; bi++)
        {
            for (int ti = 0; ti < n; ti++)
            {
                float* row = src + ((long)bi * n + ti) * 3 * d;
                for (int part = 0; part < 3; part++)
                {
                    for (int h = 0; h < heads; h++)
                    {
                        new ReadOnlySpan<float>(row + part * d + h * hd, hd)
                            .CopyTo(new Span<float>(dst[part] + (((long)bi * heads + h) * n + ti) * hd, hd));
                    }
                }
            }
        }

        return (q, k, v);
    }

    /// <summary>Merges <c>[B, heads, N, headDim]</c> attention output back to <c>[B, N, D]</c>.</summary>
    internal static Tensor MergeHeads(Tensor ctx)
    {
        int b = (int)ctx.Shape[0], heads = (int)ctx.Shape[1], n = (int)ctx.Shape[2], hd = (int)ctx.Shape[3];
        Tensor o = new(new TensorShape(b, n, heads * hd), DType.F32);
        float* src = (float*)ctx.DataPointer, dst = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int h = 0; h < heads; h++)
            {
                for (int ti = 0; ti < n; ti++)
                {
                    new ReadOnlySpan<float>(src + (((long)bi * heads + h) * n + ti) * hd, hd)
                        .CopyTo(new Span<float>(dst + ((long)bi * n + ti) * heads * hd + h * hd, hd));
                }
            }
        }

        return o;
    }

    /// <summary>Multi-head self-attention over <c>[B, N, D]</c> with a packed qkv projection and an output projection.</summary>
    internal static Tensor SelfAttention(IBackend backend, Tensor x, Tensor qkvW, Tensor qkvB, Tensor projW, Tensor projB, int heads)
    {
        using Tensor qkv = ControlFoleyOps.Linear(backend, x, qkvW, qkvB);
        (Tensor q, Tensor k, Tensor v) = SplitQkv(qkv, heads);
        using Tensor qd = q, kd = k, vd = v;
        using Tensor ctx = new(q.Shape, DType.F32);
        backend.ScaledDotProductAttention(ctx, q, k, v, null, 1f / MathF.Sqrt((float)q.Shape[3]));
        using Tensor merged = MergeHeads(ctx);
        return ControlFoleyOps.Linear(backend, merged, projW, projB);
    }

    /// <summary>Fetches <paramref name="key"/> as an F32 (or kept-BF16) weight after checking its shape.</summary>
    internal static Tensor Load(IReadOnlyDictionary<string, Tensor> weights, string key, List<Tensor> owned, params long[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
        {
            throw new KeyNotFoundException($"ControlFoley video checkpoint is missing '{key}'.");
        }

        if (t.Shape.Rank != shape.Length)
        {
            throw new InvalidDataException($"'{key}' has rank {t.Shape.Rank}, expected {shape.Length}.");
        }

        for (int i = 0; i < shape.Length; i++)
        {
            if (t.Shape[i] != shape[i])
            {
                throw new InvalidDataException($"'{key}' dimension {i} is {t.Shape[i]}, expected {shape[i]}.");
            }
        }

        return ControlFoleyOps.PrepareWeight(t, owned);
    }
}
