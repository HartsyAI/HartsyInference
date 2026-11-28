using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The BF16 arithmetic of LongCat's projections and attention on the BF16 route, computed apart from the shared kernels: the base product,
/// the LoRA branch (each of its linears, the scale, and the sum) and the final sum each round to BF16, as upstream's BF16 LoRA forward does; the
/// QK norm rounds its unit vector and then its weight multiply; and the attention runs head-major with its inputs and output on the BF16 grid.
/// CPU only: the split and the norm read their operands on the host. A weight without an active adjunct, and any non-CPU weight, takes the shared
/// linear and rounds its output.</summary>
internal static class LongCatSplitLinear
{
    /// <summary><c>bf(bf(x·Wᵀ + b) + bf(branch))</c> over the rows <c>[rowOffset, rowOffset + rowCount)</c> of <paramref name="weight"/> (all rows
    /// by default), the branch being <c>bf(bf(bf(x·Downᵀ)·Upᵀ)·scale)</c> summed over the active terms and rounded once. Without <paramref name="bf16"/>,
    /// the shared linear. A new <c>[rows, count]</c> F32 tensor, owned by the caller.</summary>
    public static unsafe Tensor Linear(IBackend backend, Tensor input, Tensor weight, Tensor? bias, bool bf16, int rowOffset = 0, int rowCount = -1)
    {
        int rows = (int)input.Shape[0], total = (int)weight.Shape[0];
        int count = rowCount < 0 ? total : rowCount;
        bool windowed = rowOffset != 0 || count != total;
        LowRankAdjunct? adjunct = weight.LowRankAdjunct;
        bool split = bf16 && weight.Device.IsCpu && adjunct is { IsActive: true };
        if (!split)
        {
            Tensor plain = new(new TensorShape(rows, count), DType.F32);
            if (windowed)
                backend.LinearWeightRows(plain, input, weight, bias, rowOffset, count);
            else
                backend.Linear(plain, input, weight, bias);
            return bf16 ? LongCatOps.RoundOwned(backend, plain, on: true) : plain;
        }

        // The base product: the weight's rows without the adjunct, as a borrowed view of the same bytes (the weight owns them).
        long rowBytes = (long)weight.Shape[1] * weight.DType.SizeInBytes;
        byte* rowStart = (byte*)weight.DataPointer + rowOffset * rowBytes;
        using Tensor baseWeight = new(rowStart, new TensorShape(count, weight.Shape[1]), weight.DType, weight.Device);
        Tensor? baseBias = null;
        if (bias is not null)
        {
            byte* biasStart = (byte*)bias.DataPointer + rowOffset * bias.DType.SizeInBytes;
            baseBias = new Tensor(biasStart, new TensorShape(count), bias.DType, bias.Device);
        }
        try
        {
            Tensor baseRaw = new(new TensorShape(rows, count), DType.F32);
            backend.Linear(baseRaw, input, baseWeight, baseBias);
            using Tensor baseBf = LongCatOps.RoundOwned(backend, baseRaw, on: true);

            LowRankAdjunct window = windowed ? adjunct!.SliceRows(rowOffset, count) : adjunct!;
            Tensor? branch = null;
            foreach (LowRankAdjunctTerm term in window.Terms)
            {
                if (!term.IsApplied)
                    continue;
                Tensor downRaw = new(new TensorShape(rows, term.Down.Shape[0]), DType.F32);
                backend.Linear(downRaw, input, term.Down, null);
                Tensor down = LongCatOps.RoundOwned(backend, downRaw, on: true);
                Tensor up = down;
                if (term.Up is not null)
                {
                    Tensor upRaw = new(new TensorShape(rows, term.Up.Shape[0]), DType.F32);
                    backend.Linear(upRaw, down, term.Up, null);
                    up = LongCatOps.RoundOwned(backend, upRaw, on: true);
                }
                Tensor scaled = Scale(up, term.EffectiveScale);
                up.Dispose();
                if (branch is null)
                {
                    branch = scaled;
                }
                else
                {
                    Tensor summed = Add(branch, scaled);
                    branch.Dispose();
                    scaled.Dispose();
                    branch = summed;
                }
            }
            if (branch is null)
                return baseBf.To(baseBf.Device);
            using Tensor branchBf = LongCatOps.RoundOwned(backend, branch, on: true);
            Tensor total2 = Add(baseBf, branchBf);
            return LongCatOps.RoundOwned(backend, total2, on: true);
        }
        finally
        {
            baseBias?.Dispose();
        }
    }

    /// <summary>Per-head RMS norm on the BF16 route: the unit vector rounds, then the weight multiply rounds, as upstream's norm does. Reads
    /// <paramref name="x"/> <c>[rows, heads·headDim]</c> and returns <c>[batch, rows / batch, heads, headDim]</c> F32 on the BF16 grid.</summary>
    public static Tensor HeadNorm(Tensor x, Tensor weight, int heads, float eps, int batch = 1)
    {
        int rows = (int)x.Shape[0], headDim = (int)weight.Shape[0];
        Tensor output = new(new TensorShape(batch, rows / batch, heads, headDim), DType.F32);
        ReadOnlySpan<float> source = x.AsReadOnlySpan<float>();
        ReadOnlySpan<float> w = weight.AsReadOnlySpan<float>();
        Span<float> destination = output.AsSpan<float>();
        for (int r = 0; r < rows; r++)
        {
            for (int h = 0; h < heads; h++)
            {
                int at = r * heads * headDim + h * headDim;
                double sum = 0;
                for (int d = 0; d < headDim; d++)
                    sum += (double)source[at + d] * source[at + d];
                float inv = (float)(1.0 / Math.Sqrt(sum / headDim + eps));
                for (int d = 0; d < headDim; d++)
                {
                    float unit = Bf16(source[at + d] * inv);
                    destination[at + d] = Bf16(unit * w[d]);
                }
            }
        }
        return output;
    }

    /// <summary>Attention on the BF16 route: each of <paramref name="q"/>, <paramref name="k"/> and <paramref name="v"/> (<c>[batch·S, heads·D]</c> F32 on the
    /// grid) is consumed. The query and the key take the two-step norm, then the rotation when <paramref name="rotate"/>, then the attention runs
    /// head-major on the layer's attention call. Returns the token-major output <c>[batch·Sq, heads·D]</c>, rounded.</summary>
    public static Tensor AttendNormed(IBackend backend, AttentionLayer layer, Tensor q, Tensor k, Tensor v, Tensor qNorm, Tensor kNorm, int batch,
        int heads, float eps, in AttentionCallContext context, bool rotate)
    {
        using Tensor queries = q;
        using Tensor keysRaw = k;
        using Tensor valuesRaw = v;
        int sq = (int)q.Shape[0] / batch, skv = (int)k.Shape[0] / batch, headDim = (int)qNorm.Shape[0];
        using Tensor qn = HeadNorm(q, qNorm, heads, eps, batch);
        using Tensor kn = HeadNorm(k, kNorm, heads, eps, batch);
        if (rotate)
            layer.RotateQk(backend, qn, kn, in context);
        using Tensor qMh = ToHeadMajor(qn, batch, sq, heads, headDim);
        using Tensor kMh = ToHeadMajor(kn, batch, skv, heads, headDim);
        using Tensor vMh = ToHeadMajor(valuesRaw, batch, skv, heads, headDim);
        using Tensor attended = layer.AttendHeadMajorOperands(backend, qMh, kMh, vMh, default, OperandOwnership.None);
        Tensor flat = FromHeadMajor(attended, batch, sq, heads, headDim);
        return LongCatOps.RoundOwned(backend, flat, on: true);
    }

    private static float Bf16(float value) => BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(value));

    /// <summary>Token-major <c>[batch, S, heads, D]</c> to head-major <c>[batch, heads, S, D]</c>, on the host.</summary>
    private static Tensor ToHeadMajor(Tensor tokenMajor, int batch, int s, int heads, int headDim)
    {
        Tensor output = new(new TensorShape(batch, heads, s, headDim), DType.F32);
        ReadOnlySpan<float> source = tokenMajor.AsReadOnlySpan<float>();
        Span<float> destination = output.AsSpan<float>();
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < s; t++)
                for (int h = 0; h < heads; h++)
                    source.Slice(((b * s + t) * heads + h) * headDim, headDim)
                        .CopyTo(destination.Slice(((b * heads + h) * s + t) * headDim, headDim));
        return output;
    }

    /// <summary>Head-major <c>[batch, heads, S, D]</c> back to token-major <c>[batch·S, heads·D]</c>, on the host.</summary>
    private static Tensor FromHeadMajor(Tensor headMajor, int batch, int s, int heads, int headDim)
    {
        Tensor output = new(new TensorShape(batch * s, heads * headDim), DType.F32);
        ReadOnlySpan<float> source = headMajor.AsReadOnlySpan<float>();
        Span<float> destination = output.AsSpan<float>();
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < s; t++)
                for (int h = 0; h < heads; h++)
                    source.Slice(((b * heads + h) * s + t) * headDim, headDim)
                        .CopyTo(destination.Slice(((b * s + t) * heads + h) * headDim, headDim));
        return output;
    }

    private static Tensor Scale(Tensor source, float factor)
    {
        Tensor output = new(source.Shape, DType.F32);
        ReadOnlySpan<float> a = source.AsReadOnlySpan<float>();
        Span<float> o = output.AsSpan<float>();
        for (int i = 0; i < a.Length; i++)
            o[i] = Bf16(a[i] * factor);
        return output;
    }

    private static Tensor Add(Tensor a, Tensor b)
    {
        Tensor output = new(a.Shape, DType.F32);
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> y = b.AsReadOnlySpan<float>();
        Span<float> o = output.AsSpan<float>();
        for (int i = 0; i < x.Length; i++)
            o[i] = x[i] + y[i];
        return output;
    }
}
