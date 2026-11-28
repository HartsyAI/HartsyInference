using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The BF16 arithmetic of a LongCat projection on the BF16 route, computed apart from the shared linear: the base product, the LoRA branch
/// (each of its linears, the scale and the sum) and the final sum each round to BF16, as upstream's BF16 LoRA forward does. CPU only: the base and
/// the branch are read on the host here. A weight without an active adjunct, and any non-CPU weight, takes the shared linear unchanged.</summary>
internal static class LongCatSplitLinear
{
    /// <summary><c>bf(bf(x·Wᵀ + b) + bf(branch))</c>, the branch being <c>bf(bf(bf(x·Downᵀ)·Upᵀ)·scale)</c> summed over the active terms and rounded
    /// once. A new <c>[rows, out]</c> F32 tensor, owned by the caller.</summary>
    public static Tensor Linear(IBackend backend, Tensor input, Tensor weight, Tensor? bias, bool bf16)
    {
        int rows = (int)input.Shape[0], outDim = (int)weight.Shape[0];
        LowRankAdjunct? adjunct = weight.LowRankAdjunct;
        bool split = bf16 && weight.Device.IsCpu && adjunct is { IsActive: true };
        if (!split)
        {
            Tensor plain = new(new TensorShape(rows, outDim), DType.F32);
            backend.Linear(plain, input, weight, bias);
            return bf16 ? LongCatOps.RoundOwned(backend, plain, on: true) : plain;
        }

        // The base product: the weight without its adjunct (a borrowed view of the same bytes; the weight owns them).
        unsafe
        {
            using Tensor baseWeight = new((void*)weight.DataPointer, weight.Shape, weight.DType, weight.Device);
            Tensor baseRaw = new(new TensorShape(rows, outDim), DType.F32);
            backend.Linear(baseRaw, input, baseWeight, bias);
            using Tensor baseBf = LongCatOps.RoundOwned(backend, baseRaw, on: true);

            Tensor? branch = null;
            foreach (LowRankAdjunctTerm term in adjunct!.Terms)
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
            Tensor total = Add(baseBf, branchBf);
            return LongCatOps.RoundOwned(backend, total, on: true);
        }
    }

    /// <summary>Per-head RMS norm on the BF16 route: the unit vector rounds, then the weight multiply rounds, as upstream's norm does. Reads
    /// <paramref name="x"/> <c>[rows, heads·headDim]</c> and returns <c>[rows, heads, headDim]</c> F32 on the BF16 grid.</summary>
    public static Tensor HeadNorm(Tensor x, Tensor weight, int heads, float eps)
    {
        int rows = (int)x.Shape[0], headDim = (int)weight.Shape[0];
        Tensor output = new(new TensorShape(rows, heads, headDim), DType.F32);
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

    private static float Bf16(float value) => BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(value));

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
