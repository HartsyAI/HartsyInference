using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Sparse;

/// <summary>Per-token mixing of attention heads before the output projection, <c>o[h] &lt;- sum_g m[h,g] * o[g]</c> (ERNIE VHA post-mix).</summary>
/// <remarks>The checkpoint stores the mix as a low-rank update, <c>m = I + V @ U^T</c> with <c>U, V</c> of shape <c>[heads, rank]</c>.
/// <see cref="FoldLowRank"/> folds it into a dense <c>[heads, heads]</c> matrix once at load, as the reference's <c>fold_vha</c> does, so the
/// forward pays one <c>heads x heads</c> product per token instead of two rank-sized ones.</remarks>
public static class HeadMix
{
    /// <summary>Dense <c>m = I + V @ U^T</c>, row-major <c>[heads, heads]</c>, each element accumulated in double.</summary>
    /// <param name="u">The checkpoint's <c>U</c>, <c>[heads, rank]</c>.</param>
    /// <param name="v">The checkpoint's <c>V</c>, <c>[heads, rank]</c>.</param>
    /// <param name="roundToBf16">Rounds each element to BF16 (nearest even), as a fold stored in the BF16 weights' dtype is.</param>
    public static float[] FoldLowRank(ReadOnlySpan<float> u, ReadOnlySpan<float> v, int heads, int rank, bool roundToBf16)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(heads, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(rank, 1);
        if (u.Length != (long)heads * rank || v.Length != u.Length) throw new ArgumentException("U and V must each hold heads x rank values.");
        float[] m = new float[heads * heads];
        for (int h = 0; h < heads; h++)
            for (int g = 0; g < heads; g++)
            {
                double sum = h == g ? 1.0 : 0.0;
                for (int r = 0; r < rank; r++) sum += (double)v[h * rank + r] * u[g * rank + r];
                m[h * heads + g] = roundToBf16 ? RoundBf16((float)sum) : (float)sum;
            }
        return m;
    }

    /// <summary>Mixes head-major values: <c>output[h, :] = sum_g mix[h, g] * input[g, :]</c> through <see cref="IBackend.MatMul"/>.</summary>
    /// <param name="mix">F32 <c>[heads, heads]</c> from <see cref="FoldLowRank"/>.</param>
    /// <param name="input">F32 <c>[heads, n]</c>: every token's values for one head in a row.</param>
    /// <param name="output">F32 <c>[heads, n]</c>; must not alias <paramref name="input"/>.</param>
    public static void Apply(IBackend backend, Tensor mix, Tensor input, Tensor output)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (mix.Shape.Rank != 2 || mix.Shape[0] != mix.Shape[1] || input.Shape.Rank != 2 || input.Shape[0] != mix.Shape[0])
            throw new ArgumentException($"HeadMix needs mix [H,H] and input [H,n]; got {mix.Shape} and {input.Shape}.");
        if (ReferenceEquals(input, output)) throw new ArgumentException("HeadMix output must not alias its input.", nameof(output));
        backend.MatMul(output, mix, input);
    }

    private static float RoundBf16(float value)
    {
        uint u = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value)) return value;
        return BitConverter.UInt32BitsToSingle(((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16) << 16);
    }
}
