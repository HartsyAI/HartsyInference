using System.Buffers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Rope;

/// <summary>Per-token, per-pair cos/sin from a <see cref="RopeFrequencyPlan"/>, and the full rows the apply ops read.</summary>
/// <remarks>Positions are token-major with one coordinate per axis. Coordinate rules (centering, offsets, zeroed axes)
/// belong to the model; this type takes the final coordinates as given.</remarks>
public static class MultiAxisRopeTables
{
    /// <summary>Writes cos and sin for <paramref name="tokens"/> rows of <c>plan.Pairs</c> values each.</summary>
    public static void ComputePairs(
        RopeFrequencyPlan plan,
        ReadOnlySpan<float> positions,
        int tokens,
        RopeAngleOrder order,
        Span<float> cos,
        Span<float> sin)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ValidateTokens(tokens);
        int axes = plan.Axes;
        int pairs = plan.Pairs;
        if (positions.Length != (long)tokens * axes)
            throw new ArgumentException($"Positions must hold {tokens} tokens of {axes} axes.", nameof(positions));
        long tableLength = (long)tokens * pairs;
        if (cos.Length != tableLength || sin.Length != tableLength)
            throw new ArgumentException($"Tables must hold {tokens} tokens of {pairs} pairs.", nameof(cos));
        for (int i = 0; i < positions.Length; i++)
        {
            if (!float.IsFinite(positions[i]))
                throw new ArgumentException("RoPE positions must be finite.", nameof(positions));
        }

        ReadOnlySpan<int> axisDims = plan.AxisDims;
        ReadOnlySpan<int> offsets = plan.AxisPairOffsets;
        ReadOnlySpan<double> inverse = plan.InverseFrequencies;
        ReadOnlySpan<double> denominators = plan.Denominators;
        bool multiply = order == RopeAngleOrder.MultiplyByInverse;
        for (int s = 0; s < tokens; s++)
        {
            int rowBase = s * pairs;
            int positionBase = s * axes;
            for (int a = 0; a < axes; a++)
            {
                double position = positions[positionBase + a];
                int pairStart = offsets[a];
                int pairCount = axisDims[a] / 2;
                for (int k = 0; k < pairCount; k++)
                {
                    int index = pairStart + k;
                    double angle = multiply ? position * inverse[index] : position / denominators[index];
                    cos[rowBase + index] = (float)Math.Cos(angle);
                    sin[rowBase + index] = (float)Math.Sin(angle);
                }
            }
        }
    }

    /// <summary>Computes per-pair values, keeps the first <c>effectiveRotaryDim / 2</c> pairs of each token, and expands
    /// them into full rows in <paramref name="layout"/>.</summary>
    public static void BuildRows(
        RopeFrequencyPlan plan,
        ReadOnlySpan<float> positions,
        int tokens,
        RopeAngleOrder order,
        RopeTableLayout layout,
        int rotaryDim,
        RopeTablePadding padding,
        Span<float> cosRows,
        Span<float> sinRows)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ValidateTokens(tokens);
        int effective = RopeTableExpander.EffectiveRotaryDim(plan.HeadDim, rotaryDim, layout);
        (float[] cosActive, float[] sinActive) = ComputeActivePairs(plan, positions, tokens, order, effective / 2);
        RopeTableExpander.Expand(cosActive, sinActive, tokens, plan.HeadDim, layout, rotaryDim, padding, cosRows, sinRows);
    }

    /// <summary>Allocates <c>[tokens, headDim]</c> F32 cos and sin rows. The caller owns and disposes both tensors.</summary>
    public static (Tensor Cos, Tensor Sin) BuildTensors(
        RopeFrequencyPlan plan,
        ReadOnlySpan<float> positions,
        int tokens,
        RopeAngleOrder order,
        RopeTableLayout layout,
        int rotaryDim,
        RopeTablePadding padding)
    {
        ArgumentNullException.ThrowIfNull(plan);
        if (tokens == 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "Tensor storage needs at least one token; use BuildRows for empty tables.");
        int headDim = plan.HeadDim;
        int count = checked(tokens * headDim);
        Tensor cos = new(new TensorShape(tokens, headDim), DType.F32);
        Tensor? sin = null;
        bool succeeded = false;
        try
        {
            sin = new Tensor(new TensorShape(tokens, headDim), DType.F32);
            unsafe
            {
                BuildRows(plan, positions, tokens, order, layout, rotaryDim, padding,
                    new Span<float>(cos.DataPointer, count), new Span<float>(sin.DataPointer, count));
            }
            succeeded = true;
            return (cos, sin);
        }
        finally
        {
            if (!succeeded)
            {
                cos.Dispose();
                sin?.Dispose();
            }
        }
    }

    /// <summary>Per-pair cos and sin for the first <paramref name="activePairs"/> pairs of each token, as compact arrays.</summary>
    internal static (float[] Cos, float[] Sin) ComputeActivePairs(
        RopeFrequencyPlan plan,
        ReadOnlySpan<float> positions,
        int tokens,
        RopeAngleOrder order,
        int activePairs)
    {
        int pairs = plan.Pairs;
        int fullLength = checked(tokens * pairs);
        float[] cosFull = ArrayPool<float>.Shared.Rent(fullLength);
        float[] sinFull = ArrayPool<float>.Shared.Rent(fullLength);
        try
        {
            ComputePairs(plan, positions, tokens, order, cosFull.AsSpan(0, fullLength), sinFull.AsSpan(0, fullLength));
            float[] cos = new float[checked(tokens * activePairs)];
            float[] sin = new float[cos.Length];
            for (int s = 0; s < tokens; s++)
            {
                cosFull.AsSpan(s * pairs, activePairs).CopyTo(cos.AsSpan(s * activePairs, activePairs));
                sinFull.AsSpan(s * pairs, activePairs).CopyTo(sin.AsSpan(s * activePairs, activePairs));
            }
            return (cos, sin);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(cosFull);
            ArrayPool<float>.Shared.Return(sinFull);
        }
    }

    private static void ValidateTokens(int tokens)
    {
        if (tokens < 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "Token count cannot be negative.");
    }
}
