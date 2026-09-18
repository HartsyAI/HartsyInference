using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Lora;

/// <summary>Applies a DoRA magnitude vector to an in-flight merge, following ComfyUI's <c>weight_adapter/base.py::weight_decompose</c> exactly. DoRA splits a weight into direction and magnitude: the LoRA moves the direction, the stored vector restores the magnitude, so the delta cannot simply be added.</summary>
/// <remarks><para>Two branches, selected the way ComfyUI selects them — by whether the magnitude vector's length
/// matches the weight's OUTPUT dimension. On that branch the normalizer is the row norm of the ORIGINAL weight, not
/// of the LoRA'd weight; on the other it is the column norm of the LoRA'd weight. Neither is the plain
/// <c>‖W + ΔW‖_row</c> the DoRA paper writes, and the difference is visible in the output, so both are transcribed
/// rather than simplified.</para>
/// <para>Strength scales the DECOMPOSED result, not the delta: at strength 1 the weight is replaced outright, and
/// below it the merge interpolates between the original weight and the fully decomposed one.</para></remarks>
public static class LoraDoraDecompose
{
    /// <summary>Machine epsilon for F32 — <c>torch.finfo(torch.float32).eps</c>. Not <see cref="float.Epsilon"/>, which is the smallest denormal and would leave a zero-norm row dividing by ~0.</summary>
    private const float F32Epsilon = 1.1920929e-7f;

    /// <summary>Rewrites <paramref name="weightF32"/> in place from <c>W</c> to the DoRA-decomposed merge of <c>W</c> and <c>scale · ΔW</c>.</summary>
    /// <param name="weightF32">The base weight as an owned F32 <c>[out, in]</c> accumulator, flattened if the target is a conv weight. Holds <c>W</c> on entry and the merged result on exit.</param>
    /// <param name="deltaF32">The unscaled delta from <see cref="LoraDelta.ComputeF32"/>, same shape. Overwritten.</param>
    /// <param name="doraScale">The file's magnitude vector; its element count must equal either the row count or the column count of <paramref name="weightF32"/>.</param>
    /// <param name="scale">The decomposition's <see cref="LoraDelta.Scale"/> (alpha / rank).</param>
    /// <param name="strength">The user's LoRA strength.</param>
    public static void Apply(Tensor weightF32, Tensor deltaF32, Tensor doraScale, float scale, float strength)
    {
        ArgumentNullException.ThrowIfNull(weightF32);
        ArgumentNullException.ThrowIfNull(deltaF32);
        ArgumentNullException.ThrowIfNull(doraScale);
        if (weightF32.Shape.Rank != 2 || weightF32.DType != DType.F32)
            throw new HartsyInferenceException(
                $"DoRA needs a rank-2 F32 weight accumulator, got {weightF32.Shape} ({weightF32.DType.Name}).");
        if (deltaF32.Shape != weightF32.Shape || deltaF32.DType != DType.F32)
            throw new HartsyInferenceException(
                $"DoRA delta {deltaF32.Shape} ({deltaF32.DType.Name}) does not match the weight {weightF32.Shape}.");

        long rows = weightF32.Shape[0];
        long columns = weightF32.Shape[1];
        long magnitudes = doraScale.Shape.ElementCount;
        if (magnitudes != rows && magnitudes != columns)
            throw new HartsyInferenceException(
                $"DoRA magnitude vector has {magnitudes} entries, which matches neither the {rows} rows nor the "
                + $"{columns} columns of the weight it scales.");

        using Tensor magnitudeF32 = doraScale.CastTo(DType.F32);
        ReadOnlySpan<float> magnitude = magnitudeF32.AsReadOnlySpan<float>();
        Span<float> weight = weightF32.AsSpan<float>();
        Span<float> delta = deltaF32.AsSpan<float>();

        // ComfyUI selects on the magnitude vector's LEADING dimension, so a [rows, 1] vector takes the output axis
        // and a [1, columns] one takes the input axis. Row count wins a tie, matching the shape[0] comparison there.
        if (magnitudes == rows)
        {
            ApplyOnOutputAxis(weight, delta, magnitude, rows, columns, scale, strength);
        }
        else
        {
            ApplyOnInputAxis(weight, delta, magnitude, rows, columns, scale, strength);
        }
    }

    /// <summary>The <c>wd_on_output_axis</c> branch: each row is renormalized by the ORIGINAL weight row's norm.</summary>
    private static void ApplyOnOutputAxis(Span<float> weight, Span<float> delta, ReadOnlySpan<float> magnitude,
        long rows, long columns, float scale, float strength)
    {
        for (long row = 0; row < rows; row++)
        {
            long rowBase = row * columns;
            double sumOfSquares = 0.0;
            for (long column = 0; column < columns; column++)
            {
                float value = weight[(int)(rowBase + column)];
                sumOfSquares += (double)value * value;
            }
            float norm = (float)Math.Sqrt(sumOfSquares) + F32Epsilon;
            float factor = magnitude[(int)row] / norm;
            for (long column = 0; column < columns; column++)
            {
                int index = (int)(rowBase + column);
                float original = weight[index];
                float decomposed = (original + delta[index] * scale) * factor;
                weight[index] = strength == 1.0f ? decomposed : original + strength * (decomposed - original);
            }
        }
    }

    /// <summary>The other branch: each column is renormalized by the LoRA'd weight column's norm, so the scaled delta has to be materialized first.</summary>
    private static void ApplyOnInputAxis(Span<float> weight, Span<float> delta, ReadOnlySpan<float> magnitude,
        long rows, long columns, float scale, float strength)
    {
        for (int index = 0; index < delta.Length; index++)
        {
            delta[index] = weight[index] + delta[index] * scale;
        }
        for (long column = 0; column < columns; column++)
        {
            double sumOfSquares = 0.0;
            for (long row = 0; row < rows; row++)
            {
                float value = delta[(int)(row * columns + column)];
                sumOfSquares += (double)value * value;
            }
            float norm = (float)Math.Sqrt(sumOfSquares) + F32Epsilon;
            float factor = magnitude[(int)column] / norm;
            for (long row = 0; row < rows; row++)
            {
                int index = (int)(row * columns + column);
                float original = weight[index];
                float decomposed = delta[index] * factor;
                weight[index] = strength == 1.0f ? decomposed : original + strength * (decomposed - original);
            }
        }
    }
}
