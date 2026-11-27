using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Core.Backends.Host;

/// <summary>Host reference for <see cref="IBackend.Mxfp4LinearF32"/>: <c>y = x · Wᵀ</c> with <c>W</c> an MXFP4 weight kept packed, decoded
/// one output row at a time. A weight element is <c>E2M1(nibble) · 2^(scale − 127)</c>, the low nibble the even column, one E8M0 scale per
/// 32 columns.</summary>
public static unsafe class Mxfp4LinearReference
{
    private static readonly float[] E2M1 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    /// <summary>Checks the operands of <see cref="IBackend.Mxfp4LinearF32"/> and returns the batch size and the weight's shape.</summary>
    public static void Validate(Tensor output, Tensor input, Tensor packedWeight, QuantRecipe recipe,
        out int n, out int rows, out int cols)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(packedWeight);
        ArgumentNullException.ThrowIfNull(recipe);
        if (recipe.Encoding != QuantEncoding.Mxfp4E8M0)
            throw new NotSupportedException($"Mxfp4Linear takes an MXFP4 recipe, got {recipe.Encoding}.");
        if (recipe.Geometry != new BlockGeometry(1, 32) || recipe.ScaleLayout != ScaleLayout.RowMajorBlocks || recipe.ScaleColOffset != 0)
            throw new NotSupportedException("Mxfp4Linear needs 32-column blocks with row-major scales and no column offset.");
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("Mxfp4Linear takes F32 activations and output.");
        rows = checked((int)recipe.LogicalRows);
        cols = checked((int)recipe.LogicalCols);
        if (cols % 32 != 0) throw new NotSupportedException($"Mxfp4Linear needs whole 32-column blocks; got {cols} columns.");
        if (input.ElementCount % cols != 0 || output.ElementCount != input.ElementCount / cols * rows)
            throw new ArgumentException($"Input {input.Shape} and output {output.Shape} do not match a weight of {rows}×{cols}.");
        n = checked((int)(input.ElementCount / cols));
        if (packedWeight.ElementCount != (long)rows * cols / 2)
            throw new ArgumentException($"Packed weight {packedWeight.Shape} must hold {rows}×{cols / 2} bytes.", nameof(packedWeight));
        Tensor scale = recipe.Scale ?? throw new ArgumentException("An MXFP4 recipe needs its scales.", nameof(recipe));
        if (scale.ElementCount < (long)rows * (cols / 32))
            throw new ArgumentException($"Scales {scale.Shape} must hold {rows}×{cols / 32} bytes.", nameof(recipe));
    }

    /// <summary>Computes <paramref name="output"/> <c>[n, rows]</c> from <paramref name="input"/> <c>[n, cols]</c>.</summary>
    /// <param name="packedWeight">U8 <c>[rows, cols / 2]</c>.</param>
    /// <param name="recipe">An MXFP4 recipe with <c>Geometry = (1, 32)</c>, row-major scales <c>[rows, cols / 32]</c>.</param>
    public static void Apply(Tensor output, Tensor input, Tensor packedWeight, QuantRecipe recipe)
    {
        Validate(output, input, packedWeight, recipe, out int n, out int rows, out int cols);
        int blocks = cols / 32;
        ReadOnlySpan<float> x = input.AsReadOnlySpan<float>();
        ReadOnlySpan<byte> packed = packedWeight.AsReadOnlySpan<byte>();
        ReadOnlySpan<byte> scales = recipe.Scale!.AsReadOnlySpan<byte>();
        Span<float> y = output.AsSpan<float>();
        for (int o = 0; o < rows; o++)
        {
            for (int r = 0; r < n; r++)
            {
                double acc = 0;
                for (int c = 0; c < cols; c++)
                {
                    int b = packed[o * (cols / 2) + c / 2];
                    int nib = (c & 1) == 0 ? b & 0xF : b >> 4;
                    float mag = E2M1[nib & 7] * (float)Math.Pow(2, scales[o * blocks + c / 32] - 127);
                    float w = (nib & 8) != 0 ? -mag : mag;
                    acc += x[r * cols + c] * w;
                }
                y[r * rows + o] = (float)acc;
            }
        }
    }
}
