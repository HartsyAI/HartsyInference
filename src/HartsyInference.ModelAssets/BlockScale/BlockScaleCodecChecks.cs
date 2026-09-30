using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.ModelAssets.BlockScale;

/// <summary>Argument validation shared by the host codecs, so a recipe that does not describe the bytes fails before any decode.</summary>
internal static class BlockScaleCodecChecks
{
    private static readonly DType[] E8M0Types = [DType.F8E8M0, DType.U8];

    /// <summary>Validates an E8M0-scaled recipe against <paramref name="packedLength"/> and the requested window; returns the scale row stride in elements.</summary>
    internal static long Validate(QuantRecipe recipe, QuantEncoding expected, long packedLength, long rowOffset, long rowCount, long destLength) =>
        Validate(recipe, expected, E8M0Types, packedLength, rowOffset, rowCount, destLength);

    /// <summary>Validates a recipe whose scale must be one of <paramref name="scaleTypes"/>; returns the scale row stride in elements.</summary>
    internal static long Validate(QuantRecipe recipe, QuantEncoding expected, DType[] scaleTypes, long packedLength, long rowOffset, long rowCount,
        long destLength)
    {
        if (recipe.Encoding != expected)
            throw new HartsyInferenceException($"Recipe encoding is {recipe.Encoding}, this codec decodes {expected}.");
        if (recipe.ScaleLayout != ScaleLayout.RowMajorBlocks)
            throw new NotSupportedException($"{expected} host decode reads {ScaleLayout.RowMajorBlocks} scales; recipe is {recipe.ScaleLayout}.");
        if (Array.IndexOf(scaleTypes, recipe.ScaleDType) < 0)
            throw new NotSupportedException(
                $"{expected} host decode needs {string.Join(" or ", scaleTypes.Select(t => t.Name))} scales; recipe has {recipe.ScaleDType}.");
        Tensor scale = recipe.Scale ?? throw new HartsyInferenceException($"{expected} recipe has no scale tensor.");
        if (scale.DType != recipe.ScaleDType || scale.Shape.Rank != 2)
            throw new HartsyInferenceException($"{expected} scale must be rank-2 {recipe.ScaleDType}; got {scale.DType} {scale.Shape}.");
        if (rowOffset < 0 || rowCount < 0 || rowOffset + rowCount > recipe.LogicalRows)
            throw new ArgumentOutOfRangeException(
                nameof(rowOffset), $"Rows [{rowOffset}..{rowOffset + rowCount}) are outside [0..{recipe.LogicalRows}).");

        BlockGeometry geometry = recipe.Geometry;
        (long scaleRows, long scaleCols) = geometry.ScaleShape(recipe.LogicalRows, recipe.LogicalCols);
        if (scale.Shape[0] < scaleRows || scale.Shape[1] < recipe.ScaleColOffset + scaleCols)
            throw new HartsyInferenceException(
                $"{expected} scale {scale.Shape} is smaller than the [{scaleRows}, {recipe.ScaleColOffset + scaleCols}] "
                + $"its {geometry} geometry needs.");

        long expectedPacked = recipe.LogicalRows * recipe.LogicalCols / recipe.ElementsPerByte;
        if (packedLength != expectedPacked)
            throw new ArgumentException(
                $"Packed weight is {packedLength} bytes; a {recipe.LogicalRows}x{recipe.LogicalCols} {expected} matrix is {expectedPacked}.",
                "packed");
        if (destLength != rowCount * recipe.LogicalCols)
            throw new ArgumentException(
                $"Destination holds {destLength} floats; {rowCount} rows of {recipe.LogicalCols} need {rowCount * recipe.LogicalCols}.", "dest");
        return scale.Shape[1];
    }
}
