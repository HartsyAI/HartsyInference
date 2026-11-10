using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>CPU reference for <see cref="IBackend.BlockDiagonalLinear"/>: a block-diagonal linear over an F32 or block-scaled FP8 weight.</summary>
public static class BlockDiagonalLinearReference
{
    /// <summary>Checks the operands and returns the rows M, the outputs per group R and the inputs per group D.</summary>
    public static (int M, int R, int D) Validate(Tensor output, Tensor input, Tensor weight, Tensor? weightScaleInv, int groups,
        int blockRows, int blockCols)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(groups);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockRows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockCols);
        if (input.DType != DType.F32 || output.DType != DType.F32)
            throw new NotSupportedException("BlockDiagonalLinear supports F32 input and output only.");
        if (weight.Shape.Rank != 2) throw new ArgumentException($"BlockDiagonalLinear needs a rank-2 weight, not {weight.Shape}.", nameof(weight));
        long rows = weight.Shape[0], d = weight.Shape[1];
        if (weight.DType == DType.F32)
        {
            if (weightScaleInv is not null)
                throw new ArgumentException("An F32 BlockDiagonalLinear weight takes no scale grid.", nameof(weightScaleInv));
        }
        else if (weight.DType == DType.F8E4M3)
        {
            if (weightScaleInv is null || weightScaleInv.DType != DType.F32)
                throw new ArgumentException("An F8_E4M3 BlockDiagonalLinear weight needs an F32 weight_scale_inv grid.", nameof(weightScaleInv));
            long gridRows = (rows + blockRows - 1) / blockRows, gridCols = (d + blockCols - 1) / blockCols;
            if (weightScaleInv.Shape.Rank != 2 || weightScaleInv.Shape[0] != gridRows || weightScaleInv.Shape[1] != gridCols)
                throw new ArgumentException(
                    $"A [{rows}, {d}] weight with {blockRows}x{blockCols} blocks needs a [{gridRows}, {gridCols}] scale, not {weightScaleInv.Shape}.",
                    nameof(weightScaleInv));
        }
        else
        {
            throw new NotSupportedException($"BlockDiagonalLinear takes an F32 or F8_E4M3 weight, not {weight.DType.Name}.");
        }
        if (d == 0 || rows % groups != 0)
            throw new ArgumentException($"BlockDiagonalLinear weight {weight.Shape} must have D > 0 and rows divisible by {groups} groups.",
                nameof(weight));
        long r = rows / groups, inWidth = groups * d;
        if (input.Shape.Rank < 1 || input.Shape[input.Shape.Rank - 1] != inWidth)
            throw new ArgumentException($"BlockDiagonalLinear input must end in G*D={inWidth}; got {input.Shape}.", nameof(input));
        long m = input.ElementCount / inWidth;
        if (output.ElementCount != m * rows || output.Shape.Rank < 1 || output.Shape[output.Shape.Rank - 1] != rows)
            throw new ArgumentException($"BlockDiagonalLinear output must be [..., {rows}] with {m} rows; got {output.Shape}.", nameof(output));
        if (m * inWidth > int.MaxValue || m * rows > int.MaxValue || rows * d > int.MaxValue)
            throw new NotSupportedException("BlockDiagonalLinear operands exceed 32-bit indexing.");
        return ((int)m, (int)r, (int)d);
    }

    /// <summary>Computes the grouped product; the input and output must not alias.</summary>
    public static unsafe void Apply(Tensor output, Tensor input, Tensor weight, Tensor? weightScaleInv, int groups, int blockRows, int blockCols)
    {
        (int m, int r, int d) = Validate(output, input, weight, weightScaleInv, groups, blockRows, blockCols);
        if (weight.DType == DType.F32)
        {
            ApplyRows(input.AsReadOnlySpan<float>(), weight.AsReadOnlySpan<float>(), 0, groups * r, m, groups, r, d, output.AsSpan<float>());
            return;
        }
        float* x = (float*)input.DataPointer, y = (float*)output.DataPointer, scale = (float*)weightScaleInv!.DataPointer;
        byte* w = (byte*)weight.DataPointer;
        int n = groups * r, gridCols = (d + blockCols - 1) / blockCols;
        float[] row = new float[d];
        for (int j = 0; j < n; j++)
        {
            float* scaleRow = scale + (long)(j / blockRows) * gridCols;
            for (int c = 0; c < d; c++) row[c] = LatentCodec.DecodeE4M3(w[(long)j * d + c]) * scaleRow[c / blockCols];
            int g = j / r;
            for (int i = 0; i < m; i++)
            {
                float* xi = x + ((long)i * groups + g) * d;
                double sum = 0.0;
                for (int c = 0; c < d; c++) sum += (double)xi[c] * row[c];
                y[(long)i * n + j] = (float)sum;
            }
        }
    }

    /// <summary>The F32 product over the flattened output rows <c>[firstRow, firstRow + count)</c>, where <paramref name="weightRows"/>
    /// holds those <c>count</c> weight rows, each <paramref name="groupDim"/> wide; each element is one in-order float dot product, so
    /// splitting the rows into windows gives the same bits as the whole matrix.</summary>
    public static void ApplyRows(ReadOnlySpan<float> x, ReadOnlySpan<float> weightRows, int firstRow, int count, int tokens, int groups,
        int rank, int groupDim, Span<float> dest)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(tokens);
        ArgumentOutOfRangeException.ThrowIfLessThan(groups, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(rank, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(groupDim, 1);
        if (firstRow < 0 || count < 0 || (long)firstRow + count > (long)groups * rank)
            throw new ArgumentOutOfRangeException(nameof(count), "Rows [firstRow, firstRow + count) must lie within groups x rank.");
        if (weightRows.Length < (long)count * groupDim)
            throw new ArgumentException("weightRows must hold count x groupDim values.", nameof(weightRows));
        if (x.Length < (long)tokens * groups * groupDim)
            throw new ArgumentException("x must hold tokens x groups x groupDim values.", nameof(x));
        if (dest.Length < (long)tokens * groups * rank)
            throw new ArgumentException("dest must hold tokens x groups x rank values.", nameof(dest));
        for (int r = 0; r < count; r++)
        {
            int g = (firstRow + r) / rank, o = (firstRow + r) % rank;
            ReadOnlySpan<float> wr = weightRows.Slice(r * groupDim, groupDim);
            for (int t = 0; t < tokens; t++)
            {
                ReadOnlySpan<float> xv = x.Slice((t * groups + g) * groupDim, groupDim);
                float sum = 0f;
                for (int d = 0; d < groupDim; d++) sum += xv[d] * wr[d];
                dest[(t * groups + g) * rank + o] = sum;
            }
        }
    }
}
