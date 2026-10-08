using System.Buffers;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.BlockScale;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>One two-dimensional weight of the V4.1 host model, either as F32 values or in the form the checkpoint stores it (FP8 block-scaled, MXFP4, BF16, ...) decoded by row window while it is used.</summary>
/// <remarks>A stored weight keeps only a borrowed view of the checkpoint tensor, so the checkpoint must outlive it. Products against a stored weight are bit-identical to the same product against its F32 widening,
/// because each output element is the same sequential dot over the same decoded values.</remarks>
public sealed class DeepSeekV41Weight
{
    // rows of a stored weight decoded at once; bounds the scratch buffer to about 16 MiB
    private const int WindowElements = 1 << 22;

    private readonly float[]? _values;
    private readonly Tensor? _stored;
    private readonly QuantWeightInfo? _quant;

    private DeepSeekV41Weight(float[]? values, Tensor? stored, QuantWeightInfo? quant, long rows, long cols)
    {
        _values = values;
        _stored = stored;
        _quant = quant;
        Rows = rows;
        Cols = cols;
    }

    /// <summary>Logical rows, or -1 for an F32 array whose shape only its caller knows.</summary>
    public long Rows { get; }

    /// <summary>Logical columns, or -1 as for <see cref="Rows"/>.</summary>
    public long Cols { get; }

    /// <summary>Logical element count; for an F32 array that is its length.</summary>
    public long Elements => _values is not null ? _values.Length : Rows * Cols;

    /// <summary>True when the weight is held as F32 values rather than decoded on use.</summary>
    public bool IsWidened => _values is not null;

    /// <summary>Bytes the weight occupies in host memory as this object holds it (a stored weight counts the checkpoint bytes it maps).</summary>
    public long ResidentBytes => _values is not null ? (long)_values.Length * sizeof(float) : _stored!.DType.ComputeByteCount(_stored.ElementCount);

    /// <summary>Wraps F32 values; their shape is checked where they are used.</summary>
    public static DeepSeekV41Weight FromF32(float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return new DeepSeekV41Weight(values, null, null, -1, -1);
    }

    /// <summary>Wraps F32 values; their shape is checked where they are used.</summary>
    [return: System.Diagnostics.CodeAnalysis.NotNullIfNotNull(nameof(values))]
    public static implicit operator DeepSeekV41Weight?(float[]? values) => values is null ? null : FromF32(values);

    /// <summary>Keeps a checkpoint tensor in its stored form after checking its logical shape is <c>[rows, cols]</c>.</summary>
    /// <param name="key">Checkpoint key, named in the refusal.</param>
    /// <param name="weight">The stored tensor (borrowed).</param>
    /// <param name="quant">Its bound recipe, or null for an unquantized F32, BF16 or F16 matrix.</param>
    /// <exception cref="HartsyInferenceException">The shape differs or the encoding has no host reader.</exception>
    public static DeepSeekV41Weight FromStored(string key, Tensor weight, QuantWeightInfo? quant, long rows, long cols)
    {
        ArgumentNullException.ThrowIfNull(weight);
        long actualRows, actualCols;
        if (quant?.Recipe is { } recipe)
        {
            actualRows = recipe.LogicalRows;
            actualCols = recipe.LogicalCols;
        }
        else
        {
            if (quant is not null) throw new HartsyInferenceException($"{key} uses quantization '{quant.Format}', which has no host reader.");
            if (weight.Shape.Rank != 2) throw new HartsyInferenceException($"{key} has rank {weight.Shape.Rank}, expected a [{rows}, {cols}] matrix.");
            if (weight.DType != DType.F32 && weight.DType != DType.BF16 && weight.DType != DType.F16)
                throw new HartsyInferenceException($"{key} is {weight.DType} without a quantization recipe, which has no host reader.");
            actualRows = weight.Shape[0];
            actualCols = weight.Shape[1];
        }
        if (actualRows != rows || actualCols != cols) throw new HartsyInferenceException($"{key} is [{actualRows}, {actualCols}], expected [{rows}, {cols}].");
        return new DeepSeekV41Weight(null, weight, quant, rows, cols);
    }

    /// <summary>Returns <paramref name="count"/> rows starting at <paramref name="firstRow"/>: a view of the values when widened, otherwise decoded into <paramref name="scratch"/>.</summary>
    /// <param name="cols">Row width; checked against the weight's own for a stored weight.</param>
    /// <param name="scratch">At least <c>count * cols</c> floats; unused when the weight is widened.</param>
    public ReadOnlySpan<float> ReadRows(long firstRow, int count, int cols, float[] scratch)
    {
        if (_values is not null) return _values.AsSpan(checked((int)(firstRow * cols)), count * cols);
        if (cols != Cols) throw new ArgumentException($"The weight is {Cols} wide, not {cols}.", nameof(cols));
        Span<float> dest = scratch.AsSpan(0, count * cols);
        WeightDequantizer.ToF32Rows(_stored!, _quant, firstRow, count, dest);
        return dest;
    }

    /// <summary><c>y[t,o] = sum_i x[t,i] * w[o,i]</c> with the weight as <c>[outDim, inDim]</c>.</summary>
    public float[] Linear(ReadOnlySpan<float> x, int tokens, int inDim, int outDim)
    {
        if (_values is not null) return DeepSeekV41HostMath.Linear(x, _values, tokens, inDim, outDim);
        CheckShape(inDim, outDim);
        if (x.Length != (long)tokens * inDim) throw new ArgumentException("Linear operands do not match the stated shape.", nameof(x));
        float[] y = new float[checked(tokens * outDim)];
        int window = WindowRows(inDim, outDim);
        float[] scratch = ArrayPool<float>.Shared.Rent(window * inDim);
        try
        {
            for (int row = 0; row < outDim; row += window)
            {
                int count = Math.Min(window, outDim - row);
                DeepSeekV41HostMath.LinearInto(x, ReadRows(row, count, inDim, scratch), tokens, inDim, count, y, outDim, row);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
        return y;
    }

    /// <summary>Copies row <paramref name="row"/> to <paramref name="dest"/> (an embedding lookup).</summary>
    public void CopyRow(long row, Span<float> dest)
    {
        if (_values is not null)
        {
            _values.AsSpan(checked((int)(row * dest.Length)), dest.Length).CopyTo(dest);
            return;
        }
        if (dest.Length != Cols) throw new ArgumentException($"The weight is {Cols} wide, not {dest.Length}.", nameof(dest));
        WeightDequantizer.ToF32Rows(_stored!, _quant, row, 1, dest);
    }

    /// <summary>Widens the whole weight; only for small weights and tests.</summary>
    public float[] ToF32() => _values ?? WeightDequantizer.ToF32(_stored!, _quant);

    /// <summary>Largest row window the grouped and per-row consumers decode at once for rows of <paramref name="cols"/> floats.</summary>
    internal static int WindowRows(int cols, int totalRows) => Math.Clamp(WindowElements / Math.Max(cols, 1), 1, Math.Max(totalRows, 1));

    private void CheckShape(int inDim, int outDim)
    {
        if (Rows != outDim || Cols != inDim) throw new ArgumentException($"The weight is [{Rows}, {Cols}], not [{outDim}, {inDim}].");
    }
}
