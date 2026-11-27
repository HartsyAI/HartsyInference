namespace HartsyInference.Core.Tensors;

/// <summary>Host I32 index tensors from the integer spans the models hold: the operand form of
/// <see cref="Backends.IBackend.GatherRows"/>, <see cref="Backends.IBackend.ScatterAddRows"/> and the other row-index ops.</summary>
public static class IndexTensor
{
    /// <summary>A <c>[rows.Length]</c> I32 tensor holding a copy of <paramref name="rows"/>.</summary>
    public static Tensor Rows(ReadOnlySpan<int> rows)
    {
        Tensor t = new(new TensorShape(rows.Length), DType.I32);
        rows.CopyTo(t.AsSpan<int>());
        return t;
    }

    /// <summary>A <c>[values.Length]</c> F32 tensor holding a copy of <paramref name="values"/>.</summary>
    public static Tensor Values(ReadOnlySpan<float> values)
    {
        Tensor t = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
