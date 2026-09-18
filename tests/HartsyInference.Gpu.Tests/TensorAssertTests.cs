using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Gpu.Tests;

/// <summary>The comparison the parity suites trust, checked against the failures it exists to catch.
///
/// <para>A tolerance check has one classic hole and one quiet one, and both let a broken kernel report success — the
/// worst failure mode available to a test helper, because every suite downstream inherits it.</para></summary>
public sealed class TensorAssertTests
{
    private static Tensor From(params float[] values)
    {
        Tensor tensor = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    /// <summary>NaN compares false to every bound, so a magnitude test silently passes it.</summary>
    [Fact]
    public void A_Nan_Against_A_Finite_Reference_Fails()
    {
        using Tensor actual = From(1f, float.NaN, 3f);
        using Tensor expected = From(1f, 2f, 3f);

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => TensorAssert.Close(actual, expected));
        Assert.Contains("non-finite", error.Message, StringComparison.Ordinal);
        Assert.Contains("[1]", error.Message, StringComparison.Ordinal);
    }

    /// <summary>Infinity is a real divergence, not an enormous number to be scaled against a finite peak.</summary>
    [Fact]
    public void An_Infinity_Against_A_Finite_Reference_Fails()
    {
        using Tensor actual = From(1f, float.PositiveInfinity, 3f);
        using Tensor expected = From(1f, 2f, 3f);

        Assert.Throws<InvalidOperationException>(() => TensorAssert.Close(actual, expected));
    }

    /// <summary>A NaN both sides is agreement, not a failure: a masked-out attention slot is legitimately NaN in
    /// both implementations, and failing there would make the helper unusable for the op it matters most for.</summary>
    [Fact]
    public void A_Nan_On_Both_Sides_Agrees()
    {
        using Tensor actual = From(1f, float.NaN, 3f);
        using Tensor expected = From(1f, float.NaN, 3f);

        TensorAssert.Close(actual, expected);
    }

    [Fact]
    public void Matching_Values_Still_Pass()
    {
        using Tensor actual = From(1f, 2f, 3f);
        using Tensor expected = From(1.000001f, 2f, 3f);

        TensorAssert.Close(actual, expected);
    }

    /// <summary>The quiet hole: widening to F32 and subtracting makes -0.0 equal +0.0, so a bit-identity check built
    /// on a zero tolerance would not notice the representation changing.</summary>
    [Fact]
    public void Identical_Distinguishes_Negative_Zero()
    {
        using Tensor actual = From(0f, -0f);
        using Tensor expected = From(0f, 0f);

        TensorAssert.Close(actual, expected);   // numerically equal, and that is correct
        Assert.Throws<InvalidOperationException>(() => TensorAssert.Identical(actual, expected));
    }

    [Fact]
    public void Identical_Rejects_A_Different_DType()
    {
        using Tensor actual = From(1f, 2f);
        using Tensor expected = new(new TensorShape(2), DType.F16);

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => TensorAssert.Identical(actual, expected));
        Assert.Contains("DType mismatch", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Identical_Accepts_The_Same_Bytes()
    {
        using Tensor actual = From(1f, 2f, 3f);
        using Tensor expected = From(1f, 2f, 3f);

        TensorAssert.Identical(actual, expected);
    }
}
