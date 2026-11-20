using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Scalar score-order and empty-row controls for the attention correctness baseline.</summary>
public sealed class FlashAttentionSemanticTests
{
    [Fact]
    public void SoftcapPrecedesAlibi()
    {
        using IBackend backend = new CpuBackend();
        using Tensor q = new(new TensorShape(1, 1, 1, 2), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 2, 2), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor slope = new(new TensorShape(1), DType.F32);
        q.AsSpan<float>().Clear();
        k.AsSpan<float>().Clear();
        v.AsSpan<float>()[0] = v.AsSpan<float>()[1] = 2;
        v.AsSpan<float>()[2] = v.AsSpan<float>()[3] = 10;
        slope.AsSpan<float>()[0] = 4;
        backend.FlashAttention(output, q, k, v, 2, 1, true, 1, 1, softcap: 1, alibiSlopes: slope);
        double expected = (2 * Math.Exp(-4) + 10) / (Math.Exp(-4) + 1);
        foreach (float actual in output.AsReadOnlySpan<float>())
            Assert.InRange(Math.Abs(actual - expected), 0, 2e-5 * Math.Abs(expected));
    }

    [Theory]
    [InlineData(null, false)]
    [InlineData(float.NegativeInfinity, false)]
    [InlineData(float.PositiveInfinity, true)]
    [InlineData(float.NaN, true)]
    public void EmptyRowsPreserveSinkSemantics(float? sinkValue, bool expectNaN)
    {
        using IBackend backend = new CpuBackend();
        using Tensor q = new(new TensorShape(1, 1, 1, 2), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 2, 2), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor? sink = sinkValue.HasValue ? new(new TensorShape(1), DType.F32) : null;
        q.AsSpan<float>().Clear();
        k.AsSpan<float>().Clear();
        v.AsSpan<float>().Fill(7);
        if (sink is not null) sink.AsSpan<float>()[0] = sinkValue!.Value;
        backend.FlashAttention(output, q, k, v, 0, 1, true, 0, 1, sink: sink);
        foreach (float actual in output.AsReadOnlySpan<float>())
            if (expectNaN) Assert.True(float.IsNaN(actual)); else Assert.Equal(0, actual);
    }

    [Fact]
    public void DenseFullyExcludedRowIsZeroAndOthersAreUntouched()
    {
        using IBackend backend = new CpuBackend();
        using Tensor q = new(new TensorShape(1, 1, 3, 2), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 2, 2), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor mask = new(new TensorShape(3, 2), DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        q.AsSpan<float>().Clear();
        k.AsSpan<float>().Clear();
        v.AsSpan<float>()[0] = v.AsSpan<float>()[1] = 2;
        v.AsSpan<float>()[2] = v.AsSpan<float>()[3] = 10;
        // Row 0 sees nothing, row 1 sees only key 1, row 2 is fully excluded except for a NaN bias that must propagate.
        float[] bias = [float.NegativeInfinity, float.NegativeInfinity, float.NegativeInfinity, 0f, float.NegativeInfinity, float.NaN];
        bias.CopyTo(mask.AsSpan<float>());
        backend.ScaledDotProductAttention(output, q, k, v, mask, 1f);
        ReadOnlySpan<float> o = output.AsReadOnlySpan<float>();
        Assert.Equal(0f, o[0]);
        Assert.Equal(0f, o[1]);
        Assert.Equal(10f, o[2]);
        Assert.Equal(10f, o[3]);
        Assert.True(float.IsNaN(o[4]) && float.IsNaN(o[5]));
    }
}
