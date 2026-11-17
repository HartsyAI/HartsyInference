using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Guards the shared no-write validation boundary used before CPU and GPU attention planning.</summary>
public sealed class AttentionCallValidationTests
{
    [Fact]
    public void InvalidScaleIsRejectedBeforeCpuAttentionWritesOutput()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call with { Scale = float.NaN };
        using IBackend backend = new CpuBackend();

        Assert.Throws<ArgumentOutOfRangeException>(() => backend.Attention(in call));

        AssertOutputIsSentinel(operands.Output);
    }

    [Fact]
    public void AliasedOutputIsRejectedBeforeCpuAttentionWritesOutput()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call with { Output = AttentionOperand.HeadMajor(operands.Query) };
        using IBackend backend = new CpuBackend();

        Assert.Throws<ArgumentException>(() => backend.Attention(in call));

        AssertOutputIsSentinel(operands.Output);
    }

    [Fact]
    public void InvalidPlanningUsesTheSameSharedPreflight()
    {
        using Operands operands = new();
        AttentionCall call = operands.Call with { Scale = float.PositiveInfinity };
        using IBackend backend = new CpuBackend();

        Assert.Throws<ArgumentOutOfRangeException>(() => backend.PlanAttention(in call));

        AssertOutputIsSentinel(operands.Output);
    }

    [Fact]
    public void NonF32SinkIsRejectedBeforeCpuAttentionWritesOutput()
    {
        using Operands operands = new();
        using Tensor sink = new(new TensorShape(1), DType.F16);
        AttentionCall call = operands.Call with
        {
            Route = AttentionRoute.Flash,
            Positions = AttentionPositions.Host(1, 0),
            Logits = new AttentionLogitTerms(0f, sink, null),
        };
        using IBackend backend = new CpuBackend();

        Assert.Throws<AttentionUnsupportedException>(() => backend.Attention(in call));

        AssertOutputIsSentinel(operands.Output);
    }

    private static void AssertOutputIsSentinel(Tensor output)
    {
        foreach (float value in output.AsReadOnlySpan<float>()) Assert.Equal(7f, value);
    }

    private sealed class Operands : IDisposable
    {
        public Tensor Query { get; } = new(new TensorShape(1, 1, 1, 2), DType.F32);
        private Tensor Key { get; } = new(new TensorShape(1, 1, 1, 2), DType.F32);
        private Tensor Value { get; } = new(new TensorShape(1, 1, 1, 2), DType.F32);
        public Tensor Output { get; } = new(new TensorShape(1, 1, 1, 2), DType.F32);

        public Operands() => Output.AsSpan<float>().Fill(7f);

        public AttentionCall Call => AttentionCall.FromScaledDotProductAttention(Output, Query, Key, Value, null, 1f, false);

        public void Dispose()
        {
            Output.Dispose();
            Value.Dispose();
            Key.Dispose();
            Query.Dispose();
        }
    }
}
