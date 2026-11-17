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
    public void NonF32SinkIsMalformedBeforeCpuAttentionWritesOutput()
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

        // The documented [Hq] F32 term, as the CUDA flash contract enforces it: a wrong dtype is malformed, not unsupported.
        ArgumentException error = Assert.Throws<ArgumentException>(() => backend.Attention(in call));
        Assert.Contains("Sinks", error.Message);

        AssertOutputIsSentinel(operands.Output);
    }

    [Fact]
    public void LegacyCpuDenseEntryRunsTheSharedPreflight()
    {
        using Operands operands = new();
        using IBackend backend = new CpuBackend();
        operands.Query.AsSpan<float>().Fill(3f);

        // The CUDA and Vulkan legacy entries reject an aliased output through the shared boundary; the CPU entry must too.
        Assert.Throws<ArgumentException>(() =>
            backend.ScaledDotProductAttention(operands.Query, operands.Query, operands.Key, operands.Value, null, 1f));

        foreach (float value in operands.Query.AsReadOnlySpan<float>()) Assert.Equal(3f, value);
    }

    public static TheoryData<string> MalformedCases =>
    [
        "negative token stride", "negative query offset", "valid length past keys", "valid length negative",
        "bias does not broadcast", "sink count", "alibi count", "alibi dtype", "softcap negative", "softcap nan",
        "window negative", "block size negative", "head dims disagree",
    ];

    [Theory]
    [MemberData(nameof(MalformedCases))]
    public void MalformedCallsAreArgumentExceptions(string name)
    {
        using Operands operands = new();
        using Tensor wideBias = new(new TensorShape(1, 1, 2, 3), DType.F32);
        using Tensor twoSinks = new(new TensorShape(2), DType.F32);
        using Tensor halfSlopes = new(new TensorShape(1), DType.F16);
        using Tensor narrowKey = new(new TensorShape(1, 1, 1, 1), DType.F32);
        AttentionCall call = operands.Call;
        AttentionCall malformed = name switch
        {
            "negative token stride" => call with { Key = call.Key with { TokenStride = -2 } },
            "negative query offset" => call with { Positions = AttentionPositions.Host(1, -1) },
            "valid length past keys" => call with { Positions = AttentionPositions.Host(2, 0) },
            "valid length negative" => call with { Positions = AttentionPositions.Host(-1, 0) },
            "bias does not broadcast" => call with { Mask = AttentionMask.Additive(wideBias) },
            "sink count" => call with { Logits = new AttentionLogitTerms(0f, twoSinks, null) },
            "alibi count" => call with { Logits = new AttentionLogitTerms(0f, null, twoSinks) },
            "alibi dtype" => call with { Logits = new AttentionLogitTerms(0f, null, halfSlopes) },
            "softcap negative" => call with { Logits = new AttentionLogitTerms(-1f, null, null) },
            "softcap nan" => call with { Logits = new AttentionLogitTerms(float.NaN, null, null) },
            "window negative" => call with { Mask = AttentionMask.FromFlash(true, -1) },
            "block size negative" => call with { Mask = call.Mask with { BlockSize = -1 } },
            "head dims disagree" => call with { Key = AttentionOperand.HeadMajor(narrowKey) },
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };

        Exception error = Assert.ThrowsAny<ArgumentException>(() => AttentionCallValidation.Validate(in malformed));
        Assert.IsNotType<AttentionUnsupportedException>(error);
    }

    [Fact]
    public void NonF32BiasIsUnsupportedNotMalformed()
    {
        using Operands operands = new();
        using Tensor bias = new(new TensorShape(1, 1), DType.F16);
        AttentionCall call = operands.Call with { Mask = AttentionMask.Additive(bias) };

        Assert.Throws<AttentionUnsupportedException>(() => AttentionCallValidation.Validate(in call));
    }

    [Fact]
    public void MalformationIsReportedBeforeARefusedFormat()
    {
        using Tensor q = new(new TensorShape(1, 1, 1, 2), DType.F16), k = new(new TensorShape(1, 1, 1, 2), DType.F16);
        using Tensor v = new(new TensorShape(1, 1, 1, 2), DType.F16), o = new(new TensorShape(1, 1, 1, 2), DType.F16);
        AttentionCall refused = AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, 1f, false)
            with { Numerics = new() { Refused = AttentionFormats.F16 } };
        AttentionCall both = refused with { Scale = float.NaN };

        Assert.Throws<AttentionUnsupportedException>(() => AttentionCallValidation.Validate(in refused));
        Assert.Throws<ArgumentOutOfRangeException>(() => AttentionCallValidation.Validate(in both));
    }

    private static void AssertOutputIsSentinel(Tensor output)
    {
        foreach (float value in output.AsReadOnlySpan<float>()) Assert.Equal(7f, value);
    }

    private sealed class Operands : IDisposable
    {
        public Tensor Query { get; } = new(new TensorShape(1, 1, 1, 2), DType.F32);
        public Tensor Key { get; } = new(new TensorShape(1, 1, 1, 2), DType.F32);
        public Tensor Value { get; } = new(new TensorShape(1, 1, 1, 2), DType.F32);
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
