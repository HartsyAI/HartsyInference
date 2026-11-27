using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Pins the shared execution boundary and legacy adapters to the direct attention-call path.</summary>
public sealed class AttentionExecutionTests
{
    [Fact]
    public void PlanningAndExecutionBindAnUnboundedFlashPrefix()
    {
        using Tensor q = new(new TensorShape(1, 1, 1, 2), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 3, 2), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromFlashAttention(output, q, k, v, 3, false, 0, 1f, 0f, null, 0, null)
            with { Positions = default };
        int plannedKeys = 0, executedKeys = 0;
        AttentionKernelCapability capability = new()
        {
            Id = "flash.test", Entry = AttentionRoute.Flash, QueryTypes = AttentionDTypes.F32,
            KvTypes = AttentionDTypes.F32, Layouts = AttentionLayouts.HeadMajor,
        };
        AttentionKernelRow[] rows =
        [
            new DelegateAttentionRow(capability, null,
                (in AttentionCall c) => { plannedKeys = c.ValidKvLength; return true; },
                (in AttentionCall c) => { executedKeys = c.ValidKvLength; return true; }),
        ];

        Assert.Same(rows[0], AttentionExecution.Plan(rows, in call).Row);
        Assert.Same(rows[0], AttentionExecution.Run(rows, in call));
        Assert.Equal(3, plannedKeys);
        Assert.Equal(3, executedKeys);
    }

    [Fact]
    public void CpuDenseLegacyAndDirectEntriesProduceIdenticalOutput()
    {
        using CpuBackend backend = new();
        using Tensor q = new(new TensorShape(1, 2, 2, 4), DType.F32);
        using Tensor k = new(new TensorShape(1, 2, 3, 4), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor legacy = new(q.Shape, DType.F32);
        using Tensor direct = new(q.Shape, DType.F32);
        Fill(q, 0.03f);
        Fill(k, -0.02f);
        Fill(v, 0.1f);

        backend.ScaledDotProductAttention(legacy, q, k, v, null, 0.5f);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(direct, q, k, v, null, 0.5f, false);
        backend.Attention(in call);

        Assert.Contains("cpu.simd", backend.PlanAttention(in call).Reason);
        Assert.True(legacy.AsReadOnlySpan<float>().SequenceEqual(direct.AsReadOnlySpan<float>()));
    }

    [Fact]
    public void CpuFlashLegacyAndDirectEntriesProduceIdenticalOutput()
    {
        using IBackend backend = new CpuBackend();
        using Tensor q = new(new TensorShape(1, 2, 1, 4), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 3, 4), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor legacy = new(q.Shape, DType.F32);
        using Tensor direct = new(q.Shape, DType.F32);
        Fill(q, 0.03f);
        Fill(k, -0.02f);
        Fill(v, 0.1f);

        backend.FlashAttention(legacy, q, k, v, 3, 2, false, 2, 0.5f);
        AttentionCall call = AttentionCall.FromFlashAttention(direct, q, k, v, 3, false, 2, 0.5f, 0f, null, 0, null);
        backend.Attention(in call);

        Assert.Contains("cpu.flash", backend.PlanAttention(in call).Reason);
        Assert.True(legacy.AsReadOnlySpan<float>().SequenceEqual(direct.AsReadOnlySpan<float>()));
    }

    [Fact]
    public void MalformedLegacyAndDirectCallsBothLeaveOutputUntouched()
    {
        using CpuBackend backend = new();
        using Tensor q = new(new TensorShape(1, 1, 1, 2), DType.F32);
        using Tensor k = new(q.Shape, DType.F32);
        using Tensor v = new(q.Shape, DType.F32);
        using Tensor legacy = new(q.Shape, DType.F32);
        using Tensor direct = new(q.Shape, DType.F32);
        legacy.AsSpan<float>().Fill(7f);
        direct.AsSpan<float>().Fill(7f);

        Assert.Throws<ArgumentOutOfRangeException>(() => backend.ScaledDotProductAttention(legacy, q, k, v, null, float.NaN));
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(direct, q, k, v, null, float.NaN, false);
        Assert.Throws<ArgumentOutOfRangeException>(() => backend.Attention(in call));

        foreach (float value in legacy.AsReadOnlySpan<float>()) Assert.Equal(7f, value);
        foreach (float value in direct.AsReadOnlySpan<float>()) Assert.Equal(7f, value);
    }

    private static void Fill(Tensor tensor, float step)
    {
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (i - values.Length / 2f) * step;
    }
}
