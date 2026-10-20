using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Cpu.Tests.Attention;

/// <summary><see cref="CpuBackend.Attention"/> honours every component of the call: what its legacy entries cannot carry
/// runs on the contract reference rather than being dropped.</summary>
public sealed unsafe class CpuAttentionEntryTests
{
    private static Tensor Rand(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [Fact]
    public void ACausalDenseCall_IsCausal()
    {
        using CpuBackend cpu = new();
        using Tensor q = Rand(new TensorShape(1, 2, 5, 8), 1), k = Rand(new TensorShape(1, 2, 5, 8), 2), v = Rand(new TensorShape(1, 2, 5, 8), 3);
        using Tensor viaEntry = new(q.Shape, DType.F32), expected = new(q.Shape, DType.F32), plain = new(q.Shape, DType.F32);
        AttentionCall dense = AttentionCall.FromScaledDotProductAttention(viaEntry, q, k, v, null, 0.35f, false);
        AttentionCall causal = dense with { Mask = dense.Mask with { Causal = true } };
        cpu.Attention(in causal);
        AttentionHostReference.Run(causal with { Output = AttentionOperand.HeadMajor(expected) });
        cpu.Attention(dense with { Output = AttentionOperand.HeadMajor(plain) });
        float* a = (float*)viaEntry.DataPointer, e = (float*)expected.DataPointer, p = (float*)plain.DataPointer;
        float worst = 0, moved = 0;
        for (long i = 0; i < q.ElementCount; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(a[i] - e[i]));
            moved = MathF.Max(moved, MathF.Abs(a[i] - p[i]));
        }
        Assert.True(worst < 1e-5f, $"causal entry vs reference {worst}");
        Assert.True(moved > 1e-3f, "the causal flag changed nothing");
    }

    [Fact]
    public void AGroupedDenseCall_ReadsEachGroupsKeyValueHead()
    {
        using CpuBackend cpu = new();
        using Tensor q = Rand(new TensorShape(1, 4, 3, 8), 1), k = Rand(new TensorShape(1, 2, 6, 8), 2), v = Rand(new TensorShape(1, 2, 6, 8), 3);
        using Tensor actual = new(q.Shape, DType.F32), expected = new(q.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(actual, q, k, v, null, 0.35f, false);
        cpu.Attention(in call);
        AttentionHostReference.Run(call with { Output = AttentionOperand.HeadMajor(expected) });
        float* a = (float*)actual.DataPointer, e = (float*)expected.DataPointer;
        for (long i = 0; i < q.ElementCount; i++) Assert.Equal(e[i], a[i]);
    }
}
