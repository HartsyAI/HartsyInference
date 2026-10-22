using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Gpu.Tests;

/// <summary>On every backend's unmasked F32 attention, with or without the BF16 grant, the output for V·2⁻⁸ scaled back by 2⁸ is
/// the output for V bit for bit while nothing is subnormal: a caller's power-of-two V damp is a no-op. The CUDA rows are pinned one by
/// one in the CUDA suite's <c>AttentionValueScaleTests</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class AttentionValueScaleTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;
    private const int Heads = 4, Tokens = 300, HeadDim = 64;
    private const float Damp = 1f / 256f;

    private static Tensor Filled(Func<long, float> value)
    {
        Tensor t = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = value(i);
        return t;
    }

    public static IEnumerable<object[]> Rows =>
        BackendGate.AllKinds.SelectMany(kind => new[] { new object[] { kind[0], false }, new object[] { kind[0], true } });

    [Theory]
    [MemberData(nameof(Rows))]
    public void APowerOfTwoValueScale_PassesThroughExactly(string kind, bool bf16Grant)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? opened)) return;
        using IBackend backend = opened!;
        float scale = 1f / MathF.Sqrt(HeadDim);
        Random rng = new(7);
        using Tensor q = Filled(_ => (float)(rng.NextDouble() * 2 - 1)), k = Filled(_ => (float)(rng.NextDouble() * 2 - 1));
        // Every 13th element sits past F16's range, as a few of Lens's late-block V columns do.
        using Tensor v = Filled(i => (float)(rng.NextDouble() * 2 - 1) * (i % 13 == 0 ? 1.4e5f : 1f));
        using Tensor damped = Filled(i => ((float*)v.DataPointer)[i] * Damp);

        float[] Run(Tensor value)
        {
            using Tensor o = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
            AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, value, null, scale, allowF16: false);
            backend.Attention(bf16Grant ? call with { Numerics = AttentionNumerics.Bf16Storage } : call);
            backend.Sync();
            return AttnInvoker.ReadF32(o);
        }

        float[] plain = Run(v), viaDamp = Run(damped);
        using Tensor expected = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(expected, q, k, v, null, scale, false));
        // A row that ignored V would pass the comparison below vacuously.
        double rel = AttnAssert.Measure(plain, AttnInvoker.ReadF32(expected), HeadDim).RelL2;
        Assert.True(rel <= 8e-3, $"the output is {rel:E2} from the contract reference");
        int differ = 0;
        for (int i = 0; i < plain.Length; i++)
            if (BitConverter.SingleToUInt32Bits(plain[i]) != BitConverter.SingleToUInt32Bits(viaDamp[i] / Damp)) differ++;
        _out.WriteLine($"{kind}{(bf16Grant ? " with the BF16 grant" : "")}: {differ} of {plain.Length} outputs differ; relL2 {rel:E2}");
        Assert.True(differ == 0, $"{differ} of {plain.Length} {kind} outputs moved when V was scaled by 2^-8 and the output by 2^8");
    }
}
