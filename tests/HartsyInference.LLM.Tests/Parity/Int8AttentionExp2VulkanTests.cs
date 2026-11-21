using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary><c>exp2Fixed</c> (Shaders/exp2_fixed.glsl) against its host twin <see cref="Int8AttentionExp2.Evaluate"/>.
/// <para>The bound was fixed before the first run: bit-equal for every argument, in range and out of it. How far the
/// device's own <c>exp2()</c> is from the twin is counted and reported, not asserted.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class Int8AttentionExp2VulkanTests(ITestOutputHelper output)
{
    private static float[] Arguments()
    {
        List<float> x = [float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f, -0f, Int8AttentionV2Reference.U8Offset];
        // Every whole number and every half, where the split into k and r changes, with the floats on each side.
        for (int k = -127; k <= 129; k++)
            foreach (float at in (float[])[k, k + 0.5f])
            {
                x.Add(at);
                x.Add(MathF.BitIncrement(at));
                x.Add(MathF.BitDecrement(at));
            }
        Random random = new(91);
        for (int i = 0; i < 400000; i++) x.Add((float)(random.NextDouble() * 258 - 129));
        // What the attention pass asks for: a score less a tile's or the running maximum.
        for (int i = 0; i < 400000; i++) x.Add((float)(random.NextDouble() * 48 - 40));
        for (int i = 0; i < 100000; i++) x.Add((float)(random.NextDouble() - 0.5));
        return [.. x];
    }

    [VulkanFact]
    public void Exp2FixedEqualsItsHostTwin()
    {
        float[] x = Arguments();
        int n = x.Length;
        using VulkanBackend backend = new();
        using Tensor arguments = new(new TensorShape(n), DType.F32);
        x.CopyTo(arguments.AsSpan<float>());
        using Tensor fixedResult = new(arguments.Shape, DType.F32), native = new(arguments.Shape, DType.F32);
        backend.Exp2FixedProbe(fixedResult, native, arguments);
        backend.Sync();
        ReadOnlySpan<float> f = fixedResult.AsReadOnlySpan<float>(), d = native.AsReadOnlySpan<float>();

        int misses = 0, first = -1, nativeDiffers = 0;
        long nativeWorst = 0;
        for (int i = 0; i < n; i++)
        {
            float twin = Int8AttentionExp2.Evaluate(x[i]);
            bool same = float.IsNaN(twin) ? float.IsNaN(f[i])
                : BitConverter.SingleToInt32Bits(twin) == BitConverter.SingleToInt32Bits(f[i]);
            if (!same)
            {
                if (first < 0) first = i;
                misses++;
            }
            if (float.IsFinite(twin) && twin > 0f && float.IsFinite(d[i]) && d[i] > 0f)
            {
                long apart = Math.Abs((long)BitConverter.SingleToInt32Bits(twin) - BitConverter.SingleToInt32Bits(d[i]));
                if (apart != 0) nativeDiffers++;
                nativeWorst = Math.Max(nativeWorst, apart);
            }
        }
        output.WriteLine($"{n} arguments; exp2Fixed differs from its twin in {misses}; the device exp2() differs from the twin "
            + $"in {nativeDiffers}, by at most {nativeWorst} units in the last place");
        Assert.True(misses == 0, first < 0 ? "" : $"exp2Fixed differs in {misses} of {n}; first is {x[first]:R}: "
            + $"{f[first]:R}, expected {Int8AttentionExp2.Evaluate(x[first]):R}");
    }
}
