using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The opt-in INT8 Linear quantizes both operands on the device — the weight once, cached beside its other casts — and its
/// result stays within INT8's own error of the F32 product; a shape the packed kernel cannot take falls back to the F32 path unchanged.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanInt8LinearTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    private static Tensor Rand(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        return t;
    }

    private float MaxRelError(Tensor a, Tensor b)
    {
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> y = b.AsReadOnlySpan<float>();
        float maxErr = 0f, maxAbs = 0f;
        for (int i = 0; i < x.Length; i++) { maxErr = MathF.Max(maxErr, MathF.Abs(x[i] - y[i])); maxAbs = MathF.Max(maxAbs, MathF.Abs(y[i])); }
        float rel = maxAbs > 0 ? maxErr / maxAbs : maxErr;
        _out.WriteLine($"maxErr={maxErr:E3} maxAbs={maxAbs:E3} rel={rel:E3}");
        return rel;
    }

    [Fact]
    public void Int8Linear_MatchesTheF32Product_AndReusesTheCachedWeight()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (!backend.Vk.HasInt8DotProduct) { _out.WriteLine("SKIPPED: no integer dot product on this device"); return; }
        const int M = 64, K = 256, N = 128;
        using Tensor x = Rand(new TensorShape(M, K), 1);
        using Tensor w = Rand(new TensorShape(N, K), 2);
        using Tensor bias = Rand(new TensorShape(N), 3);
        backend.PreloadWeights([w]);

        backend.EnableInt8Linear = false;
        using Tensor outF32 = new(new TensorShape(M, N), DType.F32);
        backend.Linear(outF32, x, w, bias);
        backend.Sync();

        backend.EnableInt8Linear = true;
        using Tensor outInt8 = new(new TensorShape(M, N), DType.F32);
        backend.Linear(outInt8, x, w, bias);
        backend.Sync();
        Assert.True(MaxRelError(outInt8, outF32) < 3e-2f, "the INT8 product is outside INT8's error of the F32 one");

        // The second call reads the cached INT8 weight; freeing the weight drops it and the third call rebuilds it.
        using Tensor outAgain = new(new TensorShape(M, N), DType.F32);
        backend.Linear(outAgain, x, w, bias);
        backend.Sync();
        Assert.Equal(outInt8.AsReadOnlySpan<float>().ToArray(), outAgain.AsReadOnlySpan<float>().ToArray());
        backend.FreeWeights([w]);
        using Tensor outRebuilt = new(new TensorShape(M, N), DType.F32);
        backend.Linear(outRebuilt, x, w, bias);
        backend.Sync();
        Assert.Equal(outInt8.AsReadOnlySpan<float>().ToArray(), outRebuilt.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Int8Linear_FallsBackOnAnUnpackableK()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = VulkanTestDevice.Create();
        const int M = 16, K = 130, N = 32;   // K % 4 != 0: no packed int8 words
        using Tensor x = Rand(new TensorShape(M, K), 4);
        using Tensor w = Rand(new TensorShape(N, K), 5);
        backend.EnableInt8Linear = false;
        using Tensor outF32 = new(new TensorShape(M, N), DType.F32);
        backend.Linear(outF32, x, w, null);
        backend.Sync();
        backend.EnableInt8Linear = true;
        using Tensor outKnob = new(new TensorShape(M, N), DType.F32);
        backend.Linear(outKnob, x, w, null);
        backend.Sync();
        Assert.Equal(outF32.AsReadOnlySpan<float>().ToArray(), outKnob.AsReadOnlySpan<float>().ToArray());
    }
}
