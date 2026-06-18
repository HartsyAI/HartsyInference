using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Correctness tests for the INT8 dot-product GEMM (<see cref="VulkanBackend.MatMulInt8"/>),
/// the cross-vendor DP4a/IMMA equivalent via <c>dotPacked4x8</c>. The kernel accumulates the int8
/// products exactly in int32, so the result is compared against an exact int64 reference (only the
/// final float scale rounds), giving a tight tolerance. The feature is HW-accelerated on
/// NVIDIA/AMD/Intel; tests self-skip on a device that lacks it (e.g. llvmpipe).</summary>
public sealed class VulkanInt8GemmTests
{
    private readonly ITestOutputHelper _out;
    public VulkanInt8GemmTests(ITestOutputHelper output) => _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    [Theory]
    [InlineData(32, 64, 48)]    // aligned to 16
    [InlineData(17, 64, 20)]    // M, N not multiples of 16 — exercises the bounds check
    [InlineData(8, 256, 8)]     // larger K
    public unsafe void MatMulInt8_MatchesExactIntegerReference(int M, int K, int N)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Vk.HasInt8DotProduct)
        {
            _out.WriteLine($"SKIPPED: {backend.Vk.DeviceName} has no integer dot-product support");
            return;
        }
        _out.WriteLine($"Device: {backend.Vk.DeviceName}, int8dot={backend.Vk.HasInt8DotProduct}");

        const float scaleA = 0.011f, scaleB = 0.019f;
        Tensor a = new(new TensorShape(M, K), DType.I8);   // activations
        Tensor b = new(new TensorShape(N, K), DType.I8);   // weights [N,K]
        Tensor c = new(new TensorShape(M, N), DType.F32);
        try
        {
            sbyte* ap = (sbyte*)a.DataPointer;
            sbyte* bp = (sbyte*)b.DataPointer;
            Random rng = new(1234);
            for (int i = 0; i < M * K; i++) ap[i] = (sbyte)rng.Next(-127, 128);
            for (int i = 0; i < N * K; i++) bp[i] = (sbyte)rng.Next(-127, 128);

            backend.MatMulInt8(c, a, b, scaleA, scaleB);

            ReadOnlySpan<float> cs = c.AsReadOnlySpan<float>();
            float maxErr = 0f;
            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                {
                    long acc = 0;
                    for (int k = 0; k < K; k++) acc += (long)ap[m * K + k] * bp[n * K + k];
                    float expected = (float)acc * scaleA * scaleB;
                    maxErr = MathF.Max(maxErr, MathF.Abs(cs[m * N + n] - expected));
                }
            }
            _out.WriteLine($"INT8 GEMM {M}x{N}x{K}: maxErr={maxErr:E3}");
            // Only the final float multiply rounds; the integer dot is exact. Tight tolerance.
            Assert.True(maxErr < 5e-3f, $"INT8 GEMM maxErr {maxErr:E3} exceeds tolerance — dotPacked4x8 result is wrong.");
        }
        finally { a.Dispose(); b.Dispose(); c.Dispose(); }
    }

    [Fact]
    public unsafe void MatMulInt8_RejectsUnalignedK()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Vk.HasInt8DotProduct) { _out.WriteLine("SKIPPED: no int8 dot"); return; }

        Tensor a = new(new TensorShape(16, 18), DType.I8);  // K=18 not a multiple of 4
        Tensor b = new(new TensorShape(16, 18), DType.I8);
        Tensor c = new(new TensorShape(16, 16), DType.F32);
        try
        {
            Assert.Throws<ArgumentException>(() => backend.MatMulInt8(c, a, b, 1f, 1f));
        }
        finally { a.Dispose(); b.Dispose(); c.Dispose(); }
    }
}
