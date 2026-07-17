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
[Trait("Category", "GpuIntegration")]
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
    [InlineData(32, 64, 48)]    // single tile, aligned
    [InlineData(17, 64, 20)]    // single tile, M/N not multiples of 16 — bounds check
    [InlineData(8, 256, 8)]     // single M/N tile, multiple K tiles
    [InlineData(128, 128, 96)]  // multiple BM/BN tiles + multiple K tiles, all tile-aligned
    [InlineData(130, 96, 70)]   // multiple tiles, none tile-aligned — exercises partial edge tiles
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
        Tensor sa = new(new TensorShape(M), DType.F32);    // per-row scale (uniform == per-tensor)
        Tensor sb = new(new TensorShape(N), DType.F32);
        try
        {
            sbyte* ap = (sbyte*)a.DataPointer;
            sbyte* bp = (sbyte*)b.DataPointer;
            Random rng = new(1234);
            for (int i = 0; i < M * K; i++) ap[i] = (sbyte)rng.Next(-127, 128);
            for (int i = 0; i < N * K; i++) bp[i] = (sbyte)rng.Next(-127, 128);
            Span<float> sas = sa.AsSpan<float>(); for (int i = 0; i < M; i++) sas[i] = scaleA;
            Span<float> sbs = sb.AsSpan<float>(); for (int i = 0; i < N; i++) sbs[i] = scaleB;

            backend.MatMulInt8(c, a, b, sa, sb);

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
        finally { a.Dispose(); b.Dispose(); c.Dispose(); sa.Dispose(); sb.Dispose(); }
    }

    [Theory]
    [InlineData(64, 128, 96)]
    [InlineData(96, 256, 128)]
    public unsafe void MatMulInt8_QuantizedWeights_ApproximatesFloatMatmul(int M, int K, int N)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Vk.HasInt8DotProduct) { _out.WriteLine("SKIPPED: no int8 dot"); return; }

        // Random FP activations and weights with per-row magnitude variation, so per-row scaling
        // actually matters (a per-tensor scale would be far less accurate here).
        Tensor aF = new(new TensorShape(M, K), DType.F32);
        Tensor bF = new(new TensorShape(N, K), DType.F32);
        Tensor c = new(new TensorShape(M, N), DType.F32);
        Tensor? aI8 = null, aScale = null, bI8 = null, bScale = null;
        try
        {
            Span<float> af = aF.AsSpan<float>();
            Span<float> bf = bF.AsSpan<float>();
            Random rng = new(77);
            for (int r = 0; r < M; r++)
            {
                float mag = 0.5f + (float)rng.NextDouble() * 3.0f;   // per-row magnitude varies
                for (int k = 0; k < K; k++) af[r * K + k] = (float)(rng.NextDouble() * 2.0 - 1.0) * mag;
            }
            for (int r = 0; r < N; r++)
            {
                float mag = 0.5f + (float)rng.NextDouble() * 3.0f;
                for (int k = 0; k < K; k++) bf[r * K + k] = (float)(rng.NextDouble() * 2.0 - 1.0) * mag;
            }

            (aI8, aScale) = Int8Quantizer.RowwiseSymmetric(aF);
            (bI8, bScale) = Int8Quantizer.RowwiseSymmetric(bF);

            backend.MatMulInt8(c, aI8, bI8, aScale, bScale);

            // FP reference in double, and relative Frobenius error vs the INT8 result.
            ReadOnlySpan<float> cs = c.AsReadOnlySpan<float>();
            double sqErr = 0.0, sqRef = 0.0;
            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                {
                    double acc = 0.0;
                    for (int k = 0; k < K; k++) acc += (double)af[m * K + k] * bf[n * K + k];
                    double diff = cs[m * N + n] - acc;
                    sqErr += diff * diff;
                    sqRef += acc * acc;
                }
            }
            double relErr = Math.Sqrt(sqErr / sqRef);
            _out.WriteLine($"INT8 quantized {M}x{N}x{K}: relative Frobenius error = {relErr:P3}");
            // Per-row symmetric INT8 (~1/127 per element) on this data lands well under 2%.
            Assert.True(relErr < 0.02, $"INT8 quantized matmul rel error {relErr:P3} too high — quantization or dequant wrong.");
        }
        finally
        {
            aF.Dispose(); bF.Dispose(); c.Dispose();
            aI8?.Dispose(); aScale?.Dispose(); bI8?.Dispose(); bScale?.Dispose();
        }
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
        Tensor sa = new(new TensorShape(16), DType.F32);
        Tensor sb = new(new TensorShape(16), DType.F32);
        try
        {
            Assert.Throws<ArgumentException>(() => backend.MatMulInt8(c, a, b, sa, sb));
        }
        finally { a.Dispose(); b.Dispose(); c.Dispose(); sa.Dispose(); sb.Dispose(); }
    }
}
