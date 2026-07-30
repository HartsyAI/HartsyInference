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

    /// <summary>Gate for <c>VulkanBackend.TryDispatchInt8Linear</c> — the opt-in wiring of the
    /// already-validated INT8 dot-product GEMM into <see cref="VulkanBackend.Linear"/> (toggled via the
    /// settable <see cref="VulkanBackend.EnableInt8Linear"/> property, matching <c>CudaBackend.EnableW8A8</c>'s
    /// pattern so tests can flip it without an env var + fresh process). Runs the SAME input twice — once
    /// with the opt-in off (exact GEMM path) and once on — so this can't pass by silently falling through
    /// to the exact path: the off-run must match the CPU reference almost exactly, while the on-run must
    /// show real (but bounded) INT8 quantization error, proving the dot-product path actually dispatched.
    /// The bound is consistent with <see cref="MatMulInt8_QuantizedWeights_ApproximatesFloatMatmul"/>
    /// above, since this re-quantizes both operands the same way (per-row symmetric INT8) before
    /// dispatching the same kernel.</summary>
    [Theory]
    [InlineData(64, 128, 96)]
    [InlineData(96, 256, 128)]
    public unsafe void Backend_Linear_Int8OptIn_ApproximatesF32Reference(int M, int K, int N)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Vk.HasInt8DotProduct) { _out.WriteLine("SKIPPED: no int8 dot"); return; }

        Tensor input = new(new TensorShape(M, K), DType.F32);
        Tensor weight = new(new TensorShape(N, K), DType.F32);
        Tensor bias = new(new TensorShape(N), DType.F32);
        Tensor outputExact = new(new TensorShape(M, N), DType.F32);
        Tensor outputInt8 = new(new TensorShape(M, N), DType.F32);
        try
        {
            Span<float> iS = input.AsSpan<float>();
            Span<float> wS = weight.AsSpan<float>();
            Span<float> bS = bias.AsSpan<float>();
            Random rng = new(99);
            for (int r = 0; r < M; r++)
            {
                float mag = 0.5f + (float)rng.NextDouble() * 3.0f;
                for (int k = 0; k < K; k++) iS[r * K + k] = (float)(rng.NextDouble() * 2.0 - 1.0) * mag;
            }
            for (int r = 0; r < N; r++)
            {
                float mag = 0.5f + (float)rng.NextDouble() * 3.0f;
                for (int k = 0; k < K; k++) wS[r * K + k] = (float)(rng.NextDouble() * 2.0 - 1.0) * mag;
            }
            for (int n = 0; n < N; n++) bS[n] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.1f;

            backend.EnableInt8Linear = false;
            backend.Linear(outputExact, input, weight, bias);
            backend.EnableInt8Linear = true;
            backend.Linear(outputInt8, input, weight, bias);

            ReadOnlySpan<float> exactS = outputExact.AsReadOnlySpan<float>();
            ReadOnlySpan<float> int8S = outputInt8.AsReadOnlySpan<float>();
            double sqErrExact = 0.0, sqErrInt8 = 0.0, sqRef = 0.0;
            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                {
                    double acc = bS[n];
                    for (int k = 0; k < K; k++) acc += (double)iS[m * K + k] * wS[n * K + k];
                    double diffExact = exactS[m * N + n] - acc;
                    double diffInt8 = int8S[m * N + n] - acc;
                    sqErrExact += diffExact * diffExact;
                    sqErrInt8 += diffInt8 * diffInt8;
                    sqRef += acc * acc;
                }
            }
            double relErrExact = Math.Sqrt(sqErrExact / sqRef);
            double relErrInt8 = Math.Sqrt(sqErrInt8 / sqRef);
            _out.WriteLine($"Linear {M}x{N}x{K}: exact relErr={relErrExact:P4}, INT8 opt-in relErr={relErrInt8:P4}");
            Assert.True(relErrExact < 1e-4, $"Exact-path (opt-in off) rel error {relErrExact:P4} should be ~0 — GEMM regression.");
            Assert.True(relErrInt8 < 0.02, $"Linear INT8 opt-in rel error {relErrInt8:P4} too high vs F32 reference.");
            Assert.True(relErrInt8 > relErrExact * 10,
                $"INT8 opt-in relErr ({relErrInt8:P4}) is barely above the exact path's ({relErrExact:P4}) — " +
                "suggests TryDispatchInt8Linear silently fell through to the exact GEMM instead of actually quantizing.");
        }
        finally { input.Dispose(); weight.Dispose(); bias.Dispose(); outputExact.Dispose(); outputInt8.Dispose(); }
    }

    /// <summary>Discriminates a real residency hazard: <c>TryDispatchInt8Linear</c>'s bias add reads
    /// <c>output.DataPointer</c> on the host AFTER the GPU GEMM already cached <c>output</c> as
    /// GPU-resident. If that host write didn't correctly invalidate the cached device buffer, a downstream
    /// GPU op consuming <c>output</c> would silently read stale (pre-bias) data — the same class of bug as
    /// the flash-attention <c>kvCapacity</c> issue, only visible with realistic buffer chaining, which is
    /// why <see cref="Backend_Linear_Int8OptIn_ApproximatesF32Reference"/> above (which only reads the
    /// host span) can't catch it. Chains <c>output</c> straight into a second GPU op (<c>Silu</c>) with no
    /// intervening host read, and checks the result reflects the bias.</summary>
    [Fact]
    public unsafe void Backend_Linear_Int8OptIn_BiasSurvivesDownstreamGpuConsumption()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Vk.HasInt8DotProduct) { _out.WriteLine("SKIPPED: no int8 dot"); return; }
        backend.EnableInt8Linear = true;

        const int M = 32, K = 64, N = 32;
        Tensor input = new(new TensorShape(M, K), DType.F32);
        Tensor weight = new(new TensorShape(N, K), DType.F32);
        Tensor bias = new(new TensorShape(N), DType.F32);
        Tensor linearOut = new(new TensorShape(M, N), DType.F32);
        Tensor siluOut = new(new TensorShape(M, N), DType.F32);
        try
        {
            Span<float> iS = input.AsSpan<float>();
            Span<float> wS = weight.AsSpan<float>();
            Span<float> bS = bias.AsSpan<float>();
            Random rng = new(11);
            for (int i = 0; i < M * K; i++) iS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
            // Large bias relative to the dot-product magnitude: if the downstream Silu sees the
            // pre-bias value instead, the mismatch is unmistakable, not lost in quantization noise.
            for (int n = 0; n < N; n++) bS[n] = 5.0f + (float)n * 0.1f;

            backend.Linear(linearOut, input, weight, bias);      // caches linearOut as GPU-resident
            backend.Silu(siluOut, linearOut);                    // consumes it with NO intervening host read

            ReadOnlySpan<float> siluS = siluOut.AsReadOnlySpan<float>();
            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                {
                    double acc = bS[n];
                    for (int k = 0; k < K; k++) acc += (double)iS[m * K + k] * wS[n * K + k];
                    double expectedSilu = acc / (1.0 + Math.Exp(-acc));
                    Assert.True(Math.Abs(siluS[m * N + n] - expectedSilu) < Math.Abs(expectedSilu) * 0.05 + 0.05,
                        $"Silu(Linear(...))[{m},{n}]={siluS[m * N + n]:G6} vs expected={expectedSilu:G6} — " +
                        "downstream GPU consumer likely saw a stale pre-bias buffer.");
                }
            }
        }
        finally { input.Dispose(); weight.Dispose(); bias.Dispose(); linearOut.Dispose(); siluOut.Dispose(); }
    }

    /// <summary>With <see cref="VulkanBackend.EnableInt8Linear"/> left at its default (off), <c>Linear</c>
    /// must take the ordinary GEMM path exactly as before — the opt-in must not change default behavior.</summary>
    [Fact]
    public void Backend_Linear_Int8OptIn_DefaultsOff()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        Assert.False(backend.EnableInt8Linear, "EnableInt8Linear must default to off unless HARTSYINFERENCE_VK_INT8=1 is set.");
    }

    /// <summary>Even with the opt-in enabled, a K not divisible by 4 must fall through to the exact
    /// GEMM path untouched (<c>TryDispatchInt8Linear</c>'s explicit guard) — verified by requiring an
    /// exact (not merely approximate) match against the CPU reference.</summary>
    [Fact]
    public unsafe void Backend_Linear_Int8OptIn_FallsBackWhenKNotMultipleOfFour()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Vk.HasInt8DotProduct) { _out.WriteLine("SKIPPED: no int8 dot"); return; }
        backend.EnableInt8Linear = true;

        const int M = 8, K = 17, N = 8;   // K % 4 != 0
        Tensor input = new(new TensorShape(M, K), DType.F32);
        Tensor weight = new(new TensorShape(N, K), DType.F32);
        Tensor output = new(new TensorShape(M, N), DType.F32);
        try
        {
            Span<float> iS = input.AsSpan<float>();
            Span<float> wS = weight.AsSpan<float>();
            Random rng = new(5);
            for (int i = 0; i < M * K; i++) iS[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < N * K; i++) wS[i] = (float)(rng.NextDouble() * 2 - 1);

            backend.Linear(output, input, weight, null);

            ReadOnlySpan<float> oS = output.AsReadOnlySpan<float>();
            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                {
                    float acc = 0;
                    for (int k = 0; k < K; k++) acc += iS[m * K + k] * wS[n * K + k];
                    Assert.InRange(oS[m * N + n] - acc, -1e-3f, 1e-3f);
                }
            }
        }
        finally { input.Dispose(); weight.Dispose(); output.Dispose(); }
    }
}
