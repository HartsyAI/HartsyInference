using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Cross-vendor correctness tests targeting the small-subgroup reduction path. With a
/// large normalized dimension every subgroup in the workgroup holds real partial sums, so the
/// cross-subgroup combine must fold ALL of them. On a device whose subgroup size is smaller than
/// the subgroup count (e.g. llvmpipe: subgroup 8 at local 256 = 32 subgroups), the previous
/// "read warp[laneId], guarded by < numSubgroups" combine silently dropped partials 8..31 and
/// produced wrong results. The strided fold fixes that for all subgroup sizes. Run these against
/// both the NVIDIA device and llvmpipe (set VK_ICD_FILENAMES to the lvp ICD) to cover both regimes.</summary>
public sealed class VulkanCrossVendorTests
{
    private readonly ITestOutputHelper _out;
    public VulkanCrossVendorTests(ITestOutputHelper output) => _out = output;

    private static bool VulkanAvailable()
    {
        try
        {
            using VulkanInstance instance = new();
            return instance.EnumeratePhysicalDevices().Length > 0;
        }
        catch { return false; }
    }

    [Fact]
    public unsafe void LayerNorm_LargeDim_AllSubgroupsHaveData_MatchesCpu()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        _out.WriteLine($"Device: {backend.Vk.DeviceName}, subgroup={backend.Vk.SubgroupSize}");

        // D=1024 with the norm local_size (256) means every thread (and thus every subgroup)
        // processes real elements — the discriminating case for the cross-subgroup fold.
        const int B = 2, T = 2, D = 1024;
        Tensor x = new(new TensorShape(B, T, D), DType.F32);
        Tensor w = new(new TensorShape(D), DType.F32);
        Tensor b = new(new TensorShape(D), DType.F32);
        Tensor y = new(new TensorShape(B, T, D), DType.F32);
        try
        {
            Span<float> xS = x.AsSpan<float>();
            Span<float> wS = w.AsSpan<float>();
            Span<float> bS = b.AsSpan<float>();
            for (int i = 0; i < B * T * D; i++) xS[i] = MathF.Sin(i * 0.017f) * 3.0f + 0.5f;
            for (int i = 0; i < D; i++) { wS[i] = 1.0f; bS[i] = 0.0f; }

            backend.LayerNorm(y, x, w, b, 1e-5f);

            ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
            float maxErr = 0f;
            for (int row = 0; row < B * T; row++)
            {
                double mean = 0.0;
                for (int d = 0; d < D; d++) mean += xS[row * D + d];
                mean /= D;
                double var_ = 0.0;
                for (int d = 0; d < D; d++) { double v = xS[row * D + d] - mean; var_ += v * v; }
                var_ /= D;
                float invStd = (float)(1.0 / Math.Sqrt(var_ + 1e-5));
                for (int d = 0; d < D; d++)
                {
                    float expected = (float)((xS[row * D + d] - mean) * invStd);
                    maxErr = MathF.Max(maxErr, MathF.Abs(yS[row * D + d] - expected));
                }
            }
            _out.WriteLine($"LayerNorm D={D} maxErr={maxErr:E3}");
            Assert.True(maxErr < 2e-3f, $"LayerNorm large-dim maxErr {maxErr:E3} too high — cross-subgroup fold likely dropping partials.");
        }
        finally { x.Dispose(); w.Dispose(); b.Dispose(); y.Dispose(); }
    }

    [Fact]
    public unsafe void RmsNorm_LargeDim_AllSubgroupsHaveData_MatchesCpu()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        _out.WriteLine($"Device: {backend.Vk.DeviceName}, subgroup={backend.Vk.SubgroupSize}");

        // Single-accumulator (sum of squares) strided-fold variant.
        const int rows = 3, D = 1024;
        Tensor x = new(new TensorShape(rows, D), DType.F32);
        Tensor w = new(new TensorShape(D), DType.F32);
        Tensor y = new(new TensorShape(rows, D), DType.F32);
        try
        {
            Span<float> xS = x.AsSpan<float>();
            Span<float> wS = w.AsSpan<float>();
            for (int i = 0; i < rows * D; i++) xS[i] = MathF.Cos(i * 0.013f) * 2.5f + 0.3f;
            for (int i = 0; i < D; i++) wS[i] = 1.0f;

            backend.RmsNorm(y, x, w, 1e-6f);

            ReadOnlySpan<float> yS = y.AsReadOnlySpan<float>();
            float maxErr = 0f;
            for (int r = 0; r < rows; r++)
            {
                double ms = 0.0;
                for (int d = 0; d < D; d++) ms += (double)xS[r * D + d] * xS[r * D + d];
                ms /= D;
                float invRms = (float)(1.0 / Math.Sqrt(ms + 1e-6));
                for (int d = 0; d < D; d++)
                {
                    float expected = xS[r * D + d] * invRms;
                    maxErr = MathF.Max(maxErr, MathF.Abs(yS[r * D + d] - expected));
                }
            }
            _out.WriteLine($"RmsNorm D={D} maxErr={maxErr:E3}");
            Assert.True(maxErr < 2e-3f, $"RmsNorm large-dim maxErr {maxErr:E3} too high — cross-subgroup fold likely dropping partials.");
        }
        finally { x.Dispose(); w.Dispose(); y.Dispose(); }
    }
}
