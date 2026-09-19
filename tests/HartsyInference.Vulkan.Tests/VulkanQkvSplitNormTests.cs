using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Parity gate for the fused QKV split + per-head QK-RMSNorm on Vulkan.
///
/// <para>This was the one op on the Flux/DiT path whose host default is a TRUE fallback rather than composition:
/// it reads <c>DataPointer</c> on six tensors, so every call cost a device-to-host sync, a scalar loop over every
/// token and head, and an upload of the results. Flux runs 19 double plus 38 single blocks per forward, each
/// calling it once per step.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanQkvSplitNormTests
{
    private readonly ITestOutputHelper _output;

    public VulkanQkvSplitNormTests(ITestOutputHelper output) => _output = output;

    private static Tensor Filled(TensorShape shape, int seed, double centre = 0.0)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> span = t.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = (float)(rng.NextDouble() * 2.0 - 1.0 + centre);
        }
        return t;
    }

    /// <summary>Head dims that straddle every plausible subgroup width, because the cross-subgroup fold is where a
    /// per-head reduction silently normalizes by the wrong denominator.</summary>
    [Theory]
    [InlineData(6, 24, 64)]    // Flux double-stream: 24 heads x 64
    [InlineData(4, 8, 128)]    // headDim twice a 64-wide subgroup
    [InlineData(3, 5, 40)]     // headDim below a subgroup and not a power of two
    [InlineData(1, 1, 256)]    // a single wide head, several subgroups deep
    public void MatchesCpuReference(int tokens, int heads, int headDim)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        int w = heads * headDim;
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();

        // Off-centre input: a norm that dropped the scale or used the wrong denominator still looks plausible on
        // well-conditioned data.
        using Tensor qkv = Filled(new TensorShape(tokens, 3 * w), seed: 5, centre: 0.4);
        using Tensor qWeight = Filled(new TensorShape(headDim), seed: 6, centre: 1.0);
        using Tensor kWeight = Filled(new TensorShape(headDim), seed: 7, centre: 1.0);

        using Tensor gq = new(new TensorShape(tokens, w), DType.F32);
        using Tensor gk = new(new TensorShape(tokens, w), DType.F32);
        using Tensor gv = new(new TensorShape(tokens, w), DType.F32);
        using Tensor cq = new(new TensorShape(tokens, w), DType.F32);
        using Tensor ck = new(new TensorShape(tokens, w), DType.F32);
        using Tensor cv = new(new TensorShape(tokens, w), DType.F32);

        gpu.QkvSplitNorm(gq, gk, gv, qkv, qWeight, kWeight, 1e-6f);
        cpu.QkvSplitNorm(cq, ck, cv, qkv, qWeight, kWeight, 1e-6f);

        double worstQ = Worst(gq, cq), worstK = Worst(gk, ck), worstV = Worst(gv, cv);
        _output.WriteLine($"[{tokens}t x {heads}h x {headDim}d] q={worstQ:E3} k={worstK:E3} v={worstV:E3}");
        Assert.True(worstQ < 1e-4, $"q diverges: {worstQ:E3}");
        Assert.True(worstK < 1e-4, $"k diverges: {worstK:E3}");
        Assert.True(worstV == 0.0, $"v is a straight copy and must be exact, got {worstV:E3}");
    }

    /// <summary>LayerNorm + per-batch modulation, the other norm every DiT block runs.</summary>
    /// <remarks>The batch index is <c>row / seqLen</c>, so the shapes below deliberately include more than one
    /// batch entry: a kernel that ignored the divide would read batch 0's modulation for every row and still look
    /// correct on a single-entry batch.</remarks>
    [Theory]
    [InlineData(1, 16, 320)]
    [InlineData(3, 7, 128)]
    [InlineData(2, 1, 64)]
    public void LayerNormModulate_MatchesCpuReference(int batch, int seqLen, int dim)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        TensorShape shape = new(batch, seqLen, dim);
        using Tensor input = Filled(shape, seed: 31, centre: 0.7);
        using Tensor scale = Filled(new TensorShape(batch, dim), seed: 32);
        using Tensor shift = Filled(new TensorShape(batch, dim), seed: 33);
        using Tensor got = new(shape, DType.F32);
        using Tensor want = new(shape, DType.F32);

        gpu.LayerNormModulate(got, input, scale, shift, 1e-6f);
        cpu.LayerNormModulate(want, input, scale, shift, 1e-6f);

        double worst = Worst(got, want);
        _output.WriteLine($"[{batch}b x {seqLen}s x {dim}d] max abs err = {worst:E3}");
        Assert.True(worst < 1e-4, $"LayerNormModulate diverges: {worst:E3}");
    }

    /// <summary>In-place rotary on one tensor, full and partial.</summary>
    /// <remarks>The partial cases are the ones worth having: a kernel that ignored rotaryDim and rotated the whole
    /// head would pass every full-rotary shape and corrupt exactly the models that use partial rotary.</remarks>
    [Theory]
    [InlineData(1, 6, 4, 64, 0)]     // full rotary, rotaryDim defaulted
    [InlineData(2, 3, 2, 128, 128)]  // full rotary, stated explicitly
    [InlineData(1, 5, 3, 128, 64)]   // partial: half the head rotates, half must be untouched
    [InlineData(1, 2, 1, 96, 32)]    // partial with an odd-ish head width
    public void ApplyRopeSingle_MatchesCpuReference(int batch, int seqLen, int heads, int headDim, int rotaryDim)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        TensorShape shape = new(batch, seqLen, heads, headDim);
        using Tensor gpuX = Filled(shape, seed: 41, centre: 0.3);
        using Tensor cpuX = Filled(shape, seed: 41, centre: 0.3);
        using Tensor cos = Filled(new TensorShape(batch, seqLen, headDim), seed: 42);
        using Tensor sin = Filled(new TensorShape(batch, seqLen, headDim), seed: 43);

        gpu.ApplyRopeSingle(gpuX, cos, sin, rotaryDim);
        cpu.ApplyRopeSingle(cpuX, cos, sin, rotaryDim);

        double worst = Worst(gpuX, cpuX);
        _output.WriteLine($"[{batch}b x {seqLen}s x {heads}h x {headDim}d rot={rotaryDim}] max abs err = {worst:E3}");
        Assert.True(worst < 1e-5, $"ApplyRopeSingle diverges: {worst:E3}");
    }

    private static double Worst(Tensor a, Tensor b)
    {
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> y = b.AsReadOnlySpan<float>();
        double worst = 0;
        for (int i = 0; i < x.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(x[i] - y[i]));
        }
        return worst;
    }

    private static string SpirvDir()
    {
        string local = Path.Combine(AppContext.BaseDirectory, "Spirv");
        return Directory.Exists(local) ? local : Path.Combine(RepoRoot.Path, "src", "HartsyInference.Vulkan", "Spirv");
    }

    private static bool VulkanAvailable(out string? reason)
    {
        try
        {
            using VulkanBackend probe = new(0, SpirvDir());
            reason = null;
            return true;
        }
        catch (Exception ex)
        {
            reason = ex.Message;
            return false;
        }
    }
}
