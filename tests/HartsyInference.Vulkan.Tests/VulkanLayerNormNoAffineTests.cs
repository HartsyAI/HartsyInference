using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Parity gate for <c>LayerNormNoAffine</c> on Vulkan.
///
/// <para>This op had no Vulkan override at all, so every call fell through to <see cref="IBackend"/>'s host
/// default — which refuses anything but F32 outright. A Flux generation on Vulkan hit exactly that: the DiT runs
/// its block activations in F16 by default, so the pre-modulation norm handed the default an F16 tensor and the
/// generation died with "LayerNormNoAffine default fallback only supports F32". Every DiT normalizes before
/// modulating, so the op sits on the hot path of the whole family.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanLayerNormNoAffineTests
{
    private readonly ITestOutputHelper _output;

    public VulkanLayerNormNoAffineTests(ITestOutputHelper output) => _output = output;

    private static Tensor Random(TensorShape shape, int seed, DType dtype)
    {
        Tensor f32 = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> span = f32.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
        {
            // Deliberately off-centre and wide: a norm that silently skipped the mean subtraction still looks
            // right on zero-mean unit-variance input.
            span[i] = (float)(rng.NextDouble() * 6.0 - 1.5);
        }
        if (dtype == DType.F32)
        {
            return f32;
        }
        Tensor cast = f32.CastTo(dtype);
        f32.Dispose();
        return cast;
    }

    [Theory]
    [InlineData(4, 320)]      // a small DiT row
    [InlineData(2, 3072)]     // Flux hidden size
    [InlineData(7, 1024)]     // rows not a multiple of the workgroup, dim that is
    [InlineData(1, 129)]      // dim not a multiple of the subgroup, to exercise the strided fold
    public void MatchesCpuReference_F32(int rows, int dim)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        TensorShape shape = new(rows, dim);
        using Tensor input = Random(shape, seed: 11, DType.F32);
        using Tensor actual = new(shape, DType.F32);
        using Tensor expected = new(shape, DType.F32);

        gpu.LayerNormNoAffine(actual, input, 1e-6f);
        cpu.LayerNormNoAffine(expected, input, 1e-6f);

        ReadOnlySpan<float> got = actual.AsReadOnlySpan<float>();
        ReadOnlySpan<float> want = expected.AsReadOnlySpan<float>();
        double worst = 0;
        for (int i = 0; i < got.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(got[i] - want[i]));
        }
        _output.WriteLine($"[{rows}x{dim}] max abs err = {worst:E3}");
        Assert.True(worst < 1e-4, $"LayerNormNoAffine diverges from the CPU reference: max abs err {worst:E3}");
    }

    /// <summary>F16 is the case that mattered: it is what the DiT actually hands this op, and what the host default
    /// refused.</summary>
    [Fact]
    public void AcceptsF16_WhichTheHostDefaultRefused()
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        TensorShape shape = new(2, 512);
        using Tensor input = Random(shape, seed: 23, DType.F16);
        using Tensor actual = new(shape, DType.F16);

        gpu.LayerNormNoAffine(actual, input, 1e-6f);

        // Compare against the F32 reference over the same values, at F16's own precision.
        using Tensor inputF32 = input.CastTo(DType.F32);
        using Tensor expected = new(shape, DType.F32);
        IBackend cpu = new CpuBackend();
        cpu.LayerNormNoAffine(expected, inputF32, 1e-6f);

        using Tensor actualF32 = actual.CastTo(DType.F32);
        ReadOnlySpan<float> got = actualF32.AsReadOnlySpan<float>();
        ReadOnlySpan<float> want = expected.AsReadOnlySpan<float>();
        double worst = 0;
        for (int i = 0; i < got.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(got[i] - want[i]));
        }
        _output.WriteLine($"F16 max abs err vs F32 reference = {worst:E3}");
        Assert.True(worst < 5e-3, $"F16 LayerNormNoAffine diverges: max abs err {worst:E3}");
    }

    private static string SpirvDir()
    {
        string local = Path.Combine(AppContext.BaseDirectory, "Spirv");
        return Directory.Exists(local)
            ? local
            : Path.Combine(RepoRoot.Path, "src", "HartsyInference.Vulkan", "Spirv");
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
