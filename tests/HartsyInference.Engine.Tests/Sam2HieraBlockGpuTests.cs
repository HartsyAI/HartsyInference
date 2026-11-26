using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Tests.Services;
using HartsyInference.Vision.Segmentation.Sam2;
using HartsyInference.Vulkan;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests;

/// <summary>The SAM 2 Hiera block on a GPU against <see cref="CpuBackend"/>, with its device-to-host syncs printed. It uses nothing newer than the
/// block, so it can be copied onto an older commit to read the count there. Synthetic weights, no downloads.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class Sam2HieraBlockGpuTests(ITestOutputHelper output)
{
    private static readonly Lazy<string?> CudaFailure = new(static () => Probe(static () => Cuda().Dispose()));
    private static readonly Lazy<string?> VulkanFailure = new(static () => Probe(static () => new VulkanBackend().Dispose()));

    private static CudaBackend Cuda() => new(0, Ernie5ModelGpuParityTests.PtxDirectory());

    [CudaRun]
    public void Cuda_HieraBlock_MatchesCpu() => HieraBlocks(Cuda(), globalSyncs: 0);

    [VulkanRun]
    public void Vulkan_HieraBlock_MatchesCpu() => HieraBlocks(new VulkanBackend());

    private void HieraBlocks(IBackend device, long? globalSyncs = null)
    {
        using (device)
        {
            using ExactGemmScope exact = new(device);
            Compare(device, dimIn: 8, dimOut: 8, heads: 2, window: 0, pool: false, expectedSyncs: globalSyncs);
            // The windowed forms still partition windows on the host, so their counts and the pooled one's are only printed.
            Compare(device, dimIn: 8, dimOut: 16, heads: 2, window: 0, pool: true);
            Compare(device, dimIn: 8, dimOut: 8, heads: 2, window: 4, pool: false);
            Compare(device, dimIn: 8, dimOut: 16, heads: 4, window: 4, pool: true);
        }
    }

    private void Compare(IBackend device, int dimIn, int dimOut, int heads, int window, bool pool, long? expectedSyncs = null)
    {
        Random rng = new(3);
        Dictionary<string, Tensor> weights = new()
        {
            ["norm1.weight"] = Filled(rng, dimIn), ["norm1.bias"] = Filled(rng, dimIn),
            ["attn.qkv.weight"] = Filled(rng, 3 * dimOut, dimIn), ["attn.qkv.bias"] = Filled(rng, 3 * dimOut),
            ["attn.proj.weight"] = Filled(rng, dimOut, dimOut), ["attn.proj.bias"] = Filled(rng, dimOut),
            ["norm2.weight"] = Filled(rng, dimOut), ["norm2.bias"] = Filled(rng, dimOut),
            ["mlp.layers.0.weight"] = Filled(rng, 4 * dimOut, dimOut), ["mlp.layers.0.bias"] = Filled(rng, 4 * dimOut),
            ["mlp.layers.1.weight"] = Filled(rng, dimOut, 4 * dimOut), ["mlp.layers.1.bias"] = Filled(rng, dimOut),
            ["proj.weight"] = Filled(rng, dimOut, dimIn), ["proj.bias"] = Filled(rng, dimOut),
        };
        using Tensor x = Filled(rng, 1, 8, 8, dimIn);
        HieraBlock block = new(dimIn, dimOut, heads, window, pool, 4, 1e-6f);
        block.LoadWeights(weights, "");
        float[] expected;
        using (CpuBackend cpu = new())
        {
            (Tensor y, _, _) = block.Forward(cpu, x, 8, 8);
            using (y) expected = y.AsReadOnlySpan<float>().ToArray();
        }

        using Tensor resident = new(x.Shape, DType.F32);
        device.Scale(resident, x, 1f);
        device.ResetD2hSyncCount();
        (Tensor result, _, _) = block.Forward(device, resident, 8, 8);
        long syncs = device.GetD2hSyncCount();
        output.WriteLine($"D2H_SYNCS hiera block window={window} pool={pool}: {syncs}");
        using (result)
        {
            ReadOnlySpan<float> actual = result.AsReadOnlySpan<float>();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-4f, $"window={window} pool={pool} element {i}: CPU {expected[i]} vs device {actual[i]}.");
        }
        foreach (Tensor weight in weights.Values) weight.Dispose();
        if (expectedSyncs is { } expectedCount) Assert.Equal(expectedCount, syncs);
    }

    private static Tensor Filled(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static string? Probe(Action open)
    {
        try
        {
            open();
            return null;
        }
        catch (Exception ex)
        {
            return $"{ex.GetType().Name}: {ex.Message}";
        }
    }

    private sealed class CudaRunAttribute : FactAttribute
    {
        public CudaRunAttribute()
        {
            if (CudaFailure.Value is { } failure) Skip = $"SKIPPED: no usable CUDA device ({failure}).";
        }
    }

    private sealed class VulkanRunAttribute : FactAttribute
    {
        public VulkanRunAttribute()
        {
            if (VulkanFailure.Value is { } failure) Skip = $"SKIPPED: no usable Vulkan device ({failure}).";
        }
    }
}
