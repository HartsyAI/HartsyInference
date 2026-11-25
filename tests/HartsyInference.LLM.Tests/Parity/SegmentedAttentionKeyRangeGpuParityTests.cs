using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary><see cref="SegmentedAttention.RunHeadMajor"/> on a GPU against <see cref="CpuBackend"/>, over device operands with keys longer than the
/// queries, and with no device-to-host sync. Synthetic; needs no weights.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SegmentedAttentionKeyRangeGpuParityTests(ITestOutputHelper output)
{
    private const int Heads = 4, HeadDim = 64, Queries = 96, Keys = 224;

    [CudaFact]
    public void Cuda_MatchesCpu_WithoutAHostSync() => Check(new CudaBackend(0, CudaPtxDirectory.Resolve()));

    [VulkanFact]
    public void Vulkan_MatchesCpu_WithoutAHostSync() => Check(new VulkanBackend());

    private void Check(IBackend device)
    {
        KeyRangeSegment[] segments =
        [
            new(0, 16, [new(0, 64), new(128, 16)]), KeyRangeSegment.Of(16, 40, 0, 200), KeyRangeSegment.Of(56, 24, 0, 200), KeyRangeSegment.Of(80, 16, 200, 24),
        ];
        Random rng = new(9);
        AttentionLayer layer = new(new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, ComputeDType = DType.F32 });
        using (device)
        using (Tensor q = Filled(rng, Queries))
        using (Tensor k = Filled(rng, Keys))
        using (Tensor v = Filled(rng, Keys))
        {
            float[] expected;
            using (Tensor cpu = SegmentedAttention.RunHeadMajor(new CpuBackend(), layer, q, k, v, segments))
                expected = cpu.AsReadOnlySpan<float>().ToArray();
            using Tensor qd = OnDevice(device, q), kd = OnDevice(device, k), vd = OnDevice(device, v);
            device.ResetD2hSyncCount();
            using Tensor result = SegmentedAttention.RunHeadMajor(device, layer, qd, kd, vd, segments);
            long syncs = device.GetD2hSyncCount();
            output.WriteLine($"D2H_SYNCS segmented attention: {syncs}");
            ReadOnlySpan<float> actual = result.AsReadOnlySpan<float>();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-4f, $"element {i}: CPU {expected[i]} vs device {actual[i]}.");
            Assert.Equal(0, syncs);
        }
    }

    private static Tensor Filled(Random rng, int tokens)
    {
        Tensor t = new(new TensorShape(1, Heads, tokens, HeadDim), DType.F32);
        foreach (ref float value in t.AsSpan<float>()) value = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor OnDevice(IBackend device, Tensor host)
    {
        Tensor copy = new(host.Shape, DType.F32);
        device.Scale(copy, host, 1f);
        Assert.False(copy.HostBytesCurrent);
        return copy;
    }
}
