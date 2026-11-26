using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Tests.Services;
using HartsyInference.Vision.Detection.Blocks;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests;

/// <summary>The whole PSA block (projection, attention, positional encoding, output projection) on a GPU backend against
/// the same block on the CPU backend, with random weights. The CPU backend's attention is checked against the block's
/// former host loop in <c>PsaAttentionTests</c>; this checks that a GPU serves the same call and agrees.
/// <para>The tolerances were fixed before the first run: 1e-4 relative to the larger of 1 and the value on Vulkan, and on
/// CUDA the contract bound for TF32, 2e-3, because CUDA multiplies F32 attention and its convolutions in TF32.</para>
/// <para>It counts attention evidence events, so it runs in the <c>AttentionEvidence</c> collection.</para></summary>
[Collection("AttentionEvidence")]
public sealed class PsaAttentionGpuTests
{
    private readonly ITestOutputHelper _output;

    public PsaAttentionGpuTests(ITestOutputHelper output) => _output = output;

    private static Tensor Random(TensorShape shape, int seed, float scale)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (random.NextSingle() * 2f - 1f) * scale;
        return tensor;
    }

    private void MatchesTheCpuBackend(IBackend backend, float tolerance)
    {
        foreach ((int batch, int dim, int heads, int height, int width) in
            ((int, int, int, int, int)[])[(1, 256, 4, 20, 20), (1, 512, 8, 20, 20), (2, 24, 3, 3, 5)])
        {
            int keyDim = dim / heads / 2, qkvChannels = dim + 2 * heads * keyDim, seed = dim;
            Dictionary<string, Tensor> weights = new()
            {
                ["psa.qkv.conv.weight"] = Random(new TensorShape(qkvChannels, dim, 1, 1), seed++, 1f / MathF.Sqrt(dim)),
                ["psa.qkv.conv.bias"] = Random(new TensorShape(qkvChannels), seed++, 0.1f),
                ["psa.proj.conv.weight"] = Random(new TensorShape(dim, dim, 1, 1), seed++, 1f / MathF.Sqrt(dim)),
                ["psa.proj.conv.bias"] = Random(new TensorShape(dim), seed++, 0.1f),
                ["psa.pe.conv.weight"] = Random(new TensorShape(dim, 1, 3, 3), seed++, 0.3f),
                ["psa.pe.conv.bias"] = Random(new TensorShape(dim), seed++, 0.1f),
            };
            PsaAttention block = new(dim, heads);
            block.LoadWeights(weights, "psa");
            using Tensor input = Random(new TensorShape(batch, dim, height, width), seed++, 1f);

            float[] expected;
            using (CpuBackend cpu = new())
            using (Tensor reference = block.Forward(cpu, input))
                expected = reference.AsReadOnlySpan<float>().ToArray();

            using Tensor output = block.Forward(backend, input);
            backend.Sync();
            ReadOnlySpan<float> actual = output.AsReadOnlySpan<float>();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1f, Math.Abs(expected[i])),
                    $"batch {batch}, {dim} channels, {heads} heads, {height}x{width}: element {i} is {actual[i]}, expected {expected[i]}");
            foreach (Tensor weight in weights.Values) weight.Dispose();
        }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnCudaTheBlockMatchesTheCpuBackend()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, Ernie5ModelGpuParityTests.PtxDirectory());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
        try { MatchesTheCpuBackend(backend, 2e-3f); }
        finally { ReportRows(evidence.Snapshot()); }
        AssertServedByTheGpu(evidence.Snapshot(), "materialized");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnVulkanTheBlockMatchesTheCpuBackend()
    {
        VulkanBackend backend;
        try { backend = new VulkanBackend(); }
        catch (Exception ex) { _output.WriteLine($"SKIPPED: Vulkan unavailable ({ex.Message})"); return; }
        using (backend)
        {
            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(256);
            try { MatchesTheCpuBackend(backend, 1e-4f); }
            finally { ReportRows(evidence.Snapshot()); }
            // Dense all-F32 attention at this head dim takes the query-tiled kernel by default on Vulkan (numerics.vkTiledAttention).
            AssertServedByTheGpu(evidence.Snapshot(), "tiled.f32");
        }
    }

    /// <summary>Each shape runs once on the CPU backend, for the reference, and once on the GPU one: three executions must
    /// carry the GPU backend's own row and three the CPU backend's, so a GPU call that fell back would fail here.</summary>
    private static void AssertServedByTheGpu(AttentionEvidenceEvent[] events, string gpuRow)
    {
        AttentionEvidenceEvent[] executions = events.Where(e => e.Kind == AttentionEvidenceKind.Execution).ToArray();
        Assert.Equal(3, executions.Count(e => e.Implementation == gpuRow));
        Assert.Equal(6, executions.Length);
        Assert.DoesNotContain(events, e => e.Kind == AttentionEvidenceKind.HostReference);
    }

    /// <summary>Which kernel rows served the attention: reported also when a comparison fails.</summary>
    private void ReportRows(AttentionEvidenceEvent[] events)
    {
        foreach (IGrouping<string, AttentionEvidenceEvent> row in events.Where(e => e.Kind == AttentionEvidenceKind.Execution)
                     .GroupBy(e => $"{e.Implementation} [{e.Branch}] heads {e.Heads}, seq {e.QueryTokens}"))
            _output.WriteLine($"{row.Key}: {row.Count()} call(s)");
    }
}
