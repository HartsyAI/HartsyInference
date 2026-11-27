using System.Text.RegularExpressions;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Ernie5;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The weight set an ERNIE-5.5 layer offers for device preload, and the preload's memory budget.</summary>
public sealed partial class Ernie5DenseWeightsTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-ernie5-dense-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Dense_set_holds_every_per_token_weight_and_no_routed_expert()
    {
        Ernie5SyntheticCheckpoint.Write(_root);
        Ernie5Config config = Ernie5Config.Load(Path.Combine(_root, "config.json"));
        int layerIndex = config.FirstKDenseReplace;
        Assert.True(layerIndex < config.NumHiddenLayers, "the synthetic checkpoint needs a MoE layer");
        using Ernie5CheckpointWeights weights = Ernie5CheckpointWeights.Open(_root);
        RecordingSource source = new(weights);
        DeepSeekV41RopeTable hca = DeepSeekV41RopeTable.Build(config.QkRopeHeadDim, 8, config.CompressRopeTheta, config.HcaRopeScaling);
        DeepSeekV41RopeTable dsa = DeepSeekV41RopeTable.Build(config.MlaQkRopeHeadDim, 8, config.RopeTheta, null);
        using CpuBackend backend = new();
        using Ernie5Layer layer = new(backend, config, layerIndex, source, hca, dsa);

        HashSet<Tensor> dense = new(layer.EnumerateDenseWeights(), ReferenceEqualityComparer.Instance);
        string prefix = $"model.layers.{layerIndex}.";
        int routed = 0, shared = 0, latent = 0;
        foreach ((string name, Tensor tensor) in source.Reads)
        {
            if (!name.StartsWith(prefix, StringComparison.Ordinal)) continue;
            if (RoutedExpert().IsMatch(name))
            {
                routed++;
                Assert.False(dense.Contains(tensor), $"routed expert {name} is in the dense set");
            }
            else if (name.Contains(".shared_experts.", StringComparison.Ordinal))
            {
                shared++;
                Assert.True(dense.Contains(tensor), $"shared expert {name} is missing from the dense set");
            }
            else if (name.Contains("_latent_proj.", StringComparison.Ordinal))
            {
                latent++;
                Assert.True(dense.Contains(tensor), $"latent projection {name} is missing from the dense set");
            }
        }
        Assert.True(routed > 0 && shared > 0 && latent > 0, $"read {routed} routed, {shared} shared, {latent} latent tensors");
        // Every FP8 weight brings its scale grid, which the block linear reads on every call too.
        foreach (Tensor t in dense)
            if (t.QuantInfo?.Fp8BlockScaleInv is { } scale) Assert.Contains(scale, dense);
        // The router, widened to F32 at load, is in the set.
        Assert.Contains(dense, t => t.DType == DType.F32 && t.Shape.Rank == 2 && t.Shape[0] == config.NRoutedExperts && t.Shape[1] == config.HiddenSize);
    }

    [Theory]
    [InlineData(20L << 30, 24L << 30, (20L << 30) - (24L << 30) * 15 / 100)]   // 15% of 24 GiB is above the 3 GiB floor
    [InlineData(10L << 30, 12L << 30, (10L << 30) - (3L << 30))]                // the 3 GiB floor
    [InlineData(2L << 30, 12L << 30, 0L)]                                        // less free than the headroom
    [InlineData(0L, 0L, 0L)]                                                     // the backend does not report memory
    public void Preload_budget_keeps_headroom(long free, long total, long expected) =>
        Assert.Equal(expected, Ernie5Model.PreloadBudget(free, total));

    [Theory]
    [InlineData(79L << 30, 80L << 30, (79L << 30) - (80L << 30) * 5 / 100)]    // 5% of 80 GiB is above the 4 GiB floor
    [InlineData(22L << 30, 24L << 30, (22L << 30) - (4L << 30))]                // the 4 GiB floor
    [InlineData(3L << 30, 24L << 30, 0L)]                                        // less free than the headroom
    public void Preload_budget_with_every_weight_keeps_a_smaller_headroom(long free, long total, long expected) =>
        Assert.Equal(expected, Ernie5Model.PreloadBudget(free, total, allWeights: true));

    // The load reads a layer's mapped weights ahead of their uploads with parallel readers. That only helps for tensors with a file
    // extent, so every routed expert of a sharded FP8 checkpoint must have one, and the read-ahead must cover each byte once.
    [Theory]
    [InlineData(4)]
    [InlineData(0)]
    public void Preload_reads_every_mapped_expert_byte_ahead_once(int readers)
    {
        Ernie5SyntheticCheckpoint.Write(_root, storage: Ernie5SyntheticCheckpoint.Storage.Fp8, shards: 3);
        Ernie5Config config = Ernie5Config.Load(Path.Combine(_root, "config.json"));
        using Ernie5CheckpointWeights weights = Ernie5CheckpointWeights.Open(_root);
        DeepSeekV41RopeTable hca = DeepSeekV41RopeTable.Build(config.QkRopeHeadDim, 8, config.CompressRopeTheta, config.HcaRopeScaling);
        DeepSeekV41RopeTable dsa = DeepSeekV41RopeTable.Build(config.MlaQkRopeHeadDim, 8, config.RopeTheta, null);
        using CpuBackend backend = new();
        using Ernie5Layer layer = new(backend, config, config.FirstKDenseReplace, weights, hca, dsa);
        List<IReadOnlyList<Tensor>> groups = [.. layer.EnumerateExpertPreloadGroups()];
        Tensor[] experts = [.. layer.Moe!.EnumerateExpertGroups().SelectMany(g => g)];
        Assert.Equal(3 * config.NRoutedExperts, experts.Length);
        Assert.All(experts, t => Assert.True(TensorFileExtents.TryGet(t, out _, out _), "a routed expert has no file extent"));

        long read = Ernie5Model.PreloadGroups(backend, groups, readers, grouped: true);

        long mapped = groups.SelectMany(g => g).Where(t => TensorFileExtents.TryGet(t, out _, out _)).Sum(t => t.DType.ComputeByteCount(t.ElementCount));
        Assert.True(mapped >= experts.Sum(t => t.DType.ComputeByteCount(t.ElementCount)));
        Assert.Equal(readers == 0 ? 0 : mapped, read);
        // The dense set goes through the same call ungrouped, one tensor per group; nothing of it is read twice either.
        List<Tensor> dense = [.. layer.EnumerateDenseWeights()];
        long denseMapped = dense.Where(t => TensorFileExtents.TryGet(t, out _, out _)).Sum(t => t.DType.ComputeByteCount(t.ElementCount));
        Assert.Equal(readers == 0 ? 0 : denseMapped,
            Ernie5Model.PreloadGroups(backend, [.. dense.Select(t => (IReadOnlyList<Tensor>)[t])], readers, grouped: false));
    }

    [GeneratedRegex(@"\.experts\.\d+\.")]
    private static partial Regex RoutedExpert();

    // Records every tensor handed out, by name, so the test can tell which instances a layer kept.
    private sealed class RecordingSource(IErnie5WeightSource inner) : IErnie5WeightSource
    {
        public List<(string Name, Tensor Tensor)> Reads { get; } = [];

        public IEnumerable<string> Names => inner.Names;

        public Tensor Read(string name)
        {
            Tensor t = inner.Read(name);
            Reads.Add((name, t));
            return t;
        }
    }
}
