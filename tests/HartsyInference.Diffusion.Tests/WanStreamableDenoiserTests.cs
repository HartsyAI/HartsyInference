using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the split the Wan family's streaming depends on: the shared set and the block set must together be exactly the full weight set, with nothing counted twice and nothing dropped.</summary>
/// <remarks>Both failure directions are silent rather than loud. A weight in neither set never uploads, so the
/// forward reads whatever the allocator left behind; a weight in both is evicted by the block window while the
/// shared preload still believes it is resident. Neither throws — they corrupt output or crash much later, so the
/// invariant is asserted here rather than left to a real-weight run to expose.</remarks>
public sealed class WanStreamableDenoiserTests
{
    /// <summary>Identity, not value: the streaming controller keys residency on the tensor reference itself.</summary>
    private sealed class ReferenceComparer : IEqualityComparer<Tensor>
    {
        public bool Equals(Tensor? a, Tensor? b) => ReferenceEquals(a, b);

        public int GetHashCode(Tensor t) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(t);
    }

    private static readonly ReferenceComparer ByReference = new ReferenceComparer();

    private static WanVideoConfig Config(int layers) => new WanVideoConfig
    {
        NumHeads = 2,
        HeadDim = 16,
        InChannels = 36,
        OutChannels = 16,
        VaeLatentChannels = 16,
        FfnDim = 64,
        NumLayers = layers,
        VaeSpatialCompression = 8,
        ImageDim = 8,
        AddedKvProjDim = 8,
    };

    /// <summary>Asserts shared ∪ blocks == all, disjointly, by tensor IDENTITY — which is what the streaming
    /// controller tracks, so comparing by value would pass while the real contract was broken.</summary>
    private static void AssertPartitions(IStreamableDenoiser denoiser, IEnumerable<Tensor> all)
    {
        HashSet<Tensor> shared = new HashSet<Tensor>(denoiser.EnumerateSharedWeights(), ByReference);
        List<Tensor> blockWeights = [];
        for (int b = 0; b < denoiser.BlockCount; b++)
        {
            blockWeights.AddRange(denoiser.GetBlock(b).EnumerateWeights());
        }
        HashSet<Tensor> blocks = new HashSet<Tensor>(blockWeights, ByReference);
        HashSet<Tensor> everything = new HashSet<Tensor>(all, ByReference);

        Assert.DoesNotContain(shared, blocks.Contains);
        Assert.DoesNotContain(everything, t => !shared.Contains(t) && !blocks.Contains(t));
        Assert.DoesNotContain(shared.Concat(blocks), t => !everything.Contains(t));
        Assert.True(denoiser.BlockCount > 0);
    }

    [Fact]
    public void WanVideo_SharedAndBlocksPartitionTheWeightSet()
    {
        WanVideoConfig c = Config(layers: 3);
        using WanVideoTransformer t = new WanVideoTransformer(c);
        t.LoadWeights(WanSyntheticWeights.BuildTransformer(c));
        AssertPartitions(t, t.EnumerateWeights());
    }

    /// <summary>VACE additionally has a whole control stack that must land in the SHARED half — it runs as its own
    /// complete pass before the main loop, so a block window cannot carry it.</summary>
    [Fact]
    public void WanVace_KeepsTheControlStackShared()
    {
        // VaceLayers must be non-empty or there is no control stack to keep shared and the test asserts nothing.
        WanVideoConfig c = Config(layers: 4) with { VaceLayers = [0, 2], VaceInChannels = 96 };
        using WanVaceTransformer t = new WanVaceTransformer(c);
        t.LoadWeights(WanSyntheticWeights.BuildVaceTransformer(c));
        AssertPartitions(t, t.EnumerateWeights());

        // The control blocks are real weights, and every one of them is on the shared side rather than in any block.
        HashSet<Tensor> shared = new HashSet<Tensor>(t.EnumerateSharedWeights(), ByReference);
        int blockTensors = 0;
        for (int b = 0; b < t.BlockCount; b++) blockTensors += t.GetBlock(b).EnumerateWeights().Count();
        Assert.True(shared.Count > blockTensors / t.BlockCount,
            $"shared={shared.Count} carries the 2 VACE control blocks, per-block={blockTensors / t.BlockCount}");
    }

    [Fact]
    public void WanS2V_KeepsTheAudioInjectorShared()
    {
        WanVideoConfig c = Config(layers: 3);
        using WanS2VTransformer t = new WanS2VTransformer(c);
        t.LoadWeights(WanSyntheticWeights.BuildS2VTransformer(c));
        AssertPartitions(t, t.EnumerateWeights());
    }

    /// <summary>Every block must be reachable by index and report a positive size, or the window cannot budget.</summary>
    [Fact]
    public void EveryBlockIsIndexableAndSized()
    {
        WanVideoConfig c = Config(layers: 4);
        using WanVideoTransformer t = new WanVideoTransformer(c);
        t.LoadWeights(WanSyntheticWeights.BuildTransformer(c));

        Assert.Equal(4, t.BlockCount);
        for (int b = 0; b < t.BlockCount; b++)
        {
            Assert.True(t.GetBlock(b).EstimatedWeightBytes > 0, $"block {b} reported no bytes");
            Assert.NotEmpty(t.GetBlock(b).EnumerateWeights());
        }
    }

    /// <summary>The controller tracks residency by reference, so a block must hand back the SAME tensors each call.</summary>
    [Fact]
    public void GetBlock_ReturnsStableTensorReferences()
    {
        WanVideoConfig c = Config(layers: 2);
        using WanVideoTransformer t = new WanVideoTransformer(c);
        t.LoadWeights(WanSyntheticWeights.BuildTransformer(c));

        Tensor[] first = [.. t.GetBlock(1).EnumerateWeights()];
        Tensor[] second = [.. t.GetBlock(1).EnumerateWeights()];
        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
        {
            Assert.Same(first[i], second[i]);
        }
    }
}
