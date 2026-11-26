using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The recompute admission against what a recompute keeps. <c>SyntheticSmoke</c>. With two blocks, one full-history block and five one-frame
/// chunks, the recompute keeps every frame in block 0 and the sink plus the recent chunk (two frames) in block 1: seven frame-layers. The old check
/// charged both blocks for all five frames (ten), so a budget of seven frame-layers was refused. The tests pin the exact figure and that a refusal leaves
/// the cache as it was.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class MovieRollRecomputeBudgetTests
{
    private const int Channels = 4;
    private const int Height = 2;
    private const int Width = 3;
    private const int Blocks = 2;
    private const int Hidden = 32;

    // One frame of one layer's K and V at this grid: 2 * 3 * 32 tokens of width 32, two tensors, four bytes each.
    private const long FrameLayerBytes = Height * Width * Hidden * 2 * sizeof(float);

    [Fact]
    public void RecomputeFitsABudgetOfWhatItKeeps()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = new(fixture.Backend, 7 * FrameLayerBytes);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);

        int frames = fixture.Dit.StreamRecompute(fixture.Backend, Latent(5), [1, 1, 1, 1, 1], cache, fullHistoryBlocks: 1);

        Assert.Equal(5, frames);
        Assert.True(cache.BytesHeld <= 7 * FrameLayerBytes, $"the cache holds {cache.BytesHeld} bytes");
    }

    [Fact]
    public void RecomputeRefusedByBudgetLeavesTheCacheAsItWas()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = new(fixture.Backend, 6 * FrameLayerBytes);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);
        fixture.Dit.StreamEncode(fixture.Backend, Latent(1), new float[1], cache, fullHistoryBlocks: 1, startFrame: 0);
        long before = cache.BytesHeld;
        IReadOnlyList<(int Start, int End)> beforeRanges = cache.FrameRanges(HunyuanVideo15Dit.StreamLayerName(0));

        // Seven frame-layers do not fit in six. The refusal happens before the cache is cleared, so the encoded frame is still there.
        Assert.Throws<InvalidOperationException>(() =>
            fixture.Dit.StreamRecompute(fixture.Backend, Latent(5), [1, 1, 1, 1, 1], cache, fullHistoryBlocks: 1));

        Assert.Equal(before, cache.BytesHeld);
        Assert.Equal(beforeRanges, cache.FrameRanges(HunyuanVideo15Dit.StreamLayerName(0)));
    }

    private static Tensor Latent(int frames, long seed = 3) =>
        TorchCompatibleNoise.Randn(new TensorShape([1L, Channels, frames, Height, Width]), seed);

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = Channels, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    private sealed class Fixture : IDisposable
    {
        public Fixture()
        {
            Backend = new CpuBackend();
            Weights = new SafeTensorsLoader();
            Weights.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
            Dit = new HunyuanVideo15Dit(TinyConfig());
            Dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(
                Weights.GetAllTensors()
                    .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
                        && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
                    .ToDictionary(pair => pair.Key, pair => pair.Value),
                Blocks, Hidden));
        }

        public CpuBackend Backend { get; }
        public SafeTensorsLoader Weights { get; }
        public HunyuanVideo15Dit Dit { get; }

        public void Dispose()
        {
            Dit.Dispose();
            Backend.Dispose();
            Weights.Dispose();
        }
    }
}
