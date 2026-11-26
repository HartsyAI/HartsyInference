using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The block-streaming hook on MovieRoll's streaming entry points. <c>SyntheticSmoke</c>: this checks that <c>StreamEncode</c>,
/// <c>StreamRecompute</c> and <c>StreamDenoise</c> call <see cref="HunyuanVideo15Dit.BeforeBlockForward"/> once per block, in order. It does not
/// check staging or eviction; those happen in <c>BlockStreamingScope</c> on a device, and nothing here runs one.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class MovieRollStreamHookTests
{
    private const int Channels = 4;
    private const int Height = 2;
    private const int Width = 3;
    private const int Blocks = 2;
    private const int Hidden = 32;

    [Fact]
    public void StreamEncodeCallsTheHookForEveryBlockInOrder()
    {
        using Fixture fixture = new();
        List<int> calls = [];
        fixture.Dit.BeforeBlockForward = calls.Add;
        using MovieRollKvCache cache = new(fixture.Backend);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);

        fixture.Dit.StreamEncode(fixture.Backend, Latent(2), new float[2], cache, fullHistoryBlocks: 1, startFrame: 0);

        Assert.Equal([0, 1], calls);
    }

    [Fact]
    public void StreamRecomputeCallsTheHookForEveryBlockInOrder()
    {
        using Fixture fixture = new();
        List<int> calls = [];
        fixture.Dit.BeforeBlockForward = calls.Add;
        using MovieRollKvCache cache = new(fixture.Backend);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);

        // Two chunks of one frame; block 0 keeps full history, block 1 is sink-recent, so each block runs once through the chunked pass.
        fixture.Dit.StreamRecompute(fixture.Backend, Latent(2), [1, 1], cache, fullHistoryBlocks: 1);

        Assert.Equal([0, 1], calls);
    }

    [Fact]
    public void StreamWindowCallsTheHookBeforeEachBlockRuns()
    {
        using Fixture fixture = new();
        List<int> calls = [];
        fixture.Dit.BeforeBlockForward = calls.Add;
        using MovieRollKvCache cache = new(fixture.Backend);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);

        // One encoded frame is in the cache first; the noisy window then goes through StreamDenoise. Only the hook calls are checked here.
        fixture.Dit.StreamEncode(fixture.Backend, Latent(1), new float[1], cache, fullHistoryBlocks: 1, startFrame: 0);
        calls.Clear();
        using Tensor velocity = fixture.Dit.StreamDenoise(fixture.Backend, Latent(1, seed: 5), new float[1], 0f, fixture.Qwen, fixture.ByT5,
            cache, fullHistoryBlocks: 1);

        Assert.Equal([0, 1], calls);
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
            Weights = new SafeTensorsLoader();
            Weights.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
            Backend = new CpuBackend();
            Dit = new HunyuanVideo15Dit(TinyConfig());
            Dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(
                Weights.GetAllTensors()
                    .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
                        && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
                    .ToDictionary(pair => pair.Key, pair => pair.Value),
                Blocks, Hidden));
        }

        public SafeTensorsLoader Weights { get; }
        public CpuBackend Backend { get; }
        public HunyuanVideo15Dit Dit { get; }
        public Tensor Qwen => Weights.GetTensor("in.qwen");
        public Tensor ByT5 => Weights.GetTensor("in.byt5");

        public void Dispose()
        {
            Dit.Dispose();
            Backend.Dispose();
            Weights.Dispose();
        }
    }
}
