using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Generations through <see cref="MovieRollPipeline"/> on the tiny DiT, resident and streamed. Each generation runs one forward per denoising
/// step and window, with KV recompute on, so the streamed run loads and releases its blocks on every forward, the recompute and the encode passes
/// included; the second generation reuses the same pipeline and DiT. The pool drains are counted through a fake streaming cache.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class MovieRollPipelineStreamingTests
{
    private const int Blocks = 2;
    private const int Hidden = 32;
    private const int Channels = 4;
    private const int Height = 2;
    private const int Width = 3;

    private static HunyuanVideo15Config TinyDitConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = Channels, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    private static HunyuanVideo15VaeConfig TinyVaeConfig() => new()
    {
        LatentChannels = 2, BlockOutChannels = [4, 8, 16, 32, 32], LayersPerBlock = 2, SpatialCompression = 16, TemporalCompression = 4, ScalingFactor = 1f,
    };

    private static MovieRollSamplerOptions Options() => new()
    {
        ChunkFrames = 1, NumChunks = 5, FullHistoryBlocks = 1, ContextNoise = 0f,
        KvRecomputeEnabled = true, KvRecomputeThreshold = 2, KvRecomputeRecentK = 1, Cfg = null,
    };

    private static Dictionary<string, Tensor> Source(SafeTensorsLoader loader) => loader.GetAllTensors()
        .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
            && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
        .ToDictionary(pair => pair.Key, pair => pair.Value);

    private static Tensor RandomTensor(Random rng, params long[] dims)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        foreach (ref float v in tensor.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        return tensor;
    }

    /// <summary>The noise of one generation: the same draws for every generation, so the runs differ only in how the DiT is loaded.</summary>
    private static Func<int, Tensor> Noise(int seed)
    {
        Random rng = new(seed);
        return frames => RandomTensor(rng, 1, Channels, frames, Height, Width);
    }

    /// <summary>A streaming cache that counts pool drains and does nothing else: the streamed path uses no upload or eviction.</summary>
    private sealed class CountingPoolCache : IStreamingWeightCache
    {
        public int Drains { get; private set; }

        public void DrainAndReleasePool() => Drains++;

        public StreamingUploadToken BeginUploadAsync(IEnumerable<Tensor> weights) => throw new NotSupportedException();

        public void AwaitWeights(StreamingUploadToken token) => throw new NotSupportedException();

        public void EvictAsync(IEnumerable<Tensor> weights) => throw new NotSupportedException();

        public long QueryAvailableWeightCacheBytes(long activationReserve) => throw new NotSupportedException();
    }

    /// <summary>After a generation no block is loaded, every load has its release, and no block tensor is held.</summary>
    private static void AssertReleased(HunyuanVideo15BlockLoader loader, string when)
    {
        Assert.True(loader.LoadedBlock == -1, $"{when}: block {loader.LoadedBlock} is still loaded.");
        Assert.True(loader.LoadedBlockTensorCount == 0, $"{when}: {loader.LoadedBlockTensorCount} block tensors are still held.");
        Assert.Equal(loader.BlockLoads, loader.BlockReleases);
    }

    private sealed class Rig : IDisposable
    {
        public Rig(CountingPoolCache pool)
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
            Loader = new SafeTensorsLoader();
            Loader.Load(path);
            Source = HunyuanVideo15BlockLoaderTestsSource.Of(Loader);
            Config = TinyDitConfig();
            Backend = new CpuBackend();
            Vae = new HunyuanVideo15Vae(TinyVaeConfig());
            Schedule = MovieRollSchedule.Distilled([1000, 800, 600, 400, 200], 5, 5.0);
            Resident = new HunyuanVideo15Dit(Config);
            Resident.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(Source, Blocks, Hidden));
            ResidentPipeline = new MovieRollPipeline(Backend, Resident, Vae, Schedule);
            Streamed = new HunyuanVideo15Dit(Config);
            BlockLoader = new HunyuanVideo15BlockLoader(Streamed, Config, Backend, Source, [], readBlocksFromFile: true);
            BlockLoader.LoadShared();
            Streamed.LoadBlock = BlockLoader.LoadBlock;
            Streamed.UnloadBlock = BlockLoader.UnloadBlock;
            StreamedPipeline = new MovieRollPipeline(Backend, Streamed, Vae, Schedule, poolCache: pool);
        }

        public SafeTensorsLoader Loader { get; }
        public Dictionary<string, Tensor> Source { get; }
        public HunyuanVideo15Config Config { get; }
        public CpuBackend Backend { get; }
        public HunyuanVideo15Vae Vae { get; }
        public MovieRollSchedule Schedule { get; }
        public HunyuanVideo15Dit Resident { get; }
        public MovieRollPipeline ResidentPipeline { get; }
        public HunyuanVideo15Dit Streamed { get; }
        public HunyuanVideo15BlockLoader BlockLoader { get; }
        public MovieRollPipeline StreamedPipeline { get; }

        public void Dispose()
        {
            BlockLoader.Dispose();
            Streamed.Dispose();
            Resident.Dispose();
            Vae.Dispose();
            Loader.Dispose();
        }
    }

    [Fact]
    public void TwoGenerationsStreamedAndResidentGiveTheSameLatents()
    {
        CountingPoolCache pool = new();
        using Rig rig = new(pool);
        Random textRng = new(3);
        using Tensor qwen = RandomTensor(textRng, 1, 3, 24);
        using Tensor byt5 = RandomTensor(textRng, 1, 2, 16);
        MovieRollSamplerOptions options = Options();

        using Tensor residentFirst = rig.ResidentPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        using Tensor residentSecond = rig.ResidentPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        using Tensor streamedFirst = rig.StreamedPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        AssertReleased(rig.BlockLoader, "first generation");
        int loadsAfterFirst = rig.BlockLoader.BlockLoads;
        Assert.True(loadsAfterFirst > 0, "The first generation loaded no blocks.");
        int drainsAfterFirst = pool.Drains;

        using Tensor streamedSecond = rig.StreamedPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        AssertReleased(rig.BlockLoader, "second generation");
        // The second generation loads the same blocks the first did, and drains the pool again.
        Assert.Equal(2 * loadsAfterFirst, rig.BlockLoader.BlockLoads);
        Assert.True(pool.Drains > drainsAfterFirst, "The second generation did not drain the pool.");

        Assert.Equal(residentFirst.Shape, streamedFirst.Shape);
        Assert.True(residentFirst.AsReadOnlySpan<float>().SequenceEqual(residentSecond.AsReadOnlySpan<float>()), "Resident: the second generation differs.");
        Assert.True(residentFirst.AsReadOnlySpan<float>().SequenceEqual(streamedFirst.AsReadOnlySpan<float>()), "Streamed first generation differs from resident.");
        Assert.True(residentFirst.AsReadOnlySpan<float>().SequenceEqual(streamedSecond.AsReadOnlySpan<float>()), "Streamed second generation differs.");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(9)]
    public void AForcedThrowMidForwardLeavesNoBlockLoaded(int throwAtBlockForward)
    {
        CountingPoolCache pool = new();
        using Rig rig = new(pool);
        Random textRng = new(3);
        using Tensor qwen = RandomTensor(textRng, 1, 3, 24);
        using Tensor byt5 = RandomTensor(textRng, 1, 2, 16);
        MovieRollSamplerOptions options = Options();

        // The Nth block forward of the run throws with its block loaded.
        int calls = 0;
        rig.Streamed.BeforeBlockForward = _ =>
        {
            if (++calls == throwAtBlockForward) throw new InvalidOperationException("forced");
        };
        Assert.Throws<InvalidOperationException>(() =>
            rig.StreamedPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11)));
        Assert.Equal(-1, rig.BlockLoader.LoadedBlock);
        AssertReleased(rig.BlockLoader, "after the forced throw");
        Assert.True(pool.Drains > 0, "The pool was not drained after the failed run.");

        // The pipeline and the DiT are still usable: the next generation runs and releases everything.
        rig.Streamed.BeforeBlockForward = null;
        using Tensor next = rig.StreamedPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        AssertReleased(rig.BlockLoader, "the generation after the throw");
    }

    [Fact]
    public void DrainsAreCountedPerStepAndAfterTheRun()
    {
        CountingPoolCache pool = new();
        using Rig rig = new(pool);
        Random textRng = new(3);
        using Tensor qwen = RandomTensor(textRng, 1, 3, 24);
        using Tensor byt5 = RandomTensor(textRng, 1, 2, 16);

        using Tensor latents = rig.StreamedPipeline.SampleLatents(Options(), qwen, byt5, null, null, Height, Width, Noise(11));
        // One drain per denoising step (five steps, five chunks) and one after the KV cache is freed.
        Assert.True(pool.Drains >= 6, $"Expected at least 6 drains, saw {pool.Drains}.");
    }
}

/// <summary>The shared (top-level and block 0) weights of the tiny fixture, as the loader takes them.</summary>
internal static class HunyuanVideo15BlockLoaderTestsSource
{
    public static Dictionary<string, Tensor> Of(SafeTensorsLoader loader) => loader.GetAllTensors()
        .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
            && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
        .ToDictionary(pair => pair.Key, pair => pair.Value);
}
