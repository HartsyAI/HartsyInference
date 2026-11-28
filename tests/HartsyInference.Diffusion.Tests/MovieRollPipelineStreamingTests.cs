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

    /// <summary>A placement that does the CPU's no-op work and throws on a chosen call of the upload or the release (1-based; 0 never throws).</summary>
    private sealed class FaultyPlacement : IBlockPlacement
    {
        private readonly BackendBlockPlacement _inner = new(new CpuBackend());

        public int FailPreloadOn { get; set; }

        public int FailFreeOn { get; set; }

        private int _preloads;
        private int _frees;

        public void Preload(IEnumerable<Tensor> weights)
        {
            if (++_preloads == FailPreloadOn) throw new InvalidOperationException("forced upload failure");
            _inner.Preload(weights);
        }

        public void Free(IEnumerable<Tensor> weights)
        {
            if (++_frees == FailFreeOn) throw new InvalidOperationException("forced release failure");
            _inner.Free(weights);
        }

        public void Sync() => _inner.Sync();
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
        public Rig(CountingPoolCache pool, IBlockPlacement? placement = null, bool trimEachStep = true)
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
            Loader = new SafeTensorsLoader();
            Loader.Load(path);
            Source = HunyuanVideo15BlockLoaderTestsSource.Of(Loader);
            Config = TinyDitConfig();
            Backend = new CpuBackend();
            VaeLoader = new SafeTensorsLoader();
            VaeLoader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_vae_tiny.safetensors"));
            Vae = new HunyuanVideo15Vae(TinyVaeConfig());
            // The fixture also holds the reference's own tensors; the VAE loads only its encoder and decoder weights.
            Vae.LoadWeights(VaeLoader.GetAllTensors()
                .Where(pair => pair.Key.StartsWith("encoder.", StringComparison.Ordinal) || pair.Key.StartsWith("decoder.", StringComparison.Ordinal))
                .ToDictionary(pair => pair.Key, pair => pair.Value));
            Schedule = MovieRollSchedule.Distilled([1000, 800, 600, 400, 200], 5, 5.0);
            Resident = new HunyuanVideo15Dit(Config);
            Resident.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(Source, Blocks, Hidden));
            ResidentPipeline = new MovieRollPipeline(Backend, Resident, Vae, Schedule);
            Streamed = new HunyuanVideo15Dit(Config);
            BlockLoader = placement is null
                ? new HunyuanVideo15BlockLoader(Streamed, Config, Backend, Source, [], readBlocksFromFile: true)
                : new HunyuanVideo15BlockLoader(Streamed, Config, placement, Source, [], readBlocksFromFile: true);
            BlockLoader.LoadShared();
            Streamed.LoadBlock = BlockLoader.LoadBlock;
            Streamed.UnloadBlock = BlockLoader.UnloadBlock;
            StreamedPipeline = new MovieRollPipeline(Backend, Streamed, Vae, Schedule, poolCache: pool, trimEachStep: trimEachStep);
        }

        public SafeTensorsLoader Loader { get; }
        public SafeTensorsLoader VaeLoader { get; }
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
            VaeLoader.Dispose();
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
        AssertReleased(rig.BlockLoader, "after the forced throw");
        Assert.True(pool.Drains > 0, "The pool was not drained after the failed run.");

        // The pipeline and the DiT are still usable: the next generation runs and releases everything.
        rig.Streamed.BeforeBlockForward = null;
        using Tensor next = rig.StreamedPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        AssertReleased(rig.BlockLoader, "the generation after the throw");
    }

    [Fact]
    public void AFailedUploadInTheRunLeavesNoBlockLoaded()
    {
        CountingPoolCache pool = new();
        FaultyPlacement placement = new() { FailPreloadOn = 7 };
        using Rig rig = new(pool, placement);
        Random textRng = new(3);
        using Tensor qwen = RandomTensor(textRng, 1, 3, 24);
        using Tensor byt5 = RandomTensor(textRng, 1, 2, 16);

        Assert.Throws<InvalidOperationException>(() =>
            rig.StreamedPipeline.SampleLatents(Options(), qwen, byt5, null, null, Height, Width, Noise(11)));
        AssertReleased(rig.BlockLoader, "after a failed upload");
    }

    [Fact]
    public void SampleLatentsDrainsOnceAfterTheRunWhenStepDrainsAreOff()
    {
        CountingPoolCache pool = new();
        using Rig rig = new(pool, trimEachStep: false);
        Random textRng = new(3);
        using Tensor qwen = RandomTensor(textRng, 1, 3, 24);
        using Tensor byt5 = RandomTensor(textRng, 1, 2, 16);

        using Tensor latents = rig.StreamedPipeline.SampleLatents(Options(), qwen, byt5, null, null, Height, Width, Noise(11));
        // Only the drain after the KV cache is freed.
        Assert.Equal(1, pool.Drains);
    }

    [Fact]
    public void GenerateDrainsAfterTheDecodeToo()
    {
        CountingPoolCache pool = new();
        using Rig rig = new(pool, trimEachStep: false);
        Random textRng = new(3);
        using Tensor qwen = RandomTensor(textRng, 1, 3, 24);
        using Tensor byt5 = RandomTensor(textRng, 1, 2, 16);

        using Tensor frames = rig.StreamedPipeline.Generate(Options(), qwen, byt5, null, null, Height, Width, Noise(11));
        // The drain after the KV cache is freed (inside SampleLatents), then the drain after the decode.
        Assert.Equal(2, pool.Drains);
        AssertReleased(rig.BlockLoader, "after Generate");

        // The same generation through the resident DiT gives the same frames.
        using Tensor residentFrames = rig.ResidentPipeline.Generate(Options(), qwen, byt5, null, null, Height, Width, Noise(11));
        Assert.Equal(residentFrames.Shape, frames.Shape);
        Assert.True(residentFrames.AsReadOnlySpan<float>().SequenceEqual(frames.AsReadOnlySpan<float>()), "Streamed Generate differs from resident.");
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
