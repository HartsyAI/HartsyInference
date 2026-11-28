using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Two generations through <see cref="MovieRollPipeline"/> on the tiny DiT, resident and streamed. Each generation runs one forward per denoising
/// step and window, so the streamed run loads and releases its blocks on every forward, and the second generation reuses the same pipeline and DiT.
/// All four latents must be bit-identical.</summary>
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

    /// <summary>After a generation no block is loaded, every load has its release, and no block tensor is held.</summary>
    private static void AssertReleased(HunyuanVideo15BlockLoader loader, string when)
    {
        Assert.True(loader.LoadedBlock == -1, $"{when}: block {loader.LoadedBlock} is still loaded.");
        Assert.True(loader.LoadedBlockTensorCount == 0, $"{when}: {loader.LoadedBlockTensorCount} block tensors are still held.");
        Assert.Equal(loader.BlockLoads, loader.BlockReleases);
    }

    [Fact]
    public void TwoGenerationsStreamedAndResidentGiveTheSameLatents()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> source = Source(loader);
        HunyuanVideo15Config config = TinyDitConfig();
        using CpuBackend backend = new();
        MovieRollSchedule schedule = MovieRollSchedule.Distilled([1000, 800, 600, 400, 200], 5, 5.0);
        MovieRollSamplerOptions options = new()
        {
            ChunkFrames = 1, NumChunks = 3, FullHistoryBlocks = 1, ContextNoise = 0f, KvRecomputeEnabled = false, Cfg = null,
        };
        Random textRng = new(3);
        using Tensor qwen = RandomTensor(textRng, 1, 3, 24);
        using Tensor byt5 = RandomTensor(textRng, 1, 2, 16);
        using HunyuanVideo15Vae vae = new(TinyVaeConfig());

        // Resident: the whole DiT converted, cast and loaded at once.
        using HunyuanVideo15Dit resident = new(config);
        resident.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(source, Blocks, Hidden));
        MovieRollPipeline residentPipeline = new(backend, resident, vae, schedule);

        // Streamed: the loader owns the blocks; the pipeline keeps only the shared weights resident.
        using HunyuanVideo15Dit streamed = new(config);
        using HunyuanVideo15BlockLoader blockLoader = new(streamed, config, backend, source, [], readBlocksFromFile: true);
        blockLoader.LoadShared();
        streamed.LoadBlock = blockLoader.LoadBlock;
        streamed.UnloadBlock = blockLoader.UnloadBlock;
        MovieRollPipeline streamedPipeline = new(backend, streamed, vae, schedule);

        using Tensor residentFirst = residentPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        using Tensor residentSecond = residentPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        using Tensor streamedFirst = streamedPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        AssertReleased(blockLoader, "first generation");
        int loadsAfterFirst = blockLoader.BlockLoads;
        Assert.True(loadsAfterFirst > 0, "The first generation loaded no blocks.");

        using Tensor streamedSecond = streamedPipeline.SampleLatents(options, qwen, byt5, null, null, Height, Width, Noise(11));
        AssertReleased(blockLoader, "second generation");
        // The second generation loads the same blocks the first did, and nothing the first left stays behind.
        Assert.Equal(2 * loadsAfterFirst, blockLoader.BlockLoads);

        Assert.Equal(residentFirst.Shape, streamedFirst.Shape);
        Assert.True(residentFirst.AsReadOnlySpan<float>().SequenceEqual(residentSecond.AsReadOnlySpan<float>()), "Resident: the second generation differs.");
        Assert.True(residentFirst.AsReadOnlySpan<float>().SequenceEqual(streamedFirst.AsReadOnlySpan<float>()), "Streamed first generation differs from resident.");
        Assert.True(residentFirst.AsReadOnlySpan<float>().SequenceEqual(streamedSecond.AsReadOnlySpan<float>()), "Streamed second generation differs.");
    }
}
