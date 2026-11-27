using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The streamed block loader must give the resident load's forward bit for bit: with the same converter and the same cast, the velocity from
/// blocks loaded as the forward reaches them equals the velocity from the whole DiT loaded at once. Uses the tiny synthetic DiT fixture on the CPU.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class HunyuanVideo15BlockLoaderTests
{
    private const int Blocks = 2;
    private const int Hidden = 32;

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = 4, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
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

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void StreamedBlocks_GiveTheResidentVelocityBitForBit(bool castToF16, bool readFromFile)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        HunyuanVideo15Config config = TinyConfig();
        Dictionary<string, Tensor> source = Source(loader);
        List<DType> castSources = castToF16 ? [DType.F32] : [];

        // Resident: the whole DiT converted, cast and loaded at once.
        Dictionary<string, Tensor> converted = HunyuanVideo15CheckpointConverter.Convert(source, Blocks, Hidden);
        if (castToF16)
        {
            foreach (string key in converted.Keys.ToList())
            {
                if (converted[key].DType == DType.F32) converted[key] = converted[key].CastTo(DType.F16);
            }
        }
        using HunyuanVideo15Dit resident = new(config);
        resident.LoadWeights(converted);

        // Streamed: the shared weights once, then each block as the forward reaches it, released after it.
        using CpuBackend backend = new();
        using HunyuanVideo15Dit streamed = new(config);
        using HunyuanVideo15BlockLoader blockLoader = new(streamed, config, backend, source, castSources, readBlocksFromFile: readFromFile);
        blockLoader.LoadShared();
        streamed.LoadBlock = blockLoader.LoadBlock;
        streamed.UnloadBlock = blockLoader.UnloadBlock;
        Assert.True(streamed.StreamsBlockWeights);

        Random rng = new(7);
        using Tensor latent = RandomTensor(rng, 1, 4, 2, 4, 4);
        using Tensor qwen = RandomTensor(rng, 1, 3, 24);
        using Tensor byt5 = RandomTensor(rng, 1, 2, 16);
        float[] frameTimesteps = [1000f, 600f];

        using Tensor expected = resident.ForwardPerFrame(backend, latent, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
        using Tensor actual = streamed.ForwardPerFrame(backend, latent, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
        Assert.Equal(expected.Shape, actual.Shape);
        Assert.True(expected.AsReadOnlySpan<float>().SequenceEqual(actual.AsReadOnlySpan<float>()),
            "The streamed forward must equal the resident forward bit for bit.");

        // A second forward reloads the blocks, and still matches.
        using Tensor again = streamed.ForwardPerFrame(backend, latent, frameTimesteps, 1000f, qwen, byt5, conditioning: null);
        Assert.True(expected.AsReadOnlySpan<float>().SequenceEqual(again.AsReadOnlySpan<float>()));
    }
}
