using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A block's footprint counts its own weights only. A gated LoRA's adapter operands count with the transformer's shared weights, which the
/// transformer preloads once; the planner sizes the resident prefix against shared plus blocks.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatLoraFootprintTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "longcat-footprint-" + Guid.NewGuid().ToString("N"));
    private readonly CpuBackend _backend = new();
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _dense;

    public LongCatLoraFootprintTests()
    {
        Directory.CreateDirectory(_directory);
        _dense = LongCatTestData.Weights(_config, 43);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_dense);
        _backend.Dispose();
        Directory.Delete(_directory, recursive: true);
    }

    [Fact]
    public void AGatedLoraAddsItsAdapterOperandsToTheSharedFootprintNotTheBlock()
    {
        long baseBytes;
        long baseSharedBytes;
        using (LongCatTransformer bare = new(_config))
        {
            bare.LoadWeights(new Dictionary<string, Tensor>(_dense, StringComparer.Ordinal));
            baseBytes = bare.GetBlock(0).EstimatedWeightBytes;
            baseSharedBytes = SharedBytes(bare);
        }

        using LongCatLoraFixture fixture = new(Path.Combine(_directory, "a.safetensors"), _config, 7);
        using LoraFile file = LoraFile.Load(fixture.Path);
        using LoraStack stack = new();
        AdjunctGate gate = new();
        stack.AddGated(file, gate);
        Dictionary<string, Tensor> patched = new(_dense, StringComparer.Ordinal);
        stack.ApplyTo(patched, LoraTarget.Transformer, _backend);

        long adapterBytes = 0;
        foreach ((string key, Tensor tensor) in patched)
        {
            if (!key.StartsWith("blocks.", StringComparison.Ordinal) || tensor.LowRankAdjunct is null)
                continue;
            foreach (LowRankAdjunctTerm term in tensor.LowRankAdjunct.Terms)
            {
                adapterBytes += term.Down.DType.ComputeByteCount(term.Down.ElementCount);
                if (term.Up is not null)
                    adapterBytes += term.Up.DType.ComputeByteCount(term.Up.ElementCount);
            }
        }
        using LongCatTransformer gated = new(_config);
        gated.LoadWeights(patched);

        Assert.True(adapterBytes > 0, "the gated LoRA must put adapter operands on the blocks");
        // The operands are resident for the run, so they count with the shared weights (preloaded once), and the block streams its base weights only.
        long blockBytes = gated.GetBlock(0).EstimatedWeightBytes;
        long sharedBytes = SharedBytes(gated);
        Assert.Equal(baseBytes, blockBytes);
        Assert.Equal(baseSharedBytes + adapterBytes, sharedBytes);
    }

    private static long SharedBytes(LongCatTransformer transformer)
    {
        long bytes = 0;
        foreach (Tensor tensor in transformer.EnumerateSharedWeights())
            bytes += tensor.DType.ComputeByteCount(tensor.ElementCount);
        return bytes;
    }
}
