using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A gated LoRA's adapter operands are shared-resident: the transformer reports them with its shared weights, so they are preloaded once for
/// the run, and a block's own footprint stays its base weights. Streaming a block therefore never drops its operands mid-run.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAdapterResidencyTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "longcat-residency-" + Guid.NewGuid().ToString("N"));
    private readonly CpuBackend _backend = new();
    private readonly LongCatConfig _config = LongCatTestData.Tiny;
    private readonly Dictionary<string, Tensor> _dense;

    public LongCatAdapterResidencyTests()
    {
        Directory.CreateDirectory(_directory);
        _dense = LongCatTestData.Weights(_config, 61);
    }

    public void Dispose()
    {
        LongCatTestData.DisposeAll(_dense);
        _backend.Dispose();
        Directory.Delete(_directory, recursive: true);
    }

    [Fact]
    public void AGatedLoraMovesItsOperandsToTheSharedWeightsAndLeavesTheBlockAtItsBaseSize()
    {
        long baseBlockBytes, baseSharedBytes;
        using (LongCatTransformer bare = new(_config))
        {
            bare.LoadWeights(new Dictionary<string, Tensor>(_dense, StringComparer.Ordinal));
            baseBlockBytes = bare.GetBlock(0).EstimatedWeightBytes;
            baseSharedBytes = SharedBytes(bare);
        }

        using LongCatLoraFixture fixture = new(Path.Combine(_directory, "a.safetensors"), _config, 7);
        using LoraFile file = LoraFile.Load(fixture.Path);
        using LoraStack stack = new();
        stack.AddGated(file, new AdjunctGate());
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
        Assert.True(adapterBytes > 0, "the gated LoRA must put adapter operands on the blocks");

        using LongCatTransformer gated = new(_config);
        gated.LoadWeights(patched);

        Assert.Equal(baseBlockBytes, gated.GetBlock(0).EstimatedWeightBytes);
        Assert.Equal(baseSharedBytes + adapterBytes, SharedBytes(gated));
    }

    private static long SharedBytes(LongCatTransformer transformer)
    {
        long bytes = 0;
        foreach (Tensor tensor in transformer.EnumerateSharedWeights())
            bytes += tensor.DType.ComputeByteCount(tensor.ElementCount);
        return bytes;
    }
}
