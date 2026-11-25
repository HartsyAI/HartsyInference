using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The block footprint the streaming planner reads counts a gated LoRA's adapter operands, not only the base weights: the planner must
/// size the resident prefix against what the device actually holds.</summary>
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
    public void AGatedLoraAddsItsAdapterOperandsToTheBlockFootprint()
    {
        long baseBytes;
        using (LongCatTransformer bare = new(_config))
        {
            bare.LoadWeights(new Dictionary<string, Tensor>(_dense, StringComparer.Ordinal));
            baseBytes = bare.GetBlock(0).EstimatedWeightBytes;
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
            if (!key.StartsWith("blocks.0.", StringComparison.Ordinal) || tensor.LowRankAdjunct is null)
                continue;
            foreach (Tensor operand in tensor.LowRankAdjunct.EnumerateOperands())
                adapterBytes += operand.DType.ComputeByteCount(operand.ElementCount);
        }
        using LongCatTransformer gated = new(_config);
        gated.LoadWeights(patched);

        Assert.True(adapterBytes > 0, "the gated LoRA must put adapter operands on block 0");
        Assert.Equal(baseBytes + adapterBytes, gated.GetBlock(0).EstimatedWeightBytes);
    }
}
