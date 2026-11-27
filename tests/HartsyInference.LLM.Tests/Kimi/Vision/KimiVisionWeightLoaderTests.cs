using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Kimi.Vision;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi.Vision;

/// <summary>The vision loader reads the checkpoint's names, refuses a missing or misshapen tensor by name, and converts BF16 to F32.</summary>
public sealed class KimiVisionWeightLoaderTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-kimi-vision-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void Load_RefusesAMissingTensorByName()
    {
        KimiVisionConfig config = KimiVisionSyntheticCheckpoint.Tiny(layers: 2);
        Dictionary<string, float[]> values = KimiVisionSyntheticCheckpoint.Values(config, seed: 3);
        string missing = "vision_tower.encoder.blocks.1.wo.weight";
        MemorySource source = new(values, KimiVisionSyntheticCheckpoint.Dims(config), omit: missing);

        KeyNotFoundException error = Assert.Throws<KeyNotFoundException>(() => KimiVisionWeightLoader.Load(config, source));

        Assert.Contains(missing, error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_RefusesAShapeThatDiffersFromTheConfigByName()
    {
        KimiVisionConfig config = KimiVisionSyntheticCheckpoint.Tiny(layers: 1);
        Dictionary<string, float[]> values = KimiVisionSyntheticCheckpoint.Values(config, seed: 4);
        Dictionary<string, long[]> dims = KimiVisionSyntheticCheckpoint.Dims(config);
        string wrong = "mm_projector.post_norm.weight";
        dims[wrong] = [config.TextHidden + 1];
        MemorySource source = new(values, dims, omit: null);

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => KimiVisionWeightLoader.Load(config, source));

        Assert.Contains(wrong, error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_ReadsBf16TensorsFromTheSameIndexAsF32()
    {
        KimiVisionConfig config = KimiVisionSyntheticCheckpoint.Tiny(layers: 1);
        string bf16 = "vision_tower.patch_embed.proj.weight";
        Dictionary<string, float[]> values = KimiVisionSyntheticCheckpoint.Values(config, seed: 5, bf16);
        KimiVisionSyntheticCheckpoint.Write(_dir, config, values, bf16);

        using KimiK31CheckpointWeights source = KimiK31CheckpointWeights.Open(_dir);
        using KimiVisionWeights weights = KimiVisionWeightLoader.Load(config, source);

        Assert.Equal(values[bf16], weights.PatchEmbed.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(values["vision_tower.encoder.blocks.0.wqkv.weight"].AsSpan(0, config.QkvHidden * config.Hidden).ToArray(),
            weights.Blocks[0].Q.AsReadOnlySpan<float>().ToArray());
        Assert.Equal(values["vision_tower.encoder.blocks.0.wqkv.weight"].AsSpan(config.QkvHidden * config.Hidden, config.QkvHidden * config.Hidden).ToArray(),
            weights.Blocks[0].K.AsReadOnlySpan<float>().ToArray());
    }

    /// <summary>An in-memory source with the checkpoint's names; each read returns a fresh tensor the caller disposes, as the contract says.</summary>
    private sealed class MemorySource : IKimiK31WeightSource
    {
        private readonly Dictionary<string, float[]> _values;
        private readonly Dictionary<string, long[]> _dims;

        public MemorySource(Dictionary<string, float[]> values, Dictionary<string, long[]> dims, string? omit)
        {
            _values = new Dictionary<string, float[]>(values, StringComparer.Ordinal);
            if (omit is not null) _values.Remove(omit);
            _dims = dims;
        }

        public bool Has(string name) => _values.ContainsKey(name);

        public Tensor Read(string name)
        {
            float[] values = _values[name];
            Tensor t = new(new TensorShape(_dims[name]), DType.F32);
            values.CopyTo(t.AsSpan<float>());
            return t;
        }
    }
}
