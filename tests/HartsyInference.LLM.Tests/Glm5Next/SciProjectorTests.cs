using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Glm5Next.SciPerception;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The SciPerception projector against upstream's <c>AtomProjector</c> (<c>tools/parity/sciperception_reference.py</c>, SciPerception revision
/// 7c228fc0, torch 2.7.1 CPU): four tiny random projectors with the option sets the released ones use, one forward pass each.</summary>
public sealed class SciProjectorTests(ITestOutputHelper output)
{
    private static string Dir(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "sciperception", name);

    [Theory]
    [InlineData("plain")]
    [InlineData("discrete")]
    [InlineData("side")]
    [InlineData("readout")]
    public void Forward_MatchesUpstreamAtomProjector(string name)
    {
        string d = Dir(name);
        SciProjectorSpec spec = SciProjectorSpec.Parse(File.ReadAllText(Path.Combine(d, "config.json")));
        using CheckpointSource source = CheckpointSource.Open(Path.Combine(d, "projector.safetensors"));
        SciProjector projector = new(new CpuBackend(), spec, source.Weights);

        (long[] shape, float[] features) = NpyReader.ReadFloat32(Path.Combine(d, "features.npy"));
        int rows = (int)shape[0];
        int[] positions = NpyReader.ReadInt32(Path.Combine(d, "positions.npy")).Data;
        int[] discrete = Optional(Path.Combine(d, "discrete.npy"), p => NpyReader.ReadInt32(p).Data);
        float[] side = Optional(Path.Combine(d, "side.npy"), p => NpyReader.ReadFloat32(p).Data);
        float[] expected = NpyReader.ReadFloat32(Path.Combine(d, "output.npy")).Data;

        float[] actual = projector.Forward(features, rows, positions, discrete, side);
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        output.WriteLine($"{name}: max abs difference {worst:E3} over {expected.Length} values");
        Assert.True(worst < 5e-6f, $"{name}: max abs difference {worst}");
    }

    [Fact]
    public void ReleasedProjectorConfigs_ParseToTheirPublishedWidths()
    {
        // projectors/sage-s2.1-flash/spectra/qm9s/config.json and protein-seq/mpbind/config.json, projector blocks only.
        SciProjectorSpec qm9s = SciProjectorSpec.Parse("""
            {"format_version":"sage-projector-v1","projector":{"kind":"atom","feature_dim":50,"hidden_dim":1024,"index_dim":32,"max_atoms":384,
             "discrete_dim":16,"discrete_sizes":[3]},"output_dim":4096}
            """);
        Assert.Equal(98, qm9s.InputWidth);
        SciProjectorSpec mpbind = SciProjectorSpec.Parse("""
            {"format_version":"sage-projector-v1","projector":{"kind":"atom","feature_dim":1280,"hidden_dim":1024,"index_dim":32,"max_atoms":512,
             "task_head_outputs":1,"readout_classes":1,"readout_uses_side":false,"side_dim":2,"side_hidden_dim":128},"output_dim":4096}
            """);
        Assert.Equal(1312, mpbind.InputWidth);
        Assert.True(mpbind.Readout);
    }

    [Theory]
    [InlineData("\"depth\":2")]
    [InlineData("\"context_layers\":1")]
    [InlineData("\"graph_layers\":1")]
    [InlineData("\"output_scale\":4.0")]
    [InlineData("\"readout_classes\":5")]
    [InlineData("\"readout_classes\":1,\"side_dim\":2")]
    public void OptionsNoReleasedProjectorUses_AreRefusedByName(string option)
    {
        string json = "{\"format_version\":\"sage-projector-v1\",\"projector\":{\"kind\":\"atom\",\"feature_dim\":4,\"hidden_dim\":8,\"index_dim\":2,"
            + "\"max_atoms\":4," + option + "},\"output_dim\":8}";
        Assert.Throws<InvalidDataException>(() => SciProjectorSpec.Parse(json));
    }

    [Fact]
    public void RowPositionBeyondWhatTheProjectorLearned_IsRefused()
    {
        string d = Dir("plain");
        SciProjectorSpec spec = SciProjectorSpec.Parse(File.ReadAllText(Path.Combine(d, "config.json")));
        using CheckpointSource source = CheckpointSource.Open(Path.Combine(d, "projector.safetensors"));
        SciProjector projector = new(new CpuBackend(), spec, source.Weights);
        Assert.Throws<ArgumentOutOfRangeException>(() => projector.Forward(new float[spec.FeatureDim], 1, [spec.MaxAtoms], [], []));
    }

    [Fact]
    public void TransposedWeight_SameElementCount_IsRefusedByShape()
    {
        // layers.1.weight is [hidden, width]; the transposed [width, hidden] holds the same 512 values and must still be refused.
        string d = Dir("plain");
        SciProjectorSpec spec = SciProjectorSpec.Parse(File.ReadAllText(Path.Combine(d, "config.json")));
        using CheckpointSource source = CheckpointSource.Open(Path.Combine(d, "projector.safetensors"));
        Dictionary<string, Tensor> weights = source.Weights.ToDictionary(p => p.Key, p => p.Value);
        Tensor original = weights["layers.1.weight"];
        float[] values = original.AsReadOnlySpan<float>().ToArray();
        weights["layers.1.weight"] = HostMath.Tensor(values, values.Length / spec.HiddenDim, spec.HiddenDim);

        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => new SciProjector(new CpuBackend(), spec, weights));

        Assert.Contains("layers.1.weight", ex.Message);
        Assert.Contains("shape", ex.Message);
    }

    private static T[] Optional<T>(string path, Func<string, T[]> read) => File.Exists(path) ? read(path) : [];
}
