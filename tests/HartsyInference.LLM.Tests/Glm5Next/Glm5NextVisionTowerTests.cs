using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Glm5Next;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The C# vision tower against transformers' <c>Glm5NextVisionModel</c> (commit 52bac5d0, torch 2.7.1 CPU F32): a tiny random tower with the real
/// <c>model.visual.*</c> keys on one image grid (1x4x6) and one two-frame clip (2x4x4), so per-frame attention segments and repeated positions are exercised.
/// Reference: <c>tools/parity/glm5_next_vision_reference.py</c>.</summary>
public sealed class Glm5NextVisionTowerTests(ITestOutputHelper output)
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next", "tiny_vision");

    [Fact]
    public void MergerRows_MatchTheReference()
    {
        Glm5NextVisionConfig config = Glm5NextHfConfig.ParseVision(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, "glm5_next");
        using Glm5NextVisionTower tower = new(new CpuBackend(), config, source);
        float[] patches = NpyReader.ReadFloat32(Path.Combine(Dir, "patches.npy")).Data;
        int[] grid = NpyReader.ReadInt32(Path.Combine(Dir, "grid_thw.npy")).Data;
        (int, int, int)[] grids = [(grid[0], grid[1], grid[2]), (grid[3], grid[4], grid[5])];
        float[] expected = NpyReader.ReadFloat32(Path.Combine(Dir, "merged.npy")).Data;

        float[] actual = tower.Forward(patches, grids);

        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f, scale = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
            scale = MathF.Max(scale, MathF.Abs(expected[i]));
        }
        output.WriteLine($"merger rows: max abs difference {worst:E3} (reference max magnitude {scale:F3}, {expected.Length} values)");
        Assert.True(worst < 1e-5f, $"max abs difference {worst}");
    }

    [Fact]
    public void DownsampledRows_MatchTheReferenceLastHiddenState()
    {
        // The fixture's tower.npy is transformers' last_hidden_state: the tower after the downsample, before the merger.
        Glm5NextVisionConfig config = Glm5NextHfConfig.ParseVision(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, "glm5_next");
        using Glm5NextVisionTower tower = new(new CpuBackend(), config, source);
        float[] patches = NpyReader.ReadFloat32(Path.Combine(Dir, "patches.npy")).Data;
        int[] grid = NpyReader.ReadInt32(Path.Combine(Dir, "grid_thw.npy")).Data;
        (int, int, int)[] grids = [(grid[0], grid[1], grid[2]), (grid[3], grid[4], grid[5])];
        float[] expected = NpyReader.ReadFloat32(Path.Combine(Dir, "tower.npy")).Data;

        float[] actual = tower.Downsample(patches, grids);

        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        output.WriteLine($"downsampled rows: max abs difference {worst:E3} ({expected.Length} values)");
        Assert.True(worst < 1e-5f, $"max abs difference {worst}");
    }

    [Fact]
    public void WeightWithTheRightCountButAnotherShape_IsRefusedByName()
    {
        // blocks.0.attn.qkv.weight is [3h, h]; the transposed [h, 3h] holds the same values and must still be refused.
        Glm5NextVisionConfig config = Glm5NextHfConfig.ParseVision(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using BlockFp8CheckpointWeights inner = BlockFp8CheckpointWeights.Open(Dir, "glm5_next");
        string name = "model.visual.blocks.0.attn.qkv.weight";
        float[] values = inner.Read(name).AsReadOnlySpan<float>().ToArray();
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() =>
            new Glm5NextVisionTower(new CpuBackend(), config, new Replacing(inner, name, () => HostMath.Tensor(values, config.Hidden, 3 * config.Hidden))));
        Assert.Contains(name, ex.Message);
    }

    [Fact]
    public void MissingTensor_IsRefusedAsInvalidData()
    {
        Glm5NextVisionConfig config = Glm5NextHfConfig.ParseVision(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using BlockFp8CheckpointWeights inner = BlockFp8CheckpointWeights.Open(Dir, "glm5_next");
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() =>
            new Glm5NextVisionTower(new CpuBackend(), config, new Missing(inner, "model.visual.merger.proj.weight")));
        Assert.Contains("merger.proj.weight", ex.Message);
    }

    [Theory]
    [InlineData(10, 4)]
    [InlineData(12, 2)]
    public void WidthsThatDoNotSplitIntoHeads_AreRefused(int hidden, int heads)
    {
        Glm5NextVisionConfig bad = Glm5NextHfConfig.ParseVision(File.ReadAllText(Path.Combine(Dir, "config.json"))) with { Hidden = hidden, Heads = heads };
        Assert.Throws<InvalidDataException>(() => bad.Validate());
    }

    // Returns a tensor from `make` for one name and passes every other read through.
    private sealed class Replacing(IModelWeightSource inner, string name, Func<Tensor> make) : IModelWeightSource
    {
        public IEnumerable<string> Names => inner.Names;
        public Tensor Read(string n) => n == name ? make() : inner.Read(n);
    }

    // Fails the read of one name as a checkpoint without that tensor does.
    private sealed class Missing(IModelWeightSource inner, string name) : IModelWeightSource
    {
        public IEnumerable<string> Names => inner.Names;
        public Tensor Read(string n) => n == name ? throw new KeyNotFoundException($"no {n}") : inner.Read(n);
    }

    [Fact]
    public void GridThatTheMergeSizeDoesNotDivide_IsRefused()
    {
        Glm5NextVisionConfig config = Glm5NextHfConfig.ParseVision(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using BlockFp8CheckpointWeights source = BlockFp8CheckpointWeights.Open(Dir, "glm5_next");
        using Glm5NextVisionTower tower = new(new CpuBackend(), config, source);
        Assert.Throws<ArgumentException>(() => tower.Forward(new float[tower.PatchRowWidth * 15], [(1, 3, 5)]));
    }
}
