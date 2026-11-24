using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Ernie5;
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
        using Ernie5CheckpointWeights source = Ernie5CheckpointWeights.Open(Dir);
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
    public void GridThatTheMergeSizeDoesNotDivide_IsRefused()
    {
        Glm5NextVisionConfig config = Glm5NextHfConfig.ParseVision(File.ReadAllText(Path.Combine(Dir, "config.json")));
        using Ernie5CheckpointWeights source = Ernie5CheckpointWeights.Open(Dir);
        using Glm5NextVisionTower tower = new(new CpuBackend(), config, source);
        Assert.Throws<ArgumentException>(() => tower.Forward(new float[tower.PatchRowWidth * 15], [(1, 3, 5)]));
    }
}
