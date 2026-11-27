using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Kimi;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The disk path: a Kimi K3.1 checkpoint directory (config, index and safetensors shard under the published names) loads through
/// <see cref="KimiK31Checkpoint.Load"/> and runs on the CPU backend.</summary>
public sealed class KimiCheckpointLoadTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-kimi-ckpt-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void Load_BuildsTheDecoderFromTheCheckpointAndRunsIt()
    {
        KimiSyntheticCheckpoint.Write(_dir);
        using KimiK31Checkpoint checkpoint = KimiK31Checkpoint.Load(_dir, new CpuBackend(), maxPositions: 16);

        Assert.Equal(KimiSyntheticCheckpoint.Hidden, checkpoint.Config.Hidden);
        Assert.Equal(KimiSyntheticCheckpoint.Layers, checkpoint.Config.Layers);
        Assert.Contains(1, checkpoint.Config.FullAttentionLayers);
        using KimiK31State state = new(checkpoint.Config, capacity: 16);
        float[] logits = checkpoint.Model.Forward([1, 2, 3], state);
        Assert.Equal(KimiSyntheticCheckpoint.Vocab, logits.Length);
        foreach (float v in logits) Assert.True(float.IsFinite(v));
    }

    [Fact]
    public void Load_RefusesACheckpointMissingATensor()
    {
        KimiSyntheticCheckpoint.Write(_dir);
        string index = Path.Combine(_dir, "model.safetensors.index.json");
        File.WriteAllText(index, File.ReadAllText(index).Replace("language_model.lm_head.weight", "language_model.lm_head.renamed", StringComparison.Ordinal));
        Assert.ThrowsAny<Exception>(() => KimiK31Checkpoint.Load(_dir, new CpuBackend(), maxPositions: 16));
    }
}
