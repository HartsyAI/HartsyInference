using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.LLM.Ernie5;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Keeps the layout the real-weight harness reads (FP8 shards named by <c>model.safetensors.index.json</c>) exercised on the CPU lane
/// without weights: the index maps every expected key, every shard it names exists, and the forward gives finite logits.</summary>
public sealed class Ernie5ShardedSyntheticTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-ernie5-sharded-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void Cpu_ShardedSyntheticErnieFp8_IndexMapsEveryKeyAndForwardIsFinite()
    {
        Ernie5SyntheticCheckpoint.Write(_dir, storage: Ernie5SyntheticCheckpoint.Storage.Fp8, shards: 4);
        Assert.False(File.Exists(Path.Combine(_dir, "model.safetensors")));

        Ernie5Config config = Ernie5Config.Load(Path.Combine(_dir, "config.json"));
        using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(_dir, "model.safetensors.index.json")));
        Dictionary<string, string> weightMap = new(StringComparer.Ordinal);
        foreach (JsonProperty entry in index.RootElement.GetProperty("weight_map").EnumerateObject())
            weightMap[entry.Name] = entry.Value.GetString()!;

        Assert.Equal(4, weightMap.Values.Distinct().Count());
        foreach (string shard in weightMap.Values.Distinct())
            Assert.True(File.Exists(Path.Combine(_dir, shard)), $"{shard} is named by the index but is not on disk.");
        Ernie5WeightCoverage coverage = Ernie5WeightMap.Check(config, weightMap.Keys, mtp: true);
        coverage.RequireComplete();
        Assert.Empty(coverage.Unmapped);

        using CpuBackend cpu = new();
        // The helper returns every row of the forward, so the last row is the next-token distribution.
        float[] logits = Ernie5ModelGpuParityTests.Logits(_dir, cpu, resident: false);
        Assert.Equal(0, logits.Length % config.VocabSize);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        float[] last = logits[(logits.Length - config.VocabSize)..];
        Assert.True(last.Max() > last.Min(), "The sharded forward produced constant logits.");
    }
}
