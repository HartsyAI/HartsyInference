using HartsyInference.LLM.Ernie5;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Writes the real-dimension synthetic ERNIE checkpoint (FP8 storage, three layers: HCA, DSA, HCA; eight routed experts; 4096 vocab)
/// into the directory named by <c>ERNIE_WRITE_SYNTHETIC_DIR</c>, for the offline per-layer dry run in tools/ernie5-reference. Skipped when the
/// variable is unset. The writer is the same one the unit tests use, so the key layout is the engine's.</summary>
[Trait("Category", "Integration")]
public sealed class Ernie5ReferenceFixtureTests
{
    private const string DirVariable = "ERNIE_WRITE_SYNTHETIC_DIR";

    /// <summary>A fact reported as skipped (never passed) when <c>ERNIE_WRITE_SYNTHETIC_DIR</c> is not set.</summary>
    private sealed class WriteFactAttribute : FactAttribute
    {
        public WriteFactAttribute()
        {
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(DirVariable)))
                Skip = $"SKIPPED: {DirVariable} is not set; the synthetic reference checkpoint was not written.";
        }
    }

    /// <summary>Optional config fields on top of the real dimensions, as <c>key=json</c> pairs split by <c>;</c>, for example
    /// <c>compress_ratios=[8, -2, 8];sliding_window=8;index_topk=16</c> to reach the compressor and the indexer's selection in a few tokens.</summary>
    private const string OverridesVariable = "ERNIE_WRITE_SYNTHETIC_OVERRIDES";

    [WriteFact]
    public void WriteRealDimensionSyntheticCheckpoint()
    {
        string dir = Environment.GetEnvironmentVariable(DirVariable)!;
        Ernie5SyntheticCheckpoint.Write(dir, seed: 7, storage: Ernie5SyntheticCheckpoint.Storage.Fp8,
            configOverrides: WithExtraOverrides(Environment.GetEnvironmentVariable(OverridesVariable)), shards: 3);
        Assert.True(File.Exists(Path.Combine(dir, "model.safetensors.index.json")));
    }

    private static Dictionary<string, string> WithExtraOverrides(string? extra)
    {
        Dictionary<string, string> all = new(Ernie5SyntheticCheckpoint.RealDimensionOverrides, StringComparer.Ordinal);
        foreach (string pair in (extra ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int eq = pair.IndexOf('=');
            if (eq <= 0) throw new ArgumentException($"{OverridesVariable}: expected key=json, got '{pair}'.");
            all[pair[..eq].Trim()] = pair[(eq + 1)..].Trim();
        }
        return all;
    }
}
