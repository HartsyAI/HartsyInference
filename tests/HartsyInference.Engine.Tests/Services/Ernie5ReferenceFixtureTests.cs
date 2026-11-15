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

    [WriteFact]
    public void WriteRealDimensionSyntheticCheckpoint()
    {
        string dir = Environment.GetEnvironmentVariable(DirVariable)!;
        Ernie5SyntheticCheckpoint.Write(dir, seed: 7, storage: Ernie5SyntheticCheckpoint.Storage.Fp8,
            configOverrides: Ernie5SyntheticCheckpoint.RealDimensionOverrides, shards: 3);
        Assert.True(File.Exists(Path.Combine(dir, "model.safetensors.index.json")));
    }
}
