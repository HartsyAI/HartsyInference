using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The query-tiled F32 attention row (the one a call with more scores than fit in memory runs) against the contract's host reference, at
/// several tile heights: the tile only changes the order of the GEMMs, so every height must agree with the reference and with each other.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("SdpaKnobs")]
public sealed class SdpaTiledGpuTests
{
    private const int Heads = 3, Tokens = 1537, HeadDim = 64;
    private readonly ITestOutputHelper _output;

    public SdpaTiledGpuTests(ITestOutputHelper output) => _output = output;

    // 1 is a tile of one query row, 37 does not divide the 1,537 rows, 512 leaves a short last tile, and 100000 is clamped to the query length, which is
    // one tile for the whole call.
    [Theory]
    [InlineData(1)]
    [InlineData(37)]
    [InlineData(512)]
    [InlineData(100000)]
    public void TiledRow_AgreesWithTheHostReferenceAtEveryTileHeight(int tileRows)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        bool forcedBefore = EngineKnobs.SdpaForceTiled.Value;
        int tileBefore = EngineKnobs.SdpaTile.Value ?? 0;
        KnobFile.Apply("{\"settings\": {\"vram.sdpaForceTiled\": true, \"numerics.sdpaTile\": " + tileRows + "}}", "SdpaTiledGpuTests");
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            Random rng = new(5);
            using Tensor q = Random(rng), k = Random(rng), v = Random(rng);
            using Tensor actual = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
            using Tensor expected = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
            float scale = 1f / MathF.Sqrt(HeadDim);

            backend.ScaledDotProductAttention(actual, q, k, v, null, scale);
            AttentionCall call = AttentionCall.FromScaledDotProductAttention(expected, q, k, v, null, scale, false);
            AttentionHostReference.Run(in call);
            backend.OffloadActivation(actual);

            // TF32 GEMMs (the engine's default for F32 attention) bound the error at the contract's TF32 tolerance.
            double error = RelativeL2(actual.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
            _output.WriteLine($"tile {tileRows}: relative L2 {error:E3}");
            Assert.True(error < 2e-3, $"relative L2 {error:E3} at tile {tileRows}");
        }
        finally
        {
            // Back to what was set before; a tile height of 0 is "no override" (TileRows ignores a non-positive height).
            KnobFile.Apply(
                "{\"settings\": {\"vram.sdpaForceTiled\": " + (forcedBefore ? "true" : "false") + ", \"numerics.sdpaTile\": " + tileBefore + "}}",
                "SdpaTiledGpuTests");
        }
    }

    private static Tensor Random(Random rng)
    {
        Tensor t = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static double RelativeL2(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double error = 0, reference = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            error += d * d;
            reference += expected[i] * (double)expected[i];
        }

        return Math.Sqrt(error / reference);
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }

        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test binary.");
    }
}

/// <summary>The tests that change attention knobs, which are process-wide, run alone.</summary>
[CollectionDefinition("SdpaKnobs", DisableParallelization = true)]
public sealed class SdpaKnobsCollection;
