using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Fibo 2.0 recipe casts a BF16 checkpoint to F16 before loading, so the denoiser must read half-precision storage
/// correctly. The learned Perceiver queries are read as raw floats, so the tiny F32 fixture is cast to F16 for that one tensor and
/// compared with its F32 run. The other weights stay F32: the CPU backend's linear kernels take F32 only.</summary>
public sealed unsafe class Fibo2HalfPrecisionTests
{
    /// <summary>F16 rounds every weight by about 5e-4 relative, and the error compounds through the blocks. A raw float read of
    /// half-precision storage misses by the full value, so this bound still separates a correct load from a broken one.</summary>
    private const float RelativeTolerance = 5e-2f;

    private static readonly string FixturePath = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Fibo2", "fibo2_tiny.safetensors");

    [Fact]
    public void Forward_HalfPrecisionWeights_MatchTheFloatRun()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(FixturePath);
        Dictionary<string, Tensor> all = loader.GetAllTensors();
        Dictionary<string, Tensor> weights32 = new(StringComparer.Ordinal);
        foreach ((string key, Tensor value) in all)
        {
            if (key.StartsWith("weights.", StringComparison.Ordinal))
                weights32[key["weights.".Length..]] = value;
        }
        Fibo2Config config = Fibo2Config.FromWeights(weights32, TinyTemplate());
        float sigma = ((float*)all["inputs.sigma"].DataPointer)[0];
        CpuBackend backend = new();

        using Fibo2Transformer reference = new(config);
        reference.LoadWeights(weights32);
        using Tensor velocity32 = reference.Forward(backend, all["inputs.latent"], sigma, all["inputs.context"]);

        List<Tensor> halfWeights = [];
        try
        {
            Dictionary<string, Tensor> weights16 = new(weights32, StringComparer.Ordinal);
            Tensor queries = weights32["perceiver.latent_queries"].CastTo(DType.F16);
            halfWeights.Add(queries);
            weights16["perceiver.latent_queries"] = queries;

            using Fibo2Transformer halfTransformer = new(config);
            halfTransformer.LoadWeights(weights16);
            using Tensor velocity16 = halfTransformer.Forward(backend, all["inputs.latent"], sigma, all["inputs.context"]);

            Assert.Equal(velocity32.Shape, velocity16.Shape);
            ReadOnlySpan<float> expected = new((void*)velocity32.DataPointer, (int)velocity32.ElementCount);
            ReadOnlySpan<float> actual = new((void*)velocity16.DataPointer, (int)velocity16.ElementCount);
            float scale = 0f;
            float worst = 0f;
            for (int i = 0; i < expected.Length; i++)
            {
                scale = MathF.Max(scale, MathF.Abs(expected[i]));
                worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
            }
            Assert.True(worst <= RelativeTolerance * scale,
                $"half-precision velocity max |Δ| = {worst} exceeds {RelativeTolerance} of its scale {scale}");
        }
        finally
        {
            foreach (Tensor half in halfWeights)
                half.Dispose();
        }
    }

    /// <summary>The fixture's dimensions, as in the parity test (tests/python-reference/fibo2_reference.py TINY).</summary>
    private static Fibo2Config TinyTemplate() => Fibo2Config.Checkpoint with
    {
        InChannels = 32,
        PatchSize = 2,
        AxesDims = [8, 12, 12],
        RopeTheta = 256f,
        NormEps = 1e-5f,
        TimestepScale = 1000f,
        MaxGist = 8,
        MinGist = 4,
        GistStep = 2,
        GistMinTextLen = 16,
        GistMaxTextLen = 64,
    };
}
