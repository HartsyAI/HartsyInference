using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// End-to-end residency gate for a complete SD3 transformer forward. Primitive-only tests can all pass while a
/// model helper still dereferences <see cref="Tensor.DataPointer"/> between them, so this test measures the whole
/// patch-embed → MMDiT blocks → final AdaLN/projection → unpatchify chain before inspecting its output.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class Sd3GpuResidencyTests
{
    private readonly ITestOutputHelper _output;

    public Sd3GpuResidencyTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void FullForward_MatchesCpuAndHasNoIntermediateD2h()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        Sd3Config config = new()
        {
            Depth = 2,
            HiddenSize = 32,
            NumHeads = 4,
            PatchSize = 2,
            InChannels = 4,
            JointAttentionDim = 16,
            PooledProjectionDim = 12,
            UseQkNorm = true,
            DualAttentionLayers = null,
        };

        Dictionary<string, Tensor> weights = Sd3WeightBuilder.Build(config);
        // Exercise the center-cropped positional path as part of the complete forward. A 7x7 source grid is
        // deliberately larger than the 4x6 latent patch grid and asymmetric in width.
        weights["pos_embed.pos_embed"] = Sd3WeightBuilder.Rand(
            new TensorShape(1, 7 * 7, config.HiddenSize), seed: 901, scale: 0.03f);

        string? priorDebugDir = Environment.GetEnvironmentVariable("SD3_DEBUG_DIR");
        Environment.SetEnvironmentVariable("SD3_DEBUG_DIR", null);
        try
        {
            using Tensor latent = Sd3WeightBuilder.Rand(
                new TensorShape(1, config.InChannels, 8, 12), seed: 902, scale: 0.5f);
            using Tensor context = Sd3WeightBuilder.Rand(
                new TensorShape(1, 5, config.HiddenSize), seed: 903, scale: 0.2f);
            using Tensor pooled = Sd3WeightBuilder.Rand(
                new TensorShape(1, config.PooledProjectionDim), seed: 904, scale: 0.2f);

            float[] expected;
            using (CpuBackend cpu = new())
            using (Sd3Transformer reference = new(config))
            {
                reference.LoadWeights(weights);
                using Tensor output = reference.Forward(cpu, latent, 500.0f, context, pooled);
                expected = Snapshot(output);
            }

            using CudaBackend cuda = new(0, PtxDir()) { HighPrecisionGemm = true };
            using Sd3Transformer transformer = new(config);
            transformer.LoadWeights(weights);
            cuda.PreloadWeights(transformer.EnumerateWeights());
            cuda.Sync();
            cuda.ResetD2hSyncCount();

            using Tensor actualTensor = transformer.Forward(cuda, latent, 500.0f, context, pooled);
            cuda.Sync();
            Assert.Equal(0, cuda.GetD2hSyncCount());

            float[] actual = Snapshot(actualTensor);
            Assert.Equal(1, cuda.GetD2hSyncCount());
            Assert.Equal(new TensorShape(1, config.InChannels, 8, 12), actualTensor.Shape);
            AssertFiniteClose(expected, actual, tolerance: 3e-3f);

            cuda.FreeWeights(transformer.EnumerateWeights());
        }
        finally
        {
            Environment.SetEnvironmentVariable("SD3_DEBUG_DIR", priorDebugDir);
            Sd3WeightBuilder.DisposeAll(weights);
        }
    }

    private static string PtxDir()
    {
        string directory = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(directory))
        {
            directory = Path.Combine(
                HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        }
        return directory;
    }

    private static float[] Snapshot(Tensor tensor)
    {
        float[] values = new float[checked((int)tensor.ElementCount)];
        new ReadOnlySpan<float>(tensor.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private void AssertFiniteClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxError = 0f;
        int maxIndex = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]), $"CPU output[{i}] is non-finite: {expected[i]}.");
            Assert.True(float.IsFinite(actual[i]), $"CUDA output[{i}] is non-finite: {actual[i]}.");
            float error = MathF.Abs(expected[i] - actual[i]);
            Assert.True(float.IsFinite(error), $"Absolute error[{i}] is non-finite.");
            if (error > maxError)
            {
                maxError = error;
                maxIndex = i;
            }
        }

        _output.WriteLine($"Full SD3 forward: max abs error {maxError:E6} at {maxIndex}; intermediate D2H=0.");
        Assert.True(maxError <= tolerance,
            $"Full SD3 forward max error {maxError:E6} at {maxIndex} exceeds {tolerance:E2}; " +
            $"expected={expected[maxIndex]:G9}, actual={actual[maxIndex]:G9}.");
    }
}
