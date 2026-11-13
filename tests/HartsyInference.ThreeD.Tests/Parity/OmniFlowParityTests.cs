using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>The Hunyuan3D flow sampler that Omni uses (guidance 4.5, 50 steps) against upstream's own <c>Sampler.sample_ode(euler)</c> with the <c>denoise_with_cfg</c> of <c>pipeline_generation_sit_omni.py</c>
/// (<c>tests/python-reference/dump_omni_flow.py</c>, torchdiffeq's fixed-grid Euler transcribed), with the same toy velocity in place of the DiT: the time grid, the timestep handed to the model, the guidance
/// and the Euler update.</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding <c>omni_flow_fp32.npz</c>. No weights.</remarks>
[Trait("Category", "Integration")]
public sealed class OmniFlowParityTests(ITestOutputHelper output)
{
    private const string Fixture = "omni_flow_fp32.npz";

    [Fact]
    public void TheTimeGridTheGuidanceAndTheEulerUpdateMatchUpstreamsSampleOde()
    {
        string parityDir = Environment.GetEnvironmentVariable(ParityFixtures.ParityDirVariable) ?? string.Empty;
        string path = Path.Combine(parityDir, Fixture);
        if (!File.Exists(path))
        {
            string reason = $"{ParityFixtures.ParityDirVariable} does not hold {Fixture}";
            if (Environment.GetEnvironmentVariable(ParityFixtures.RequireWeightsVariable) == "1")
                throw new InvalidOperationException($"{ParityFixtures.RequireWeightsVariable}=1 but {reason}.");
            output.WriteLine($"SKIPPED: {reason}");
            return;
        }

        NpzArchive fixture = NpzArchive.Open(path);
        float[] grid = fixture.Get("t").ToFloat32(), expected = fixture.Get("final").ToFloat32();
        float guidance = fixture.Get("guidance").ToFloat32()[0];
        NpzArray initArray = fixture.Get("init");

        // upstream integrates over linspace(0, 1, 50) and stops at the last point; this repository's schedule is that grid with a trailing 1 (a last step with no width).
        float[] sigmas = Hunyuan3DFlowSchedule.Sigmas(grid.Length);
        Assert.Equal(grid.Length + 1, sigmas.Length);
        double gridError = grid.Zip(sigmas, (a, b) => Math.Abs(a - b)).Max();
        Assert.Equal(1f, sigmas[^1]);
        output.WriteLine($"time grid: largest difference {gridError:E1} over {grid.Length} points");
        Assert.True(gridError < 1e-7, $"the time grid differs by {gridError}");

        List<float> given = [];
        using CpuBackend backend = new();
        Tensor latents = new(new TensorShape(initArray.Shape), DType.F32);
        initArray.ToFloat32().CopyTo(latents.AsSpan<float>());

        // The toy velocity of the oracle: (0.3 c - 0.5 t) x + 0.1 c t^2, c = 1 for the conditional branch and 0.3 for the unconditional one.
        Tensor Velocity(Tensor x, float t, float c)
        {
            Tensor v = new(x.Shape, DType.F32);
            ReadOnlySpan<float> input = x.AsSpan<float>();
            Span<float> values = v.AsSpan<float>();
            for (int i = 0; i < input.Length; i++)
                values[i] = (0.3f * c - 0.5f * t) * input[i] + 0.1f * c * t * t;
            return v;
        }

        int evaluations;
        using (Tensor result = ShapeIntegrate(backend, latents, sigmas, (x, t) => { given.Add(t); return Velocity(x, t, 1f); }, (x, t) => Velocity(x, t, 0.3f), guidance, out evaluations))
        {
            Assert.Equal(grid.Length - 1, evaluations);
            double timestepError = given.Zip(grid, (a, b) => Math.Abs(a - b)).Max();
            Assert.Equal(expected.Length, result.ElementCount);
            (double relative, double worst) = ParityCompare.RelativeL2(result.AsSpan<float>(), expected);
            output.WriteLine($"guidance {guidance}: timestep given to the model differs by at most {timestepError:E1}; final latents largest absolute difference {worst:E2}, relative L2 {relative:E2}");
            Assert.True(timestepError < 1e-7, $"timesteps differ by {timestepError}");
            Assert.True(relative < 1e-6, $"relative L2 {relative}");
        }
    }

    // Hunyuan3DShapePipeline.Integrate is internal; the call is wrapped so the evaluation count can come back.
    private static Tensor ShapeIntegrate(CpuBackend backend, Tensor latents, float[] sigmas, Func<Tensor, float, Tensor> conditional, Func<Tensor, float, Tensor> unconditional, float guidance, out int evaluations)
    {
        evaluations = Hunyuan3DShapePipeline.Integrate(backend, latents, sigmas, conditional, unconditional, guidance);
        return latents;
    }
}
