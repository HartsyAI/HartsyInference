using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>The X-Part flow loop against upstream's own <c>FlowMatchEulerDiscreteScheduler</c> and denoising loop (<c>tests/python-reference/dump_xpart_flow.py</c>), with the same toy velocity in place of the DiT and the same
/// noise: the schedule, the timestep handed to the model and the Euler update, for 50 steps (the default) and 7.</summary>
/// <remarks>Integration lane: needs <c>HARTSY_PARITY_DIR</c> holding <c>xpart_flow_fp32.npz</c>. No weights.</remarks>
[Trait("Category", "Integration")]
public sealed class XPartFlowParityTests(ITestOutputHelper output)
{
    private const string Fixture = "xpart_flow_fp32.npz";

    [Theory]
    [InlineData(50)]
    [InlineData(7)]
    public void SchedulesTimestepsAndTheEulerUpdateMatchTheScheduler(int steps)
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
        NpzArray noiseArray = fixture.Get("noise");
        float[] referenceSigmas = fixture.Get($"sigmas{steps}").ToFloat32(), referenceTimesteps = fixture.Get($"timesteps{steps}").ToFloat32(), expected = fixture.Get($"latents{steps}").ToFloat32();

        float[] sigmas = Hunyuan3DFlowSchedule.Sigmas(steps);
        Assert.Equal(referenceSigmas.Length, sigmas.Length);
        double sigmaError = sigmas.Zip(referenceSigmas, (a, b) => Math.Abs(a - b)).Max();
        output.WriteLine($"{steps} steps: sigmas differ by at most {sigmaError:E1}");
        Assert.True(sigmaError < 1e-7, $"sigmas differ by {sigmaError}");

        List<float> given = [];
        using CpuBackend backend = new();
        Tensor latents = new(new TensorShape(noiseArray.Shape), DType.F32);
        noiseArray.ToFloat32().CopyTo(latents.AsSpan<float>());
        using Tensor result = XPartPipeline.Integrate(backend, latents, sigmas, (x, t) =>
        {
            given.Add(t);
            Tensor v = new(x.Shape, DType.F32);
            ReadOnlySpan<float> input = x.AsSpan<float>();
            Span<float> output = v.AsSpan<float>();
            for (int i = 0; i < input.Length; i++)
                output[i] = -0.5f * input[i] + 0.3f * MathF.Sin(3f * t) + 0.1f * MathF.Cos(7f * input[i]);
            return v;
        });

        // The scheduler runs a model forward at every timestep, including the last one whose step has no width; the loop skips that forward.
        Assert.Equal(referenceTimesteps.Length - 1, given.Count);
        double timestepError = given.Zip(referenceTimesteps, (a, b) => Math.Abs(a - b)).Max();
        double worst = 0, sumSquares = 0, sumReference = 0;
        ReadOnlySpan<float> actual = result.AsSpan<float>();
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs(actual[i] - expected[i]);
            worst = Math.Max(worst, d);
            sumSquares += d * d;
            sumReference += (double)expected[i] * expected[i];
        }

        double relative = Math.Sqrt(sumSquares / sumReference);
        output.WriteLine($"{steps} steps: timestep given to the model differs by at most {timestepError:E1}; final latents largest absolute difference {worst:E2}, relative L2 {relative:E2}");
        Assert.True(timestepError < 1e-6);
        Assert.True(relative < 1e-5, $"relative L2 {relative}");
    }
}
