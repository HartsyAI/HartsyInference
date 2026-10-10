using Xunit;
using Xunit.Abstractions;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// Cross-runtime validation of the Euler scheduler against Python reference values. Self-contained: needs no model
/// files, only the constants recorded below.
/// </summary>
public class CrossRuntimeValidationTests
{
    private readonly ITestOutputHelper _output;

    public CrossRuntimeValidationTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Validates that the C# scheduler produces identical sigma values, timesteps, and
    /// InitialNoiseSigma compared to the Python reference.
    /// </summary>
    [Fact]
    public void SchedulerMatchesPythonReference()
    {
        // Python reference values from reference_stats.json
        float[] expectedTimesteps =
        [
            950f, 900f, 850f, 800f, 750f, 700f, 650f, 600f, 550f, 500f,
            450f, 400f, 350f, 300f, 250f, 200f, 150f, 100f, 50f, 0f
        ];

        float[] expectedSigmas =
        [
            10.96606731f, 8.34659958f, 6.47457790f, 5.11103010f, 4.09916592f,
            3.33438993f, 2.74580836f, 2.28464103f, 1.91683412f, 1.61827970f,
            1.37166870f, 1.16439164f, 0.98710895f, 0.83275318f, 0.69579947f,
            0.57166636f, 0.45607531f, 0.34393182f, 0.22558255f, 0.02916753f,
            0.0f
        ];

        float expectedInitNoiseSigma = 11.01156807f;

        // Compute C# values
        EulerDiscreteScheduler scheduler = new();
        scheduler.SetTimesteps(20);
        ReadOnlySpan<float> timesteps = scheduler.Timesteps;
        float initSigma = scheduler.InitialNoiseSigma;

        // Validate timesteps
        _output.WriteLine("=== Timestep Comparison ===");
        Assert.Equal(expectedTimesteps.Length, timesteps.Length);
        for (int i = 0; i < expectedTimesteps.Length; i++)
        {
            float diff = MathF.Abs(timesteps[i] - expectedTimesteps[i]);
            _output.WriteLine($"  t[{i}]: C#={timesteps[i]:F4}, Python={expectedTimesteps[i]:F4}, diff={diff:E4}");
            Assert.True(diff < 0.01f, $"Timestep {i} diverged: C#={timesteps[i]}, Python={expectedTimesteps[i]}");
        }

        // Validate InitialNoiseSigma
        float sigmaDiff = MathF.Abs(initSigma - expectedInitNoiseSigma);
        _output.WriteLine($"\n=== InitialNoiseSigma ===");
        _output.WriteLine($"  C#={initSigma:F6}, Python={expectedInitNoiseSigma:F6}, diff={sigmaDiff:E4}");
        Assert.True(sigmaDiff < 0.001f, $"InitialNoiseSigma diverged: C#={initSigma}, Python={expectedInitNoiseSigma}");

        // Validate sigmas (requires accessing private field - use ScaleModelInput to infer)
        _output.WriteLine($"\n=== Sigma Validation (via ScaleModelInput) ===");
        for (int i = 0; i < 20; i++)
        {
            // ScaleModelInput returns 1/sqrt(sigma^2 + 1), so sigma = sqrt(1/scale^2 - 1)
            float scale = scheduler.ScaleModelInput(i);
            float inferredSigma = MathF.Sqrt(1.0f / (scale * scale) - 1.0f);
            float sigDiff = MathF.Abs(inferredSigma - expectedSigmas[i]);
            _output.WriteLine($"  sigma[{i}]: C#={inferredSigma:F6}, Python={expectedSigmas[i]:F6}, diff={sigDiff:E4}");
            Assert.True(sigDiff < 0.01f, $"Sigma {i} diverged: C#={inferredSigma}, Python={expectedSigmas[i]}");
        }

        _output.WriteLine("\nScheduler validation PASSED — all values match Python reference.");
    }
}
