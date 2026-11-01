using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The engine's corrections against the reference adapter's own <c>_correct</c>, on the fixed tensors the ComfyUI dump wrote. Needs
/// <c>HARTSY_ANIMA_REF_DUMP</c> (the output of <c>tests/python-reference/dump_anima_ref_adapter.py</c>); skips without it. Runs on the CPU.</summary>
public sealed class ReferenceCorrectionFixtureTests
{
    private const float Phi = 0.5f;
    private const float Tolerance = 1e-4f;

    private readonly ITestOutputHelper _output;

    public ReferenceCorrectionFixtureTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Corrections_MatchTheReferenceAdapter()
    {
        string? dumpPath = Environment.GetEnvironmentVariable("HARTSY_ANIMA_REF_DUMP");
        if (dumpPath is null)
        {
            _output.WriteLine("SKIPPED: set HARTSY_ANIMA_REF_DUMP to the dump_anima_ref_adapter.py output");
            return;
        }

        using SafeTensorsLoader dump = new();
        dump.Load(dumpPath);
        using CpuBackend cpu = new();
        Tensor guided = dump.GetTensor("corr.guided");
        Tensor positive = dump.GetTensor("corr.pos");
        using Tensor output = new(guided.Shape, DType.F32);

        foreach ((ReferenceCorrection correction, string key) in new[]
        {
            (ReferenceCorrection.None, "corr.none"),
            (ReferenceCorrection.Apg, "corr.apg"),
            (ReferenceCorrection.StdRescale, "corr.rescale"),
        })
        {
            ReferenceGuidance.Correct(cpu, correction, Phi, guided, positive, output);
            Assert.True(MaxRelativeError(dump.GetTensor(key), output, out float worst) <= Tolerance,
                $"{correction}: relative error {worst} exceeds {Tolerance}");
            _output.WriteLine($"{correction}: max relative error {worst:G6}");
        }
    }

    private static float MaxRelativeError(Tensor expected, Tensor actual, out float worst)
    {
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        worst = 0f;
        for (int i = 0; i < e.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(e[i] - a[i]) / MathF.Max(1f, MathF.Abs(e[i])));
        }
        return worst;
    }
}
