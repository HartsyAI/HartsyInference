using System.Diagnostics;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Dinov2;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests.Dinov2;

/// <summary>Integration lane: the DINOv2-giant encoder against Hugging Face <c>transformers</c> on the CPU. Set <c>HARTSY_DINOV2_GIANT</c> (a Hunyuan3D shape checkpoint that bundles the
/// F16 weights, <c>conditioner.main_image_encoder.model.*</c>) and <c>HARTSY_DINOV2_GIANT_REF</c> (the fixture of <c>tests/python-reference/dump_dinov2_giant.py</c>).</summary>
[Trait("Category", "Integration")]
public sealed class Dinov2GiantReferenceTests(ITestOutputHelper output)
{
    private static unsafe float[] Values(Tensor t)
    {
        float[] values = new float[t.ElementCount];
        new ReadOnlySpan<float>((float*)t.DataPointer, values.Length).CopyTo(values);
        return values;
    }

    private void Compare(string stage, float[] actual, float[] expected, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double max = 0, diff2 = 0, ref2 = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = actual[i], b = expected[i], d = a - b;
            max = Math.Max(max, Math.Abs(d));
            diff2 += d * d;
            ref2 += b * b;
            sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
        }

        double n = expected.Length, rel = Math.Sqrt(diff2 / ref2);
        double corr = (sab / n - sa / n * (sb / n)) / Math.Sqrt((saa / n - sa / n * (sa / n)) * (sbb / n - sb / n * (sb / n)));
        output.WriteLine($"{stage,-20} n={expected.Length,9}  max|diff|={max:E3}  relRMS={rel:E3}  corr={corr:F6}");
        Assert.True(rel < tolerance && corr > 0.9999, $"{stage}: relRMS {rel:E3}, correlation {corr:F6}");
    }

    [RealWeightsFact("HARTSY_DINOV2_GIANT", "HARTSY_DINOV2_GIANT_REF")]
    public void Giant_MatchesTransformersOnTheCpu()
    {
        using SafeTensorsLoader loader = new(), refLoader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_DINOV2_GIANT")!);
        refLoader.Load(Environment.GetEnvironmentVariable("HARTSY_DINOV2_GIANT_REF")!);
        Dictionary<string, Tensor> fixture = refLoader.GetAllTensors();
        Dinov2VisionEncoder encoder = new(Dinov2Preset.Giant);
        Stopwatch watch = Stopwatch.StartNew();
        encoder.LoadWeights(loader.GetAllTensors(), "conditioner.main_image_encoder.model");
        output.WriteLine($"loaded in {watch.Elapsed.TotalSeconds:F1} s");
        using CpuBackend backend = new();
        watch.Restart();
        using Tensor normed = encoder.Encode(backend, fixture["pixels"]);
        output.WriteLine($"forward with the final norm: {watch.Elapsed.TotalSeconds:F1} s");
        Compare("last_hidden_state", Values(normed), Values(fixture["last_hidden_state"]), 1e-4);
        using Tensor prenorm = encoder.Encode(backend, fixture["pixels"], applyFinalNorm: false);
        Compare("prenorm (hidden[-1])", Values(prenorm), Values(fixture["prenorm"]), 1e-4);
    }
}
