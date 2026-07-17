using System;
using System.IO;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Detection.GroundingDino;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Real-weight numerical parity for Grounding DINO (<c>IDEA-Research/grounding-dino-tiny</c>) against a
/// Hugging Face <c>GroundingDinoForObjectDetection</c> oracle. Gated on <c>GROUNDING_DINO_PATH</c> (the
/// <c>model.safetensors</c>) and <c>GROUNDING_DINO_REF</c> (raw-binary boundary dumps from <c>oracle.py</c>).
/// Validates each ported stage against the corresponding dumped intermediate.</summary>
public sealed class GroundingDinoRealWeightParityTests
{
    private readonly ITestOutputHelper _out;

    public GroundingDinoRealWeightParityTests(ITestOutputHelper output) => _out = output;

    [Trait("Category", "Integration")]
    [Fact]
    public void TextTower_MatchesReference()
    {
        if (!TryPaths(out string ckpt, out string refDir))
            return;

        using IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        int[] ids = LoadI64(Path.Combine(refDir, "input_input_ids.i64"));
        GroundingDinoConfig cfg = GroundingDinoConfig.Tiny;
        using GroundingDinoTextTower tower = new(cfg);
        tower.LoadWeights(w, "model");

        using Tensor got = tower.Encode(backend, ids);
        float[] reference = LoadF32(Path.Combine(refDir, "text_features.bin"));
        (double corr, double maxAbs) = Compare(got, reference);
        _out.WriteLine($"text_features: corr={corr:F6} maxAbs={maxAbs:E3} (n={reference.Length}, T={ids.Length})");
        Assert.True(corr > 0.999, $"text-tower correlation too low: {corr:F6}");
        Assert.True(maxAbs < 5e-3, $"text-tower max-abs diff too high: {maxAbs:E3}");
    }

    private static bool TryPaths(out string ckpt, out string refDir)
    {
        ckpt = Environment.GetEnvironmentVariable("GROUNDING_DINO_PATH") ?? "";
        refDir = Environment.GetEnvironmentVariable("GROUNDING_DINO_REF") ?? "";
        return !string.IsNullOrEmpty(ckpt) && File.Exists(ckpt)
            && !string.IsNullOrEmpty(refDir) && Directory.Exists(refDir);
    }

    private static int[] LoadI64(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int[] outv = new int[bytes.Length / 8];
        for (int i = 0; i < outv.Length; i++)
            outv[i] = (int)BitConverter.ToInt64(bytes, i * 8);
        return outv;
    }

    private static float[] LoadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] outv = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, outv, 0, bytes.Length);
        return outv;
    }

    private static unsafe (double corr, double maxAbs) Compare(Tensor got, float[] reference)
    {
        long n = got.ElementCount;
        if (n != reference.Length)
            throw new InvalidOperationException($"element count mismatch: got {n} vs ref {reference.Length}");
        float* g = (float*)got.DataPointer;
        double sumG = 0, sumR = 0, sumGG = 0, sumRR = 0, sumGR = 0, maxAbs = 0;
        for (long i = 0; i < n; i++)
        {
            double a = g[i], b = reference[i];
            sumG += a; sumR += b; sumGG += a * a; sumRR += b * b; sumGR += a * b;
            double d = Math.Abs(a - b);
            if (d > maxAbs) maxAbs = d;
        }
        double covar = sumGR - sumG * sumR / n;
        double varG = sumGG - sumG * sumG / n;
        double varR = sumRR - sumR * sumR / n;
        double corr = covar / Math.Sqrt(varG * varR + 1e-12);
        return (corr, maxAbs);
    }
}
