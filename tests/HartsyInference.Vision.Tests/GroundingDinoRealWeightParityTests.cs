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

    [Trait("Category", "Integration")]
    [Fact]
    public void SwinBackbone_MatchesReference()
    {
        if (!TryPaths(out string ckpt, out string refDir))
            return;

        using IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        GroundingDinoConfig cfg = GroundingDinoConfig.Tiny;
        using SwinBackbone swin = new(cfg);
        swin.LoadWeights(w);

        using Tensor pixels = LoadF32Tensor(Path.Combine(refDir, "input_pixel_values.bin"), new TensorShape(1, 3, 800, 1066));
        Tensor[] feats = swin.Forward(backend, pixels);
        try
        {
            for (int i = 0; i < feats.Length; i++)
            {
                float[] reference = LoadF32(Path.Combine(refDir, $"backbone_feat_{i}.bin"));
                (double corr, double maxAbs) = Compare(feats[i], reference);
                _out.WriteLine($"backbone_feat_{i} {feats[i].Shape}: corr={corr:F6} maxAbs={maxAbs:E3} (n={reference.Length})");
                Assert.True(corr > 0.99, $"backbone_feat_{i} correlation too low: {corr:F6}");
            }
        }
        finally
        {
            foreach (Tensor t in feats) t.Dispose();
        }
    }

    [Trait("Category", "Integration")]
    [Fact]
    public void Neck_MatchesReference()
    {
        if (!TryPaths(out string ckpt, out string refDir))
            return;

        using IBackend backend = new CpuBackend();
        using SafeTensorsLoader loader = new();
        loader.Load(ckpt);
        System.Collections.Generic.Dictionary<string, Tensor> w = loader.GetAllTensors();

        GroundingDinoConfig cfg = GroundingDinoConfig.Tiny;
        using GroundingDinoNeck neck = new(cfg);
        neck.LoadWeights(w);

        Tensor[] feats =
        [
            LoadF32Tensor(Path.Combine(refDir, "backbone_feat_0.bin"), new TensorShape(1, 192, 100, 134)),
            LoadF32Tensor(Path.Combine(refDir, "backbone_feat_1.bin"), new TensorShape(1, 384, 50, 67)),
            LoadF32Tensor(Path.Combine(refDir, "backbone_feat_2.bin"), new TensorShape(1, 768, 25, 34)),
        ];
        GroundingDinoNeck.Result r = neck.Forward(backend, feats);
        foreach (Tensor t in feats) t.Dispose();

        float[] refSource = LoadF32(Path.Combine(refDir, "enc_in_vision.bin"));
        float[] refPos = LoadF32(Path.Combine(refDir, "enc_in_vpos.bin"));
        (double cS, double mS) = Compare(r.SourceFlatten, refSource);
        (double cP, double mP) = Compare(r.PositionFlatten, refPos);
        _out.WriteLine($"source_flatten: corr={cS:F6} maxAbs={mS:E3} (tokens={r.TotalTokens})");
        _out.WriteLine($"lvl_pos_embed:  corr={cP:F6} maxAbs={mP:E3}");
        r.SourceFlatten.Dispose();
        r.PositionFlatten.Dispose();
        Assert.True(cS > 0.999, $"source_flatten corr too low: {cS:F6}");
        Assert.True(cP > 0.999, $"lvl_pos_embed corr too low: {cP:F6}");
    }

    private static unsafe Tensor LoadF32Tensor(string path, TensorShape shape)
    {
        float[] data = LoadF32(path);
        Tensor t = new(shape, DType.F32);
        if (data.Length != t.ElementCount)
            throw new InvalidOperationException($"size mismatch loading {path}: {data.Length} vs {t.ElementCount}");
        fixed (float* src = data)
            Buffer.MemoryCopy(src, (void*)t.DataPointer, (long)data.Length * 4, (long)data.Length * 4);
        return t;
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
