using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.DepthAnything;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Real-image A/B of the full Flux-Depth conditioning map path (preprocess → DA-V2 ViT-L →
/// resize-to-gen-res + scaling) against the official Python implementation on the SAME photo.
/// <c>tests/python-reference/dump_flux_depth_ab.py</c> dumps the official transform output, the raw
/// network-res depth, the <c>infer_image</c>-style source-res map, and the BFL-style gen-res map; this
/// test replays the engine path (<see cref="DepthAnythingPreprocessor"/> + <see cref="DepthAnythingV2Model"/>)
/// and compares each stage. Engine maps are also dumped for the edge-band fringe analysis script
/// (<c>compare_flux_depth_ab.py</c>). Skips cleanly when the checkpoint or the Python dump is absent.</summary>
public sealed unsafe class FluxDepthAbTests
{
    private readonly ITestOutputHelper _output;
    public FluxDepthAbTests(ITestOutputHelper output) => _output = output;

    private static string ReferenceDir =>
        Path.Combine(RepoRoot.Path, "tests", "python-reference", "flux_depth_reference_tensors");

    [Fact]
    [Trait("Category", "Integration")]
    public void FluxDepthMap_MatchesOfficialReference_OnRealImage()
    {
        string checkpoint = TestPaths.DepthAnything.VitL;
        string metaPath = Path.Combine(ReferenceDir, "meta.json");
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: checkpoint not found: {checkpoint}");
            return;
        }
        if (!File.Exists(metaPath))
        {
            _output.WriteLine($"SKIPPED: Python reference not found in {ReferenceDir} — run dump_flux_depth_ab.py first.");
            return;
        }

        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(metaPath));
        string imagePath = meta.RootElement.GetProperty("image").GetString()!;
        int srcW = meta.RootElement.GetProperty("src_w").GetInt32();
        int srcH = meta.RootElement.GetProperty("src_h").GetInt32();
        int netW = meta.RootElement.GetProperty("net_w").GetInt32();
        int netH = meta.RootElement.GetProperty("net_h").GetInt32();
        int genW = meta.RootElement.GetProperty("gen_w").GetInt32();
        int genH = meta.RootElement.GetProperty("gen_h").GetInt32();
        if (!File.Exists(imagePath))
        {
            _output.WriteLine($"SKIPPED: source image not found: {imagePath}");
            return;
        }

        (byte[] rgb, int w, int h) = PngDecoder.DecodeFromFile(imagePath);
        Assert.Equal((srcW, srcH), (w, h));

        DepthAnythingPreprocessor pre = new();
        (int tw, int th) = pre.ComputeTargetSize(srcW, srcH);
        _output.WriteLine($"target size: C# {tw}x{th}, python {netW}x{netH}");
        Assert.Equal((netW, netH), (tw, th));

        using Tensor pixels = pre.Preprocess(rgb, srcW, srcH);
        (double preCorr, double preAvg, double preMax) = CompareFile(pixels.AsReadOnlySpan<float>(),
            Path.Combine(ReferenceDir, "net_input.bin"));
        _output.WriteLine($"preprocess: corr {preCorr:F6}, avg abs {preAvg:E3}, max abs {preMax:E3}");

        using PytorchPickleLoader loader = new();
        loader.Load(checkpoint);
        DepthAnythingV2Model model = new(DepthAnythingPreset.Large);
        model.LoadWeights(loader.GetAllTensors());

        using IBackend backend = new CpuBackend();
        Tensor depth = model.Forward(backend, pixels);
        Assert.Equal(new TensorShape(1, 1, netH, netW), depth.Shape);
        (double netCorr, double netAvg, double netMax) = CompareFile(depth.AsReadOnlySpan<float>(),
            Path.Combine(ReferenceDir, "net_depth.bin"));
        _output.WriteLine($"net-res depth: corr {netCorr:F6}, avg abs {netAvg:E3}, max abs {netMax:E3}");

        string dumpDir = Path.Combine(TestPaths.OutputDir, "fluxdepth_redux_ab", "depth_csharp");
        Directory.CreateDirectory(dumpDir);
        WriteSpan(Path.Combine(dumpDir, "engine_net_depth.bin"), depth.AsReadOnlySpan<float>());

        // Source-res map, SD-ControlNet convention (min-max stretch) — infer_image + NormalizeToUnit oracle.
        float[] srcUnit = DepthAnythingPreprocessor.PostprocessToUnit(depth, srcW, srcH, minMaxNormalize: true);
        float[] refSrc = ReadF32(Path.Combine(ReferenceDir, "src_depth.bin"), (long)srcW * srcH);
        DepthAnythingPreprocessor.NormalizeToUnit(refSrc);
        (double srcCorr, double srcAvg, double srcMax) = Compare(srcUnit, refSrc);
        _output.WriteLine($"src-res unit map: corr {srcCorr:F6}, avg abs {srcAvg:E3}, max abs {srcMax:E3}");
        WriteSpan(Path.Combine(dumpDir, "engine_src_unit.bin"), srcUnit);

        // Gen-res map, FLUX scaling convention (max-only, BFL bicubic-antialias kernel) — the exact tensor
        // the Swarm extension feeds the Flux-Depth VAE encode (before the [0,1] → [-1,1] shift).
        float[] genUnit = DepthAnythingPreprocessor.PostprocessToUnit(depth, genW, genH, minMaxNormalize: false);
        float[] refGenBilinear = ReadF32(Path.Combine(ReferenceDir, "gen_bilinear.bin"), (long)genW * genH);
        float[] refGenBfl = ReadF32(Path.Combine(ReferenceDir, "gen_bfl.bin"), (long)genW * genH);
        (double genCorr, double genAvg, double genMax) = Compare(genUnit, refGenBilinear);
        (double bflCorr, double bflAvg, double bflMax) = Compare(genUnit, refGenBfl);
        _output.WriteLine($"gen-res map vs bilinear ref (informational): corr {genCorr:F6}, avg abs {genAvg:E3}, max abs {genMax:E3}");
        _output.WriteLine($"gen-res map vs BFL bicubic-antialias ref: corr {bflCorr:F6}, avg abs {bflAvg:E3}, max abs {bflMax:E3}");
        WriteSpan(Path.Combine(dumpDir, "engine_gen_flux.bin"), genUnit);
        depth.Dispose();

        Assert.True(preCorr > 0.999, $"Preprocess diverges on real image: corr {preCorr:F6}.");
        Assert.True(netCorr > 0.999, $"Net-res depth diverges: corr {netCorr:F6}.");
        Assert.True(srcCorr > 0.999, $"Source-res unit map diverges: corr {srcCorr:F6}.");
        Assert.True(bflCorr > 0.9999 && bflMax < 1e-3,
            $"Gen-res flux map diverges from the BFL bicubic-antialias reference: corr {bflCorr:F6}, max abs {bflMax:E3}.");
    }

    private static float[] ReadF32(string path, long expectedCount)
    {
        byte[] raw = File.ReadAllBytes(path);
        Assert.Equal(expectedCount * 4, raw.Length);
        float[] data = new float[expectedCount];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return data;
    }

    private static void WriteSpan(string path, ReadOnlySpan<float> data)
    {
        byte[] bytes = new byte[data.Length * 4];
        Buffer.BlockCopy(data.ToArray(), 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    private static (double Corr, double AvgAbs, double MaxAbs) CompareFile(ReadOnlySpan<float> actual, string refPath)
    {
        byte[] raw = File.ReadAllBytes(refPath);
        Assert.Equal((long)actual.Length * 4, raw.Length);
        float[] reference = new float[actual.Length];
        Buffer.BlockCopy(raw, 0, reference, 0, raw.Length);
        return Compare(actual, reference);
    }

    private static (double Corr, double AvgAbs, double MaxAbs) Compare(ReadOnlySpan<float> a, ReadOnlySpan<float> r)
    {
        Assert.Equal(r.Length, a.Length);
        double sumA = 0, sumR = 0, sumAA = 0, sumRR = 0, sumAR = 0, sumAbs = 0, maxAbs = 0;
        int n = a.Length;
        for (int i = 0; i < n; i++)
        {
            double d = Math.Abs(a[i] - r[i]);
            sumAbs += d;
            if (d > maxAbs) maxAbs = d;
            sumA += a[i]; sumR += r[i];
            sumAA += (double)a[i] * a[i]; sumRR += (double)r[i] * r[i]; sumAR += (double)a[i] * r[i];
        }
        double cov = sumAR / n - sumA / n * (sumR / n);
        double varA = sumAA / n - sumA / n * (sumA / n);
        double varR = sumRR / n - sumR / n * (sumR / n);
        return (cov / Math.Sqrt(Math.Max(varA * varR, 1e-30)), sumAbs / n, maxAbs);
    }
}
