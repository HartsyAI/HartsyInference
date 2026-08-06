using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.Annotators;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Real-weight parity for the UperNet-ConvNeXt-Small ADE20K segmentation annotator against the
/// transformers reference. <c>dump_upernet_seg.py</c> runs <c>UperNetForSemanticSegmentation</c>
/// (<c>openmmlab/upernet-convnext-small</c>) on a real image (bus.png, resized to 512×512) and dumps the
/// backbone features, head intermediates, 1/4-resolution logits, argmax class map and ADE20K-palette RGB
/// map; this test loads the same <c>pytorch_model.bin</c> via <see cref="PytorchPickleLoader"/>, runs
/// <see cref="UperNetSegModel"/> + <see cref="UperNetSegPreprocessor"/> on the CPU backend, and compares
/// stage-by-stage plus per-pixel class agreement (≥ 99%; the residual pixels sit on interpolation/argmax
/// tie boundaries where float noise flips the winning class). Skips cleanly when the checkpoint or dump
/// is absent.</summary>
public sealed unsafe class UperNetSegParityTests
{
    private readonly ITestOutputHelper _output;
    public UperNetSegParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void MatchesTransformersReference()
    {
        string checkpoint = TestPaths.Annotators.UperNetSeg;
        string refDir = Path.Combine(RepoRoot.Path, "tests", "python-reference", "upernet_seg_reference_tensors");
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: checkpoint not found: {checkpoint}");
            return;
        }
        if (!File.Exists(Path.Combine(refDir, "seg_u8.bin")))
        {
            _output.WriteLine($"SKIPPED: Python reference not found in {refDir} — run dump_upernet_seg.py first.");
            return;
        }

        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "meta.json")));
        int h = meta.RootElement.GetProperty("H").GetInt32();
        int w = meta.RootElement.GetProperty("W").GetInt32();

        Stopwatch sw = Stopwatch.StartNew();
        using PytorchPickleLoader loader = new();
        loader.Load(checkpoint);
        UperNetSegModel model = new();
        model.LoadWeights(loader.GetAllTensors());
        _output.WriteLine($"[load] {sw.ElapsedMilliseconds} ms");

        byte[] image = File.ReadAllBytes(Path.Combine(refDir, "image.bin"));
        Assert.Equal((long)h * w * 3, image.Length);

        using Tensor input = UperNetSegPreprocessor.Preprocess(image, w, h);
        float[] refInput = AnnotatorParityIo.ReadF32(Path.Combine(refDir, "input.bin"), input.ElementCount);
        (double preAvg, _, _) = AnnotatorParityIo.Compare(input.AsReadOnlySpan<float>(), refInput);
        _output.WriteLine($"input: avg err {preAvg:E3}");
        Assert.True(preAvg < 1e-6, $"Preprocess diverges: avg err {preAvg:E3}.");

        List<(string Name, double AvgErr, double RefMeanAbs)> stages = [];
        Action<string, Tensor> tap = (name, t) =>
        {
            string path = Path.Combine(refDir, name + ".bin");
            if (!File.Exists(path)) return;
            float[] reference = AnnotatorParityIo.ReadF32(path, t.ElementCount);
            (double avg, _, double refMean) = AnnotatorParityIo.Compare(t.AsReadOnlySpan<float>(), reference);
            stages.Add((name, avg, refMean));
        };

        using IBackend backend = new CpuBackend();
        sw.Restart();
        Tensor logits = model.Forward(backend, input, tap);
        _output.WriteLine($"[forward] {sw.ElapsedMilliseconds} ms — output {logits.Shape}");
        Assert.Equal(new TensorShape(1, UperNetSegModel.NumClasses, h, w), logits.Shape);

        foreach ((string name, double avgErr, double refMean) in stages)
            _output.WriteLine($"  {name,-9} avg err {avgErr:E3} (ref mean abs {refMean:E3})");
        foreach ((string name, double avgErr, double refMean) in stages)
            Assert.True(avgErr < Math.Max(1e-2 * refMean, 1e-3), $"Stage {name} diverges: avg err {avgErr:E3} vs ref mean abs {refMean:E3}.");

        byte[] classMap = UperNetSegPreprocessor.Argmax(logits);
        logits.Dispose();
        byte[] refClasses = File.ReadAllBytes(Path.Combine(refDir, "seg_u8.bin"));
        double mismatch = AnnotatorParityIo.MismatchFraction(classMap, refClasses);
        _output.WriteLine($"class map: agreement {1 - mismatch:P3}");
        Assert.True(mismatch < 0.01, $"Per-pixel class agreement {1 - mismatch:P3} < 99%.");

        // Exact palette mapping: colorizing the reference class map must reproduce the reference RGB
        // dump byte-for-byte.
        byte[] refRgb = File.ReadAllBytes(Path.Combine(refDir, "seg_rgb.bin"));
        byte[] palettized = Ade20kPalette.Colorize(refClasses);
        Assert.Equal(refRgb.Length, palettized.Length);
        Assert.True(refRgb.AsSpan().SequenceEqual(palettized), "ADE20K palette mapping diverges from controlnet_aux ade_palette().");

        // And the e2e RGB output only differs from the reference where the class disagreed.
        byte[] ourRgb = Ade20kPalette.Colorize(classMap);
        double rgbMismatch = AnnotatorParityIo.MismatchFraction(ourRgb, refRgb);
        _output.WriteLine($"seg_rgb: byte mismatch fraction {rgbMismatch:E3}");
        Assert.True(rgbMismatch <= mismatch * 3 + 1e-9, "RGB map mismatches exceed class-map mismatches — palette bug.");
    }
}
