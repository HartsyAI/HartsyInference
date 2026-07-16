using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.PyTorch;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.Annotators;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Real-weight parity for the lineart Generator against controlnet_aux's <c>LineartDetector</c>.
/// <c>dump_lineart.py</c> runs the official Generator on a fixed deterministic image and dumps every stage
/// plus the e2e inverted uint8 conditioning map; this test loads the same checkpoint via
/// <see cref="PytorchPickleLoader"/>, runs <see cref="LineartGenerator"/> on the CPU backend, and compares
/// stage-by-stage. Skips cleanly when the checkpoint or dump is absent.</summary>
public sealed unsafe class LineartParityTests
{
    private readonly ITestOutputHelper _output;
    public LineartParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void Realistic_MatchesControlnetAuxReference() =>
        RunParity("realistic", LineartPreset.Realistic, TestPaths.Annotators.LineartRealistic);

    [Fact]
    [Trait("Category", "Integration")]
    public void Coarse_MatchesControlnetAuxReference() =>
        RunParity("coarse", LineartPreset.Coarse, TestPaths.Annotators.LineartCoarse);

    private void RunParity(string variant, LineartPreset preset, string checkpoint)
    {
        string refDir = Path.Combine(RepoRoot.Path, "tests", "python-reference", "lineart_reference_tensors", variant);
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: checkpoint not found: {checkpoint}");
            return;
        }
        if (!File.Exists(Path.Combine(refDir, "line.bin")))
        {
            _output.WriteLine($"SKIPPED: Python reference not found in {refDir} — run dump_lineart.py first.");
            return;
        }

        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "meta.json")));
        int h = meta.RootElement.GetProperty("H").GetInt32();
        int w = meta.RootElement.GetProperty("W").GetInt32();

        Stopwatch sw = Stopwatch.StartNew();
        using PytorchPickleLoader loader = new();
        loader.Load(checkpoint);
        LineartGenerator model = new(preset);
        model.LoadWeights(loader.GetAllTensors());
        _output.WriteLine($"[load] {sw.ElapsedMilliseconds} ms");

        byte[] image = File.ReadAllBytes(Path.Combine(refDir, "image.bin"));
        Assert.Equal((long)h * w * 3, image.Length);

        using Tensor input = LineartPreprocessor.Preprocess(image, w, h);
        float[] refInput = AnnotatorParityIo.ReadF32(Path.Combine(refDir, "input.bin"), input.ElementCount);
        (double preAvg, _, _) = AnnotatorParityIo.Compare(input.AsReadOnlySpan<float>(), refInput);
        _output.WriteLine($"input: avg err {preAvg:E3}");
        Assert.True(preAvg < 1e-6, $"Preprocess diverges: avg err {preAvg:E3}.");

        string dumpDir = Path.Combine(RepoRoot.Path, "Output", "lineart_csharp_dump", variant);
        Directory.CreateDirectory(dumpDir);
        List<(string Name, double AvgErr)> stages = [];
        Action<string, Tensor> tap = (name, t) =>
        {
            AnnotatorParityIo.WriteF32(Path.Combine(dumpDir, name + ".bin"), t);
            string path = Path.Combine(refDir, name + ".bin");
            if (!File.Exists(path)) return;
            float[] reference = AnnotatorParityIo.ReadF32(path, t.ElementCount);
            (double avg, _, _) = AnnotatorParityIo.Compare(t.AsReadOnlySpan<float>(), reference);
            stages.Add((name, avg));
        };

        using IBackend backend = new CpuBackend();
        sw.Restart();
        Tensor line = model.Forward(backend, input, tap);
        _output.WriteLine($"[forward] {sw.ElapsedMilliseconds} ms — output {line.Shape}");
        AnnotatorParityIo.WriteF32(Path.Combine(dumpDir, "line.bin"), line);

        foreach ((string name, double avgErr) in stages)
            _output.WriteLine($"  {name,-4} avg err {avgErr:E3}");

        Assert.Equal(new TensorShape(1, 1, h, w), line.Shape);
        float[] refLine = AnnotatorParityIo.ReadF32(Path.Combine(refDir, "line.bin"), line.ElementCount);
        (double avg, double max, _) = AnnotatorParityIo.Compare(line.AsReadOnlySpan<float>(), refLine);
        _output.WriteLine($"line: avg err {avg:E3}, max err {max:E3}");
        Assert.True(avg < 1e-3, $"Line diverges: avg err {avg:E3} >= 1e-3.");
        foreach ((string name, double avgErr) in stages)
            Assert.True(avgErr < 1e-2, $"Stage {name} diverges: avg err {avgErr:E3}.");

        // e2e inverted uint8 conditioning map (255 - truncated quantization).
        byte[] cond = AnnotatorParityIo.UnitToBytes(LineartPreprocessor.PostprocessToUnit(line));
        byte[] refCond = File.ReadAllBytes(Path.Combine(refDir, "cond_u8.bin"));
        double condMismatch = AnnotatorParityIo.MismatchFraction(cond, refCond);
        _output.WriteLine($"cond_u8: mismatch fraction {condMismatch:E3}");
        Assert.True(condMismatch < 1e-3, $"cond_u8 mismatch fraction {condMismatch:E3} >= 1e-3.");
        line.Dispose();
    }
}
