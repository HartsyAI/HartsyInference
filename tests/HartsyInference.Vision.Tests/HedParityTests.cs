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

/// <summary>Real-weight parity for the ControlNetHED soft-edge annotator against controlnet_aux.
/// <c>dump_hed.py</c> runs the official <c>ControlNetHED_Apache2</c> + <c>HEDdetector</c> on a fixed
/// deterministic image and dumps the 5 side logits, the fused edge map, and the e2e uint8 outputs
/// (softedge / safe / scribble); this test loads the same <c>ControlNetHED.pth</c> via
/// <see cref="PytorchPickleLoader"/>, runs <see cref="HedModel"/> + <see cref="HedPreprocessor"/> on the
/// CPU backend, and compares stage-by-stage. Skips cleanly when the checkpoint or dump is absent.</summary>
public sealed unsafe class HedParityTests
{
    private readonly ITestOutputHelper _output;
    public HedParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void MatchesControlnetAuxReference()
    {
        string checkpoint = TestPaths.Annotators.Hed;
        string refDir = Path.Combine(RepoRoot.Path, "tests", "python-reference", "hed_reference_tensors");
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: checkpoint not found: {checkpoint}");
            return;
        }
        if (!File.Exists(Path.Combine(refDir, "edge.bin")))
        {
            _output.WriteLine($"SKIPPED: Python reference not found in {refDir} — run dump_hed.py first.");
            return;
        }

        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "meta.json")));
        int h = meta.RootElement.GetProperty("H").GetInt32();
        int w = meta.RootElement.GetProperty("W").GetInt32();

        Stopwatch sw = Stopwatch.StartNew();
        using PytorchPickleLoader loader = new();
        loader.Load(checkpoint);
        HedModel model = new();
        model.LoadWeights(loader.GetAllTensors());
        _output.WriteLine($"[load] {sw.ElapsedMilliseconds} ms");

        byte[] image = File.ReadAllBytes(Path.Combine(refDir, "image.bin"));
        Assert.Equal((long)h * w * 3, image.Length);

        using Tensor input = HedPreprocessor.Preprocess(image, w, h);
        float[] refInput = AnnotatorParityIo.ReadF32(Path.Combine(refDir, "input.bin"), input.ElementCount);
        (double preAvg, _, _) = AnnotatorParityIo.Compare(input.AsReadOnlySpan<float>(), refInput);
        _output.WriteLine($"input: avg err {preAvg:E3}");
        Assert.True(preAvg < 1e-6, $"Preprocess diverges: avg err {preAvg:E3}.");

        string dumpDir = Path.Combine(RepoRoot.Path, "Output", "hed_csharp_dump");
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
        Tensor edge = model.Forward(backend, input, tap);
        _output.WriteLine($"[forward] {sw.ElapsedMilliseconds} ms — output {edge.Shape}");
        AnnotatorParityIo.WriteF32(Path.Combine(dumpDir, "edge.bin"), edge);

        foreach ((string name, double avgErr) in stages)
            _output.WriteLine($"  {name,-8} avg err {avgErr:E3}");

        Assert.Equal(new TensorShape(1, 1, h, w), edge.Shape);
        float[] refEdge = AnnotatorParityIo.ReadF32(Path.Combine(refDir, "edge.bin"), edge.ElementCount);
        (double avg, double max, _) = AnnotatorParityIo.Compare(edge.AsReadOnlySpan<float>(), refEdge);
        _output.WriteLine($"edge: avg err {avg:E3}, max err {max:E3}");
        Assert.True(avg < 1e-3, $"Fused edge diverges: avg err {avg:E3} >= 1e-3.");
        foreach ((string name, double avgErr) in stages)
            Assert.True(avgErr < 1e-2, $"Stage {name} diverges: avg err {avgErr:E3}.");

        // e2e uint8 forms: softedge, safe, scribble — compare code-for-code with a small allowance for
        // float noise on quantization boundaries.
        byte[] soft = AnnotatorParityIo.UnitToBytes(HedPreprocessor.PostprocessToUnit(edge));
        byte[] refSoft = File.ReadAllBytes(Path.Combine(refDir, "softedge_u8.bin"));
        double softMismatch = AnnotatorParityIo.MismatchFraction(soft, refSoft);
        _output.WriteLine($"softedge_u8: mismatch fraction {softMismatch:E3}");
        Assert.True(softMismatch < 1e-3, $"softedge_u8 mismatch fraction {softMismatch:E3} >= 1e-3.");

        byte[] safe = AnnotatorParityIo.UnitToBytes(HedPreprocessor.PostprocessToUnit(edge, safe: true));
        byte[] refSafe = File.ReadAllBytes(Path.Combine(refDir, "safe_u8.bin"));
        double safeMismatch = AnnotatorParityIo.MismatchFraction(safe, refSafe);
        _output.WriteLine($"safe_u8: mismatch fraction {safeMismatch:E3}");
        Assert.True(safeMismatch < 1e-3, $"safe_u8 mismatch fraction {safeMismatch:E3} >= 1e-3.");

        byte[] scribble = HedPreprocessor.ScribbleFromSoftEdge(soft, w, h);
        byte[] refScribble = File.ReadAllBytes(Path.Combine(refDir, "scribble_u8.bin"));
        double scribbleMismatch = AnnotatorParityIo.MismatchFraction(scribble, refScribble);
        _output.WriteLine($"scribble_u8: mismatch fraction {scribbleMismatch:E3}");
        Assert.True(scribbleMismatch < 1e-2, $"scribble_u8 mismatch fraction {scribbleMismatch:E3} >= 1e-2.");
        edge.Dispose();
    }
}
