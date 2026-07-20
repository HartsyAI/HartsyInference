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

/// <summary>Real-weight parity for the NormalBAE surface-normal annotator against controlnet_aux.
/// <c>dump_normalbae.py</c> runs the official NNET (EfficientNet-B5 + normal decoder) on a fixed
/// deterministic image and dumps encoder taps, decoder pyramid, per-scale normalized outputs and the e2e
/// uint8 conditioning image; this test loads the same <c>scannet.pt</c> via
/// <see cref="PytorchPickleLoader"/>, runs <see cref="NormalBaeModel"/> on the CPU backend, and compares
/// stage-by-stage. Skips cleanly when the checkpoint or dump is absent.</summary>
public sealed unsafe class NormalBaeParityTests
{
    private readonly ITestOutputHelper _output;
    public NormalBaeParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void MatchesControlnetAuxReference()
    {
        string checkpoint = TestPaths.Annotators.NormalBae;
        string refDir = Path.Combine(RepoRoot.Path, "tests", "python-reference", "normalbae_reference_tensors");
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: checkpoint not found: {checkpoint}");
            return;
        }
        if (!File.Exists(Path.Combine(refDir, "out_res1.bin")))
        {
            _output.WriteLine($"SKIPPED: Python reference not found in {refDir} — run dump_normalbae.py first.");
            return;
        }

        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "meta.json")));
        int h = meta.RootElement.GetProperty("H").GetInt32();
        int w = meta.RootElement.GetProperty("W").GetInt32();

        Stopwatch sw = Stopwatch.StartNew();
        using PytorchPickleLoader loader = new();
        loader.Load(checkpoint);
        NormalBaeModel model = new(NormalBaePreset.Default);
        model.LoadWeights(loader.GetAllTensors());
        _output.WriteLine($"[load] {sw.ElapsedMilliseconds} ms");

        byte[] image = File.ReadAllBytes(Path.Combine(refDir, "image.bin"));
        Assert.Equal((long)h * w * 3, image.Length);

        using Tensor input = NormalBaePreprocessor.Preprocess(image, w, h);
        float[] refInput = AnnotatorParityIo.ReadF32(Path.Combine(refDir, "input.bin"), input.ElementCount);
        (double preAvg, _, _) = AnnotatorParityIo.Compare(input.AsReadOnlySpan<float>(), refInput);
        _output.WriteLine($"input: avg err {preAvg:E3}");
        Assert.True(preAvg < 1e-6, $"Preprocess diverges: avg err {preAvg:E3}.");

        string dumpDir = Path.Combine(RepoRoot.Path, "Output", "normalbae_csharp_dump");
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
        Tensor normals = model.Forward(backend, input, tap);
        _output.WriteLine($"[forward] {sw.ElapsedMilliseconds} ms — output {normals.Shape}");

        foreach ((string name, double avgErr) in stages)
            _output.WriteLine($"  {name,-10} avg err {avgErr:E3}");

        Assert.Equal(new TensorShape(1, 4, h, w), normals.Shape);
        float[] refOut = AnnotatorParityIo.ReadF32(Path.Combine(refDir, "out_res1.bin"), normals.ElementCount);
        (double avg, double max, _) = AnnotatorParityIo.Compare(normals.AsReadOnlySpan<float>(), refOut);
        _output.WriteLine($"out_res1: avg err {avg:E3}, max err {max:E3}");
        Assert.True(avg < 1e-3, $"Normals diverge: avg err {avg:E3} >= 1e-3.");
        foreach ((string name, double avgErr) in stages)
            Assert.True(avgErr < 1e-2, $"Stage {name} diverges: avg err {avgErr:E3}.");

        // e2e uint8 conditioning image ((normal+1)/2 with truncation).
        byte[] cond = NormalBaePreprocessor.PostprocessToRgb24(normals);
        byte[] refCond = File.ReadAllBytes(Path.Combine(refDir, "cond_u8.bin"));
        double condMismatch = AnnotatorParityIo.MismatchFraction(cond, refCond);
        _output.WriteLine($"cond_u8: mismatch fraction {condMismatch:E3}");
        Assert.True(condMismatch < 1e-3, $"cond_u8 mismatch fraction {condMismatch:E3} >= 1e-3.");
        normals.Dispose();
    }
}
