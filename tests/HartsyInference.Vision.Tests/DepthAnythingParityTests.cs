using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.PyTorch;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.DepthAnything;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>Real-weight parity for Depth-Anything-V2 against the official PyTorch implementation.
/// <c>dump_depth_anything.py</c> runs the official model (DepthAnything/Depth-Anything-V2 repo code + the
/// official <c>.pth</c>) on a fixed non-square input (280×378, exercising pos-embed interpolation) and dumps
/// every DPT-head intermediate; this test loads the same checkpoint via <see cref="PytorchPickleLoader"/>,
/// runs <see cref="DepthAnythingV2Model"/> on the CPU backend, compares stage-by-stage in-process, and also
/// writes the C# taps to <c>Output/depth_anything_csharp_dump/&lt;encoder&gt;</c> for
/// <c>diff_depth_anything.py</c>. Skips cleanly when the checkpoint or Python dump is absent.</summary>
public sealed unsafe class DepthAnythingParityTests
{
    private readonly ITestOutputHelper _output;
    public DepthAnythingParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void VitSmall_MatchesOfficialReference() =>
        RunParity("vits", DepthAnythingPreset.Small, TestPaths.DepthAnything.VitS);

    [Fact]
    [Trait("Category", "Integration")]
    public void VitLarge_MatchesOfficialReference() =>
        RunParity("vitl", DepthAnythingPreset.Large, TestPaths.DepthAnything.VitL);

    [Fact]
    [Trait("Category", "Integration")]
    public void Preprocessor_MatchesOfficialTransform()
    {
        string refDir = ReferenceDir("vits");
        string inputPath = Path.Combine(refDir, "preprocess_input.bin");
        string refPath = Path.Combine(refDir, "preprocess_ref.bin");
        if (!File.Exists(inputPath) || !File.Exists(refPath))
        {
            _output.WriteLine($"SKIPPED: preprocessor reference not found in {refDir} — run dump_depth_anything.py first.");
            return;
        }
        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "meta.json")));
        int srcH = meta.RootElement.GetProperty("pre_src_h").GetInt32();
        int srcW = meta.RootElement.GetProperty("pre_src_w").GetInt32();
        int outH = meta.RootElement.GetProperty("pre_out_h").GetInt32();
        int outW = meta.RootElement.GetProperty("pre_out_w").GetInt32();

        DepthAnythingPreprocessor pre = new();
        (int w, int h) = pre.ComputeTargetSize(srcW, srcH);
        _output.WriteLine($"target size: C# {w}x{h}, python {outW}x{outH}");
        Assert.Equal((outW, outH), (w, h));

        byte[] rgb = File.ReadAllBytes(inputPath);
        using Tensor pixels = pre.Preprocess(rgb, srcW, srcH);
        Assert.Equal(new TensorShape(1, 3, outH, outW), pixels.Shape);

        float[] reference = ReadF32(refPath, 3L * outH * outW);
        (double avgErr, double maxErr, _) = Compare(pixels.AsReadOnlySpan<float>(), reference);
        _output.WriteLine($"preprocess: avg err {avgErr:E3}, max err {maxErr:E3}");
        Assert.True(avgErr < 1e-3, $"Preprocessor diverges from official transform: avg err {avgErr:E3} >= 1e-3.");
    }

    private void RunParity(string encoder, DepthAnythingPreset preset, string checkpoint)
    {
        string refDir = ReferenceDir(encoder);
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: checkpoint not found: {checkpoint}");
            return;
        }
        if (!File.Exists(Path.Combine(refDir, "depth.bin")))
        {
            _output.WriteLine($"SKIPPED: Python reference not found in {refDir} — run dump_depth_anything.py first.");
            return;
        }

        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(refDir, "meta.json")));
        int h = meta.RootElement.GetProperty("H").GetInt32();
        int w = meta.RootElement.GetProperty("W").GetInt32();

        Stopwatch sw = Stopwatch.StartNew();
        using PytorchPickleLoader loader = new();
        loader.Load(checkpoint);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        _output.WriteLine($"[load] {sw.ElapsedMilliseconds} ms — {weights.Count} tensors");

        DepthAnythingV2Model model = new(preset);
        model.LoadWeights(weights);

        Tensor input = new(new TensorShape(1, 3, h, w), DType.F32);
        byte[] raw = File.ReadAllBytes(Path.Combine(refDir, "input.bin"));
        Assert.Equal(input.ElementCount * 4, raw.Length);
        raw.AsSpan().CopyTo(new Span<byte>((void*)input.DataPointer, raw.Length));

        string dumpDir = Path.Combine(RepoRoot.Path, "Output", "depth_anything_csharp_dump", encoder);
        Directory.CreateDirectory(dumpDir);
        List<(string Name, double AvgErr)> stages = [];
        Action<string, Tensor> tap = (name, t) =>
        {
            WriteF32(Path.Combine(dumpDir, name + ".bin"), t);
            CompareStage(refDir, name, t, stages);
        };

        using IBackend backend = new CpuBackend();
        sw.Restart();
        Tensor depth = model.Forward(backend, input, tap);
        _output.WriteLine($"[forward] {sw.ElapsedMilliseconds} ms — output {depth.Shape}");
        WriteF32(Path.Combine(dumpDir, "depth.bin"), depth);
        input.Dispose();

        foreach ((string name, double avgErr) in stages)
            _output.WriteLine($"  {name,-12} avg err {avgErr:E3}");

        Assert.Equal(new TensorShape(1, 1, h, w), depth.Shape);
        float[] refDepth = ReadF32(Path.Combine(refDir, "depth.bin"), depth.ElementCount);
        (double avg, double max, double refMeanAbs) = Compare(depth.AsReadOnlySpan<float>(), refDepth);
        _output.WriteLine($"depth: avg err {avg:E3} (rel {avg / refMeanAbs:E3}), max err {max:E3}, ref mean|x| {refMeanAbs:F4}");
        depth.Dispose();

        Assert.True(avg < 1e-3, $"Depth diverges from official reference: avg err {avg:E3} >= 1e-3.");
        foreach ((string name, double avgErr) in stages)
            Assert.True(avgErr < 1e-2, $"Stage {name} diverges: avg err {avgErr:E3}.");
    }

    private void CompareStage(string refDir, string name, Tensor t, List<(string, double)> stages)
    {
        string path = Path.Combine(refDir, name + ".bin");
        if (!File.Exists(path)) return;
        float[] reference = ReadF32(path, t.ElementCount);
        (double avg, _, _) = Compare(t.AsReadOnlySpan<float>(), reference);
        stages.Add((name, avg));
    }

    private static string ReferenceDir(string encoder) =>
        Path.Combine(RepoRoot.Path, "tests", "python-reference", "depth_anything_reference_tensors", encoder);

    private static float[] ReadF32(string path, long expectedCount)
    {
        byte[] raw = File.ReadAllBytes(path);
        Assert.Equal(expectedCount * 4, raw.Length);
        float[] data = new float[expectedCount];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return data;
    }

    private static void WriteF32(string path, Tensor t)
    {
        ReadOnlySpan<byte> bytes = new((void*)t.DataPointer, (int)(t.ElementCount * 4));
        File.WriteAllBytes(path, bytes.ToArray());
    }

    private static (double AvgErr, double MaxErr, double RefMeanAbs) Compare(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double sumErr = 0, maxErr = 0, sumRefAbs = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double d = Math.Abs(actual[i] - expected[i]);
            sumErr += d;
            if (d > maxErr) maxErr = d;
            sumRefAbs += Math.Abs(expected[i]);
        }
        return (sumErr / actual.Length, maxErr, sumRefAbs / actual.Length);
    }
}
