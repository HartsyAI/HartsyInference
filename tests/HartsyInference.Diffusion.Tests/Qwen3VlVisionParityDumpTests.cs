using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight dump harness for Qwen3-VL vision-tower parity vs the HF reference
/// (<c>tests/python-reference/diff_qwen3vl_vision.py</c>). Loads the BF16 <c>visual.*</c> weights from the checkpoint
/// named by <c>HARTSY_QWEN3VL_PATH</c>, runs the image processor + vision tower on a deterministic synthetic image, and
/// writes <c>pixel_values</c> / merged tokens / deepstack features as raw F32 .bin files to
/// <c>HARTSY_QWEN3VL_DUMP_DIR</c>. Skips cleanly when the env vars are unset.</summary>
public sealed class Qwen3VlVisionParityDumpTests
{
    private readonly ITestOutputHelper _output;
    public Qwen3VlVisionParityDumpTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public unsafe void DumpVisionTowerOutputs()
    {
        string? path = Environment.GetEnvironmentVariable("HARTSY_QWEN3VL_PATH");
        string? dumpDir = Environment.GetEnvironmentVariable("HARTSY_QWEN3VL_DUMP_DIR");
        if (string.IsNullOrEmpty(path) || string.IsNullOrEmpty(dumpDir) || !File.Exists(path))
            return;
        Directory.CreateDirectory(dumpDir);

        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> vw = new();
        foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
        {
            int v = kvp.Key.LastIndexOf(".visual.", StringComparison.Ordinal);
            string? bare = v >= 0 ? kvp.Key[(v + ".visual.".Length)..]
                : kvp.Key.StartsWith("visual.", StringComparison.Ordinal) ? kvp.Key["visual.".Length..]
                : null;
            // CPU backend is F32-only — cast the BF16 vision weights up front (CUDA handles BF16 natively).
            if (bare is not null) vw[bare] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);
        }
        Assert.True(vw.ContainsKey("patch_embed.proj.weight"), "checkpoint has no vision tower");

        Qwen3VlVisionConfig cfg = Qwen3VlVisionConfig.Qwen3Vl8B;
        using Qwen3VlVisionEncoder vision = new(cfg);
        vision.LoadWeights(vw);
        // HARTSY_QWEN3VL_CUDA=1 runs on the CUDA backend (the live loader path, exercising the host-glue ⇄
        // activation-cache coherence); default is the CPU backend.
        bool useCuda = Environment.GetEnvironmentVariable("HARTSY_QWEN3VL_CUDA") == "1";
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        using IBackend backend = useCuda ? new CudaBackend(0, ptxDir) : new CpuBackend();
        _output.WriteLine($"backend: {(useCuda ? "CUDA" : "CPU")}");

        // Deterministic synthetic RGB image [3, 224, 168] in [0, 1] (small grid keeps the CPU forward fast).
        int srcH = 224, srcW = 168;
        Tensor rgb = new(new TensorShape(3, srcH, srcW), DType.F32);
        Random rng = new(12345);
        float* rp = (float*)rgb.DataPointer;
        for (long i = 0; i < rgb.ElementCount; i++) rp[i] = (float)rng.NextDouble();

        Qwen3VlImageProcessor proc = new(cfg);
        (Tensor pix, int gt, int gh, int gw) = proc.Preprocess(rgb);
        _output.WriteLine($"grid = ({gt}, {gh}, {gw}), pixel_values {pix.Shape}");
        WriteBin(Path.Combine(dumpDir, "pixel_values.bin"), pix);
        File.WriteAllText(Path.Combine(dumpDir, "grid.txt"), $"{gt} {gh} {gw}");
        WriteBin(Path.Combine(dumpDir, "rgb.bin"), rgb);
        rgb.Dispose();

        Qwen3VlVisionEncoder.VisionOutput vo = vision.Forward(backend, pix, gt, gh, gw);
        pix.Dispose();
        WriteBin(Path.Combine(dumpDir, "merged_tokens.bin"), vo.MergedTokens);
        for (int i = 0; i < vo.DeepstackFeatures.Length; i++)
            WriteBin(Path.Combine(dumpDir, $"deepstack_{i}.bin"), vo.DeepstackFeatures[i]);
        _output.WriteLine($"merged {vo.MergedTokens.Shape}, deepstack x{vo.DeepstackFeatures.Length} dumped to {dumpDir}");

        vo.MergedTokens.Dispose();
        foreach (Tensor d in vo.DeepstackFeatures) d.Dispose();
    }

    private static unsafe void WriteBin(string file, Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        long bytes = f32.ElementCount * sizeof(float);
        using FileStream fs = File.Create(file);
        fs.Write(new ReadOnlySpan<byte>((void*)f32.DataPointer, checked((int)bytes)));
        if (!ReferenceEquals(f32, t)) f32.Dispose();
    }
}
