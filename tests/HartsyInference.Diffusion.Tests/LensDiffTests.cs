using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer diff harness for the Microsoft Lens DiT against the ComfyUI reference
/// (<c>comfy/ldm/lens/model.py</c> run on the REAL checkpoint by
/// <c>tests/python-reference/dump_lens_reference.py</c>). Loads the Python-saved synthetic inputs,
/// runs the C# <see cref="LensTransformer"/> on CPU with <c>LENS_DEBUG_DIR</c> set, and compares
/// every dumped stage (corr + maxAbs). Skips cleanly when the reference dump or the checkpoint
/// (env <c>LENS_DIT_CHECKPOINT</c>) is missing.</summary>
public sealed unsafe class LensDiffTests
{
    private const int HPacked = 16;
    private const int WPacked = 16;
    private const int STxt = 20;
    private const float Timestep = 0.7f;

    private readonly ITestOutputHelper _output;
    public LensDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "Integration")]
    public void Transformer_Matches_ComfyReference_LayerByLayer_Cpu()
    {
        string repoRoot = RepoRoot.Path;
        string refDir = Path.Combine(repoRoot, "tests/python-reference/lens_reference_tensors/full_forward");
        string inputsDir = Path.Combine(refDir, "inputs");
        if (!Directory.Exists(inputsDir))
        {
            _output.WriteLine($"SKIPPED: reference dump not found at {inputsDir} — run dump_lens_reference.py first.");
            return;
        }

        string? checkpoint = Environment.GetEnvironmentVariable("LENS_DIT_CHECKPOINT");
        if (string.IsNullOrEmpty(checkpoint) || !File.Exists(checkpoint))
        {
            _output.WriteLine("SKIPPED: LENS_DIT_CHECKPOINT not set or file missing.");
            return;
        }

        string csDumpDir = Path.Combine(repoRoot, "Output", "lens_csharp_dump");
        if (Directory.Exists(csDumpDir)) Directory.Delete(csDumpDir, recursive: true);
        Directory.CreateDirectory(Path.Combine(csDumpDir, "layers"));
        Environment.SetEnvironmentVariable("LENS_DEBUG_DIR", csDumpDir);

        Tensor packed = LoadF32(Path.Combine(inputsDir, "packed_latent.bin"), new TensorShape(1, HPacked * WPacked, 128));
        Tensor[] encLayers = new Tensor[4];
        for (int i = 0; i < 4; i++)
            encLayers[i] = LoadF32(Path.Combine(inputsDir, $"enc_layer_{i}.bin"), new TensorShape(1, STxt, 2880));

        _output.WriteLine($"Loading DiT from {checkpoint} ...");
        Stopwatch sw = Stopwatch.StartNew();
        SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        Dictionary<string, Tensor> converted = LensCheckpointConverter.ConvertComfyDit(loader.GetAllTensors());

        // CPU kernels are F32-only (BF16 biases/norm weights would throw; BF16 GEMM weights would
        // re-cast per call) — cast everything up front.
        Dictionary<string, Tensor> f32 = new(converted.Count);
        foreach (KeyValuePair<string, Tensor> kvp in converted)
            f32[kvp.Key] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);
        _output.WriteLine($"  Converted {f32.Count} keys in {sw.ElapsedMilliseconds}ms");

        LensTransformer transformer = new(LensConfig.Turbo);
        transformer.LoadWeights(f32);

        using CpuBackend backend = new();
        sw.Restart();
        Tensor velocity = transformer.Forward(backend, packed, encLayers, Timestep, HPacked, WPacked);
        _output.WriteLine($"  Forward done in {sw.ElapsedMilliseconds}ms");

        // ── Compare every stage the reference dumped ──
        string refLayers = Path.Combine(refDir, "layers");
        string csLayers = Path.Combine(csDumpDir, "layers");
        bool anyFail = false;
        List<string> stages = ["img_in", "txt_concat", "txt_in", "time_text_embed"];
        for (int i = 0; i < 48; i++)
        {
            stages.Add($"block_{i}_image");
            stages.Add($"block_{i}_text");
        }
        stages.Add("norm_out");
        stages.Add("proj_out");

        foreach (string stage in stages)
        {
            string refPath = Path.Combine(refLayers, stage + ".bin");
            string csPath = Path.Combine(csLayers, stage + ".bin");
            if (!File.Exists(refPath) || !File.Exists(csPath))
            {
                _output.WriteLine($"  {stage,-20} MISSING ({(File.Exists(refPath) ? "cs" : "ref")})");
                anyFail = true;
                continue;
            }
            (double corr, double maxAbs) = Compare(refPath, csPath);
            bool ok = corr >= 0.999;
            if (!ok) anyFail = true;
            _output.WriteLine($"  {stage,-20} corr={corr:F6} maxAbs={maxAbs:E3} {(ok ? "" : "  <-- DIVERGES")}");
        }

        (double vCorr, double vMax) = Compare(Path.Combine(refDir, "output_velocity.bin"), Path.Combine(csDumpDir, "output_velocity.bin"));
        _output.WriteLine($"  output_velocity      corr={vCorr:F6} maxAbs={vMax:E3}");

        velocity.Dispose();
        packed.Dispose();
        for (int i = 0; i < 4; i++) encLayers[i].Dispose();
        loader.Dispose();

        Assert.False(anyFail, "one or more stages diverged (corr < 0.999) — see output for the first divergent stage");
        Assert.True(vCorr >= 0.999, $"final velocity corr {vCorr:F6} < 0.999");
    }

    private static Tensor LoadF32(string path, TensorShape shape)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != shape.ElementCount * sizeof(float))
            throw new InvalidOperationException($"{path}: {bytes.Length} bytes != expected {shape.ElementCount * sizeof(float)}");
        Tensor t = new(shape, DType.F32);
        fixed (byte* src = bytes)
        {
            Buffer.MemoryCopy(src, (void*)t.DataPointer, bytes.Length, bytes.Length);
        }
        return t;
    }

    private static (double corr, double maxAbs) Compare(string refPath, string csPath)
    {
        float[] a = LoadFloats(refPath);
        float[] b = LoadFloats(csPath);
        if (a.Length != b.Length)
            return (double.NaN, double.NaN);
        double sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0, maxAbs = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double x = a[i], y = b[i];
            sumA += x; sumB += y; sumAA += x * x; sumBB += y * y; sumAB += x * y;
            double d = Math.Abs(x - y);
            if (d > maxAbs) maxAbs = d;
        }
        int n = a.Length;
        double cov = sumAB - sumA * sumB / n;
        double varA = sumAA - sumA * sumA / n;
        double varB = sumBB - sumB * sumB / n;
        double corr = cov / Math.Sqrt(Math.Max(varA * varB, 1e-30));
        return (corr, maxAbs);
    }

    private static float[] LoadFloats(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] data = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return data;
    }
}
