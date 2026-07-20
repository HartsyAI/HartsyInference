using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer diff test for SD3.5 against a diffusers reference. Loads the Python-saved synthetic inputs (latent + raw context + pooled + timestep), runs the C# Sd3Transformer through the CPU backend with <c>SD3_DEBUG_DIR</c> set, then writes per-block dumps for diff_sd35_layers.py to compare. Mirrors <c>ZImageDiffTests.Transformer_Matches_PythonReference_LayerByLayer</c> from PHASE_3_DEVIATIONS #28.</summary>
[Trait("Category", "Integration")]
public unsafe class Sd35DiffTests
{
    private readonly ITestOutputHelper _output;

    public Sd35DiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Cpu() =>
        RunDiff(useGpu: false, dumpDirName: "sd35_csharp_dump");

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Gpu() =>
        RunDiff(useGpu: true, dumpDirName: "sd35_csharp_gpu_dump");

    private void RunDiff(bool useGpu, string dumpDirName)
    {
        string repoRoot = RepoRoot.Path;
        string refDir = Path.Combine(repoRoot, "tests/python-reference/sd35_reference_tensors/full_forward");
        string inputsDir = Path.Combine(refDir, "inputs");

        if (!Directory.Exists(inputsDir))
        {
            _output.WriteLine($"SKIPPED: Python reference dump not found at {inputsDir}");
            _output.WriteLine("  Run: tests/python-reference/.venv/bin/python tests/python-reference/dump_sd35_full_forward.py");
            return;
        }

        string ckpt = TestPaths.Sd35.Medium;
        if (!File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {ckpt}");
            return;
        }

        string csDumpDir = Path.Combine(repoRoot, "Output", dumpDirName);
        if (Directory.Exists(csDumpDir))
            Directory.Delete(csDumpDir, recursive: true);
        Directory.CreateDirectory(Path.Combine(csDumpDir, "layers"));
        Environment.SetEnvironmentVariable("SD3_DEBUG_DIR", csDumpDir);

        string assemblyDir = Path.GetDirectoryName(typeof(Sd35DiffTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (useGpu && !Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        // ── Load inputs ──
        Tensor latent = LoadF32Tensor(Path.Combine(inputsDir, "latent.bin"), 1, 16, 64, 64);
        Tensor contextPre = LoadF32Tensor(Path.Combine(inputsDir, "context_pre.bin"), 1, 154, 4096);
        Tensor pooled = LoadF32Tensor(Path.Combine(inputsDir, "pooled.bin"), 1, 2048);
        float timestep = LoadF32Scalar(Path.Combine(inputsDir, "timestep.bin"));
        _output.WriteLine($"Loaded inputs: latent={latent.Shape}, context_pre={contextPre.Shape}, pooled={pooled.Shape}, timestep={timestep}");

        // ── Load + convert checkpoint (cast all to F32 for reference accuracy) ──
        _output.WriteLine($"Loading checkpoint: {Path.GetFileName(ckpt)}");
        Stopwatch sw = Stopwatch.StartNew();
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd3CheckpointConverter.LoadAndConvert(ckpt);
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            // CPU diff: cast weights to F32 for max precision (matches Python reference).
            // GPU diff: keep weights at native FP8/F16 — this is the path the end-to-end test uses,
            // so we want to surface any FP8/F16-specific GPU bug here, not paper over it with F32 casts.
            Dictionary<string, Tensor> trans = useGpu ? converted.Transformer : CastWeightsToF32(converted.Transformer);

            Sd3Config config = Sd3Config.AutoDetect(trans);
            _output.WriteLine($"  Config: depth={config.Depth}, hidden={config.HiddenSize}, heads={config.NumHeads}, qkNorm={config.UseQkNorm}, dual={(config.DualAttentionLayers?.Length ?? 0)} layers");

            using Sd3Transformer transformer = new(config);
            transformer.LoadWeights(trans);
            _output.WriteLine("  Weights loaded.");

            using IBackend backend = useGpu
                ? (IBackend)new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir)
                : new CpuBackend();
            if (useGpu)
            {
                ((CudaBackend)backend).PreloadWeights(transformer.EnumerateWeights());
                _output.WriteLine($"  GPU backend ready: {backend.Capabilities.Name}");
            }

            // ── Project context through context_embedder (matches the pipeline behavior) ──
            // Sd3Pipeline.GenerateFromTokens calls _transformer.ProjectContext once before the denoise loop.
            Tensor contextProjected = transformer.ProjectContext(backend, contextPre);
            _output.WriteLine($"  Context projected: {contextProjected.Shape}");

            // ── Run forward (this populates the SD3_DEBUG_DIR with per-block dumps) ──
            sw.Restart();
            _output.WriteLine($"\nRunning Sd3Transformer.Forward (CPU) ...");
            Tensor velocity = transformer.Forward(backend, latent, timestep, contextProjected, pooled);
            sw.Stop();
            _output.WriteLine($"Forward done in {sw.ElapsedMilliseconds}ms.");
            _output.WriteLine($"Output velocity: shape={velocity.Shape}");

            // Quick stats
            float* vptr = (float*)velocity.DataPointer;
            int n = (int)velocity.ElementCount;
            double sum = 0, sumAbs = 0, sumSq = 0;
            for (int i = 0; i < n; i++) { sum += vptr[i]; sumAbs += Math.Abs(vptr[i]); sumSq += vptr[i] * vptr[i]; }
            double mean = sum / n;
            double absMean = sumAbs / n;
            double std = Math.Sqrt(sumSq / n - mean * mean);
            _output.WriteLine($"  Stats: mean={mean:F6}, abs_mean={absMean:F6}, std={std:F6}");

            velocity.Dispose();
            contextProjected.Dispose();
        }

        latent.Dispose();
        contextPre.Dispose();
        pooled.Dispose();

        Environment.SetEnvironmentVariable("SD3_DEBUG_DIR", null);

        _output.WriteLine($"\nC# dump written to: {csDumpDir}");
        _output.WriteLine("Now run: tests/python-reference/.venv/bin/python tests/python-reference/diff_sd35_layers.py");
    }

    private static unsafe Tensor LoadF32Tensor(string path, params long[] dims)
    {
        TensorShape shape = dims.Length switch
        {
            1 => new TensorShape(dims[0]),
            2 => new TensorShape(dims[0], dims[1]),
            3 => new TensorShape(dims[0], dims[1], dims[2]),
            4 => new TensorShape(dims[0], dims[1], dims[2], dims[3]),
            _ => throw new ArgumentException($"Unsupported rank: {dims.Length}"),
        };
        Tensor t = new(shape, DType.F32);
        byte[] raw = File.ReadAllBytes(path);
        long expected = shape.ElementCount * sizeof(float);
        if (raw.LongLength != expected)
            throw new InvalidDataException($"{path}: expected {expected} bytes for shape {shape}, got {raw.LongLength}");
        fixed (byte* src = raw)
        {
            Buffer.MemoryCopy(src, (void*)t.DataPointer, raw.LongLength, raw.LongLength);
        }
        return t;
    }

    private static unsafe float LoadF32Scalar(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        if (raw.Length < sizeof(float))
            throw new InvalidDataException($"{path}: too small for a single float");
        fixed (byte* src = raw)
        {
            return *(float*)src;
        }
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f32[kvp.Key] = kvp.Value.DType != DType.F32
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }
}
