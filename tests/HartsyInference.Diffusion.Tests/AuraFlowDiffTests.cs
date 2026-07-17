using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer diff harness for AuraFlow. Loads the Python-saved synthetic inputs
/// (latent + raw context + timestep), runs the C# `AuraFlowTransformer` through CPU/GPU with
/// `AURAFLOW_DEBUG_DIR` set, then leaves the dump for `diff_auraflow_layers.py`.
///
/// Mirrors `Sd35DiffTests` from PHASE_3_DEVIATIONS #28 methodology. Skips cleanly when the
/// Python reference dump or the AuraFlow checkpoint is missing.</summary>
[Trait("Category", "Integration")]
public unsafe class AuraFlowDiffTests
{
    private readonly ITestOutputHelper _output;
    public AuraFlowDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Cpu() =>
        RunDiff(useGpu: false, dumpDirName: "auraflow_csharp_dump");

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Gpu() =>
        RunDiff(useGpu: true, dumpDirName: "auraflow_csharp_gpu_dump");

    private void RunDiff(bool useGpu, string dumpDirName)
    {
        string repoRoot = RepoRoot.Path;
        string refDir = Path.Combine(repoRoot, "tests/python-reference/auraflow_reference_tensors/full_forward");
        string inputsDir = Path.Combine(refDir, "inputs");

        if (!Directory.Exists(inputsDir))
        {
            _output.WriteLine($"SKIPPED: Python reference dump not found at {inputsDir}");
            _output.WriteLine("  Run: tests/python-reference/.venv/bin/python tests/python-reference/dump_auraflow_full_forward.py");
            return;
        }

        string ckpt = TestPaths.AuraFlow.V03;
        if (!File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: AuraFlow checkpoint not found: {ckpt}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(AuraFlowDiffTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (useGpu && !Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        string csDumpDir = Path.Combine(repoRoot, "Output", dumpDirName);
        if (Directory.Exists(csDumpDir)) Directory.Delete(csDumpDir, recursive: true);
        Directory.CreateDirectory(Path.Combine(csDumpDir, "layers"));
        Environment.SetEnvironmentVariable("AURAFLOW_DEBUG_DIR", csDumpDir);

        // ── Load synthetic inputs ──
        Tensor latent = LoadF32Tensor(Path.Combine(inputsDir, "latent.bin"), 1, 4, 64, 64);
        Tensor contextPre = LoadF32Tensor(Path.Combine(inputsDir, "context_pre.bin"), 1, 256, 2048);
        float timestep = LoadF32Scalar(Path.Combine(inputsDir, "timestep.bin"));
        _output.WriteLine($"Loaded inputs: latent={latent.Shape}, context_pre={contextPre.Shape}, t={timestep}");

        // ── Load + convert AuraFlow checkpoint ──
        _output.WriteLine($"Loading checkpoint: {Path.GetFileName(ckpt)}");
        Stopwatch sw = Stopwatch.StartNew();
        (AuraFlowCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            AuraFlowCheckpointConverter.LoadAndConvert(ckpt);
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms ({converted.Transformer.Count} keys)");

        using (loader)
        {
            Dictionary<string, Tensor> trans = useGpu ? converted.Transformer : CastWeightsToF32(converted.Transformer);
            AuraFlowConfig config = AuraFlowConfig.V03;

            using AuraFlowTransformer transformer = new(config);
            transformer.LoadWeights(trans);
            _output.WriteLine("  Weights loaded.");

            using IBackend backend = useGpu
                ? new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir)
                : (IBackend)new CpuBackend();
            if (useGpu)
            {
                ((CudaBackend)backend).PreloadWeights(transformer.EnumerateWeights());
                _output.WriteLine($"  GPU backend ready: {backend.Capabilities.Name}");
            }

            sw.Restart();
            _output.WriteLine($"\nRunning AuraFlowTransformer.Forward ({(useGpu ? "GPU" : "CPU")}) ...");
            Tensor velocity = transformer.Forward(backend, latent, timestep, contextPre);
            sw.Stop();
            _output.WriteLine($"Forward done in {sw.ElapsedMilliseconds}ms.");
            _output.WriteLine($"Output velocity: shape={velocity.Shape}");

            float* vptr = (float*)velocity.DataPointer;
            int n = (int)velocity.ElementCount;
            double sum = 0, sumAbs = 0, sumSq = 0;
            for (int i = 0; i < n; i++) { sum += vptr[i]; sumAbs += Math.Abs(vptr[i]); sumSq += vptr[i] * vptr[i]; }
            double mean = sum / n, absMean = sumAbs / n, std = Math.Sqrt(sumSq / n - mean * mean);
            _output.WriteLine($"  Stats: mean={mean:F6}, abs_mean={absMean:F6}, std={std:F6}");

            velocity.Dispose();
        }

        latent.Dispose();
        contextPre.Dispose();
        Environment.SetEnvironmentVariable("AURAFLOW_DEBUG_DIR", null);

        _output.WriteLine($"\nC# dump written to: {csDumpDir}");
        _output.WriteLine("Now run: tests/python-reference/.venv/bin/python tests/python-reference/diff_auraflow_layers.py");
    }

    private static unsafe Tensor LoadF32Tensor(string path, params long[] dims)
    {
        TensorShape shape = dims.Length switch
        {
            2 => new TensorShape(dims[0], dims[1]),
            3 => new TensorShape(dims[0], dims[1], dims[2]),
            4 => new TensorShape(dims[0], dims[1], dims[2], dims[3]),
            _ => throw new ArgumentException($"Unsupported rank: {dims.Length}"),
        };
        Tensor t = new(shape, DType.F32);
        byte[] raw = File.ReadAllBytes(path);
        long expected = shape.ElementCount * sizeof(float);
        if (raw.LongLength != expected)
            throw new InvalidDataException($"{path}: expected {expected} bytes, got {raw.LongLength}");
        fixed (byte* src = raw) Buffer.MemoryCopy(src, (void*)t.DataPointer, raw.LongLength, raw.LongLength);
        return t;
    }

    private static unsafe float LoadF32Scalar(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        fixed (byte* src = raw) return *(float*)src;
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dt = kvp.Value.DType;
            f32[kvp.Key] = (dt == DType.F16 || dt == DType.BF16) ? kvp.Value.CastTo(DType.F32) : kvp.Value;
        }
        return f32;
    }
}
