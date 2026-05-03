using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>Layer-by-layer diff harness for ERNIE-Image. Loads the Python-saved synthetic inputs
/// (latent + text embeds + text_lens + timestep), runs the C# `ErnieImageTransformer` on CPU/GPU
/// with `ERNIE_IMAGE_DEBUG_DIR` set, then leaves the dump for `diff_ernie_image_layers.py`.
///
/// Skips cleanly when the Python reference dump or the ERNIE-Image checkpoint dir is missing.</summary>
public unsafe class ErnieImageDiffTests
{
    private readonly ITestOutputHelper _output;
    public ErnieImageDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Cpu() =>
        RunDiff(useGpu: false, dumpDirName: "ernie_image_csharp_dump");

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Gpu() =>
        RunDiff(useGpu: true, dumpDirName: "ernie_image_csharp_gpu_dump");

    private void RunDiff(bool useGpu, string dumpDirName)
    {
        string repoRoot = RepoRoot.Path;
        string refDir = Path.Combine(repoRoot, "tests/python-reference/ernie_image_reference_tensors/full_forward");
        string inputsDir = Path.Combine(refDir, "inputs");

        if (!Directory.Exists(inputsDir))
        {
            _output.WriteLine($"SKIPPED: Python reference dump not found at {inputsDir}");
            _output.WriteLine("  Run: tests/python-reference/.venv/bin/python tests/python-reference/dump_ernie_image_full_forward.py");
            return;
        }

        string modelDir = TestPaths.ErnieImage.V1Dir;
        if (!Directory.Exists(modelDir))
        {
            _output.WriteLine($"SKIPPED: ERNIE-Image model directory not found: {modelDir}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(ErnieImageDiffTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (useGpu && !Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        string csDumpDir = Path.Combine(repoRoot, "Output", dumpDirName);
        if (Directory.Exists(csDumpDir)) Directory.Delete(csDumpDir, recursive: true);
        Directory.CreateDirectory(Path.Combine(csDumpDir, "layers"));
        Environment.SetEnvironmentVariable("ERNIE_IMAGE_DEBUG_DIR", csDumpDir);

        // ── Load synthetic inputs (must match dump_ernie_image_full_forward.py) ──
        Tensor latent = LoadF32Tensor(Path.Combine(inputsDir, "latent.bin"), 1, 128, 32, 32);
        Tensor textEmbeds = LoadF32Tensor(Path.Combine(inputsDir, "text_embeds.bin"), 1, 64, 3072);
        int[] textLens = LoadInt32Array(Path.Combine(inputsDir, "text_lens.bin"), 1);
        float timestep = LoadF32Scalar(Path.Combine(inputsDir, "timestep.bin"));
        _output.WriteLine($"Loaded inputs: latent={latent.Shape}, text={textEmbeds.Shape}, textLens=[{string.Join(",", textLens)}], t={timestep}");

        _output.WriteLine($"Loading transformer from {modelDir}/transformer ...");
        Stopwatch sw = Stopwatch.StartNew();
        (ErnieImageCheckpointConverter.ConvertedWeights converted, IReadOnlyList<SafeTensorsLoader> loaders) =
            ErnieImageCheckpointConverter.LoadAndConvert(modelDir);
        _output.WriteLine($"  Loaded {converted.Transformer.Count} keys in {sw.ElapsedMilliseconds}ms");

        try
        {
            Dictionary<string, Tensor> trans = useGpu ? converted.Transformer : CastWeightsToF32(converted.Transformer);
            ErnieImageConfig config = ErnieImageConfig.V1;

            using ErnieImageTransformer transformer = new(config);
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
            _output.WriteLine($"\nRunning ErnieImageTransformer.Forward ({(useGpu ? "GPU" : "CPU")}) ...");
            Tensor velocity = transformer.Forward(backend, latent, timestep, textEmbeds, textLens);
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
        finally
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
        }

        latent.Dispose();
        textEmbeds.Dispose();
        Environment.SetEnvironmentVariable("ERNIE_IMAGE_DEBUG_DIR", null);

        _output.WriteLine($"\nC# dump written to: {csDumpDir}");
        _output.WriteLine("Now run: tests/python-reference/.venv/bin/python tests/python-reference/diff_ernie_image_layers.py");
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

    private static unsafe int[] LoadInt32Array(string path, int n)
    {
        byte[] raw = File.ReadAllBytes(path);
        if (raw.LongLength != (long)n * sizeof(int))
            throw new InvalidDataException($"{path}: expected {(long)n * sizeof(int)} bytes, got {raw.LongLength}");
        int[] arr = new int[n];
        fixed (byte* src = raw) fixed (int* dst = arr)
            Buffer.MemoryCopy(src, dst, raw.LongLength, raw.LongLength);
        return arr;
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
