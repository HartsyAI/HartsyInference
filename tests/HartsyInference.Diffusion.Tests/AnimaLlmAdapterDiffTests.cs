using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer diff harness for Anima's <see cref="AnimaLlmAdapter"/>. Loads the Python reference
/// inputs (T5 token IDs + Qwen-3 hidden states), runs the C# adapter with
/// <c>ANIMA_LLM_ADAPTER_DEBUG_DIR</c> set so every layer dumps to disk, then leaves the dump for
/// <c>diff_anima_llm_adapter_layers.py</c> to compare against the canonical
/// <c>tdrussell/diffusion-pipe/models/llm_adapter.py</c> forward.
///
/// Skips cleanly when the Python reference dump or the Anima checkpoint is missing.</summary>
[Trait("Category", "Integration")]
public unsafe class AnimaLlmAdapterDiffTests
{
    private readonly ITestOutputHelper _output;
    public AnimaLlmAdapterDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Adapter_Matches_PythonReference_LayerByLayer_Cpu() =>
        RunDiff(dumpDirName: "anima_llm_adapter_csharp_dump");

    private void RunDiff(string dumpDirName)
    {
        string repoRoot = RepoRoot.Path;
        string refDir = Path.Combine(repoRoot, "tests/python-reference/anima_llm_adapter_reference");
        string inputsDir = Path.Combine(refDir, "inputs");

        if (!Directory.Exists(inputsDir))
        {
            _output.WriteLine($"SKIPPED: Python reference dump not found at {inputsDir}");
            _output.WriteLine("  Run: tests/python-reference/.venv/bin/python tests/python-reference/dump_anima_llm_adapter.py");
            return;
        }

        string ckpt = TestPaths.Anima.Transformer;
        if (!File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: Anima checkpoint not found: {ckpt}");
            return;
        }

        string csDumpDir = Path.Combine(repoRoot, "Output", dumpDirName);
        if (Directory.Exists(csDumpDir)) Directory.Delete(csDumpDir, recursive: true);
        Directory.CreateDirectory(Path.Combine(csDumpDir, "layers"));
        Environment.SetEnvironmentVariable("ANIMA_LLM_ADAPTER_DEBUG_DIR", csDumpDir);

        // Load Python-saved inputs.
        int[] t5Ids = LoadInt64TensorAsInt32(Path.Combine(inputsDir, "t5_input_ids.bin"));
        _output.WriteLine($"Loaded T5 token IDs: [{string.Join(", ", t5Ids)}] (len={t5Ids.Length})");

        // qwen3_hidden shape is [1, S_qwen3, 1024]. Infer S_qwen3 from file size.
        string qwenPath = Path.Combine(inputsDir, "qwen3_hidden.bin");
        long qwenBytes = new FileInfo(qwenPath).Length;
        long qwenFloats = qwenBytes / sizeof(float);
        if (qwenFloats % 1024 != 0)
            throw new InvalidDataException($"qwen3_hidden.bin has {qwenFloats} floats, not divisible by 1024.");
        int sQwen = (int)(qwenFloats / 1024);
        _output.WriteLine($"qwen3_hidden: inferred shape=[1, {sQwen}, 1024]");
        Tensor qwenHidden = LoadF32Tensor(qwenPath, 1, sQwen, 1024);

        _output.WriteLine($"\nLoading Anima checkpoint: {Path.GetFileName(ckpt)}");
        Stopwatch sw = Stopwatch.StartNew();
        (AnimaCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            AnimaCheckpointConverter.LoadAndConvert(ckpt);
        sw.Stop();
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms ({converted.LlmAdapter.Count} llm_adapter keys)");

        using (loader)
        {
            Dictionary<string, Tensor> adapterWeights = CastWeightsToF32(converted.LlmAdapter);
            AnimaLlmAdapterConfig cfg = AnimaLlmAdapterConfig.Default;
            using AnimaLlmAdapter adapter = new(cfg);
            adapter.LoadWeights(adapterWeights);
            _output.WriteLine($"  Adapter ready: {cfg.NumLayers} blocks, hidden={cfg.HiddenSize}, heads={cfg.NumHeads}");

            using IBackend backend = new CpuBackend();

            sw.Restart();
            _output.WriteLine("\nRunning AnimaLlmAdapter.Forward (CPU, F32, layer dumps enabled)...");
            Tensor output = adapter.Forward(backend, qwenHidden, t5Ids);
            sw.Stop();
            _output.WriteLine($"Forward done in {sw.ElapsedMilliseconds}ms. Output shape={output.Shape}");

            float* p = (float*)output.DataPointer;
            int n = (int)output.ElementCount;
            double sum = 0, sumAbs = 0, sumSq = 0;
            for (int i = 0; i < n; i++) { sum += p[i]; sumAbs += Math.Abs(p[i]); sumSq += p[i] * p[i]; }
            double mean = sum / n, absMean = sumAbs / n, std = Math.Sqrt(sumSq / n - mean * mean);
            _output.WriteLine($"  Stats: mean={mean:F6}, abs_mean={absMean:F6}, std={std:F6}");

            output.Dispose();
        }

        qwenHidden.Dispose();
        Environment.SetEnvironmentVariable("ANIMA_LLM_ADAPTER_DEBUG_DIR", null);

        _output.WriteLine($"\nC# dump written to: {csDumpDir}");
        _output.WriteLine("Now run: tests/python-reference/.venv/bin/python tests/python-reference/diff_anima_llm_adapter_layers.py");
    }

    private static unsafe int[] LoadInt64TensorAsInt32(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        long count = raw.LongLength / sizeof(long);
        int[] result = new int[count];
        fixed (byte* src = raw)
        {
            long* lp = (long*)src;
            for (long i = 0; i < count; i++) result[i] = checked((int)lp[i]);
        }
        return result;
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
