using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer diff harness for Anima (Cosmos-Predict2 2B derivative). Loads the Python-saved
/// synthetic inputs (5-D latent + encoder hidden states + scalar timestep), runs the C# <c>AnimaTransformer</c>
/// with <c>ANIMA_DEBUG_DIR</c> set to dump every layer, then leaves the dump for
/// <c>diff_anima_layers.py</c> to compare against the diffusers reference.
///
/// <para>The Python reference deliberately excludes Anima's in-checkpoint <c>net.llm_adapter</c>
/// (diffusers' <c>CosmosTransformer3DModel</c> has no equivalent). So the C# test must run with the
/// LlmAdapter <b>bypassed</b> — we don't construct one and we feed the raw encoder hidden states to
/// <c>AnimaTransformer.Forward</c> directly. Any divergence pinpoints a bug in the DiT trunk
/// (separately from any LlmAdapter issues).</para>
///
/// Skips cleanly when the Python reference dump or the Anima checkpoint is missing.</summary>
[Trait("Category", "Integration")]
public unsafe class AnimaDiffTests
{
    private readonly ITestOutputHelper _output;
    public AnimaDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Cpu() =>
        RunDiff(dumpDirName: "anima_csharp_dump");

    private void RunDiff(string dumpDirName)
    {
        string repoRoot = RepoRoot.Path;
        string refDir = Path.Combine(repoRoot, "tests/python-reference/anima_reference_tensors/full_forward");
        string inputsDir = Path.Combine(refDir, "inputs");

        if (!Directory.Exists(inputsDir))
        {
            _output.WriteLine($"SKIPPED: Python reference dump not found at {inputsDir}");
            _output.WriteLine("  Run: tests/python-reference/.venv/bin/python tests/python-reference/dump_anima_full_forward.py");
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
        Environment.SetEnvironmentVariable("ANIMA_DEBUG_DIR", csDumpDir);

        // Synthetic inputs MUST match dump_anima_full_forward.py:
        //   latent  [1, 16, 1, 32, 32] F32 (5-D Cosmos layout; T = 1 for image mode)
        //   encoder [1, 6, 1024]       F32
        //   timestep = sigma / (sigma + 1) = 0.5 / 1.5 ≈ 0.3333
        // The C# AnimaTransformer.Forward takes a 4-D latent [B, C, H, W] (T is implicit = 1), so
        // we collapse the singleton T dimension when loading.
        Tensor latent = Load5DAs4D(Path.Combine(inputsDir, "latent.bin"), 1, 16, 1, 32, 32);
        Tensor encoderHidden = LoadF32Tensor(Path.Combine(inputsDir, "encoder_hidden.bin"), 1, 6, 1024);
        const float Sigma = 0.5f;
        float timestep = Sigma / (Sigma + 1.0f);

        _output.WriteLine($"Loaded inputs: latent={latent.Shape}, encoder_hidden={encoderHidden.Shape}, t={timestep:F6} (sigma={Sigma})");

        _output.WriteLine($"Loading Anima checkpoint: {Path.GetFileName(ckpt)}");
        Stopwatch sw = Stopwatch.StartNew();
        (AnimaCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            AnimaCheckpointConverter.LoadAndConvert(ckpt);
        sw.Stop();
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms ({converted.Transformer.Count} DiT keys, {converted.LlmAdapter.Count} llm_adapter keys ignored)");

        using (loader)
        {
            // The Anima safetensors stores weights as BF16. Cast to F32 for the CPU forward to match
            // the Python reference's F32 capture path. (Python loads BF16 weights and upcasts captures.)
            Dictionary<string, Tensor> trans = CastWeightsToF32(converted.Transformer);

            AnimaConfig config = AnimaConfig.AnimaPreview3;
            using AnimaTransformer transformer = new(config);
            transformer.LoadWeights(trans);
            _output.WriteLine($"  Weights loaded ({config.NumLayers} blocks, hidden={config.HiddenSize}).");

            using IBackend backend = new CpuBackend();

            sw.Restart();
            _output.WriteLine("\nRunning AnimaTransformer.Forward (CPU, BYPASS llm_adapter — raw encoder features as cross-attn K/V)...");
            Tensor velocity = transformer.Forward(backend, latent, timestep, encoderHidden);
            sw.Stop();
            _output.WriteLine($"Forward done in {sw.ElapsedMilliseconds}ms. Output shape={velocity.Shape}");

            float* vptr = (float*)velocity.DataPointer;
            int n = (int)velocity.ElementCount;
            double sum = 0, sumAbs = 0, sumSq = 0;
            for (int i = 0; i < n; i++) { sum += vptr[i]; sumAbs += Math.Abs(vptr[i]); sumSq += vptr[i] * vptr[i]; }
            double mean = sum / n, absMean = sumAbs / n, std = Math.Sqrt(sumSq / n - mean * mean);
            _output.WriteLine($"  Stats: mean={mean:F6}, abs_mean={absMean:F6}, std={std:F6}");

            velocity.Dispose();
        }

        latent.Dispose();
        encoderHidden.Dispose();
        Environment.SetEnvironmentVariable("ANIMA_DEBUG_DIR", null);

        _output.WriteLine($"\nC# dump written to: {csDumpDir}");
        _output.WriteLine("Now run: tests/python-reference/.venv/bin/python tests/python-reference/diff_anima_layers.py");
    }

    /// <summary>Loads an F32 binary saved as <c>[B, C, T, H, W]</c> and collapses the <c>T=1</c> dim to
    /// produce a 4-D <c>[B, C, H, W]</c> tensor (the format <c>AnimaTransformer.Forward</c> expects).
    /// Since data is contiguous and T=1, the only change is the rank of the shape descriptor.</summary>
    private static unsafe Tensor Load5DAs4D(string path, long b, long c, long t, long h, long w)
    {
        if (t != 1) throw new ArgumentException($"Expected T=1, got T={t}", nameof(t));
        TensorShape shape = new(b, c, h, w);
        Tensor tensor = new(shape, DType.F32);
        byte[] raw = File.ReadAllBytes(path);
        long expected = shape.ElementCount * sizeof(float);
        if (raw.LongLength != expected)
            throw new InvalidDataException($"{path}: expected {expected} bytes, got {raw.LongLength}");
        fixed (byte* src = raw) Buffer.MemoryCopy(src, (void*)tensor.DataPointer, raw.LongLength, raw.LongLength);
        return tensor;
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
