using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Layer-by-layer diff harness for the Qwen-Image VAE decoder (Anima's image-output VAE).
/// Loads the Python-saved synthetic latent, runs the C# <see cref="QwenImageVaeDecoder"/> with
/// <c>QWEN_VAE_DEBUG_DIR</c> set to dump every layer, then leaves the dump for
/// <c>diff_qwen_image_vae_layers.py</c> to compare against the diffusers
/// <c>AutoencoderKLQwenImage</c> reference.
///
/// <para>The C# decoder is fed the <b>raw</b> latent (pre-rescale); its internal <c>UndoScaling</c>
/// applies the per-channel <c>latents_mean</c> / <c>latents_std</c> from <see cref="VaeConfig.QwenImage"/>.
/// The Python reference rescales OUTSIDE the model and passes the rescaled tensor into
/// <c>vae.decode()</c>. The first dump (<c>post_quant_conv</c>) compares the output of the 1×1 conv
/// on those two paths — if it diverges, the per-channel rescale itself is wrong; if it agrees and a
/// later layer diverges, the bug is downstream in the decoder body.</para>
///
/// Skips cleanly when the Python reference dump or the Qwen-Image VAE checkpoint is missing.</summary>
[Trait("Category", "Integration")]
public unsafe class QwenImageVaeDiffTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageVaeDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Decoder_Matches_PythonReference_LayerByLayer_Cpu() =>
        RunDiff(dumpDirName: "qwen_image_vae_csharp_dump");

    private void RunDiff(string dumpDirName)
    {
        string repoRoot = RepoRoot.Path;
        string refDir = Path.Combine(repoRoot, "tests/python-reference/qwen_image_vae_reference");
        string inputsDir = Path.Combine(refDir, "inputs");

        if (!Directory.Exists(inputsDir))
        {
            _output.WriteLine($"SKIPPED: Python reference dump not found at {inputsDir}");
            _output.WriteLine("  Run: tests/python-reference/.venv/bin/python tests/python-reference/dump_qwen_image_vae.py");
            return;
        }

        string vaePath = TestPaths.QwenImage.Vae;
        if (!File.Exists(vaePath))
        {
            _output.WriteLine($"SKIPPED: Qwen-Image VAE checkpoint not found: {vaePath}");
            return;
        }

        string csDumpDir = Path.Combine(repoRoot, "Output", dumpDirName);
        if (Directory.Exists(csDumpDir)) Directory.Delete(csDumpDir, recursive: true);
        Directory.CreateDirectory(Path.Combine(csDumpDir, "layers"));
        Environment.SetEnvironmentVariable("QWEN_VAE_DEBUG_DIR", csDumpDir);

        // Synthetic input from dump_qwen_image_vae.py: raw latent [1, 16, 32, 32] F32 (pre-rescale).
        // We pass this directly to the C# decoder; its UndoScaling applies the per-channel
        // latents_mean / latents_std internally (mirroring what the diffusers pipeline does
        // OUTSIDE vae.decode in the Python reference).
        Tensor latentRaw = LoadF32Tensor(Path.Combine(inputsDir, "latent_raw.bin"), 1, 16, 32, 32);
        _output.WriteLine($"Loaded latent_raw: {latentRaw.Shape}");

        _output.WriteLine($"Loading Qwen-Image VAE: {Path.GetFileName(vaePath)}");
        Stopwatch sw = Stopwatch.StartNew();
        using SafeTensorsLoader loader = new();
        loader.Load(vaePath);
        Dictionary<string, Tensor> weights = CastWeightsToF32(loader.GetAllTensors());
        sw.Stop();
        _output.WriteLine($"  Loaded {weights.Count} tensors in {sw.ElapsedMilliseconds}ms");

        using QwenImageVaeDecoder vae = new(VaeConfig.QwenImage);
        vae.LoadWeights(weights);
        _output.WriteLine("  Decoder weights bound.");

        using IBackend backend = new CpuBackend();

        sw.Restart();
        _output.WriteLine("\nRunning QwenImageVaeDecoder.Decode (CPU, F32, layer dumps enabled)...");
        Tensor decoded = vae.Decode(backend, latentRaw);
        sw.Stop();
        _output.WriteLine($"Decode done in {sw.ElapsedMilliseconds}ms. Output shape={decoded.Shape}");

        float* dptr = (float*)decoded.DataPointer;
        int n = (int)decoded.ElementCount;
        double sum = 0, sumAbs = 0, sumSq = 0;
        float minV = float.MaxValue, maxV = float.MinValue;
        for (int i = 0; i < n; i++)
        {
            float v = dptr[i];
            sum += v; sumAbs += Math.Abs(v); sumSq += v * v;
            if (v < minV) minV = v;
            if (v > maxV) maxV = v;
        }
        double mean = sum / n, absMean = sumAbs / n, std = Math.Sqrt(sumSq / n - mean * mean);
        _output.WriteLine($"  Stats: mean={mean:F6}, abs_mean={absMean:F6}, std={std:F6}, range=[{minV:F4}, {maxV:F4}]");

        decoded.Dispose();
        latentRaw.Dispose();
        Environment.SetEnvironmentVariable("QWEN_VAE_DEBUG_DIR", null);

        _output.WriteLine($"\nC# dump written to: {csDumpDir}");
        _output.WriteLine("Now run: tests/python-reference/.venv/bin/python tests/python-reference/diff_qwen_image_vae_layers.py");
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
