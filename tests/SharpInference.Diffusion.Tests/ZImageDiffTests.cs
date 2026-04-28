using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.Diffusion.Tests;

/// <summary>Layer-by-layer C# vs Python (diffusers) Z-Image transformer diff. Loads the Tongyi-MAI/Z-Image-Turbo diffusers BF16 shards (cast to F32), runs a single forward pass on the SAME synthetic inputs the Python reference saw (<c>tests/python-reference/zimage_reference_tensors/full_forward/inputs/{latent,caption}.bin</c>), and dumps per-layer outputs to <c>Output/zimage_csharp_dump/layers/*.bin</c>. Then a Python diff utility compares the two dumps to find the first divergent layer.</summary>
public sealed class ZImageDiffTests
{
    private static readonly string LinuxRepoRoot =
        Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

    private static readonly string DiffusersTransformerDir =
        Path.Combine(LinuxRepoRoot, "tests", "test-models", "zimage-turbo", "transformer");

    private static readonly string ReferenceDumpDir =
        Path.Combine(LinuxRepoRoot, "tests", "python-reference", "zimage_reference_tensors", "full_forward");

    private static readonly string CSharpDumpDir =
        Path.Combine(LinuxRepoRoot, "Output", "zimage_csharp_dump");

    private readonly ITestOutputHelper _output;
    public ZImageDiffTests(ITestOutputHelper output) => _output = output;

    /// <summary>Runs a single Z-Image transformer forward in C# (CPU backend, F32 weights converted from diffusers BF16 shards) on the same synthetic inputs the Python reference used. Writes layer-by-layer dumps to disk for the Python diff utility to compare against.</summary>
    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer()
    {
        string latentBin = Path.Combine(ReferenceDumpDir, "inputs", "latent.bin");
        string captionBin = Path.Combine(ReferenceDumpDir, "inputs", "caption.bin");
        if (!File.Exists(latentBin) || !File.Exists(captionBin))
        {
            _output.WriteLine($"SKIPPED: reference inputs not found in {ReferenceDumpDir}/inputs/. Run dump_zimage_full_forward.py first.");
            return;
        }
        if (!Directory.Exists(DiffusersTransformerDir))
        {
            _output.WriteLine($"SKIPPED: diffusers transformer dir not found: {DiffusersTransformerDir}");
            return;
        }

        // Configure C# dump dir BEFORE Forward (env var is read once, so set it first).
        Directory.CreateDirectory(Path.Combine(CSharpDumpDir, "layers"));
        Environment.SetEnvironmentVariable("Z_IMAGE_DEBUG_DIR", CSharpDumpDir);
        _output.WriteLine($"C# dump dir: {CSharpDumpDir}");

        Stopwatch sw = Stopwatch.StartNew();

        // Load diffusers shards (3 files), merge, remap to my C# naming, fuse QKV, cast to F32.
        // Loaders must live for the full forward pass — F32 tensors borrow their mmap memory.
        _output.WriteLine($"[1/4] Loading diffusers BF16 shards from {DiffusersTransformerDir}");
        (Dictionary<string, Tensor> weights, List<SafeTensorsLoader> loaders) =
            LoadAndConvertDiffusersShards(DiffusersTransformerDir);
        _output.WriteLine($"  Converted weights: {weights.Count} tensors in {sw.ElapsedMilliseconds}ms");

        try
        {
            // Build the transformer. The synthetic inputs use 32x32 latent (256 image tokens) and 64-token caption — both already multiples of 32, so no padding.
            ZImageConfig config = ZImageConfig.Turbo;
            _output.WriteLine($"  Config: dim={config.HiddenSize}, heads={config.NumHeads}, layers={config.NumLayers}, refiners={config.NumRefinerLayers}");

            sw.Restart();
            ZImageTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            _output.WriteLine($"[2/4] Transformer built in {sw.ElapsedMilliseconds}ms");

            // Read synthetic inputs.
            Tensor latent = LoadF32Bin(latentBin, new TensorShape(1, 16, 32, 32));
            Tensor caption = LoadF32Bin(captionBin, new TensorShape(1, 64, 2560));
            _output.WriteLine($"[3/4] Loaded inputs: latent {latent.Shape}, caption {caption.Shape}");

            // Run forward pass on CPU backend (no FP16/CUDA noise).
            IBackend backend = new CpuBackend();
            sw.Restart();
            const float sigma = 0.5f;  // Same as Python reference; transformer multiplies by t_scale=1000 internally.
            Tensor velocity = transformer.Forward(backend, latent, caption, sigma);
            _output.WriteLine($"[4/4] Forward done in {sw.ElapsedMilliseconds}ms — output {velocity.Shape}");

            // Quick sanity stats on the velocity output.
            ReadOnlySpan<float> v = velocity.AsReadOnlySpan<float>();
            double sum = 0, sumSq = 0;
            for (int i = 0; i < v.Length; i++) { sum += v[i]; sumSq += (double)v[i] * v[i]; }
            double mean = sum / v.Length;
            double std = Math.Sqrt(sumSq / v.Length - mean * mean);
            _output.WriteLine($"  velocity: mean={mean:F6}, std={std:F6}");

            latent.Dispose();
            caption.Dispose();
            velocity.Dispose();
            transformer.Dispose();
        }
        finally
        {
            foreach (SafeTensorsLoader l in loaders) l.Dispose();
        }

        _output.WriteLine($"\n>>> C# dump complete. Run the diff utility:");
        _output.WriteLine($">>>   tests/python-reference/.venv/bin/python tests/python-reference/diff_zimage_layers.py");
    }

    /// <summary>Loads the 3 diffusers safetensor shards, merges them, fuses to_q/to_k/to_v into a single qkv tensor per attention block, and remaps diffusers naming to my C# naming. Returns the converted dict and the loaders that own the underlying mmap memory — the caller MUST dispose the loaders only after Forward completes (the dict's tensors borrow mmap pages).</summary>
    private static (Dictionary<string, Tensor> Weights, List<SafeTensorsLoader> Loaders)
        LoadAndConvertDiffusersShards(string transformerDir)
    {
        string[] shards = Directory.GetFiles(transformerDir, "diffusion_pytorch_model-*.safetensors")
            .OrderBy(s => s).ToArray();
        if (shards.Length == 0)
            throw new FileNotFoundException($"No diffusers shards found in {transformerDir}");

        Dictionary<string, Tensor> raw = new();
        List<SafeTensorsLoader> loaders = new();
        foreach (string shard in shards)
        {
            SafeTensorsLoader loader = new();
            loader.Load(shard);
            loaders.Add(loader);
            foreach (KeyValuePair<string, Tensor> kvp in loader.GetAllTensors())
                raw[kvp.Key] = kvp.Value;
        }

        // Cast non-F32 weights only. F32 tensors borrow the mmap directly (avoids ~22 GB redundant allocation
        // when the diffusers shards are already stored as F32, which Tongyi-MAI/Z-Image-Turbo is).
        Dictionary<string, Tensor> f32 = new(raw.Count);
        foreach (KeyValuePair<string, Tensor> kvp in raw)
            f32[kvp.Key] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);

        Dictionary<string, Tensor> result = new();

        CopyIfPresent(f32, result, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.weight");
        CopyIfPresent(f32, result, "t_embedder.mlp.0.bias", "t_embedder.mlp.0.bias");
        CopyIfPresent(f32, result, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.weight");
        CopyIfPresent(f32, result, "t_embedder.mlp.2.bias", "t_embedder.mlp.2.bias");

        CopyIfPresent(f32, result, "cap_embedder.0.weight", "cap_embedder.0.weight");
        CopyIfPresent(f32, result, "cap_embedder.1.weight", "cap_embedder.1.weight");
        CopyIfPresent(f32, result, "cap_embedder.1.bias", "cap_embedder.1.bias");

        CopyIfPresent(f32, result, "all_x_embedder.2-1.weight", "x_embedder.weight");
        CopyIfPresent(f32, result, "all_x_embedder.2-1.bias", "x_embedder.bias");

        CopyIfPresent(f32, result, "all_final_layer.2-1.adaLN_modulation.1.weight", "final_layer.adaLN_modulation.1.weight");
        CopyIfPresent(f32, result, "all_final_layer.2-1.adaLN_modulation.1.bias", "final_layer.adaLN_modulation.1.bias");
        CopyIfPresent(f32, result, "all_final_layer.2-1.linear.weight", "final_layer.linear.weight");
        CopyIfPresent(f32, result, "all_final_layer.2-1.linear.bias", "final_layer.linear.bias");

        CopyIfPresent(f32, result, "cap_pad_token", "cap_pad_token");
        CopyIfPresent(f32, result, "x_pad_token", "x_pad_token");

        foreach (string prefix in EnumerateBlockPrefixes(f32))
            ConvertBlock(f32, result, prefix);

        return (result, loaders);
    }

    private static IEnumerable<string> EnumerateBlockPrefixes(Dictionary<string, Tensor> dict)
    {
        // A "block prefix" is one of: layers.{i}, noise_refiner.{i}, context_refiner.{i}.
        HashSet<string> prefixes = new();
        foreach (string key in dict.Keys)
        {
            if (key.StartsWith("layers.") || key.StartsWith("noise_refiner.") || key.StartsWith("context_refiner."))
            {
                int dot1 = key.IndexOf('.');
                int dot2 = key.IndexOf('.', dot1 + 1);
                if (dot2 > 0)
                    prefixes.Add(key.Substring(0, dot2));
            }
        }
        return prefixes;
    }

    private static void ConvertBlock(Dictionary<string, Tensor> src, Dictionary<string, Tensor> dst, string prefix)
    {
        // Fused QKV: concat to_q, to_k, to_v along axis 0 → [3*hidden, hidden].
        Tensor toQ = src[$"{prefix}.attention.to_q.weight"];
        Tensor toK = src[$"{prefix}.attention.to_k.weight"];
        Tensor toV = src[$"{prefix}.attention.to_v.weight"];
        dst[$"{prefix}.attention.qkv.weight"] = ConcatAxis0(toQ, toK, toV);

        // Output projection: to_out.0 → out
        dst[$"{prefix}.attention.out.weight"] = src[$"{prefix}.attention.to_out.0.weight"];

        // QK norm rename
        dst[$"{prefix}.attention.q_norm.weight"] = src[$"{prefix}.attention.norm_q.weight"];
        dst[$"{prefix}.attention.k_norm.weight"] = src[$"{prefix}.attention.norm_k.weight"];

        // Surrounding RMS norms (same name)
        dst[$"{prefix}.attention_norm1.weight"] = src[$"{prefix}.attention_norm1.weight"];
        dst[$"{prefix}.attention_norm2.weight"] = src[$"{prefix}.attention_norm2.weight"];
        dst[$"{prefix}.ffn_norm1.weight"] = src[$"{prefix}.ffn_norm1.weight"];
        dst[$"{prefix}.ffn_norm2.weight"] = src[$"{prefix}.ffn_norm2.weight"];

        // SwiGLU FFN
        dst[$"{prefix}.feed_forward.w1.weight"] = src[$"{prefix}.feed_forward.w1.weight"];
        dst[$"{prefix}.feed_forward.w2.weight"] = src[$"{prefix}.feed_forward.w2.weight"];
        dst[$"{prefix}.feed_forward.w3.weight"] = src[$"{prefix}.feed_forward.w3.weight"];

        // AdaLN (only present on layers and noise_refiner — context_refiner has modulation=False)
        if (src.TryGetValue($"{prefix}.adaLN_modulation.0.weight", out Tensor? adaW))
            dst[$"{prefix}.adaLN_modulation.0.weight"] = adaW;
        if (src.TryGetValue($"{prefix}.adaLN_modulation.0.bias", out Tensor? adaB))
            dst[$"{prefix}.adaLN_modulation.0.bias"] = adaB;
    }

    private static void CopyIfPresent(Dictionary<string, Tensor> src, Dictionary<string, Tensor> dst, string srcKey, string dstKey)
    {
        if (src.TryGetValue(srcKey, out Tensor? t))
            dst[dstKey] = t;
    }

    /// <summary>Concatenates 3 [out, in] F32 tensors along axis 0, producing a [3*out, in] tensor with [Q | K | V] layout.</summary>
    private static unsafe Tensor ConcatAxis0(Tensor a, Tensor b, Tensor c)
    {
        long outA = a.Shape[0];
        long inA = a.Shape[1];
        long total = outA + b.Shape[0] + c.Shape[0];
        TensorShape outShape = new TensorShape(total, inA);
        Tensor result = new Tensor(outShape, DType.F32);

        long bytesA = outA * inA * sizeof(float);
        long bytesB = b.Shape[0] * b.Shape[1] * sizeof(float);
        long bytesC = c.Shape[0] * c.Shape[1] * sizeof(float);

        byte* dst = (byte*)result.DataPointer;
        Buffer.MemoryCopy(a.DataPointer, dst, bytesA, bytesA);
        Buffer.MemoryCopy(b.DataPointer, dst + bytesA, bytesB, bytesB);
        Buffer.MemoryCopy(c.DataPointer, dst + bytesA + bytesB, bytesC, bytesC);
        return result;
    }

    private static unsafe Tensor LoadF32Bin(string path, TensorShape shape)
    {
        long expectedBytes = shape.ElementCount * sizeof(float);
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.LongLength != expectedBytes)
            throw new InvalidDataException($"Expected {expectedBytes} bytes for shape {shape}, got {bytes.LongLength} from {path}");

        Tensor t = new Tensor(shape, DType.F32);
        fixed (byte* src = bytes)
        {
            Buffer.MemoryCopy(src, t.DataPointer, expectedBytes, expectedBytes);
        }
        return t;
    }
}
