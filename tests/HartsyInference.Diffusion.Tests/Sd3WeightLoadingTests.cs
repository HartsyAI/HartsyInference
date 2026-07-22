using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SD3 weight loading tests. Validates that SD3 safetensors checkpoints load correctly into all model components (MMDiT, CLIP-L, CLIP-G, T5-XXL, VAE). Supports both diffusers format and single-file checkpoints.</summary>
public class Sd3WeightLoadingTests
{
    /// <summary>Path to SD3 in HuggingFace diffusers layout.</summary>
    private static string Sd3DiffusersDir => TestPaths.Sd3.DiffusersDir;

    /// <summary>Path to a single-file SD3 safetensors checkpoint.</summary>
    private static string Sd3SingleFilePath => TestPaths.Sd3.SingleFile;

    private readonly ITestOutputHelper _output;

    public Sd3WeightLoadingTests(ITestOutputHelper output) => _output = output;

    #region Single-File Checkpoint Tests

    [Fact]
    public void SingleFile_EnumerateKeys()
    {
        if (!File.Exists(Sd3SingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Single-file checkpoint not found: {Sd3SingleFilePath}");
            return;
        }

        _output.WriteLine($"Loading: {Path.GetFileName(Sd3SingleFilePath)}");
        using SafeTensorsLoader loader = new();
        loader.Load(Sd3SingleFilePath);

        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors = loader.Descriptors;
        _output.WriteLine($"Total tensors: {descriptors.Count}");

        // Group by top-level prefix
        Dictionary<string, int> prefixCounts = [];
        foreach (string key in descriptors.Keys)
        {
            string prefix = key.Contains('.') ? key[..key.IndexOf('.')] : key;
            prefixCounts.TryGetValue(prefix, out int count);
            prefixCounts[prefix] = count + 1;
        }

        _output.WriteLine("\n=== Key Prefix Groups ===");
        foreach (KeyValuePair<string, int> kvp in prefixCounts.OrderByDescending(x => x.Value))
        {
            _output.WriteLine($"  {kvp.Key}: {kvp.Value} tensors");
        }

        // SD3 should have diffusion model keys
        bool hasDiffusion = descriptors.Keys.Any(k => k.StartsWith("model.diffusion_model."));
        _output.WriteLine($"\nHas diffusion model: {hasDiffusion}");

        Assert.True(descriptors.Count > 0, "Checkpoint should contain tensors");
    }

    [Fact]
    public void SingleFile_ConvertAndVerifyTransformerKeys()
    {
        if (!File.Exists(Sd3SingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {Sd3SingleFilePath}");
            return;
        }

        _output.WriteLine($"Loading and converting: {Path.GetFileName(Sd3SingleFilePath)}");
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd3CheckpointConverter.LoadAndConvert(Sd3SingleFilePath);

        using (loader)
        {
            _output.WriteLine($"Transformer keys: {converted.Transformer.Count}");
            _output.WriteLine($"CLIP-L keys: {converted.ClipL.Count}");
            _output.WriteLine($"CLIP-G keys: {converted.ClipG.Count}");
            _output.WriteLine($"T5 keys: {converted.T5.Count}");
            _output.WriteLine($"VAE keys: {converted.Vae.Count}");

            // Verify essential transformer keys exist
            string[] requiredKeys =
            [
                "pos_embed.proj.weight",
                "pos_embed.proj.bias",
                "context_embedder.weight",
                "context_embedder.bias",
                "time_text_embed.timestep_embedder.linear_1.weight",
                "time_text_embed.text_embedder.linear_1.weight",
                "norm_out.linear.weight",
                "proj_out.weight",
            ];

            List<string> missing = [];
            foreach (string key in requiredKeys)
            {
                if (!converted.Transformer.ContainsKey(key))
                    missing.Add(key);
            }

            if (missing.Count > 0)
            {
                _output.WriteLine($"\nMissing required keys:");
                foreach (string key in missing)
                    _output.WriteLine($"  {key}");
            }
            Assert.Empty(missing);

            // Verify QKV was split (should have to_q, not qkv)
            bool hasToQ = converted.Transformer.ContainsKey("transformer_blocks.0.attn.to_q.weight");
            bool hasFusedQkv = converted.Transformer.Keys.Any(k => k.Contains("qkv"));
            _output.WriteLine($"\nHas split Q projection: {hasToQ}");
            _output.WriteLine($"Has fused QKV (should be false): {hasFusedQkv}");

            Assert.True(hasToQ, "QKV should be split into separate to_q/to_k/to_v");
            Assert.False(hasFusedQkv, "Fused QKV keys should not remain after conversion");

            // Detect and report depth
            int depth = Sd3CheckpointConverter.DetectDepth(converted.Transformer);
            _output.WriteLine($"\nDetected depth: {depth}");
            Assert.True(depth > 0, "Should detect at least 1 transformer block");

            // Verify block 0 has all expected sub-keys
            string[] block0Keys =
            [
                "transformer_blocks.0.norm1.linear.weight",
                "transformer_blocks.0.attn.to_q.weight",
                "transformer_blocks.0.attn.to_k.weight",
                "transformer_blocks.0.attn.to_v.weight",
                "transformer_blocks.0.attn.to_out.0.weight",
                "transformer_blocks.0.attn.add_q_proj.weight",
                "transformer_blocks.0.attn.add_k_proj.weight",
                "transformer_blocks.0.attn.add_v_proj.weight",
                "transformer_blocks.0.attn.to_add_out.weight",
                "transformer_blocks.0.ff.net.0.proj.weight",
                "transformer_blocks.0.ff.net.2.weight",
                "transformer_blocks.0.norm1_context.linear.weight",
                "transformer_blocks.0.ff_context.net.0.proj.weight",
            ];

            List<string> missingBlock0 = [];
            foreach (string key in block0Keys)
            {
                if (!converted.Transformer.ContainsKey(key))
                    missingBlock0.Add(key);
            }

            if (missingBlock0.Count > 0)
            {
                _output.WriteLine($"\nMissing block 0 keys:");
                foreach (string key in missingBlock0)
                    _output.WriteLine($"  {key}");
            }
            Assert.Empty(missingBlock0);

            // Print sample shapes
            _output.WriteLine("\n=== Sample Tensor Shapes ===");
            foreach (string key in block0Keys.Take(5))
            {
                if (converted.Transformer.TryGetValue(key, out Tensor? t))
                    _output.WriteLine($"  {key}: [{string.Join(", ", Enumerable.Range(0, t.Shape.Rank).Select(d => t.Shape[d]))}]");
            }
        }
    }

    #endregion

    #region Diffusers Format Tests

    [Fact]
    public void DiffusersFormat_LoadTransformer()
    {
        string transformerDir = Path.Combine(Sd3DiffusersDir, "transformer");
        string safetensorsPath = Path.Combine(transformerDir, "diffusion_pytorch_model.safetensors");

        // Also try FP16 variant
        if (!File.Exists(safetensorsPath))
            safetensorsPath = Path.Combine(transformerDir, "diffusion_pytorch_model.fp16.safetensors");

        if (!File.Exists(safetensorsPath))
        {
            _output.WriteLine($"SKIPPED: Transformer safetensors not found in {transformerDir}");
            return;
        }

        _output.WriteLine($"Loading transformer: {safetensorsPath}");
        using SafeTensorsLoader loader = new();
        loader.Load(safetensorsPath);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        // Cast to F32 if needed
        Dictionary<string, Tensor> f32Weights = CastWeightsToF32(weights);

        _output.WriteLine($"Total tensors: {f32Weights.Count}");

        // Create transformer with medium config and try loading
        Sd3Config config = Sd3Config.Medium;
        Sd3Transformer transformer = new Sd3Transformer(config);

        try
        {
            transformer.LoadWeights(f32Weights);
            _output.WriteLine("Transformer weights loaded successfully!");
        }
        catch (KeyNotFoundException ex)
        {
            _output.WriteLine($"Missing key: {ex.Message}");
            Assert.Fail($"Weight loading failed: {ex.Message}");
        }

        transformer.Dispose();
    }

    #endregion

    #region Config Auto-Detection Tests

    [Fact]
    public void Sd3Config_Medium_HasCorrectValues()
    {
        Sd3Config config = Sd3Config.Medium;
        Assert.Equal(24, config.Depth);
        Assert.Equal(1536, config.HiddenSize);
        Assert.Equal(24, config.NumHeads);
        Assert.Equal(64, config.HeadDim);
        Assert.Equal(2, config.PatchSize);
        Assert.Equal(16, config.InChannels);
        Assert.Equal(4096, config.JointAttentionDim);
        Assert.Equal(2048, config.PooledProjectionDim);
    }

    [Fact]
    public void VaeConfig_Sd3_Has16Channels()
    {
        VaeConfig config = VaeConfig.Sd3;
        Assert.Equal(16, config.LatentChannels);
        Assert.InRange(config.ScalingFactor, 1.530f, 1.531f);
        Assert.NotNull(config.ShiftFactor);
        Assert.InRange(config.ShiftFactor!.Value, 0.060f, 0.061f);
        Assert.False(config.UsePostQuantConv);
        Assert.False(config.UseQuantConv);
    }

    #endregion

    /// <summary>Casts all tensors to F32 if they are F16 or BF16.</summary>
    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            if (kvp.Value.DType == DType.F32)
            {
                result[kvp.Key] = kvp.Value;
            }
            else
            {
                result[kvp.Key] = CastToF32(kvp.Value);
            }
        }
        return result;
    }

    private static unsafe Tensor CastToF32(Tensor source)
    {
        Tensor f32 = new Tensor(source.Shape, DType.F32);
        long count = source.ElementCount;
        float* dstPtr = (float*)f32.DataPointer;

        if (source.DType == DType.F16)
        {
            Half* srcPtr = (Half*)source.DataPointer;
            for (long i = 0; i < count; i++)
                dstPtr[i] = (float)srcPtr[i];
        }
        else if (source.DType == DType.BF16)
        {
            ushort* srcPtr = (ushort*)source.DataPointer;
            for (long i = 0; i < count; i++)
            {
                uint bits = (uint)srcPtr[i] << 16;
                dstPtr[i] = BitConverter.UInt32BitsToSingle(bits);
            }
        }
        else
        {
            throw new NotSupportedException($"Cannot cast {source.DType} to F32");
        }

        return f32;
    }
}
