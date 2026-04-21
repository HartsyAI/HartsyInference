using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.ModelHandler.CheckpointConverters;

namespace SharpInference.Diffusion.Tests;

/// <summary>
/// SD1.5 single-file checkpoint conversion and weight loading tests.
/// Validates that SD1.5 safetensors checkpoints load correctly into all model components (UNet, CLIP-L, VAE).
/// Set SD15_SINGLE_FILE_PATH environment variable or edit the default path below.
/// </summary>
public class Sd15WeightLoadingTests
{
    private static readonly string Sd15SingleFilePath =
        Environment.GetEnvironmentVariable("SD15_SINGLE_FILE_PATH")
        ?? @"C:\Users\AI Overlord\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\v1-5-pruned-emaonly.safetensors";

    private readonly ITestOutputHelper _output;

    public Sd15WeightLoadingTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Loads a single-file SD1.5 checkpoint, converts keys to diffusers format via Sd15CheckpointConverter,
    /// and loads all three components (UNet, CLIP-L, VAE).
    /// </summary>
    [Fact]
    public void SingleFile_ConvertAndLoadAllComponents()
    {
        if (!File.Exists(Sd15SingleFilePath))
        {
            _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {Sd15SingleFilePath}");
            return;
        }

        _output.WriteLine($"Loading and converting: {Sd15SingleFilePath}");
        (Sd15CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd15CheckpointConverter.LoadAndConvert(Sd15SingleFilePath);

        using (loader)
        {
            _output.WriteLine($"  UNet keys: {converted.UNet.Count}");
            _output.WriteLine($"  CLIP-L keys: {converted.ClipL.Count}");
            _output.WriteLine($"  VAE keys: {converted.Vae.Count}");

            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);
            Dictionary<string, Tensor> clipLF32 = CastWeightsToF32(converted.ClipL);
            Dictionary<string, Tensor> vaeF32 = CastWeightsToF32(converted.Vae);

            // --- UNet ---
            _output.WriteLine("\nLoading UNet from converted keys...");
            UNet unet = new(UNetConfig.Sd15);
            try
            {
                unet.LoadWeights(unetF32);
                _output.WriteLine("  UNet: SUCCESS");
            }
            catch (KeyNotFoundException ex)
            {
                _output.WriteLine($"  UNet: FAILED — {ex.Message}");
                _output.WriteLine("  Available keys with similar prefix:");
                string missingKey = ex.Message;
                string prefix = missingKey.Contains('.') ? missingKey[..missingKey.LastIndexOf('.')] : "";
                foreach (string key in unetF32.Keys.Where(k => k.Contains(prefix)).Take(10))
                    _output.WriteLine($"    {key}");
                Assert.Fail($"Missing UNet key: {ex.Message}");
            }

            // Validate critical UNet shapes
            Assert.True(unetF32.ContainsKey("conv_in.weight"), "UNet missing conv_in.weight");
            Assert.Equal(320, (int)unetF32["conv_in.weight"].Shape[0]);
            Assert.Equal(4, (int)unetF32["conv_in.weight"].Shape[1]);

            // SD1.5 has NO add_embedding (no ADM conditioning)
            Assert.False(unetF32.ContainsKey("add_embedding.linear_1.weight"),
                "SD1.5 should not have add_embedding");

            // Validate time_embedding exists
            Assert.True(unetF32.ContainsKey("time_embedding.linear_1.weight"));

            // --- CLIP-L ---
            _output.WriteLine("\nLoading CLIP-L from converted keys...");
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd15);
            try
            {
                clipL.LoadWeights(clipLF32, "text_model");
                _output.WriteLine("  CLIP-L: SUCCESS");
            }
            catch (KeyNotFoundException ex)
            {
                _output.WriteLine($"  CLIP-L: FAILED — {ex.Message}");
                _output.WriteLine("  Available CLIP-L keys:");
                foreach (string key in clipLF32.Keys.OrderBy(k => k).Take(20))
                    _output.WriteLine($"    {key} {clipLF32[key].Shape}");
                Assert.Fail($"Missing CLIP-L key: {ex.Message}");
            }

            // Validate CLIP-L embedding shape (768-dim for SD1.5)
            Assert.True(clipLF32.ContainsKey("text_model.embeddings.token_embedding.weight"));
            Assert.Equal(768, (int)clipLF32["text_model.embeddings.token_embedding.weight"].Shape[1]);

            // SD1.5 CLIP-L should NOT have text_projection
            Assert.False(clipLF32.ContainsKey("text_projection.weight"),
                "SD1.5 CLIP-L should not have text_projection");

            // --- VAE ---
            _output.WriteLine("\nLoading VAE from converted keys...");
            VaeDecoder vae = new(VaeConfig.Sd15);
            try
            {
                vae.LoadWeights(vaeF32);
                _output.WriteLine("  VAE: SUCCESS");
            }
            catch (KeyNotFoundException ex)
            {
                _output.WriteLine($"  VAE: FAILED — {ex.Message}");
                _output.WriteLine("  Available VAE keys:");
                foreach (string key in vaeF32.Keys.OrderBy(k => k).Take(30))
                    _output.WriteLine($"    {key} {vaeF32[key].Shape}");
                Assert.Fail($"Missing VAE key: {ex.Message}");
            }

            Assert.True(vaeF32.ContainsKey("post_quant_conv.weight"), "VAE missing post_quant_conv");
            Assert.True(vaeF32.ContainsKey("decoder.conv_in.weight"), "VAE missing decoder.conv_in");

            _output.WriteLine("\n=== All 3 components loaded from single-file SD1.5 checkpoint: PASSED ===");
        }
    }

    /// <summary>
    /// Validates all expected UNet keys are present after conversion.
    /// </summary>
    [Fact]
    public void SingleFile_ConvertedUNetHasAllExpectedKeys()
    {
        if (!File.Exists(Sd15SingleFilePath))
        {
            _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {Sd15SingleFilePath}");
            return;
        }

        (Sd15CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd15CheckpointConverter.LoadAndConvert(Sd15SingleFilePath);

        using (loader)
        {
            Dictionary<string, Tensor> unet = converted.UNet;
            _output.WriteLine($"Converted UNet key count: {unet.Count}");

            int missing = 0;

            // Core structural keys
            string[] requiredKeys =
            [
                "conv_in.weight", "conv_in.bias",
                "time_embedding.linear_1.weight", "time_embedding.linear_1.bias",
                "time_embedding.linear_2.weight", "time_embedding.linear_2.bias",
                "conv_norm_out.weight", "conv_norm_out.bias",
                "conv_out.weight", "conv_out.bias",
            ];

            foreach (string key in requiredKeys)
                CheckKeyExists(unet, key, ref missing);

            // Down blocks: 4 levels, 2 resnets each
            for (int level = 0; level < 4; level++)
            {
                for (int r = 0; r < 2; r++)
                {
                    string prefix = $"down_blocks.{level}.resnets.{r}";
                    CheckKeyExists(unet, $"{prefix}.norm1.weight", ref missing);
                    CheckKeyExists(unet, $"{prefix}.conv1.weight", ref missing);
                    CheckKeyExists(unet, $"{prefix}.norm2.weight", ref missing);
                    CheckKeyExists(unet, $"{prefix}.conv2.weight", ref missing);
                    CheckKeyExists(unet, $"{prefix}.time_emb_proj.weight", ref missing);
                }

                // Downsamplers for levels 0, 1, 2 (not level 3)
                if (level < 3)
                    CheckKeyExists(unet, $"down_blocks.{level}.downsamplers.0.conv.weight", ref missing);
            }

            // Attention blocks on down levels 0, 1, 2 (not level 3)
            // SD1.5: DownBlockHasAttention = [true, true, true, false]
            for (int level = 0; level < 3; level++)
            {
                for (int a = 0; a < 2; a++)
                {
                    string attPrefix = $"down_blocks.{level}.attentions.{a}";
                    CheckKeyExists(unet, $"{attPrefix}.norm.weight", ref missing);
                    CheckKeyExists(unet, $"{attPrefix}.proj_in.weight", ref missing);
                    CheckKeyExists(unet, $"{attPrefix}.proj_out.weight", ref missing);

                    // SD1.5: 1 transformer block per attention
                    string tbPrefix = $"{attPrefix}.transformer_blocks.0";
                    CheckKeyExists(unet, $"{tbPrefix}.attn1.to_q.weight", ref missing);
                    CheckKeyExists(unet, $"{tbPrefix}.attn2.to_q.weight", ref missing);
                    CheckKeyExists(unet, $"{tbPrefix}.ff.net.0.proj.weight", ref missing);
                }
            }

            // Mid block
            CheckKeyExists(unet, "mid_block.resnets.0.norm1.weight", ref missing);
            CheckKeyExists(unet, "mid_block.resnets.1.norm1.weight", ref missing);
            CheckKeyExists(unet, "mid_block.attentions.0.norm.weight", ref missing);
            CheckKeyExists(unet, "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.weight", ref missing);
            CheckKeyExists(unet, "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.weight", ref missing);

            // Up blocks: 4 levels, 3 resnets each
            // UpBlockHasAttention = [false, true, true, true]
            for (int level = 0; level < 4; level++)
            {
                for (int r = 0; r < 3; r++)
                {
                    string prefix = $"up_blocks.{level}.resnets.{r}";
                    CheckKeyExists(unet, $"{prefix}.norm1.weight", ref missing);
                    CheckKeyExists(unet, $"{prefix}.conv1.weight", ref missing);
                }

                // Upsamplers for levels 0, 1, 2 (not level 3)
                if (level < 3)
                    CheckKeyExists(unet, $"up_blocks.{level}.upsamplers.0.conv.weight", ref missing);

                // Attention on up levels 1, 2, 3 (not level 0)
                if (level > 0)
                {
                    for (int a = 0; a < 3; a++)
                    {
                        string attPrefix = $"up_blocks.{level}.attentions.{a}";
                        CheckKeyExists(unet, $"{attPrefix}.norm.weight", ref missing);
                        CheckKeyExists(unet, $"{attPrefix}.transformer_blocks.0.attn1.to_q.weight", ref missing);
                    }
                }
            }

            _output.WriteLine($"\nTotal missing keys: {missing}");
            Assert.Equal(0, missing);
            _output.WriteLine("All expected SD1.5 UNet keys present after conversion.");
        }
    }

    /// <summary>
    /// Converts a single-file checkpoint, loads UNet, runs a forward pass, validates output.
    /// </summary>
    [Fact]
    public unsafe void SingleFile_UNetForwardPass()
    {
        if (!File.Exists(Sd15SingleFilePath))
        {
            _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {Sd15SingleFilePath}");
            return;
        }

        _output.WriteLine($"Loading and converting: {Sd15SingleFilePath}");
        (Sd15CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd15CheckpointConverter.LoadAndConvert(Sd15SingleFilePath);

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);

            UNet unet = new(UNetConfig.Sd15);
            unet.LoadWeights(unetF32);

            using CpuBackend backend = new();

            // SD1.5: 512x512 → 64x64 latent, use small 16x16 for speed
            TensorShape latentShape = new(1, 4, 16, 16);
            Tensor latent = SeedGenerator.CreateNoise(latentShape, 42);

            // SD1.5: single CLIP context [1, 77, 768]
            Tensor textEmb = new(new TensorShape(1, 77, 768), DType.F32);

            // SD1.5: no pooled embedding, no size conditioning
            _output.WriteLine("Running UNet forward pass with converted weights...");
            Tensor output = unet.Forward(backend, latent, 999.0f, textEmb);

            float mean = ComputeMean(output);
            float std = ComputeStd(output);
            _output.WriteLine($"  Output: shape={output.Shape}, mean={mean:F6}, std={std:F6}");

            Assert.False(float.IsNaN(mean), "Output contains NaN");
            Assert.False(float.IsInfinity(std), "Output contains Inf");
            Assert.True(std > 0.001f, $"Output std too small: {std}");
            Assert.Equal(4, (int)output.Shape[1]);
            Assert.Equal(16, (int)output.Shape[2]);

            _output.WriteLine("SD1.5 UNet forward pass: PASSED");

            output.Dispose();
            latent.Dispose();
            textEmb.Dispose();
        }
    }

    /// <summary>
    /// Enumerates all keys in the single-file checkpoint to validate SD1.5 key patterns.
    /// </summary>
    [Fact]
    public void SingleFile_EnumerateKeys()
    {
        if (!File.Exists(Sd15SingleFilePath))
        {
            _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {Sd15SingleFilePath}");
            return;
        }

        using SafeTensorsLoader loader = new();
        loader.Load(Sd15SingleFilePath);
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

        foreach (KeyValuePair<string, int> kvp in prefixCounts.OrderByDescending(x => x.Value))
            _output.WriteLine($"  {kvp.Key}: {kvp.Value} tensors");

        // SD1.5-specific checks
        bool hasModel = descriptors.Keys.Any(k => k.StartsWith("model.diffusion_model."));
        bool hasCondStage = descriptors.Keys.Any(k => k.StartsWith("cond_stage_model."));
        bool hasVae = descriptors.Keys.Any(k => k.StartsWith("first_stage_model."));
        bool hasConditioner = descriptors.Keys.Any(k => k.StartsWith("conditioner."));

        _output.WriteLine($"\n  Has model.diffusion_model (UNet): {hasModel}");
        _output.WriteLine($"  Has cond_stage_model (SD1.5 CLIP): {hasCondStage}");
        _output.WriteLine($"  Has first_stage_model (VAE): {hasVae}");
        _output.WriteLine($"  Has conditioner (SDXL-style): {hasConditioner}");

        Assert.True(hasModel, "SD1.5 checkpoint should have model.diffusion_model.*");
        Assert.True(hasCondStage, "SD1.5 checkpoint should have cond_stage_model.*");
        Assert.True(hasVae, "SD1.5 checkpoint should have first_stage_model.*");
        Assert.False(hasConditioner, "SD1.5 checkpoint should NOT have conditioner.* (that's SDXL)");
    }

    #region Helpers

    private void CheckKeyExists(Dictionary<string, Tensor> weights, string key, ref int missingCount)
    {
        if (!weights.ContainsKey(key))
        {
            _output.WriteLine($"  MISSING: {key}");
            missingCount++;
        }
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            f32[kvp.Key] = (kvp.Value.DType == DType.F16 || kvp.Value.DType == DType.BF16)
                ? kvp.Value.CastTo(DType.F32)
                : kvp.Value;
        }
        return f32;
    }

    private static unsafe float ComputeMean(Tensor tensor)
    {
        float* ptr = (float*)tensor.DataPointer;
        long count = tensor.ElementCount;
        double sum = 0;
        for (long i = 0; i < count; i++) sum += ptr[i];
        return (float)(sum / count);
    }

    private static unsafe float ComputeStd(Tensor tensor)
    {
        float* ptr = (float*)tensor.DataPointer;
        long count = tensor.ElementCount;
        double sum = 0, sumSq = 0;
        for (long i = 0; i < count; i++)
        {
            sum += ptr[i];
            sumSq += (double)ptr[i] * ptr[i];
        }
        double mean = sum / count;
        return (float)Math.Sqrt(Math.Max(0, sumSq / count - mean * mean));
    }

    #endregion
}
