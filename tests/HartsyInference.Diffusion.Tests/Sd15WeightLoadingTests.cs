using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>
/// SD1.5 single-file checkpoint conversion and weight loading tests.
/// Validates that SD1.5 safetensors checkpoints load correctly into all model components (UNet, CLIP-L, VAE).
/// Set SD15_SINGLE_FILE_PATH environment variable or edit the default path below.
/// </summary>
public class Sd15WeightLoadingTests
{
    private static string Sd15SingleFilePath => TestPaths.Sd15.SingleFile;

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
    /// SD1.5 UNet forward pass on GPU. Compares GPU output against CPU output.
    /// If this fails, the CUDA backend has a fundamental issue (not SDXL-specific).
    /// </summary>
    [Fact]
    public unsafe void SingleFile_UNetForwardPass_GpuVsCpu()
    {
        if (!File.Exists(Sd15SingleFilePath))
        {
            _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {Sd15SingleFilePath}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Sd15WeightLoadingTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        _output.WriteLine($"Loading and converting: {Sd15SingleFilePath}");
        (Sd15CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd15CheckpointConverter.LoadAndConvert(Sd15SingleFilePath);

        using (loader)
        {
            Dictionary<string, Tensor> unetF32 = CastWeightsToF32(converted.UNet);

            UNet cpuUnet = new(UNetConfig.Sd15);
            cpuUnet.LoadWeights(unetF32);
            UNet gpuUnet = new(UNetConfig.Sd15);
            gpuUnet.LoadWeights(unetF32);

            // SD1.5: 128x128 → 16x16 latent
            TensorShape latentShape = new(1, 4, 16, 16);
            Tensor latent = SeedGenerator.CreateNoise(latentShape, 42);

            // SD1.5: single CLIP context [1, 77, 768] — use random data for quick test
            Tensor textEmb = new(new TensorShape(1, 77, 768), DType.F32);
            float* tePtr = (float*)textEmb.DataPointer;
            Random rng = new(123);
            for (long i = 0; i < textEmb.ElementCount; i++)
                tePtr[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;

            // CPU forward
            _output.WriteLine("Running SD1.5 UNet on CPU...");
            using CpuBackend cpuBackend = new();
            Tensor cpuOutput = cpuUnet.Forward(cpuBackend, latent, 999.0f, textEmb);

            float cpuMean = ComputeMean(cpuOutput);
            float cpuStd = ComputeStd(cpuOutput);
            _output.WriteLine($"  CPU output: mean={cpuMean:F6}, std={cpuStd:F6}");

            // GPU forward
            _output.WriteLine("Running SD1.5 UNet on GPU...");
            using CudaBackend gpuBackend = new(deviceOrdinal: 0, ptxDir: ptxDir);
            Tensor gpuOutput = gpuUnet.Forward(gpuBackend, latent, 999.0f, textEmb);

            float gpuMean = ComputeMean(gpuOutput);
            float gpuStd = ComputeStd(gpuOutput);
            _output.WriteLine($"  GPU output: mean={gpuMean:F6}, std={gpuStd:F6}");

            // Compare
            float* cPtr = (float*)cpuOutput.DataPointer;
            float* gPtr = (float*)gpuOutput.DataPointer;
            long count = cpuOutput.ElementCount;
            double sumAbsErr = 0, maxAbsErr = 0;
            for (long i = 0; i < count; i++)
            {
                double err = Math.Abs(cPtr[i] - gPtr[i]);
                sumAbsErr += err;
                if (err > maxAbsErr) maxAbsErr = err;
            }
            double avgErr = sumAbsErr / count;
            _output.WriteLine($"\n=== GPU vs CPU comparison ===");
            _output.WriteLine($"  avg_err={avgErr:E3}, max_err={maxAbsErr:E3}");
            _output.WriteLine($"  (Expected < 1e-3 for correctly-working GPU backend)");

            if (avgErr < 1e-3)
                _output.WriteLine("  RESULT: PASS — GPU matches CPU within tolerance");
            else
                _output.WriteLine($"  RESULT: FAIL — GPU diverges from CPU (avg_err={avgErr:E3})");

            cpuOutput.Dispose();
            gpuOutput.Dispose();
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

    /// <summary>
    /// Layer-by-layer GPU vs CPU diagnostic. Tests each UNet sub-block independently
    /// with CPU reference inputs so GPU errors don't compound. Finds the FIRST divergent block.
    /// </summary>
    [Fact]
    public unsafe void SingleFile_LayerByLayer_GpuVsCpu_Diagnostic()
    {
        if (!File.Exists(Sd15SingleFilePath))
        {
            _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {Sd15SingleFilePath}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Sd15WeightLoadingTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        _output.WriteLine($"Loading SD1.5 checkpoint: {Sd15SingleFilePath}");
        (Sd15CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd15CheckpointConverter.LoadAndConvert(Sd15SingleFilePath);

        using (loader)
        {
            Dictionary<string, Tensor> w = CastWeightsToF32(converted.UNet);

            using CpuBackend cpu = new();
            using CudaBackend gpu = new(deviceOrdinal: 0, ptxDir: ptxDir);

            // Shared inputs
            TensorShape latentShape = new(1, 4, 16, 16);
            Tensor latent = SeedGenerator.CreateNoise(latentShape, 42);

            Tensor textEmb = new(new TensorShape(1, 77, 768), DType.F32);
            Random rng = new(123);
            float* tePtr = (float*)textEmb.DataPointer;
            for (long i = 0; i < textEmb.ElementCount; i++)
                tePtr[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;

            Span<float> timesteps = stackalloc float[] { 999.0f };

            _output.WriteLine("\n=== Layer-by-layer GPU vs CPU diagnostic ===\n");
            bool foundDivergence = false;

            // ── Layer 0: TimestepEmbedding ──
            TimestepEmbedding timeEmbBlock = new(320, 1280);
            timeEmbBlock.LoadWeights(w, "time_embedding");
            Tensor cpuTemb = timeEmbBlock.Forward(cpu, timesteps, 1);
            Tensor gpuTemb = timeEmbBlock.Forward(gpu, timesteps, 1);
            foundDivergence |= CompareAndReport("time_embedding", cpuTemb, gpuTemb);
            gpuTemb.Dispose();

            // ── Layer 1: conv_in (raw Conv2D) ──
            TensorShape convOutShape = new(1, 320, 16, 16);
            Tensor cpuHidden = new(convOutShape, DType.F32);
            cpu.Conv2D(cpuHidden, latent, w["conv_in.weight"], w["conv_in.bias"], 1, 1, 1, 1);
            Tensor gpuHidden = new(convOutShape, DType.F32);
            gpu.Conv2D(gpuHidden, latent, w["conv_in.weight"], w["conv_in.bias"], 1, 1, 1, 1);
            foundDivergence |= CompareAndReport("conv_in", cpuHidden, gpuHidden);
            gpuHidden.Dispose();

            // ── Drill into down_blocks.0.resnets.0 sub-operations ──
            _output.WriteLine("\n  --- Drilling into down_blocks.0.resnets.0 sub-ops ---");

            // Sub-op 1: GroupNorm(norm1)
            TensorShape hidShape = new(1, 320, 16, 16);
            Tensor cpuNorm1 = new(hidShape, DType.F32);
            cpu.GroupNorm(cpuNorm1, cpuHidden, w["down_blocks.0.resnets.0.norm1.weight"], w["down_blocks.0.resnets.0.norm1.bias"], 32, 1e-5f);
            Tensor gpuNorm1 = new(hidShape, DType.F32);
            gpu.GroupNorm(gpuNorm1, cpuHidden, w["down_blocks.0.resnets.0.norm1.weight"], w["down_blocks.0.resnets.0.norm1.bias"], 32, 1e-5f);
            foundDivergence |= CompareAndReport("  resnet0.norm1 (GroupNorm)", cpuNorm1, gpuNorm1);
            gpuNorm1.Dispose();

            // Sub-op 2: SiLU
            Tensor cpuSilu1 = new(hidShape, DType.F32);
            cpu.Silu(cpuSilu1, cpuNorm1);
            Tensor gpuSilu1 = new(hidShape, DType.F32);
            gpu.Silu(gpuSilu1, cpuNorm1);
            foundDivergence |= CompareAndReport("  resnet0.silu1", cpuSilu1, gpuSilu1);
            gpuSilu1.Dispose();

            // Sub-op 3: Conv2D(conv1)
            Tensor cpuConv1 = new(hidShape, DType.F32);
            cpu.Conv2D(cpuConv1, cpuSilu1, w["down_blocks.0.resnets.0.conv1.weight"], w["down_blocks.0.resnets.0.conv1.bias"], 1, 1, 1, 1);
            Tensor gpuConv1 = new(hidShape, DType.F32);
            gpu.Conv2D(gpuConv1, cpuSilu1, w["down_blocks.0.resnets.0.conv1.weight"], w["down_blocks.0.resnets.0.conv1.bias"], 1, 1, 1, 1);
            foundDivergence |= CompareAndReport("  resnet0.conv1 (Conv2D)", cpuConv1, gpuConv1);
            gpuConv1.Dispose();

            // Sub-op 4: Timestep projection — SiLU(temb) then Linear
            TensorShape tembShape = new(1, 1280);
            Tensor cpuTembSilu = new(tembShape, DType.F32);
            cpu.Silu(cpuTembSilu, cpuTemb);
            Tensor gpuTembSilu = new(tembShape, DType.F32);
            gpu.Silu(gpuTembSilu, cpuTemb);
            foundDivergence |= CompareAndReport("  resnet0.temb_silu", cpuTembSilu, gpuTembSilu);
            gpuTembSilu.Dispose();

            TensorShape projShape = new(1, 320);
            Tensor cpuTembProj = new(projShape, DType.F32);
            cpu.Linear(cpuTembProj, cpuTembSilu, w["down_blocks.0.resnets.0.time_emb_proj.weight"], w["down_blocks.0.resnets.0.time_emb_proj.bias"]);
            Tensor gpuTembProj = new(projShape, DType.F32);
            gpu.Linear(gpuTembProj, cpuTembSilu, w["down_blocks.0.resnets.0.time_emb_proj.weight"], w["down_blocks.0.resnets.0.time_emb_proj.bias"]);
            foundDivergence |= CompareAndReport("  resnet0.temb_proj (Linear)", cpuTembProj, gpuTembProj);
            gpuTembProj.Dispose();

            // Sub-op 5: AddTimestepEmbedding (CPU in-place) — both use CPU conv1 output
            // Clone cpuConv1 so we don't corrupt it
            Tensor cpuWithTemb = cpuConv1.To(cpuConv1.Device);
            {
                float* hPtr = (float*)cpuWithTemb.DataPointer;
                float* tPtr = (float*)cpuTembProj.DataPointer;
                for (int c = 0; c < 320; c++)
                {
                    float tVal = tPtr[c];
                    int offset = c * 16 * 16;
                    for (int s = 0; s < 256; s++)
                        hPtr[offset + s] += tVal;
                }
            }

            // Sub-op 6: GroupNorm(norm2) — THIS is the key test
            Tensor cpuNorm2 = new(hidShape, DType.F32);
            cpu.GroupNorm(cpuNorm2, cpuWithTemb, w["down_blocks.0.resnets.0.norm2.weight"], w["down_blocks.0.resnets.0.norm2.bias"], 32, 1e-5f);
            Tensor gpuNorm2 = new(hidShape, DType.F32);
            gpu.GroupNorm(gpuNorm2, cpuWithTemb, w["down_blocks.0.resnets.0.norm2.weight"], w["down_blocks.0.resnets.0.norm2.bias"], 32, 1e-5f);
            foundDivergence |= CompareAndReport("  resnet0.norm2 (GroupNorm)", cpuNorm2, gpuNorm2);
            gpuNorm2.Dispose();

            // ── Chain test: run all ResNet ops on GPU, feeding GPU outputs forward ──
            _output.WriteLine("\n  --- Chain test: GPU ops feeding into each other ---");

            Tensor chA = new(hidShape, DType.F32);
            gpu.GroupNorm(chA, cpuHidden, w["down_blocks.0.resnets.0.norm1.weight"], w["down_blocks.0.resnets.0.norm1.bias"], 32, 1e-5f);
            Tensor chB = new(hidShape, DType.F32);
            gpu.Silu(chB, chA);
            foundDivergence |= CompareAndReport("  CHAIN: norm1→silu1", cpuSilu1, chB);
            chA.Dispose();

            Tensor chC = new(hidShape, DType.F32);
            gpu.Conv2D(chC, chB, w["down_blocks.0.resnets.0.conv1.weight"], w["down_blocks.0.resnets.0.conv1.bias"], 1, 1, 1, 1);
            foundDivergence |= CompareAndReport("  CHAIN: →conv1", cpuConv1, chC);
            chB.Dispose();

            // Chain: temb SiLU → Linear
            Tensor chD = new(tembShape, DType.F32);
            gpu.Silu(chD, cpuTemb);
            Tensor chE = new(projShape, DType.F32);
            gpu.Linear(chE, chD, w["down_blocks.0.resnets.0.time_emb_proj.weight"], w["down_blocks.0.resnets.0.time_emb_proj.bias"]);
            chD.Dispose();

            // Add temb in-place on CPU
            {
                float* hPtr = (float*)chC.DataPointer;
                float* tPtr = (float*)chE.DataPointer;
                for (int c = 0; c < 320; c++)
                {
                    float tVal = tPtr[c];
                    int offset = c * 16 * 16;
                    for (int s = 0; s < 256; s++)
                        hPtr[offset + s] += tVal;
                }
            }
            chE.Dispose();
            foundDivergence |= CompareAndReport("  CHAIN: →+temb", cpuWithTemb, chC);

            Tensor chF = new(hidShape, DType.F32);
            gpu.GroupNorm(chF, chC, w["down_blocks.0.resnets.0.norm2.weight"], w["down_blocks.0.resnets.0.norm2.bias"], 32, 1e-5f);
            foundDivergence |= CompareAndReport("  CHAIN: →norm2", cpuNorm2, chF);
            chC.Dispose();

            Tensor chG = new(hidShape, DType.F32);
            gpu.Silu(chG, chF); chF.Dispose();
            Tensor chH = new(hidShape, DType.F32);
            gpu.Conv2D(chH, chG, w["down_blocks.0.resnets.0.conv2.weight"], w["down_blocks.0.resnets.0.conv2.bias"], 1, 1, 1, 1);
            chG.Dispose();
            Tensor chOut = new(hidShape, DType.F32);
            gpu.Add(chOut, chH, cpuHidden);
            chH.Dispose();

            // Compare chain vs CPU reference for full resnet
            // Build CPU reference resnet output manually
            Tensor cr1 = new(hidShape, DType.F32);
            cpu.GroupNorm(cr1, cpuHidden, w["down_blocks.0.resnets.0.norm1.weight"], w["down_blocks.0.resnets.0.norm1.bias"], 32, 1e-5f);
            Tensor cr2 = new(hidShape, DType.F32);
            cpu.Silu(cr2, cr1); cr1.Dispose();
            Tensor cr3 = new(hidShape, DType.F32);
            cpu.Conv2D(cr3, cr2, w["down_blocks.0.resnets.0.conv1.weight"], w["down_blocks.0.resnets.0.conv1.bias"], 1, 1, 1, 1);
            cr2.Dispose();
            Tensor cr4 = new(tembShape, DType.F32);
            cpu.Silu(cr4, cpuTemb);
            Tensor cr5 = new(projShape, DType.F32);
            cpu.Linear(cr5, cr4, w["down_blocks.0.resnets.0.time_emb_proj.weight"], w["down_blocks.0.resnets.0.time_emb_proj.bias"]);
            cr4.Dispose();
            {
                float* hPtr = (float*)cr3.DataPointer;
                float* tPtr = (float*)cr5.DataPointer;
                for (int c = 0; c < 320; c++)
                {
                    float tVal = tPtr[c];
                    int offset = c * 16 * 16;
                    for (int s = 0; s < 256; s++)
                        hPtr[offset + s] += tVal;
                }
            }
            cr5.Dispose();
            Tensor cr6 = new(hidShape, DType.F32);
            cpu.GroupNorm(cr6, cr3, w["down_blocks.0.resnets.0.norm2.weight"], w["down_blocks.0.resnets.0.norm2.bias"], 32, 1e-5f);
            cr3.Dispose();
            Tensor cr7 = new(hidShape, DType.F32);
            cpu.Silu(cr7, cr6); cr6.Dispose();
            Tensor cr8 = new(hidShape, DType.F32);
            cpu.Conv2D(cr8, cr7, w["down_blocks.0.resnets.0.conv2.weight"], w["down_blocks.0.resnets.0.conv2.bias"], 1, 1, 1, 1);
            cr7.Dispose();
            Tensor crOut = new(hidShape, DType.F32);
            cpu.Add(crOut, cr8, cpuHidden);
            cr8.Dispose();

            foundDivergence |= CompareAndReport("  CHAIN: full resnet result", crOut, chOut);
            chOut.Dispose();
            crOut.Dispose();

            _output.WriteLine("  --- End chain test ---\n");

            // Cleanup sub-op tensors
            cpuNorm1.Dispose();
            cpuSilu1.Dispose();
            cpuTembSilu.Dispose();
            cpuTembProj.Dispose();
            cpuWithTemb.Dispose();
            cpuNorm2.Dispose();
            cpuConv1.Dispose();

            // Full ResNet block test still for reference
            UNetResNetBlock res00 = new(320, 320, 1280);
            res00.LoadWeights(w, "down_blocks.0.resnets.0");
            Tensor cpuRes00 = res00.Forward(cpu, cpuHidden, cpuTemb);
            Tensor gpuRes00 = res00.Forward(gpu, cpuHidden, cpuTemb);
            foundDivergence |= CompareAndReport("down_blocks.0.resnets.0 (full)", cpuRes00, gpuRes00);
            gpuRes00.Dispose();

            CrossAttentionBlock attn00 = new(320, 8, 768, 1);
            attn00.LoadWeights(w, "down_blocks.0.attentions.0");
            Tensor cpuAttn00 = attn00.Forward(cpu, cpuRes00, textEmb);
            Tensor gpuAttn00 = attn00.Forward(gpu, cpuRes00, textEmb);
            foundDivergence |= CompareAndReport("down_blocks.0.attentions.0", cpuAttn00, gpuAttn00);
            gpuAttn00.Dispose();

            UNetResNetBlock res01 = new(320, 320, 1280);
            res01.LoadWeights(w, "down_blocks.0.resnets.1");
            Tensor cpuRes01 = res01.Forward(cpu, cpuAttn00, cpuTemb);
            Tensor gpuRes01 = res01.Forward(gpu, cpuAttn00, cpuTemb);
            foundDivergence |= CompareAndReport("down_blocks.0.resnets.1", cpuRes01, gpuRes01);
            gpuRes01.Dispose();

            CrossAttentionBlock attn01 = new(320, 8, 768, 1);
            attn01.LoadWeights(w, "down_blocks.0.attentions.1");
            Tensor cpuAttn01 = attn01.Forward(cpu, cpuRes01, textEmb);
            Tensor gpuAttn01 = attn01.Forward(gpu, cpuRes01, textEmb);
            foundDivergence |= CompareAndReport("down_blocks.0.attentions.1", cpuAttn01, gpuAttn01);
            gpuAttn01.Dispose();

            // downsamplers.0: Conv2D stride=2
            TensorShape ds0Shape = new(1, 320, 8, 8);
            Tensor cpuDs0 = new(ds0Shape, DType.F32);
            cpu.Conv2D(cpuDs0, cpuAttn01, w["down_blocks.0.downsamplers.0.conv.weight"], w["down_blocks.0.downsamplers.0.conv.bias"], 2, 2, 1, 1);
            Tensor gpuDs0 = new(ds0Shape, DType.F32);
            gpu.Conv2D(gpuDs0, cpuAttn01, w["down_blocks.0.downsamplers.0.conv.weight"], w["down_blocks.0.downsamplers.0.conv.bias"], 2, 2, 1, 1);
            foundDivergence |= CompareAndReport("down_blocks.0.downsamplers.0", cpuDs0, gpuDs0);
            gpuDs0.Dispose();

            // ── down_blocks.1: resnets.0 (320→640 channel change) ──
            UNetResNetBlock res10 = new(320, 640, 1280);
            res10.LoadWeights(w, "down_blocks.1.resnets.0");
            Tensor cpuRes10 = res10.Forward(cpu, cpuDs0, cpuTemb);
            Tensor gpuRes10 = res10.Forward(gpu, cpuDs0, cpuTemb);
            foundDivergence |= CompareAndReport("down_blocks.1.resnets.0", cpuRes10, gpuRes10);
            gpuRes10.Dispose();

            CrossAttentionBlock attn10 = new(640, 8, 768, 1);
            attn10.LoadWeights(w, "down_blocks.1.attentions.0");
            Tensor cpuAttn10 = attn10.Forward(cpu, cpuRes10, textEmb);
            Tensor gpuAttn10 = attn10.Forward(gpu, cpuRes10, textEmb);
            foundDivergence |= CompareAndReport("down_blocks.1.attentions.0", cpuAttn10, gpuAttn10);
            gpuAttn10.Dispose();

            // ── mid_block (skip to mid for speed if no divergence yet) ──
            // Use full down_blocks.1-3 via DownBlock if needed later

            _output.WriteLine($"\n=== Summary: {(foundDivergence ? "DIVERGENCE FOUND" : "All tested layers PASS")} ===");

            // Cleanup
            latent.Dispose();
            textEmb.Dispose();
            cpuTemb.Dispose();
            cpuHidden.Dispose();
            cpuRes00.Dispose();
            cpuAttn00.Dispose();
            cpuRes01.Dispose();
            cpuAttn01.Dispose();
            cpuDs0.Dispose();
            cpuRes10.Dispose();
            cpuAttn10.Dispose();
        }
    }

    /// <summary>
    /// Full UNet forward pass checkpoint test. Steps through every major block comparing CPU vs GPU
    /// to find the first divergent block.
    /// </summary>
    [Fact]
    public unsafe void SingleFile_FullForward_Checkpoint_GpuVsCpu()
    {
        if (!File.Exists(Sd15SingleFilePath))
        {
            _output.WriteLine($"SKIPPED: SD1.5 checkpoint not found: {Sd15SingleFilePath}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(Sd15WeightLoadingTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        _output.WriteLine($"Loading SD1.5 checkpoint: {Sd15SingleFilePath}");
        (Sd15CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd15CheckpointConverter.LoadAndConvert(Sd15SingleFilePath);

        using (loader)
        {
            Dictionary<string, Tensor> w = CastWeightsToF32(converted.UNet);
            UNetConfig config = UNetConfig.Sd15;

            using CpuBackend cpu = new();
            using CudaBackend gpu = new(deviceOrdinal: 0, ptxDir: ptxDir);

            // ── Build blocks ──
            TimestepEmbedding timeEmb = new(config.ModelChannels, config.ModelChannels * 4);
            timeEmb.LoadWeights(w, "time_embedding");

            DownBlock[] downBlocks = new DownBlock[config.BlockOutChannels.Length];
            for (int i = 0; i < downBlocks.Length; i++)
            {
                int inCh = i == 0 ? config.ModelChannels : config.BlockOutChannels[i - 1];
                int outCh = config.BlockOutChannels[i];
                bool hasAttn = config.DownBlockHasAttention[i];
                bool hasDown = i < downBlocks.Length - 1;
                downBlocks[i] = new DownBlock(inCh, outCh, config.ModelChannels * 4, config.LayersPerBlock,
                    hasAttn, hasDown, config.NumAttentionHeads[i], config.CrossAttentionDim, config.TransformerLayersPerBlock[i]);
                downBlocks[i].LoadWeights(w, $"down_blocks.{i}");
            }

            int midCh = config.BlockOutChannels[^1];
            UNetResNetBlock midRes0 = new(midCh, midCh, config.ModelChannels * 4);
            midRes0.LoadWeights(w, "mid_block.resnets.0");
            CrossAttentionBlock midAttn = new(midCh, config.NumAttentionHeads[^1], config.CrossAttentionDim, config.TransformerLayersPerBlock[^1]);
            midAttn.LoadWeights(w, "mid_block.attentions.0");
            UNetResNetBlock midRes1 = new(midCh, midCh, config.ModelChannels * 4);
            midRes1.LoadWeights(w, "mid_block.resnets.1");

            UpBlock[] upBlocks = new UpBlock[config.BlockOutChannels.Length];
            int[] reversedCh = ((IEnumerable<int>)config.BlockOutChannels).Reverse().ToArray();
            for (int i = 0; i < upBlocks.Length; i++)
            {
                int outCh = reversedCh[i];
                int prevOutCh = i == 0 ? midCh : reversedCh[i - 1];
                int inputCh = reversedCh[Math.Min(i + 1, upBlocks.Length - 1)];
                bool hasAttn = config.UpBlockHasAttention[i];
                bool hasUp = i < upBlocks.Length - 1;
                int upIdx = upBlocks.Length - 1 - i;
                upBlocks[i] = new UpBlock(inputCh, outCh, prevOutCh, config.ModelChannels * 4, config.LayersPerBlock + 1,
                    hasAttn, hasUp, config.NumAttentionHeads[upIdx], config.CrossAttentionDim, config.TransformerLayersPerBlock[upIdx]);
                upBlocks[i].LoadWeights(w, $"up_blocks.{i}");
            }

            // ── Inputs ──
            TensorShape latentShape = new(1, 4, 16, 16);
            Tensor latent = SeedGenerator.CreateNoise(latentShape, 42);
            Tensor textEmb = new(new TensorShape(1, 77, 768), DType.F32);
            Random rng = new(123);
            float* tePtr = (float*)textEmb.DataPointer;
            for (long i = 0; i < textEmb.ElementCount; i++)
                tePtr[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;

            Span<float> timesteps = stackalloc float[] { 999.0f };

            _output.WriteLine("\n=== Full Forward Checkpoint Test ===\n");

            // ── Step 1: time_embedding ──
            Tensor cpuTemb = timeEmb.Forward(cpu, timesteps, 1);
            Tensor gpuTemb = timeEmb.Forward(gpu, timesteps, 1);
            CompareAndReport("time_embedding", cpuTemb, gpuTemb);
            gpuTemb.Dispose();

            // ── Step 2: conv_in ──
            TensorShape convInShape = new(1, config.ModelChannels, 16, 16);
            Tensor cpuHidden = new(convInShape, DType.F32);
            cpu.Conv2D(cpuHidden, latent, w["conv_in.weight"], w["conv_in.bias"], 1, 1, 1, 1);
            Tensor gpuHidden = new(convInShape, DType.F32);
            gpu.Conv2D(gpuHidden, latent, w["conv_in.weight"], w["conv_in.bias"], 1, 1, 1, 1);
            CompareAndReport("conv_in", cpuHidden, gpuHidden);
            gpuHidden.Dispose();

            // ── Step 3: Down blocks — run each block on BOTH backends with same CPU input ──
            List<Tensor> cpuSkips = new() { cpuHidden.To(cpuHidden.Device) };
            Tensor cpuDown = cpuHidden;
            for (int i = 0; i < downBlocks.Length; i++)
            {
                // CPU first, then GPU — to avoid any side-effect from GPU forward
                (Tensor cpuDownOut, List<Tensor> cpuDownSkips) = downBlocks[i].Forward(cpu, cpuDown, cpuTemb, textEmb);
                (Tensor gpuDownOut, List<Tensor> gpuDownSkips) = downBlocks[i].Forward(gpu, cpuDown, cpuTemb, textEmb);
                CompareAndReport($"down_blocks.{i} output", cpuDownOut, gpuDownOut);
                gpuDownOut.Dispose();
                foreach (Tensor s in gpuDownSkips) s.Dispose();

                if (cpuDown != cpuHidden) cpuDown.Dispose();
                cpuDown = cpuDownOut;
                cpuSkips.AddRange(cpuDownSkips);
            }

            // ── Step 4: Mid block ──
            Tensor cpuMidR0 = midRes0.Forward(cpu, cpuDown, cpuTemb);
            Tensor gpuMidR0 = midRes0.Forward(gpu, cpuDown, cpuTemb);
            CompareAndReport("mid_block.resnets.0", cpuMidR0, gpuMidR0);
            gpuMidR0.Dispose();

            Tensor cpuMidA = midAttn.Forward(cpu, cpuMidR0, textEmb);
            Tensor gpuMidA = midAttn.Forward(gpu, cpuMidR0, textEmb);
            CompareAndReport("mid_block.attentions.0", cpuMidA, gpuMidA);
            gpuMidA.Dispose();

            Tensor cpuMidR1 = midRes1.Forward(cpu, cpuMidA, cpuTemb);
            Tensor gpuMidR1 = midRes1.Forward(gpu, cpuMidA, cpuTemb);
            CompareAndReport("mid_block.resnets.1", cpuMidR1, gpuMidR1);
            gpuMidR1.Dispose();

            // ── Step 5: Up blocks — each block compared independently with same CPU input + skips ──
            Tensor cpuUp = cpuMidR1;
            for (int i = 0; i < upBlocks.Length; i++)
            {
                // Clone skips for GPU (Forward mutates the list)
                List<Tensor> gpuUpSkips = new();
                int skipsNeeded = upBlocks[i] == upBlocks[0] ? 3 : 3; // LayersPerBlock+1 = 3
                for (int s = cpuSkips.Count - skipsNeeded; s < cpuSkips.Count; s++)
                    gpuUpSkips.Add(cpuSkips[s].To(cpuSkips[s].Device));

                Tensor gpuUpOut = upBlocks[i].Forward(gpu, cpuUp, cpuTemb, textEmb, gpuUpSkips);
                Tensor cpuUpOut = upBlocks[i].Forward(cpu, cpuUp, cpuTemb, textEmb, cpuSkips);
                CompareAndReport($"up_blocks.{i} output", cpuUpOut, gpuUpOut);
                gpuUpOut.Dispose();

                if (cpuUp != cpuMidR1) cpuUp.Dispose();
                cpuUp = cpuUpOut;
            }

            // ── Step 6: Final norm + SiLU + conv_out ──
            TensorShape finalShape = cpuUp.Shape;
            int finalCh = config.ModelChannels;

            Tensor cpuNormOut = new(finalShape, DType.F32);
            cpu.GroupNorm(cpuNormOut, cpuUp, w["conv_norm_out.weight"], w["conv_norm_out.bias"], config.NormNumGroups, config.NormEps);
            Tensor gpuNormOut = new(finalShape, DType.F32);
            gpu.GroupNorm(gpuNormOut, cpuUp, w["conv_norm_out.weight"], w["conv_norm_out.bias"], config.NormNumGroups, config.NormEps);
            CompareAndReport("conv_norm_out", cpuNormOut, gpuNormOut);
            gpuNormOut.Dispose();

            Tensor cpuSiluOut = new(finalShape, DType.F32);
            cpu.Silu(cpuSiluOut, cpuNormOut);
            Tensor gpuSiluOut = new(finalShape, DType.F32);
            gpu.Silu(gpuSiluOut, cpuNormOut);
            CompareAndReport("final SiLU", cpuSiluOut, gpuSiluOut);
            gpuSiluOut.Dispose();

            TensorShape outShape = new(1, config.OutChannels, (int)finalShape[2], (int)finalShape[3]);
            Tensor cpuOut = new(outShape, DType.F32);
            cpu.Conv2D(cpuOut, cpuSiluOut, w["conv_out.weight"], w["conv_out.bias"], 1, 1, 1, 1);
            Tensor gpuOut = new(outShape, DType.F32);
            gpu.Conv2D(gpuOut, cpuSiluOut, w["conv_out.weight"], w["conv_out.bias"], 1, 1, 1, 1);
            CompareAndReport("conv_out", cpuOut, gpuOut);
            gpuOut.Dispose();

            _output.WriteLine("\n=== Checkpoint test complete ===");

            // Cleanup
            latent.Dispose();
            textEmb.Dispose();
            cpuTemb.Dispose();
            cpuHidden.Dispose();
            cpuDown.Dispose();
            cpuMidR0.Dispose();
            cpuMidA.Dispose();
            cpuMidR1.Dispose();
            cpuUp.Dispose();
            cpuNormOut.Dispose();
            cpuSiluOut.Dispose();
            cpuOut.Dispose();
        }
    }

    /// <summary>
    /// Tests individual GPU ops at shapes used by the mid block (1280 channels, 2×2 spatial)
    /// to isolate which op diverges at this shape.
    /// </summary>
    [Fact]
    public unsafe void MidBlock_Ops_GpuVsCpu()
    {
        string assemblyDir = Path.GetDirectoryName(typeof(Sd15WeightLoadingTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA driver available");
            return;
        }

        using CpuBackend cpu = new();
        using CudaBackend gpu = new(deviceOrdinal: 0, ptxDir: ptxDir);

        _output.WriteLine("=== Testing ops at mid-block shapes (1280ch, 2×2) ===\n");

        // Test at multiple shapes to see where things break
        (int channels, int spatial)[] shapes = [(320, 16), (640, 8), (1280, 4), (1280, 2)];

        foreach ((int ch, int sp) in shapes)
        {
            TensorShape shape = new(1, ch, sp, sp);
            int elemCount = ch * sp * sp;

            // Create random input
            Tensor input = new(shape, DType.F32);
            float* ptr = (float*)input.DataPointer;
            Random rng = new(42);
            for (int i = 0; i < elemCount; i++)
                ptr[i] = (float)(rng.NextDouble() * 2 - 1);

            // Create dummy weights for GroupNorm (ch elements each)
            Tensor gnWeight = new(new TensorShape(ch), DType.F32);
            Tensor gnBias = new(new TensorShape(ch), DType.F32);
            float* gw = (float*)gnWeight.DataPointer;
            float* gb = (float*)gnBias.DataPointer;
            for (int i = 0; i < ch; i++) { gw[i] = 1.0f; gb[i] = 0.0f; }

            _output.WriteLine($"  Shape: (1, {ch}, {sp}, {sp}) — {elemCount} elements");

            // Test GroupNorm
            Tensor cpuGn = new(shape, DType.F32);
            cpu.GroupNorm(cpuGn, input, gnWeight, gnBias, 32, 1e-5f);
            Tensor gpuGn = new(shape, DType.F32);
            gpu.GroupNorm(gpuGn, input, gnWeight, gnBias, 32, 1e-5f);
            CompareAndReport($"    GroupNorm({ch}ch, {sp}×{sp})", cpuGn, gpuGn);
            gpuGn.Dispose();

            // Test SiLU
            Tensor cpuSilu = new(shape, DType.F32);
            cpu.Silu(cpuSilu, input);
            Tensor gpuSilu = new(shape, DType.F32);
            gpu.Silu(gpuSilu, input);
            CompareAndReport($"    SiLU({ch}ch, {sp}×{sp})", cpuSilu, gpuSilu);
            gpuSilu.Dispose();

            // Test Conv2D (ch→ch, 3×3, pad=1, stride=1)
            Tensor convW = new(new TensorShape(ch, ch, 3, 3), DType.F32);
            Tensor convB = new(new TensorShape(ch), DType.F32);
            float* cw = (float*)convW.DataPointer;
            for (long i = 0; i < convW.ElementCount; i++)
                cw[i] = (float)(rng.NextDouble() * 0.01);

            Tensor cpuConv = new(shape, DType.F32);
            cpu.Conv2D(cpuConv, input, convW, convB, 1, 1, 1, 1);
            Tensor gpuConv = new(shape, DType.F32);
            gpu.Conv2D(gpuConv, input, convW, convB, 1, 1, 1, 1);
            CompareAndReport($"    Conv2D({ch}ch, {sp}×{sp})", cpuConv, gpuConv);
            gpuConv.Dispose();

            // Cleanup
            input.Dispose();
            gnWeight.Dispose();
            gnBias.Dispose();
            cpuGn.Dispose();
            cpuSilu.Dispose();
            cpuConv.Dispose();
            convW.Dispose();
            convB.Dispose();

            _output.WriteLine("");
        }
    }

    private unsafe bool CompareAndReport(string name, Tensor cpuTensor, Tensor gpuTensor)
    {
        float* cPtr = (float*)cpuTensor.DataPointer;
        float* gPtr = (float*)gpuTensor.DataPointer;
        long count = cpuTensor.ElementCount;

        double sumAbsErr = 0, maxAbsErr = 0;
        double cpuSum = 0, gpuSum = 0;
        for (long i = 0; i < count; i++)
        {
            double err = Math.Abs(cPtr[i] - gPtr[i]);
            sumAbsErr += err;
            if (err > maxAbsErr) maxAbsErr = err;
            cpuSum += cPtr[i];
            gpuSum += gPtr[i];
        }
        double avgErr = sumAbsErr / count;
        double cpuMean = cpuSum / count;
        double gpuMean = gpuSum / count;

        string status = avgErr < 1e-3 ? "OK" : "DIVERGENT";
        _output.WriteLine($"  {name,-40} avg_err={avgErr:E3}  max_err={maxAbsErr:E3}  cpu_mean={cpuMean:F6}  gpu_mean={gpuMean:F6}  [{status}]");

        return avgErr >= 1e-3;
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
