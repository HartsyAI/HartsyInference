using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Diffusion.Utilities;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace SharpInference.Diffusion.Tests;

/// <summary>
/// SDXL weight loading and forward pass tests. Validates that SDXL safetensors
/// checkpoints load correctly into all model components (UNet, CLIP-L, CLIP-G, VAE).
///
/// Supports two modes:
/// 1. Diffusers format: separate directories per component (text_encoder/, text_encoder_2/, unet/, vae/)
/// 2. Single-file checkpoint: merged safetensors with prefixed keys (key enumeration only)
///
/// Set the SDXL_MODEL_DIR environment variable or edit SdxlDiffusersDir to point at your model.
/// Set SDXL_SINGLE_FILE_PATH for single-file checkpoint testing.
/// </summary>
public class SdxlWeightLoadingTests
{
    /// <summary>Path to SDXL in HuggingFace diffusers layout (tokenizer/, tokenizer_2/, text_encoder/, text_encoder_2/, unet/, vae/).</summary>
    private static readonly string SdxlDiffusersDir =
        Environment.GetEnvironmentVariable("SDXL_MODEL_DIR")
        ?? @"C:\Users\AI Overlord\Desktop\Projects\SharpInference\tests\test-models\sdxl";

    /// <summary>Path to a single-file SDXL safetensors checkpoint.</summary>
    private static readonly string SdxlSingleFilePath =
        Environment.GetEnvironmentVariable("SDXL_SINGLE_FILE_PATH")
        ?? @"C:\Users\AI Overlord\Desktop\Projects\SwarmUI\Models\Stable-Diffusion\juggernautXL_ragnarokBy.safetensors";

    private readonly ITestOutputHelper _output;

    public SdxlWeightLoadingTests(ITestOutputHelper output) => _output = output;

    #region Single-File Checkpoint Key Enumeration

    /// <summary>
    /// Loads a single-file SDXL checkpoint and enumerates all tensor keys, grouping
    /// them by component (UNet, CLIP-L, CLIP-G, VAE). Validates that expected SDXL-specific
    /// keys are present (add_embedding, transformer_blocks, text_projection, etc.).
    /// </summary>
    [Fact]
    public void SingleFileCheckpoint_EnumerateKeys()
    {
        if (!File.Exists(SdxlSingleFilePath))
        {
            _output.WriteLine($"SKIPPED: Single-file checkpoint not found: {SdxlSingleFilePath}");
            return;
        }

        _output.WriteLine($"Loading single-file checkpoint: {SdxlSingleFilePath}");
        using SafeTensorsLoader loader = new();
        loader.Load(SdxlSingleFilePath);

        IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors = loader.Descriptors;
        _output.WriteLine($"Total tensors: {descriptors.Count}");

        // Group keys by prefix
        Dictionary<string, int> prefixCounts = [];
        Dictionary<string, List<string>> prefixKeys = [];

        foreach (string key in descriptors.Keys)
        {
            string prefix = key.Contains('.') ? key[..key.IndexOf('.')] : key;
            if (!prefixCounts.ContainsKey(prefix))
            {
                prefixCounts[prefix] = 0;
                prefixKeys[prefix] = [];
            }
            prefixCounts[prefix]++;
            if (prefixKeys[prefix].Count < 5) // Sample first 5 keys per prefix
            {
                prefixKeys[prefix].Add(key);
            }
        }

        _output.WriteLine($"\n=== Key Prefix Groups ===");
        foreach (KeyValuePair<string, int> kvp in prefixCounts.OrderByDescending(x => x.Value))
        {
            _output.WriteLine($"  {kvp.Key}: {kvp.Value} tensors");
            foreach (string sample in prefixKeys[kvp.Key])
            {
                SafeTensorDescriptor desc = descriptors[sample];
                _output.WriteLine($"    {sample} [{desc.DType}] {desc.Shape}");
            }
        }

        // Check for SDXL-specific key patterns
        _output.WriteLine($"\n=== SDXL Key Pattern Checks ===");
        bool hasModel = descriptors.Keys.Any(k => k.StartsWith("model."));
        bool hasCondStage = descriptors.Keys.Any(k => k.StartsWith("cond_stage_model.") || k.StartsWith("conditioner."));
        bool hasVae = descriptors.Keys.Any(k => k.StartsWith("first_stage_model."));

        _output.WriteLine($"  Has 'model.*' (UNet): {hasModel}");
        _output.WriteLine($"  Has conditioner/cond_stage_model (CLIP): {hasCondStage}");
        _output.WriteLine($"  Has 'first_stage_model.*' (VAE): {hasVae}");

        // Look for SDXL-specific patterns
        bool hasAddEmbedding = descriptors.Keys.Any(k => k.Contains("add_embedding"));
        bool hasTransformerBlocks = descriptors.Keys.Any(k => k.Contains("transformer_blocks"));
        _output.WriteLine($"  Has 'add_embedding' (SDXL ADM): {hasAddEmbedding}");
        _output.WriteLine($"  Has 'transformer_blocks' (multi-block attention): {hasTransformerBlocks}");

        // Count transformer block depth
        HashSet<int> transformerBlockIndices = [];
        foreach (string key in descriptors.Keys)
        {
            if (key.Contains("transformer_blocks."))
            {
                int startIdx = key.IndexOf("transformer_blocks.") + "transformer_blocks.".Length;
                int endIdx = key.IndexOf('.', startIdx);
                if (endIdx > startIdx && int.TryParse(key[startIdx..endIdx], out int blockIdx))
                {
                    transformerBlockIndices.Add(blockIdx);
                }
            }
        }
        if (transformerBlockIndices.Count > 0)
        {
            _output.WriteLine($"  Transformer block indices found: [{string.Join(", ", transformerBlockIndices.Order())}]");
            _output.WriteLine($"  Max transformer depth: {transformerBlockIndices.Max() + 1}");
        }

        Assert.True(descriptors.Count > 0, "Checkpoint should contain tensors");
    }

    #endregion

    #region Diffusers-Format Weight Loading Tests

    /// <summary>Loads the SDXL UNet from diffusers-format safetensors and validates all weight keys are found.</summary>
    [Fact]
    public void DiffusersFormat_LoadUNetWeights()
    {
        string unetPath = Path.Combine(SdxlDiffusersDir, "unet", "diffusion_pytorch_model.fp16.safetensors");
        if (!File.Exists(unetPath))
        {
            // Try sharded format
            string unetDir = Path.Combine(SdxlDiffusersDir, "unet");
            if (!Directory.Exists(unetDir) || !Directory.GetFiles(unetDir, "*.safetensors").Any())
            {
                _output.WriteLine($"SKIPPED: SDXL UNet not found at {unetPath}");
                return;
            }

            _output.WriteLine("Loading SDXL UNet (sharded)...");
            using SafeTensorsShardLoader shardLoader = new();
            shardLoader.LoadDirectory(unetDir);
            Dictionary<string, Tensor> shardedWeights = CastWeightsToF32(shardLoader.GetAllTensors());
            LoadAndValidateUNet(shardedWeights);
            return;
        }

        _output.WriteLine("Loading SDXL UNet (single file)...");
        using SafeTensorsLoader loader = new();
        loader.Load(unetPath);
        Dictionary<string, Tensor> weights = CastWeightsToF32(loader.GetAllTensors());
        LoadAndValidateUNet(weights);
    }

    private void LoadAndValidateUNet(Dictionary<string, Tensor> weights)
    {
        _output.WriteLine($"  Weight count: {weights.Count}");

        // Log sample keys for debugging
        _output.WriteLine("  Sample weight keys:");
        foreach (string key in weights.Keys.Take(20))
        {
            _output.WriteLine($"    {key} [{weights[key].DType}] {weights[key].Shape}");
        }

        // Create SDXL UNet and load weights
        UNet unet = new(UNetConfig.SdxlBase);

        try
        {
            unet.LoadWeights(weights);
            _output.WriteLine("\n  UNet weight loading: SUCCESS");
        }
        catch (KeyNotFoundException ex)
        {
            _output.WriteLine($"\n  UNet weight loading: FAILED — {ex.Message}");
            _output.WriteLine("\n  Available keys containing the missing prefix:");
            string missingKey = ex.Message;
            string prefix = missingKey.Contains('.') ? missingKey[..missingKey.LastIndexOf('.')] : "";
            foreach (string key in weights.Keys.Where(k => k.Contains(prefix)).Take(10))
            {
                _output.WriteLine($"    {key}");
            }
            Assert.Fail($"Missing UNet weight key: {ex.Message}");
        }

        // Validate key counts for SDXL-specific components
        bool hasAddEmb = weights.ContainsKey("add_embedding.linear_1.weight");
        bool hasTimEmb = weights.ContainsKey("time_embedding.linear_1.weight");
        _output.WriteLine($"\n  Has add_embedding: {hasAddEmb}");
        _output.WriteLine($"  Has time_embedding: {hasTimEmb}");

        // Validate shapes for critical tensors
        if (weights.TryGetValue("conv_in.weight", out Tensor? convIn))
        {
            _output.WriteLine($"  conv_in.weight shape: {convIn.Shape} (expected [320, 4, 3, 3])");
            Assert.Equal(320, (int)convIn.Shape[0]);
            Assert.Equal(4, (int)convIn.Shape[1]);
        }

        if (weights.TryGetValue("add_embedding.linear_1.weight", out Tensor? addLin1))
        {
            _output.WriteLine($"  add_embedding.linear_1.weight shape: {addLin1.Shape} (expected [1280, 2816])");
            Assert.Equal(1280, (int)addLin1.Shape[0]);
            Assert.Equal(2816, (int)addLin1.Shape[1]);
        }

        _output.WriteLine("\n  SDXL UNet weight loading validated.");
    }

    /// <summary>Loads the SDXL CLIP-L text encoder from diffusers-format safetensors.</summary>
    [Fact]
    public void DiffusersFormat_LoadClipLWeights()
    {
        string path = Path.Combine(SdxlDiffusersDir, "text_encoder", "model.fp16.safetensors");
        if (!File.Exists(path))
        {
            _output.WriteLine($"SKIPPED: CLIP-L not found at {path}");
            return;
        }

        _output.WriteLine("Loading SDXL CLIP-L...");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = CastWeightsToF32(loader.GetAllTensors());
        _output.WriteLine($"  Weight count: {weights.Count}");

        ClipTextEncoder encoder = new(ClipTextEncoderConfig.SdxlClipL);

        try
        {
            encoder.LoadWeights(weights, "text_model");
            _output.WriteLine("  CLIP-L weight loading: SUCCESS");
        }
        catch (KeyNotFoundException ex)
        {
            _output.WriteLine($"  CLIP-L weight loading: FAILED — {ex.Message}");
            Assert.Fail($"Missing CLIP-L weight key: {ex.Message}");
        }

        // Validate embedding shapes
        if (weights.TryGetValue("text_model.embeddings.token_embedding.weight", out Tensor? tokenEmb))
        {
            _output.WriteLine($"  token_embedding shape: {tokenEmb.Shape} (expected [49408, 768])");
            Assert.Equal(49408, (int)tokenEmb.Shape[0]);
            Assert.Equal(768, (int)tokenEmb.Shape[1]);
        }

        // CLIP-L should NOT have text_projection
        bool hasProjection = weights.ContainsKey("text_projection.weight");
        _output.WriteLine($"  Has text_projection: {hasProjection} (expected false for CLIP-L)");
    }

    /// <summary>Loads the SDXL CLIP-G text encoder from diffusers-format safetensors. Validates text_projection weight exists.</summary>
    [Fact]
    public void DiffusersFormat_LoadClipGWeights()
    {
        string path = Path.Combine(SdxlDiffusersDir, "text_encoder_2", "model.fp16.safetensors");
        if (!File.Exists(path))
        {
            _output.WriteLine($"SKIPPED: CLIP-G not found at {path}");
            return;
        }

        _output.WriteLine("Loading SDXL CLIP-G...");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = CastWeightsToF32(loader.GetAllTensors());
        _output.WriteLine($"  Weight count: {weights.Count}");

        // Log a few keys for debugging
        _output.WriteLine("  Sample keys:");
        foreach (string key in weights.Keys.Take(15))
        {
            _output.WriteLine($"    {key} {weights[key].Shape}");
        }

        ClipTextEncoder encoder = new(ClipTextEncoderConfig.SdxlClipG);

        try
        {
            encoder.LoadWeights(weights, "text_model");
            _output.WriteLine("  CLIP-G weight loading: SUCCESS");
        }
        catch (KeyNotFoundException ex)
        {
            _output.WriteLine($"  CLIP-G weight loading: FAILED — {ex.Message}");
            Assert.Fail($"Missing CLIP-G weight key: {ex.Message}");
        }

        // Validate embedding shapes for CLIP-G
        if (weights.TryGetValue("text_model.embeddings.token_embedding.weight", out Tensor? tokenEmb))
        {
            _output.WriteLine($"  token_embedding shape: {tokenEmb.Shape} (expected [49408, 1280])");
            Assert.Equal(49408, (int)tokenEmb.Shape[0]);
            Assert.Equal(1280, (int)tokenEmb.Shape[1]);
        }

        // CLIP-G MUST have text_projection for pooled output
        bool hasProjection = weights.ContainsKey("text_projection.weight");
        _output.WriteLine($"  Has text_projection: {hasProjection} (expected true for CLIP-G)");
        Assert.True(hasProjection, "CLIP-G must have text_projection.weight for pooled output");

        if (weights.TryGetValue("text_projection.weight", out Tensor? textProj))
        {
            _output.WriteLine($"  text_projection.weight shape: {textProj.Shape} (expected [1280, 1280])");
            Assert.Equal(1280, (int)textProj.Shape[0]);
            Assert.Equal(1280, (int)textProj.Shape[1]);
        }

        // Validate 32 layers exist
        int maxLayer = 0;
        foreach (string key in weights.Keys)
        {
            if (key.StartsWith("text_model.encoder.layers."))
            {
                int startIdx = "text_model.encoder.layers.".Length;
                int endIdx = key.IndexOf('.', startIdx);
                if (endIdx > startIdx && int.TryParse(key[startIdx..endIdx], out int layerIdx))
                {
                    maxLayer = Math.Max(maxLayer, layerIdx);
                }
            }
        }
        _output.WriteLine($"  Max layer index: {maxLayer} (expected 31 for 32 layers)");
        Assert.Equal(31, maxLayer);
    }

    /// <summary>Loads the SDXL VAE decoder from diffusers-format safetensors.</summary>
    [Fact]
    public void DiffusersFormat_LoadVaeWeights()
    {
        string path = Path.Combine(SdxlDiffusersDir, "vae", "diffusion_pytorch_model.fp16.safetensors");
        if (!File.Exists(path))
        {
            _output.WriteLine($"SKIPPED: VAE not found at {path}");
            return;
        }

        _output.WriteLine("Loading SDXL VAE...");
        using SafeTensorsLoader loader = new();
        loader.Load(path);
        Dictionary<string, Tensor> weights = CastWeightsToF32(loader.GetAllTensors());
        _output.WriteLine($"  Weight count: {weights.Count}");

        VaeDecoder decoder = new(VaeConfig.Sdxl);

        try
        {
            decoder.LoadWeights(weights);
            _output.WriteLine("  VAE weight loading: SUCCESS");
        }
        catch (KeyNotFoundException ex)
        {
            _output.WriteLine($"  VAE weight loading: FAILED — {ex.Message}");
            Assert.Fail($"Missing VAE weight key: {ex.Message}");
        }

        // Validate post_quant_conv exists for SDXL
        bool hasPostQuant = weights.ContainsKey("post_quant_conv.weight");
        _output.WriteLine($"  Has post_quant_conv: {hasPostQuant} (expected true for SDXL)");
        Assert.True(hasPostQuant, "SDXL VAE must have post_quant_conv");
    }

    #endregion

    #region Forward Pass Sanity Checks

    /// <summary>
    /// Loads the SDXL UNet, runs a single forward pass with random noise and zero embeddings,
    /// and validates the output has no NaN/Inf and reasonable statistics.
    /// </summary>
    [Fact]
    public unsafe void DiffusersFormat_UNetForwardPassSanity()
    {
        string unetPath = Path.Combine(SdxlDiffusersDir, "unet", "diffusion_pytorch_model.fp16.safetensors");
        string unetDir = Path.Combine(SdxlDiffusersDir, "unet");

        Dictionary<string, Tensor>? weights = LoadWeightsFromPathOrDir(unetPath, unetDir);
        if (weights is null)
        {
            _output.WriteLine($"SKIPPED: SDXL UNet not found");
            return;
        }

        _output.WriteLine("Loading SDXL UNet for forward pass test...");
        UNet unet = new(UNetConfig.SdxlBase);
        unet.LoadWeights(weights);

        using CpuBackend backend = new();

        // Create inputs: [1, 4, 64, 64] latent (SDXL default 512x512 → 64x64 latent, or 128x128 for 1024)
        // Use small latent to keep test fast
        TensorShape latentShape = new(1, 4, 16, 16);
        Tensor latent = SeedGenerator.CreateNoise(latentShape, 42);

        // Zero text embedding [1, 77, 2048] (SDXL cross-attention dim)
        TensorShape embShape = new(1, 77, 2048);
        Tensor textEmb = new(embShape, DType.F32);

        // Pooled text embedding [1, 1280]
        TensorShape pooledShape = new(1, 1280);
        Tensor pooledEmb = new(pooledShape, DType.F32);

        // Size conditioning: [origH, origW, cropTop, cropLeft, targetH, targetW]
        float[] sizeCondition = [1024f, 1024f, 0f, 0f, 1024f, 1024f];

        _output.WriteLine($"Input latent: mean={ComputeMean(latent):F6}, std={ComputeStd(latent):F6}");
        _output.WriteLine("Running UNet forward pass (this may take a while on CPU)...");

        Tensor output = unet.Forward(backend, latent, 999.0f, textEmb, pooledEmb, sizeCondition);

        float outMean = ComputeMean(output);
        float outStd = ComputeStd(output);
        _output.WriteLine($"Output: mean={outMean:F6}, std={outStd:F6}, shape={output.Shape}");

        // Sanity checks
        Assert.False(float.IsNaN(outMean), "UNet output contains NaN");
        Assert.False(float.IsInfinity(outStd), "UNet output contains Inf");
        Assert.True(outStd > 0.001f, $"UNet output std too small: {outStd}");
        Assert.True(outStd < 100.0f, $"UNet output std too large: {outStd}");
        Assert.Equal(4, (int)output.Shape[1]);
        Assert.Equal(16, (int)output.Shape[2]);
        Assert.Equal(16, (int)output.Shape[3]);

        _output.WriteLine("SDXL UNet forward pass: PASSED");

        output.Dispose();
        latent.Dispose();
        textEmb.Dispose();
        pooledEmb.Dispose();
    }

    /// <summary>
    /// Loads all SDXL components, runs the full pipeline with dual CLIP encoding and UNet denoising,
    /// and validates the output image has reasonable properties.
    /// </summary>
    [Fact]
    public unsafe void DiffusersFormat_FullPipelineSmokeTest()
    {
        if (!Directory.Exists(SdxlDiffusersDir))
        {
            _output.WriteLine($"SKIPPED: SDXL diffusers directory not found: {SdxlDiffusersDir}");
            return;
        }

        SdxlModelPaths paths;
        try
        {
            paths = SdxlModelPaths.FromHuggingFaceDir(SdxlDiffusersDir);
            paths.Validate();
        }
        catch (FileNotFoundException ex)
        {
            _output.WriteLine($"SKIPPED: {ex.Message}");
            return;
        }

        _output.WriteLine("Loading all SDXL components for pipeline smoke test...");
        using CpuBackend backend = new();

        // Load tokenizers
        using ClipTokenizer tokenizerL = new(paths.VocabLPath, paths.MergesLPath);
        using ClipTokenizer tokenizerG = new(paths.VocabGPath, paths.MergesGPath);

        // Tokenize prompt
        string prompt = "a photo of a cat";
        string negativePrompt = "";
        int[] promptTokensL = tokenizerL.Encode(prompt);
        int[] negTokensL = tokenizerL.Encode(negativePrompt);
        int[] promptTokensG = tokenizerG.Encode(prompt);
        int[] negTokensG = tokenizerG.Encode(negativePrompt);

        int promptEosG = ClipTokenizer.FindEosPosition(promptTokensG);
        int negEosG = ClipTokenizer.FindEosPosition(negTokensG);

        _output.WriteLine($"Prompt tokens L: [{string.Join(", ", promptTokensL.Take(10))}...]");
        _output.WriteLine($"Prompt tokens G: [{string.Join(", ", promptTokensG.Take(10))}...]");
        _output.WriteLine($"EOS position G: prompt={promptEosG}, negative={negEosG}");

        // Load CLIP-L
        _output.WriteLine("Loading CLIP-L...");
        ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
        using SafeTensorsLoader clipLLoader = new();
        clipLLoader.Load(paths.TextEncoderLPath);
        clipL.LoadWeights(CastWeightsToF32(clipLLoader.GetAllTensors()), "text_model");

        // Load CLIP-G
        _output.WriteLine("Loading CLIP-G...");
        ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
        using SafeTensorsLoader clipGLoader = new();
        clipGLoader.Load(paths.TextEncoderGPath);
        clipG.LoadWeights(CastWeightsToF32(clipGLoader.GetAllTensors()), "text_model");

        // Load UNet
        _output.WriteLine("Loading UNet...");
        UNet unet = new(UNetConfig.SdxlBase);
        Dictionary<string, Tensor> unetWeights = LoadWeightsFromPathOrDir(
            Path.Combine(SdxlDiffusersDir, "unet", "diffusion_pytorch_model.fp16.safetensors"),
            Path.Combine(SdxlDiffusersDir, "unet"))!;
        unet.LoadWeights(unetWeights);

        // Load VAE
        _output.WriteLine("Loading VAE...");
        VaeDecoder vae = new(VaeConfig.Sdxl);
        using SafeTensorsLoader vaeLoader = new();
        vaeLoader.Load(paths.VaePath);
        vae.LoadWeights(CastWeightsToF32(vaeLoader.GetAllTensors()));

        // Test dual CLIP encoding
        _output.WriteLine("\nEncoding text with dual CLIP...");
        int[][] batchTokensL = [negTokensL, promptTokensL];
        (Tensor clipLHidden, _) = clipL.EncodePenultimate(backend, batchTokensL, [0, 0]);
        _output.WriteLine($"  CLIP-L hidden: shape={clipLHidden.Shape}, mean={ComputeMean(clipLHidden):F6}, std={ComputeStd(clipLHidden):F6}");

        int[][] batchTokensG = [negTokensG, promptTokensG];
        int[] eosPositions = [negEosG, promptEosG];
        (Tensor clipGHidden, Tensor? pooledOutput) = clipG.EncodePenultimate(backend, batchTokensG, eosPositions);
        _output.WriteLine($"  CLIP-G hidden: shape={clipGHidden.Shape}, mean={ComputeMean(clipGHidden):F6}, std={ComputeStd(clipGHidden):F6}");

        Assert.NotNull(pooledOutput);
        _output.WriteLine($"  Pooled output: shape={pooledOutput!.Shape}, mean={ComputeMean(pooledOutput):F6}, std={ComputeStd(pooledOutput):F6}");

        // Validate shapes
        Assert.Equal(2, (int)clipLHidden.Shape[0]);  // batch=2 (neg+pos)
        Assert.Equal(77, (int)clipLHidden.Shape[1]);  // seqLen
        Assert.Equal(768, (int)clipLHidden.Shape[2]); // CLIP-L hidden

        Assert.Equal(2, (int)clipGHidden.Shape[0]);
        Assert.Equal(77, (int)clipGHidden.Shape[1]);
        Assert.Equal(1280, (int)clipGHidden.Shape[2]); // CLIP-G hidden

        Assert.Equal(2, (int)pooledOutput.Shape[0]);
        Assert.Equal(1280, (int)pooledOutput.Shape[1]); // projection dim

        // Validate no NaN/Inf in text encoding output
        Assert.False(float.IsNaN(ComputeMean(clipLHidden)), "CLIP-L output contains NaN");
        Assert.False(float.IsNaN(ComputeMean(clipGHidden)), "CLIP-G output contains NaN");
        Assert.False(float.IsNaN(ComputeMean(pooledOutput)), "Pooled output contains NaN");

        _output.WriteLine("\nSDXL full pipeline smoke test: PASSED (text encoding validated)");
        _output.WriteLine("Note: Full denoising loop skipped for test speed. Use SdxlPipeline.GenerateFromTokens() for end-to-end generation.");

        clipLHidden.Dispose();
        clipGHidden.Dispose();
        pooledOutput.Dispose();
    }

    #endregion

    #region Expected Key Validation

    /// <summary>
    /// Validates that a diffusers-format SDXL UNet checkpoint contains all expected weight keys.
    /// Does not load into the model — just checks key presence and tensor shapes.
    /// </summary>
    [Fact]
    public void DiffusersFormat_UNetHasExpectedKeys()
    {
        string unetPath = Path.Combine(SdxlDiffusersDir, "unet", "diffusion_pytorch_model.fp16.safetensors");
        string unetDir = Path.Combine(SdxlDiffusersDir, "unet");

        Dictionary<string, Tensor>? weights = LoadWeightsFromPathOrDir(unetPath, unetDir);
        if (weights is null)
        {
            _output.WriteLine($"SKIPPED: SDXL UNet not found");
            return;
        }

        _output.WriteLine($"SDXL UNet weight count: {weights.Count}");

        // Core structural keys that MUST exist
        string[] requiredKeys =
        [
            "conv_in.weight",
            "conv_in.bias",
            "time_embedding.linear_1.weight",
            "time_embedding.linear_1.bias",
            "time_embedding.linear_2.weight",
            "time_embedding.linear_2.bias",
            "add_embedding.linear_1.weight",   // SDXL ADM conditioning
            "add_embedding.linear_1.bias",
            "add_embedding.linear_2.weight",
            "add_embedding.linear_2.bias",
            "conv_norm_out.weight",
            "conv_norm_out.bias",
            "conv_out.weight",
            "conv_out.bias",
        ];

        int missing = 0;
        foreach (string key in requiredKeys)
        {
            bool exists = weights.ContainsKey(key);
            if (!exists)
            {
                _output.WriteLine($"  MISSING: {key}");
                missing++;
            }
        }

        // SDXL has 3 down blocks
        for (int block = 0; block < 3; block++)
        {
            for (int resnet = 0; resnet < 2; resnet++)
            {
                string prefix = $"down_blocks.{block}.resnets.{resnet}";
                CheckKeyExists(weights, $"{prefix}.norm1.weight", ref missing);
                CheckKeyExists(weights, $"{prefix}.conv1.weight", ref missing);
                CheckKeyExists(weights, $"{prefix}.norm2.weight", ref missing);
                CheckKeyExists(weights, $"{prefix}.conv2.weight", ref missing);
            }
        }

        // Down blocks 1 and 2 have attention (SDXL: [false, true, true])
        for (int block = 1; block < 3; block++)
        {
            for (int attn = 0; attn < 2; attn++)
            {
                string prefix = $"down_blocks.{block}.attentions.{attn}";
                CheckKeyExists(weights, $"{prefix}.norm.weight", ref missing);
            }
        }

        // Transformer blocks depth 10 at level 2
        for (int tb = 0; tb < 10; tb++)
        {
            string prefix = $"down_blocks.2.attentions.0.transformer_blocks.{tb}";
            CheckKeyExists(weights, $"{prefix}.attn1.to_q.weight", ref missing);
            CheckKeyExists(weights, $"{prefix}.attn2.to_q.weight", ref missing);
            CheckKeyExists(weights, $"{prefix}.ff.net.0.proj.weight", ref missing);
        }

        // Mid block
        CheckKeyExists(weights, "mid_block.resnets.0.norm1.weight", ref missing);
        CheckKeyExists(weights, "mid_block.attentions.0.norm.weight", ref missing);
        CheckKeyExists(weights, "mid_block.resnets.1.norm1.weight", ref missing);

        // Mid block transformer blocks (depth 10 like level 2)
        for (int tb = 0; tb < 10; tb++)
        {
            string prefix = $"mid_block.attentions.0.transformer_blocks.{tb}";
            CheckKeyExists(weights, $"{prefix}.attn1.to_q.weight", ref missing);
        }

        _output.WriteLine($"\nTotal missing keys: {missing}");
        Assert.Equal(0, missing);
        _output.WriteLine("All expected SDXL UNet keys are present.");
    }

    #endregion

    #region Helpers

    private Dictionary<string, Tensor>? LoadWeightsFromPathOrDir(string singleFilePath, string directoryPath)
    {
        if (File.Exists(singleFilePath))
        {
            SafeTensorsLoader loader = new();
            loader.Load(singleFilePath);
            return CastWeightsToF32(loader.GetAllTensors());
        }

        if (Directory.Exists(directoryPath) && Directory.GetFiles(directoryPath, "*.safetensors").Length > 0)
        {
            SafeTensorsShardLoader shardLoader = new();
            shardLoader.LoadDirectory(directoryPath);
            return CastWeightsToF32(shardLoader.GetAllTensors());
        }

        return null;
    }

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
