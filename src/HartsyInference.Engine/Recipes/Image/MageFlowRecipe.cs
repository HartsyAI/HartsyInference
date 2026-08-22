using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae.Mage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Microsoft Mage-Flow recipe (4B NR-MMDiT, arXiv 2607.19064). The transformer is a dual-stream MMDiT with diffusers-standard keys, so it reuses <see cref="QwenImageTransformer"/> with the <see cref="QwenImageConfig.MageFlow"/> preset (12 blocks, patch-1, 128-ch, image-only RoPE). The Qwen3-VL-4B text encoder (<see cref="SideModels.Qwen3VL_4B"/>, already fp8) and the bespoke one-step <see cref="MageVaeDecoder"/> (<see cref="SideModels.MageVae"/>) resolve as side models when not bundled. Base vs Edit-Turbo is auto-detected from the checkpoint filename. Drives through <see cref="MageFlowRecipePipeline"/>.</summary>
/// <remarks><b>Quant-native</b>: a GGUF (Q4/Q5) or fp8 DiT loads without upcasting so a 4B backbone streams onto a 12 GB card — the official repo only ships bf16, but the community quant/GGUF path is what you run on smaller GPUs.</remarks>
public sealed class MageFlowRecipe : IArchitectureRecipe
{
    public string Name => "mage-flow";
    public bool Matches(string familyId) => string.Equals(familyId, "mage-flow", StringComparison.OrdinalIgnoreCase);

    /// <summary>Mage-Flow-Edit-Turbo rides this recipe: the init image is the edit reference (VAE-encoded to in-context ref latents). Declared for both variants — the recipe encodes a reference only when one is supplied.</summary>
    /// <remarks>Reference editing, not strength-based img2img: MageFlowPipeline appends the encoded init image as
    /// in-context reference tokens rather than noising it, so <c>Creativity</c> has nothing to select. Declaring
    /// <see cref="ImageFeatures.Img2Img"/> here would accept a creativity value the family cannot honour.
    /// <para><see cref="ImageFeatures.Lora"/> added 2026-08-20. Mage-Flow drives <see cref="HartsyInference.Diffusion.Models.Denoisers.QwenImageTransformer"/> (<c>QwenImageConfig.MageFlow</c>), so its blocks are the canonical <c>transformer_blocks.{i}</c> and Qwen-Image LoRA files map identically. A K-quant GGUF build refuses the merge by name, same boundary as every other family.</para></remarks>
    public ImageFeatures Supports => ImageFeatures.RefEdit | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;

    /// <summary>Base Mage-Flow: 30 steps at CFG 5.0, 1024×1024 (model card).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 30, CfgScale = 5.0f, Width = 1024, Height = 1024 };
    /// <summary>Mage-Flow-*-Turbo: 4 distilled steps, guidance-free (CFG 1.0).</summary>
    public static ImageDefaults TurboDefaults { get; } = new ImageDefaults { Steps = 4, CfgScale = 1.0f, Width = 1024, Height = 1024 };
    public ImageDefaults Defaults => FamilyDefaults;

    public IRecipePipeline Construct(RecipeContext context)
    {
        string fileName = Path.GetFileName(context.CheckpointPath);
        string lower = fileName.ToLowerInvariant();
        bool isTurbo = lower.Contains("turbo") || lower.Contains("tdm") || lower.Contains("distill");
        bool isEdit = lower.Contains("edit");

        List<SafeTensorsLoader> loaders = new();
        IDisposable? ggufHandle = null;
        try
        {
            // ── DiT: quant-native (GGUF stays quantized for the per-GEMM dequant path; safetensors fp8/bf16 kept). ──
            Dictionary<string, Tensor> ditWeights;
            bool isGguf = context.CheckpointPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
            if (isGguf)
            {
                GgufModelLoader.LoadedGgufModel gguf = GgufModelLoader.Load(context.CheckpointPath);
                ggufHandle = gguf;
                ditWeights = Remap(GgufModelLoader.RelabelRank2ToPyTorchOrder(gguf.Weights), StripTransformerPrefix);
            }
            else
            {
                (ditWeights, SafeTensorsLoader ditLoader) = LoadComponent(context.CheckpointPath, StripTransformerPrefix, applyFp8Dequant: true);
                loaders.Add(ditLoader);
            }
            if (ditWeights.Count == 0)
                throw new InvalidOperationException($"Mage-Flow checkpoint '{fileName}' contains no transformer weights (looked for transformer_blocks.* / img_in.*).");
            Logs.Info($"[MageFlowRecipe] Parsed DiT: {ditWeights.Count} tensors ({(isEdit ? "edit" : "t2i")}{(isTurbo ? ", turbo" : "")}).");

            QwenImageConfig config = QwenImageConfig.MageFlow;
            QwenImageTransformer transformer = new QwenImageTransformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: ditWeights);
            transformer.LoadWeights(ditWeights);

            // ── Text encoder: Qwen3-VL-4B (fp8_scaled), vision tower dropped. ──
            string encoderPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3VL_4B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            (Dictionary<string, Tensor> teWeights, SafeTensorsLoader teLoader) = LoadComponent(encoderPath, RemapQwenKey, applyFp8Dequant: true);
            loaders.Add(teLoader);
            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_VL_4B);
            textEncoder.LoadWeights(teWeights);

            // ── MageVAE: split the file into decoder (`pipeline.*`) and encoder (`student.dconv_encoder.*`). Encoder is
            // only needed for edit; a decode-only VAE file simply has no encoder keys. ──
            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.MageVae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            (Dictionary<string, Tensor> allVae, SafeTensorsLoader vaeLoader) = LoadComponent(vaePath, key => key, applyFp8Dequant: false);
            loaders.Add(vaeLoader);
            (Dictionary<string, Tensor> decW, Dictionary<string, Tensor> encW) = SplitMageVae(allVae);
            MageVaeDecoder vae = new MageVaeDecoder();
            vae.LoadWeights(decW);
            MageVaeEncoder? vaeEncoder = null;
            if (encW.Count > 0) { vaeEncoder = new MageVaeEncoder(); vaeEncoder.LoadWeights(encW); }

            MageFlowPipeline pipeline = new MageFlowPipeline(context.Backend, textEncoder, transformer, vae, config, vaeEncoder);
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 512);
            Logs.Info($"[MageFlowRecipe] Mage-Flow ready (Qwen3-VL-4B; static shift 6.0; VAE dec={decW.Count}/enc={encW.Count}{(vaeEncoder is not null ? ", edit-capable" : "")}).");
            return new MageFlowRecipePipeline(pipeline, tokenizer, textEncoder, transformer, vae, vaeEncoder, isTurbo, loaders, ggufHandle, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[MageFlowRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders) loader.Dispose();
            ggufHandle?.Dispose();
            throw;
        }
    }

    private static (Dictionary<string, Tensor>, SafeTensorsLoader) LoadComponent(string filePath, Func<string, string?> keyTransform, bool applyFp8Dequant)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(filePath);
        try
        {
            Dictionary<string, Tensor> merged = Remap(loader.GetAllTensors(), keyTransform);
            return (applyFp8Dequant ? CheckpointConvertUtils.ApplyFp8ScaledDequant(merged) : merged, loader);
        }
        catch (Exception ex)
        {
            Logs.Error($"[MageFlowRecipe] Failed to load component '{Path.GetFileName(filePath)}'.", ex);
            loader.Dispose();
            throw;
        }
    }

    private static Dictionary<string, Tensor> Remap(IReadOnlyDictionary<string, Tensor> src, Func<string, string?> keyTransform)
    {
        Dictionary<string, Tensor> merged = new();
        foreach (KeyValuePair<string, Tensor> kv in src)
        {
            if (kv.Key.EndsWith(".scaled_fp8", StringComparison.Ordinal) || kv.Key == "scaled_fp8") continue;
            string? mapped = keyTransform(kv.Key);
            if (mapped is not null) merged[mapped] = kv.Value;
        }
        return merged;
    }

    private static string StripTransformerPrefix(string key)
    {
        if (key.StartsWith("model.diffusion_model.", StringComparison.Ordinal)) return key["model.diffusion_model.".Length..];
        if (key.StartsWith("diffusion_model.", StringComparison.Ordinal)) return key["diffusion_model.".Length..];
        if (key.StartsWith("transformer.", StringComparison.Ordinal)) return key["transformer.".Length..];
        return key;
    }

    // Qwen3-VL → LlamaStyleEncoder: drop vision tower + lm_head, keep language_model.model.* as model.*.
    private static string? RemapQwenKey(string key)
    {
        if (key.Contains(".visual.", StringComparison.Ordinal) || key.StartsWith("visual.", StringComparison.Ordinal)) return null;
        if (key.Contains("lm_head", StringComparison.Ordinal)) return null;
        int lm = key.LastIndexOf("language_model.", StringComparison.Ordinal);
        string suffix = lm >= 0 ? key[(lm + "language_model.".Length)..] : key;
        if (suffix.StartsWith("model.", StringComparison.Ordinal)) suffix = suffix["model.".Length..];
        if (suffix.StartsWith("layers.", StringComparison.Ordinal) || suffix.StartsWith("embed_tokens.", StringComparison.Ordinal) || suffix == "norm.weight")
            return "model." + suffix;
        return null;
    }

    // Split a loaded MageVAE file: decoder keys are `pipeline.*` (minus the FLUX2 encoder side y_embedder.encoder/
    // bottleneck); encoder keys are `student.dconv_encoder.*` (edit path). Both are returned with their prefix stripped.
    private static (Dictionary<string, Tensor> dec, Dictionary<string, Tensor> enc) SplitMageVae(IReadOnlyDictionary<string, Tensor> all)
    {
        Dictionary<string, Tensor> dec = new(), enc = new();
        foreach (KeyValuePair<string, Tensor> kv in all)
        {
            if (kv.Key.StartsWith("pipeline.", StringComparison.Ordinal))
            {
                string k = kv.Key["pipeline.".Length..];
                if (k.StartsWith("y_embedder.encoder.", StringComparison.Ordinal) || k.StartsWith("y_embedder.bottleneck.", StringComparison.Ordinal))
                    continue;
                dec[k] = kv.Value;
            }
            else if (kv.Key.StartsWith("student.dconv_encoder.", StringComparison.Ordinal))
            {
                enc[kv.Key["student.dconv_encoder.".Length..]] = kv.Value;
            }
        }
        return (dec, enc);
    }
}
