using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Flux.2 recipe (BFL; Klein 4B / Klein 9B / Dev). Variant is detected from the converted transformer's hidden size (3072 → Klein 4B, 4096 → Klein 9B, 6144 → Dev). Lifted from the SwarmUI backend's <c>Flux2Loader</c>: the checkpoint is the transformer; the matching text encoder — Qwen3-4B (<see cref="SideModels.Qwen3_4B"/>), Qwen3-8B (<see cref="SideModels.Qwen3_8B_Fp4Mixed"/>), or Mistral-Small-3 (<see cref="SideModels.MistralSmallFlux2"/>) — and the Flux.2 VAE (<see cref="SideModels.Flux2Vae"/>, 32-channel latent + BatchNorm stats) resolve as side models. Constructs and drives through <see cref="Flux2RecipePipeline"/>.</summary>
public sealed class Flux2Recipe : IArchitectureRecipe
{
    /// <summary>Flux.2 Dev system prompt — verbatim from ComfyUI <c>comfy/text_encoders/flux.py</c> (the newline mid-sentence is in the original).</summary>
    private const string MistralDevSystemPrompt =
        "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.";

    /// <inheritdoc/>
    public string Name => "flux2";


    /// <inheritdoc/>
    /// <remarks>Flux.2's BN-style latent normalization is applied at the pipeline boundary; the encode-side inverse already lives in Flux2Pipeline.
    /// <para><see cref="ImageFeatures.Regional"/> added 2026-08-11 (Tier 3.7): <see cref="Flux2Transformer"/>'s double/single blocks
    /// gained an <c>attnBias</c> slot (mirroring <see cref="FluxTransformer"/>'s), wired through <see cref="Flux2Pipeline.GenerateFromTokens"/>
    /// and <see cref="Flux2RecipePipeline.BuildRegionalPlan"/> — real-weight verified with a two-region prompt.</para></remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.Regional | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "flux2", StringComparison.OrdinalIgnoreCase);

    /// <summary>Flux.2 Dev's official sampling settings: 50 steps at guidance 4.0, 1024x1024 (<c>GenerationDefaults.Flux2</c>); the distilled Klein variants narrow this via <see cref="Flux2RecipePipeline.VariantDefaults"/>.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 4.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <summary>Flux.2 Klein's sampling settings: 10 steps at guidance 4.0 — the Klein checkpoints are CFG-distilled few-step models, so they carry no guidance embedding.</summary>
    public static ImageDefaults KleinDefaults { get; } = new ImageDefaults { Steps = 10, CfgScale = 4.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): split-file / user overrides from ImageRequest.Components, and img2img are deferred —
        // this ports the text-to-image core.
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        IDisposable? ggufHandle = null;
        try
        {
            // GGUF repacks ship BFL-native keys (Flux2KeyMapper: "we just pass through") — the quants stay
            // native (transient per-GEMM dequant); BF16 GGUF tensors still need the F16 cast below.
            bool isGguf = context.CheckpointPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
            Dictionary<string, Tensor> rawWeights;
            if (isGguf)
            {
                GgufModelLoader.LoadedGgufModel gguf = GgufModelLoader.Load(context.CheckpointPath);
                ggufHandle = gguf;
                rawWeights = GgufModelLoader.RelabelRank2ToPyTorchOrder(gguf.Weights);
            }
            else
            {
                SafeTensorsLoader transformerLoader = new SafeTensorsLoader();
                transformerLoader.Load(context.CheckpointPath);
                loaders.Add(transformerLoader);
                rawWeights = transformerLoader.GetAllTensors();
            }
            // Pre-cast BF16 → F16 on CPU (F16 keeps the same footprint as BF16 and F16↔F32 is supported on every op).
            // GGUF-quantized tensors (Q4_K etc.) stay native — they dequant transiently per-GEMM elsewhere, and
            // a plain CastTo throws for quantized dtypes (needs GgufDequantizer instead).
            Dictionary<string, Tensor> castWeights = new Dictionary<string, Tensor>(rawWeights.Count);
            foreach (KeyValuePair<string, Tensor> kvp in rawWeights)
            {
                castWeights[kvp.Key] = kvp.Value.DType == DType.BF16 ? kvp.Value.CastTo(DType.F16) : kvp.Value;
            }

            Flux2Config config = DetectConfigFromTransformerWeights(castWeights, GuessConfigFromFilename(Path.GetFileName(context.CheckpointPath)) ?? Flux2Config.Klein4B);
            Logs.Info($"[Flux2Recipe] Architecture: hidden={config.HiddenSize}, depth={config.Depth} double + {config.DepthSingleBlocks} single → {DescribeConfig(config)}.");

            int mlpInner = (int)(config.HiddenSize * config.MlpRatio);
            Flux2CheckpointConverter converter = new Flux2CheckpointConverter(config.HiddenSize, mlpInner);
            Dictionary<string, Tensor> converted = converter.ConvertTransformer(castWeights);
            castWeights.Clear();

            Flux2Transformer transformer = new Flux2Transformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: converted);
            transformer.LoadWeights(converted);
            converted.Clear();

            // Resolve + load the variant's text encoder. Klein 9B's canonical file is FP4 quantized — the FP4
            // refusal below fires cleanly (HartsyInference has no FP4 GEMM path yet), exactly as the loader notes.
            (LlamaStyleEncoderConfig encoderConfig, ModelAsset encoderAsset, string encoderLabel) = ResolveTextEncoderForVariant(config);
            string encoderPath = ModelDownloader.EnsureSideModelAsync(encoderAsset, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
            qwenLoader.Load(encoderPath);
            loaders.Add(qwenLoader);
            Dictionary<string, Tensor> qwenRaw = qwenLoader.GetAllTensors();
            foreach (KeyValuePair<string, Tensor> kvp in qwenRaw)
            {
                string dtypeName = kvp.Value.DType.Name;
                if (dtypeName.StartsWith("F4", StringComparison.Ordinal) || dtypeName.Contains("FP4", StringComparison.Ordinal))
                {
                    throw new NotSupportedException(
                        $"{encoderLabel} weights at '{Path.GetFileName(encoderPath)}' contain FP4 tensors (e.g. '{kvp.Key}' is {dtypeName}). " +
                        "HartsyInference doesn't support FP4 GEMM yet — swap to an fp8/fp16 variant, or pick Klein 4B (Qwen3-4B, fp8-mixed) which works today.");
                }
            }
            // LoadWeights runs the raw dict through TextEncoderQuantNormalizer (folds fp8 weight_scale companions,
            // dequantizes U8-packed NVFP4, drops .comfy_quant blobs) — hand it the RAW dict, do not pre-cast.
            LlamaStyleEncoder encoder = new LlamaStyleEncoder(encoderConfig);
            encoder.LoadWeights(qwenRaw);

            // Flux.2 VAE (distinct from Flux.1's ae): 32-channel latent + BatchNorm running stats stored alongside.
            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Flux2Vae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaePath);
            loaders.Add(vaeLoader);
            Dictionary<string, Tensor> vaeWeights = vaeLoader.GetAllTensors();
            if (!vaeWeights.TryGetValue("bn.running_mean", out Tensor? bnMean))
            {
                throw new InvalidOperationException(
                    $"Flux.2 VAE '{Path.GetFileName(vaePath)}' is missing 'bn.running_mean'. Verify this is a Flux.2 VAE, not a Flux.1 ae.safetensors.");
            }
            if (!vaeWeights.TryGetValue("bn.running_var", out Tensor? bnVar))
            {
                throw new InvalidOperationException($"Flux.2 VAE '{Path.GetFileName(vaePath)}' is missing 'bn.running_var'.");
            }
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Flux2);
            vaeDecoder.LoadWeights(vaeWeights);
            VaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.Flux2, vaeWeights, "Flux2Recipe");

            // Tokenizer: Klein → embedded Qwen3 vocab/merges; Dev → Mistral tekken (HF tokenizer.json).
            Qwen3Tokenizer? qwenTokenizer = null;
            ErnieTokenizer? mistralTokenizer = null;
            if (config.TextEncoderType == Flux2TextEncoderType.Mistral)
            {
                string mistralTokPath = EnsureMistralTokenizerJson();
                Logs.Info($"[Flux2Recipe] Loading Mistral tekken tokenizer: {Path.GetFileName(mistralTokPath)}.");
                mistralTokenizer = new ErnieTokenizer(mistralTokPath);
            }
            else
            {
                qwenTokenizer = new Qwen3Tokenizer(maxLength: 512);
            }

            Flux2Pipeline pipeline = new Flux2Pipeline(
                context.Backend, encoder, transformer, vaeDecoder, vaeEncoder,
                bnMean, bnVar, config,
                hiddenLayers: null,
                bnEps: 1e-5f);
            Logs.Info($"[Flux2Recipe] Flux.2 ready ({DescribeConfig(config)}).");
            return new Flux2RecipePipeline(pipeline, config, qwenTokenizer, mistralTokenizer, MistralDevSystemPrompt, encoder, loaders, ggufHandle, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[Flux2Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            ggufHandle?.Dispose();
            throw;
        }
    }

    /// <summary>Filename-based variant guess, used only as the fallback when hidden-size detection is inconclusive.</summary>
    private static Flux2Config? GuessConfigFromFilename(string? name)
    {
        if (string.IsNullOrEmpty(name))
        {
            return null;
        }
        string n = name.ToLowerInvariant();
        if (n.Contains("klein-4b") || n.Contains("klein_4b") || n.Contains("klein4b"))
        {
            return Flux2Config.Klein4B;
        }
        if (n.Contains("klein-9b") || n.Contains("klein_9b") || n.Contains("klein9b"))
        {
            return Flux2Config.Klein9B;
        }
        if (n.Contains("flux2_dev") || n.Contains("flux-2-dev") || n.Contains("flux2-dev"))
        {
            return Flux2Config.Dev;
        }
        return null;
    }

    /// <summary>Confirms the variant by reading the actual hidden size from the converted transformer's <c>img_in.weight</c> (the authoritative answer — filename hints can lie).</summary>
    private static Flux2Config DetectConfigFromTransformerWeights(Dictionary<string, Tensor> weights, Flux2Config fallback)
    {
        Tensor? probe = null;
        foreach (string k in new[] { "img_in.weight", "x_embedder.weight", "patch_embed.proj.weight" })
        {
            if (weights.TryGetValue(k, out probe))
            {
                break;
            }
        }
        if (probe is null || probe.Shape.Rank < 1)
        {
            return fallback;
        }
        int hidden = (int)probe.Shape[0];
        return hidden switch
        {
            3072 => Flux2Config.Klein4B,
            4096 => Flux2Config.Klein9B,
            6144 => Flux2Config.Dev,
            _ => fallback,
        };
    }

    /// <summary>Picks the LlamaStyleEncoder preset + side-model asset for the variant. Klein 4B → Qwen3-4B (only verified path); Klein 9B → Qwen3-8B (FP4 file, refuses at load); Dev → Mistral-Small-3.</summary>
    private static (LlamaStyleEncoderConfig encoder, ModelAsset sideModel, string label) ResolveTextEncoderForVariant(Flux2Config config)
    {
        return config.HiddenSize switch
        {
            3072 => (LlamaStyleEncoderConfig.Qwen3_4B, SideModels.Qwen3_4B, "Qwen3-4B encoder"),
            4096 => (LlamaStyleEncoderConfig.Qwen3_8B, SideModels.Qwen3_8B_Fp4Mixed, "Qwen3-8B encoder (Klein 9B)"),
            6144 => (LlamaStyleEncoderConfig.MistralSmall3, SideModels.MistralSmallFlux2, "Mistral-Small-3 encoder (Flux.2 Dev)"),
            _ => throw new NotSupportedException(
                $"Flux.2 {DescribeConfig(config)} (hidden={config.HiddenSize}) is not a recognized variant. Expected hidden ∈ {{3072, 4096, 6144}}."),
        };
    }

    private static string DescribeConfig(Flux2Config config)
    {
        return config.HiddenSize switch
        {
            3072 => "Klein 4B",
            4096 => "Klein 9B",
            6144 => "Dev (32B)",
            _ => $"unknown variant (hidden={config.HiddenSize})",
        };
    }

    /// <summary>Ensures the Mistral-Small-3 HF tokenizer.json is present under the models root, downloading the official conversion on first use (Flux.2 Dev only).</summary>
    private static string EnsureMistralTokenizerJson()
    {
        string path = Path.Combine(RepoPaths.ModelsRoot(), "text_encoders", "mistral3_flux2_tokenizer.json");
        if (!File.Exists(path))
        {
            Logs.Info("[Flux2Recipe] Downloading Mistral tokenizer.json (unsloth/Mistral-Small-3.1-24B-Instruct-2503)...");
            using HuggingFaceClient client = new HuggingFaceClient();
            client.DownloadFileAsync("unsloth/Mistral-Small-3.1-24B-Instruct-2503", "tokenizer.json", path, progress: null, sha256: null, CancellationToken.None)
                .GetAwaiter().GetResult();
        }
        return path;
    }
}
