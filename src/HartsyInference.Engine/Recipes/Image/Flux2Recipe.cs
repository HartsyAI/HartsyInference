using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.Core.Memory;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
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
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.Regional | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora
        // Declared only because Flux2RecipePipeline builds a ScheduledPrompt and Flux2Pipeline selects per step.
        // The bit is what keeps <alternate:>/<fromto[N]:> in the prompt at all, so declaring it without consuming
        // one would hand the encoder the literal tag text as prose.
        | ImageFeatures.PromptScheduling;

    /// <summary>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>: every Flux.2 text stack — Klein's Qwen3, the
    /// 8B Klein and Dev's Mistral — disables weights in ComfyUI, so the prompt is encoded at weight 1 and each token's
    /// cond row is scaled afterwards. Nothing is trimmed off this family's encoder output, so the alignment offset
    /// is 0.</summary>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting => Diffusion.Prompting.PromptWeightingMode.CondScale;

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
        // IDisposable, not SafeTensorsLoader: the text encoder now opens through CheckpointSource. Same widening
        // a66afece made for MiniMax-H3 — a foreach over the narrower type compiles and throws at teardown.
        List<IDisposable> loaders = new List<IDisposable>();
        IDisposable? checkpoint = null;
        try
        {
            // One container for either format: a Flux.2 GGUF is a repack of this same file and keeps its BFL key
            // names, so nothing below needs to know which one arrived. Quantized tensors stay packed and dequantize
            // transiently per GEMM.
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath);
            checkpoint = source;
            // Pre-cast BF16 → F16 on CPU (F16 keeps the same footprint as BF16 and F16↔F32 is supported on every op).
            // Quantized tensors stay native — they dequant transiently per-GEMM elsewhere, and a plain CastTo throws
            // for quantized dtypes (needs GgufDequantizer instead).
            Dictionary<string, Tensor> castWeights = new Dictionary<string, Tensor>(source.Weights.Count);
            foreach (KeyValuePair<string, Tensor> kvp in source.Weights)
            {
                castWeights[kvp.Key] = kvp.Value.DType == DType.BF16 ? kvp.Value.CastTo(DType.F16) : kvp.Value;
            }

            Flux2Config config = DetectConfigFromTransformerWeights(castWeights, GuessConfigFromFilename(Path.GetFileName(context.CheckpointPath)) ?? Flux2Config.Klein4B);
            Logs.Info($"[Flux2Recipe] Architecture: hidden={config.HiddenSize}, depth={config.Depth} double + {config.DepthSingleBlocks} single → {DescribeConfig(config)}.");

            int mlpInner = (int)(config.HiddenSize * config.MlpRatio);
            Flux2CheckpointConverter converter = new Flux2CheckpointConverter(config.HiddenSize, mlpInner);
            Dictionary<string, Tensor> converted = converter.ConvertTransformer(castWeights);
            castWeights.Clear();
            // Any quant this backend has no packed-weight kernel for widens here rather than failing inside the
            // first GEMM, minutes into a generation.
            QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareForBackends(converted, context.TransformerBackends);
            // Tracked immediately so a failure further down frees the widened copies rather than
            // leaving them to the finalizer.
            checkpoint = new CompositeDisposable(source, prepared);

            Flux2Transformer transformer = new Flux2Transformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Transformer = converted },
                "Flux2Recipe");
            transformer.LoadWeights(converted);
            converted.Clear();

            // Resolve + load the variant's text encoder. Opened through the container with Nvfp4ToFp8 so the
            // nvfp4 groups land at fp8 rather than F16 — Klein 9B's encoder is 173 nvfp4 groups plus 76 fp8, and
            // the F16 expansion costs ~3 GB more for no accuracy the fp8 GEMM path does not already give. The
            // container also OWNS what it allocates, which the raw loader route did not: LlamaStyleEncoder.Dispose
            // never frees projection tensors, so every dequantized weight leaked until process exit.
            (LlamaStyleEncoderConfig encoderConfig, ModelAsset encoderAsset, string encoderLabel) = ResolveTextEncoderForVariant(config);
            string encoderPath = ModelDownloader.EnsureSideModelAsync(encoderAsset, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            CheckpointSource encoderSource = CheckpointSource.Open(
                encoderPath, new CheckpointOpenOptions { Nvfp4ToFp8 = true });
            loaders.Add(encoderSource);
            Dictionary<string, Tensor> qwenRaw = new Dictionary<string, Tensor>(encoderSource.Weights, StringComparer.Ordinal);

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
            return new Flux2RecipePipeline(pipeline, config, qwenTokenizer, mistralTokenizer, MistralDevSystemPrompt, encoder, loaders, checkpoint, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[Flux2Recipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            checkpoint?.Dispose();
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

    /// <summary>Picks the LlamaStyleEncoder preset + side-model asset for the variant. Klein 4B → Qwen3-4B; Klein 9B → Qwen3-8B (nvfp4-mixed, dequantized at load); Dev → Mistral-Small-3.</summary>
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
