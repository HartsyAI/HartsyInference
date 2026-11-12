using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;
using HartsyInference.Engine.Variants;
using HartsyInference.Core.Runtime;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Flux.2 recipe (BFL; Klein 4B / Klein 9B / Dev). Variant comes from <see cref="Flux2Variants"/>, which reads the transformer's hidden size (3072 → Klein 4B, 4096 → Klein 9B, 6144 → Dev). Lifted from the SwarmUI backend's <c>Flux2Loader</c>: the checkpoint is the transformer; the matching text encoder — Qwen3-4B (<see cref="SideModels.Qwen3_4B"/>), Qwen3-8B (<see cref="SideModels.Qwen3_8B_Fp4Mixed"/>), or Mistral-Small-3 (<see cref="SideModels.MistralSmallFlux2"/>) — and the Flux.2 VAE (<see cref="SideModels.Flux2Vae"/>, 32-channel latent + BatchNorm stats) resolve as side models. Constructs and drives through <see cref="Flux2RecipePipeline"/>.</summary>
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
    /// <remarks>Matches the catalog id <c>flux2-klein-4b</c> as well: the registry resolves a catalog entry by its id, and the
    /// variant still comes from the transformer's hidden size.</remarks>
    public bool Matches(string familyId) =>
        string.Equals(familyId, "flux2", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "flux2-klein-4b", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "iris-z-small", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "iris-z-big", StringComparison.OrdinalIgnoreCase);

    /// <summary>What the Iris-Z builds apply: the init image is a given grid (the photo for <c>depth</c>, the depth map for
    /// <c>paint</c>), never a noised start, and LoRAs merge as for their FLUX.2 bases. Image-to-image, inpaint, regions,
    /// prompt scheduling and the refiner are not part of the model.</summary>
    public const ImageFeatures IrisZSupports = ImageFeatures.RefEdit | ImageFeatures.Lora;

    /// <summary>Iris-Z-small's official generation settings (speridlabs/iris-z model card): 50 steps, true CFG 4.0.</summary>
    public static ImageDefaults IrisZSmallDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 4.0f, Width = 1024, Height = 1024 };

    /// <summary>Iris-Z-big's official generation settings (speridlabs/iris-z model card): 50 steps, true CFG 3.5.</summary>
    public static ImageDefaults IrisZBigDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageFeatures SupportsFor(ResolvedModelVariant? variant) => Flux2Variants.IsIrisZ(variant) ? IrisZSupports : Supports;

    /// <summary>Flux.2 Dev's official sampling settings: 50 steps at guidance 3.5, 1024x1024 (SwarmUI <c>Model Support.md</c>, Flux.2: Flux Guidance defaults to 3.5, 4 is the optional tweak); the distilled Klein variants narrow this via <see cref="DefaultsFor"/>.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Revision = 2, Steps = 50, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <summary>Flux.2 Klein's sampling settings: 8 steps, guidance-free (CFG 1.0), SwarmUI's settings for the distilled
    /// Klein checkpoints (<c>Model Support.md</c>, Flux.2 Klein), which carry no guidance embedding. <see cref="Flux2RecipePipeline"/> does not read CfgScale yet.</summary>
    public static ImageDefaults KleinDefaults { get; } = new ImageDefaults { Revision = 2, Steps = 8, CfgScale = 1.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ModelVariantCatalog? Variants => Flux2Variants.Catalog;

    /// <inheritdoc/>
    /// <remarks>Keyed on the variant's guidance embedding, the property that makes Klein guidance-free, so a variant added
    /// to <see cref="Flux2Variants"/> gets the right defaults without a second list here.</remarks>
    public ImageDefaults DefaultsFor(ResolvedModelVariant? variant)
    {
        if (variant is null)
        {
            return FamilyDefaults;
        }
        if (Flux2Variants.IsIrisZ(variant))
        {
            return variant.Is(Flux2Variants.IrisZSmall) ? IrisZSmallDefaults : IrisZBigDefaults;
        }
        return Flux2Variants.ConfigFor(variant).GuidanceEmbed ? FamilyDefaults : KleinDefaults;
    }

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): split-file / user overrides from ImageRequest.Components, and img2img are deferred —
        // this ports the text-to-image core.
        // IDisposable, not SafeTensorsLoader: the text encoder now opens through CheckpointSource. Same widening
        // a66afece made for MiniMax-H3 — a foreach over the narrower type compiles and throws at teardown.
        List<IDisposable> loaders = new List<IDisposable>();
        IDisposable? checkpoint = null;
        // Handed to the recipe pipeline on success; released here when construction fails part-way.
        MergedLoraStack? loraStack = null;
        LlamaStyleEncoder? encoder = null;
        Qwen3Tokenizer? qwenTokenizer = null;
        ErnieTokenizer? mistralTokenizer = null;
        try
        {
            // One container for either format: a Flux.2 GGUF is a repack of this same file and keeps its BFL key
            // names, so nothing below needs to know which one arrived. Quantized tensors stay packed and dequantize
            // transiently per GEMM.
            // A sharded release (Iris-Z ships a transformer index and its shards) opens as one key set.
            IReadOnlyList<string> files = HfCheckpointDirectory.CheckpointFiles(context.CheckpointPath);
            CheckpointSource source = files.Count == 1 ? CheckpointSource.Open(files[0]) : CheckpointSource.OpenShards(files);
            checkpoint = source;
            // Pre-cast BF16 → F16 on CPU (F16 keeps the same footprint as BF16 and F16↔F32 is supported on every op).
            // Quantized tensors stay native — they dequant transiently per-GEMM elsewhere, and a plain CastTo throws
            // for quantized dtypes (needs GgufDequantizer instead).
            Dictionary<string, Tensor> castWeights = new Dictionary<string, Tensor>(source.Weights.Count);
            foreach (KeyValuePair<string, Tensor> kvp in source.Weights)
            {
                castWeights[kvp.Key] = kvp.Value.DType == DType.BF16 ? kvp.Value.CastTo(DType.F16) : kvp.Value;
            }

            ResolvedModelVariant variant = context.ResolveVariant(Flux2Variants.Catalog);
            Flux2Config config = Flux2Variants.ConfigFor(variant);
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

            // The release's empty-prompt tensor loads first: it can need a download, and nothing heavier exists yet to strand.
            Tensor? irisEmptyPrompt = Flux2Variants.IsIrisZ(variant) ? LoadIrisZEmptyPrompt(context, variant, loaders) : null;

            Flux2Transformer transformer = new Flux2Transformer(config);
            // The transformer owns its fixed step-graph buffers and modality rows; the pipeline releases it with the loaders.
            loaders.Add(transformer);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Transformer = converted },
                "Flux2Recipe");
            transformer.LoadWeights(converted);
            converted.Clear();

            // Resolve + load the variant's text encoder. Opened through the container so the nvfp4 groups land where
            // the backend can use them — packed on a card that multiplies them natively, at fp8 elsewhere (Klein 9B's
            // encoder is 173 nvfp4 groups plus 76 fp8; the F16 expansion costs ~3 GB more for no accuracy the fp8
            // GEMM path does not already give). The container also OWNS what it allocates, which the raw loader route
            // did not: LlamaStyleEncoder.Dispose never frees projection tensors, so every dequantized weight leaked.
            (LlamaStyleEncoderConfig encoderConfig, ModelAsset encoderAsset, string encoderLabel) = ResolveTextEncoderForVariant(config);
            string encoderPath = ModelDownloader.EnsureSideModelAsync(encoderAsset, onProgress: null, context.Cancel).GetAwaiter().GetResult();
            CheckpointSource encoderSource = CheckpointSource.Open(
                encoderPath, CheckpointOpenOptions.ForNvfp4Consumer(context.Backend));
            loaders.Add(encoderSource);
            Dictionary<string, Tensor> qwenRaw = new Dictionary<string, Tensor>(encoderSource.Weights, StringComparer.Ordinal);

            // LoadWeights runs the raw dict through TextEncoderQuantNormalizer (folds fp8 weight_scale companions,
            // dequantizes U8-packed NVFP4, drops .comfy_quant blobs) — hand it the RAW dict, do not pre-cast.
            encoder = new LlamaStyleEncoder(encoderConfig);
            encoder.LoadWeights(qwenRaw);

            // Flux.2 VAE (distinct from Flux.1's ae): 32-channel latent + BatchNorm running stats stored alongside.
            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Flux2Vae, onProgress: null, context.Cancel).GetAwaiter().GetResult();
            Flux2VaeComponents vae = Flux2VaeComponents.Load(vaePath, "Flux2Recipe");
            loaders.Add(vae);

            // Tokenizer: Klein → embedded Qwen3 vocab/merges; Dev → Mistral tekken (HF tokenizer.json).
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

            if (irisEmptyPrompt is not null)
            {
                if (vae.Encoder is null)
                {
                    throw new NotSupportedException("Iris-Z needs the FLUX.2 VAE encoder; the VAE file found is decode-only.");
                }
                IrisZPipeline irisPipeline = new IrisZPipeline(
                    context.Backend, encoder, transformer, vae.Decoder, vae.Encoder, vae.BnMean, vae.BnVar, config, irisEmptyPrompt,
                    IrisZSkyCeiling(variant), IrisZTextEncoderLayers(config));
                Logs.Info($"[Flux2Recipe] Iris-Z ready ({DescribeConfig(config)}).");
                return new IrisZRecipePipeline(irisPipeline,
                    new Flux2PromptTokenizer(config, qwenTokenizer, mistralTokenizer, MistralDevSystemPrompt), IrisZSkyCeiling(variant),
                    variant.Id, qwenTokenizer, mistralTokenizer, encoder, loaders, checkpoint, loraStack);
            }

            Flux2Pipeline pipeline = new Flux2Pipeline(
                context.Backend, encoder, transformer, vae.Decoder, vae.Encoder,
                vae.BnMean, vae.BnVar, config,
                hiddenLayers: null,
                bnEps: 1e-5f);
            Logs.Info($"[Flux2Recipe] Flux.2 ready ({DescribeConfig(config)}).");
            return new Flux2RecipePipeline(pipeline, config, qwenTokenizer, mistralTokenizer, MistralDevSystemPrompt, encoder, loaders, checkpoint, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[Flux2Recipe] Construction failed.", ex);
            // Every handle is released even if one throws; the LoRA stack goes last, it owns the merged weights.
            new CompositeDisposable([qwenTokenizer, mistralTokenizer, encoder, .. loaders, checkpoint, loraStack]).Dispose();
            throw;
        }
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

    /// <summary>The sky ceiling each Iris-Z build was trained with (its <c>config.json</c>): Iris-Z-big keeps the top of
    /// the range for sky, Iris-Z-small has no sky class.</summary>
    private static float IrisZSkyCeiling(ResolvedModelVariant variant) => variant.Is(Flux2Variants.IrisZBig) ? 0.95f : 1.0f;

    /// <summary>The text-encoder taps the reference uses: 10/20/30 for Mistral-Small-3, 9/18/27 for Qwen3.</summary>
    private static int[] IrisZTextEncoderLayers(Flux2Config config) =>
        config.TextEncoderType == Flux2TextEncoderType.Mistral ? [10, 20, 30] : [9, 18, 27];

    /// <summary>Loads the release's <c>empty_prompt</c> tensor as F32: the file beside the checkpoint when it is there, the
    /// variant's side model otherwise. Whatever keeps it alive joins <paramref name="loaders"/>.</summary>
    private static Tensor LoadIrisZEmptyPrompt(RecipeContext context, ResolvedModelVariant variant, List<IDisposable> loaders)
    {
        ModelAsset asset = variant.Is(Flux2Variants.IrisZBig) ? SideModels.IrisZBigEmptyPrompt : SideModels.IrisZSmallEmptyPrompt;
        string? folder = Directory.Exists(context.CheckpointPath) ? context.CheckpointPath : Path.GetDirectoryName(context.CheckpointPath);
        string beside = Path.Combine(folder ?? "", asset.FileName);
        string path = File.Exists(beside)
            ? beside
            : ModelDownloader.EnsureSideModelAsync(asset, onProgress: null, context.Cancel).GetAwaiter().GetResult();
        CheckpointSource source = CheckpointSource.Open(path);
        loaders.Add(source);
        if (!source.Weights.TryGetValue("empty_prompt", out Tensor? stored))
        {
            throw new InvalidDataException($"'{path}' has no 'empty_prompt' tensor.");
        }
        Tensor emptyPrompt = TensorCasts.EnsureF32(stored);
        if (!ReferenceEquals(emptyPrompt, stored))
        {
            loaders.Add(emptyPrompt);
        }
        return emptyPrompt;
    }

    private static string DescribeConfig(Flux2Config config)
    {
        if (config.ModalityCount > 0)
        {
            return config.HiddenSize == 4096 ? "Iris-Z-small" : "Iris-Z-big";
        }
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
        string path = ModelFolders.Under(RepoPaths.ModelsRoot(), ModelFolders.TextEncoders, "mistral3_flux2_tokenizer.json");
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
