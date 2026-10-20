using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>MOSS-Image recipe (OpenMOSS, 5B single-stream DiT with a frozen Qwen3.5-9B text encoder and the FLUX.2 VAE;
/// text-to-image, image-to-image, inpaint and native layout prompts). The checkpoint is OpenMOSS's diffusers folder
/// (<c>transformer/</c>, <c>text_encoder/</c>, <c>tokenizer/</c>, <c>vae/</c>; the folder or any file in it), its
/// ComfyUI export's <c>diffusion_models</c> file, or a GGUF of either. The encoder, tokenizer and VAE come from the
/// folder when the checkpoint sits in one, else from the caller's component overrides, else the side models
/// (<see cref="SideModels.Qwen35_9B"/>, which carries its tokenizer, and <see cref="SideModels.Flux2Vae"/>). Base vs SFT
/// is <see cref="MossImageVariants"/>. Drives through <see cref="MossImageRecipePipeline"/>.</summary>
public sealed class MossImageRecipe : IArchitectureRecipe
{
    /// <summary>The diffusers pipeline folder's component subfolders.</summary>
    public const string TransformerFolder = "transformer", TextEncoderFolder = "text_encoder", TokenizerFolder = "tokenizer", VaeFolder = "vae";

    /// <summary>The reference sampling settings (OpenMOSS README): 50 steps at CFG 3, shift 5, 1024².</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 3.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public string Name => "moss-image";

    /// <inheritdoc/>
    /// <remarks>Regional prompts become the model's own layout JSON (<see cref="LayoutPromptBuilder.BuildSceneLayout"/>)
    /// rather than an attention bias. Reference images follow the Ostris / ai-toolkit edit layout, which edit LoRAs are
    /// trained on (the base model ignores them otherwise); they need a Qwen3.5 encoder that carries its vision tower.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.RefEdit | ImageFeatures.Regional
        | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Lora;

    /// <inheritdoc/>
    /// <remarks>Three references, plus an init image that is denoised; an init image in Reference mode is one of the three.</remarks>
    public ImageInputLimits InputLimits => new ImageInputLimits(MossImageConfig.MaxReferenceImages + 1, ReferencesRequireInitImage: false,
        MaxReferences: MossImageConfig.MaxReferenceImages);

    /// <inheritdoc/>
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>: OpenMOSS's ComfyUI tokenizer passes
    /// <c>disable_weights=True</c>, so the prompt is encoded at weight 1 and each token's conditioning row is scaled
    /// afterwards. Nothing is trimmed off the encoder output, so the alignment offset is 0.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting => Diffusion.Prompting.PromptWeightingMode.CondScale;

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public ModelVariantCatalog? Variants => MossImageVariants.Catalog;

    /// <inheritdoc/>
    public bool Matches(string familyId) =>
        string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "moss_image", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        string? root = PipelineRoot(context.CheckpointPath);
        ResolvedModelVariant variant = context.ResolveVariant(MossImageVariants.Catalog);
        List<IDisposable> owned = new List<IDisposable>();
        try
        {
            IReadOnlyList<string> files = HfCheckpointDirectory.CheckpointFiles(TransformerPath(context.CheckpointPath, root));
            CheckpointOpenOptions options = CheckpointOpenOptions.ForNvfp4Consumer(context.Backend);
            CheckpointSource source = files.Count == 1 ? CheckpointSource.Open(files[0], options) : CheckpointSource.OpenShards(files, options);
            owned.Add(source);
            Dictionary<string, Tensor> converted = MossImageCheckpointConverter.ConvertTransformer(source.Weights);
            owned.Add(QuantizedWeightPolicy.PrepareForBackends(converted, context.TransformerBackends));
            MossImageConfig config = MossImageConfig.FromWeights(converted);
            Logs.Info($"[MossImageRecipe] {variant.Variant.DisplayName} ({variant.Source}): {files.Count} file(s) under "
                + $"{Path.GetDirectoryName(files[0])}, hidden={config.HiddenSize}, {config.NumLayers} joint + {config.NumTextBlocks} text blocks.");

            MossImageTransformer transformer = new MossImageTransformer(config);
            owned.Add(transformer);
            // Merged before LoadWeights: device caches are identity-keyed, so a later merge would never be seen.
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(context, new LoraMergeTargets { Transformer = converted }, "MossImageRecipe");
            if (loraStack is not null) owned.Add(loraStack);
            transformer.LoadWeights(converted);

            string encoderPath = ResolveEncoderPath(context, root);
            Qwen35TextEncoder encoder = Qwen35TextEncoder.Load(encoderPath, MossImageConfig.MaxTextTokens, "MOSS-Image");
            owned.Add(encoder);
            ILlmTokenizer tokenizer = ResolveTokenizer(encoder, encoderPath, root, context.Cancel);
            VisionPlaceholderPrompt? placeholders = AttachVision(encoder, tokenizer, encoderPath, context.Components?.TextEncoderVision);

            string vaePath = ResolveVaePath(context, root);
            Flux2VaeComponents vae = Flux2VaeComponents.Load(vaePath, "MossImageRecipe");
            owned.Add(vae);

            MossImagePipeline pipeline = new MossImagePipeline(context.Backend, encoder, transformer, vae.Decoder, vae.Encoder, vae.BnMean, vae.BnVar)
            {
                TextEncoderBackend = context.TextEncoderBackendOrDefault,
                VaeBackend = context.VaeBackendOrDefault,
            };
            owned.Add(pipeline);
            Logs.Info($"[MossImageRecipe] {variant.Variant.DisplayName} ready (encoder {Path.GetFileName(Path.TrimEndingDirectorySeparator(encoderPath))}, "
                + $"VAE {Path.GetFileName(vaePath)}, reference images {(placeholders is null ? "unavailable (no vision tower)" : "on")}).");
            return new MossImageRecipePipeline(pipeline, new MossImagePromptTokenizer(tokenizer), config, variant, owned)
            {
                ReferenceEncoder = placeholders is null ? null : encoder,
                Placeholders = placeholders,
                HasLora = loraStack is not null,
            };
        }
        catch (Exception ex)
        {
            Logs.Error("[MossImageRecipe] Construction failed.", ex);
            new CompositeDisposable([.. Enumerable.Reverse(owned)]).Dispose();
            throw;
        }
    }

    /// <summary>Gives <paramref name="encoder"/> its Qwen3.5 vision tower for edits — from the encoder's own weights (the
    /// Comfy-Org and HF files carry it), else <paramref name="overridePath"/> — and returns the placeholder renderer for
    /// its tokenizer. Null leaves the model text-to-image only: OpenMOSS's own <c>text_encoder/</c> omits the tower.</summary>
    private static VisionPlaceholderPrompt? AttachVision(Qwen35TextEncoder encoder, ILlmTokenizer tokenizer, string encoderPath, string? overridePath)
    {
        string? configJson = Directory.Exists(encoderPath) ? Path.Combine(encoderPath, HfCheckpointDirectory.ConfigFileName) : null;
        Qwen35VisionTower? tower = Qwen35VisionTower.Resolve(encoder.SourceWeights, overridePath,
            configJson is not null && File.Exists(configJson) ? configJson : null, "MOSS-Image");
        if (tower is null)
            return null;
        encoder.AttachVision(tower);
        foreach (string literal in VisionPlaceholderPrompt.RequiredLiterals)
        {
            if (tokenizer.SpecialId(literal) is null)
            {
                Logs.Warning($"[MossImageRecipe] The Qwen3.5 tokenizer has no '{literal}' token; reference images are unavailable.");
                return null;
            }
        }
        VisionPlaceholderPrompt placeholders = new(tokenizer);
        encoder.ImageTokenId = placeholders.ImagePadId;
        return placeholders;
    }

    /// <summary>The OpenMOSS pipeline folder <paramref name="checkpointPath"/> belongs to: the folder itself, or the one
    /// above the <c>transformer/</c> folder it is or lies in. Null for a checkpoint anywhere else (a ComfyUI export in
    /// <c>diffusion_models/</c>), so a <c>transformer</c> folder that merely sits near it is never taken for its components.</summary>
    public static string? PipelineRoot(string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(checkpointPath);
        string? root = HfCheckpointDirectory.FindPipelineRoot(checkpointPath, TransformerFolder);
        if (root is null)
            return null;
        string path = Path.GetFullPath(Path.TrimEndingDirectorySeparator(checkpointPath));
        string transformer = Path.Combine(root, TransformerFolder);
        return path == root || path == transformer || path.StartsWith(transformer + Path.DirectorySeparatorChar, StringComparison.Ordinal)
            ? root
            : null;
    }

    /// <summary>The transformer weights <paramref name="checkpointPath"/> names: the pipeline folder's
    /// <c>transformer/</c> when given the folder itself, else the path as given (a file, a shard or the component folder).</summary>
    public static string TransformerPath(string checkpointPath, string? root) =>
        root is not null && Path.GetFullPath(Path.TrimEndingDirectorySeparator(checkpointPath)) == root
            ? Path.Combine(root, TransformerFolder)
            : checkpointPath;

    /// <summary>The Qwen3.5-9B encoder: the caller's override (a GGUF, a safetensors file or a Hugging Face folder), the
    /// pipeline folder's <c>text_encoder/</c>, else the side model.</summary>
    private static string ResolveEncoderPath(RecipeContext context, string? root)
    {
        string? requested = context.Components?.Qwen;
        if (!string.IsNullOrWhiteSpace(requested) && Directory.Exists(requested))
            return Path.GetFullPath(requested);
        string? folder = root is null ? null : Path.Combine(root, TextEncoderFolder);
        if (string.IsNullOrWhiteSpace(requested) && folder is not null && Directory.Exists(folder))
            return folder;
        // The bf16 safetensors, not the Q8 GGUF: the GGUF's hidden states diverge from the reference model, and the image
        // comes out unrelated to the prompt.
        return ModelFileLocator.RequireOrEnsure(requested, SideModels.Qwen35_9BBf16, "Qwen3.5-9B text encoder", context.Cancel,
            "text_encoders", "clip", "LLM");
    }

    /// <summary>The Qwen3.5 tokenizer: a GGUF encoder's own, the <c>tokenizer.json</c> of the pipeline folder or of an
    /// encoder folder, else the side model. A stray <c>tokenizer.json</c> beside a single encoder file is never read: it
    /// may belong to any model in that folder.</summary>
    private static ILlmTokenizer ResolveTokenizer(Qwen35TextEncoder encoder, string encoderPath, string? root, CancellationToken cancel)
    {
        ILlmTokenizer tokenizer;
        if (encoder.EmbeddedTokenizer is not null)
        {
            tokenizer = encoder.EmbeddedTokenizer;
        }
        else
        {
            string[] candidates =
            [
                .. root is null ? [] : new[] { Path.Combine(root, TokenizerFolder, HfCheckpointDirectory.TokenizerFileName) },
                .. Directory.Exists(encoderPath) ? new[] { Path.Combine(encoderPath, HfCheckpointDirectory.TokenizerFileName) } : [],
            ];
            string json = candidates.FirstOrDefault(File.Exists)
                ?? ModelDownloader.EnsureSideModelAsync(SideModels.Qwen35Tokenizer, downloadIfMissing: true, onProgress: null, cancel)
                    .GetAwaiter().GetResult();
            using FileStream stream = File.OpenRead(json);
            tokenizer = HfTokenizerJson.LoadByteLevelBpe(stream);
            Logs.Info($"[MossImageRecipe] Qwen3.5 tokenizer: {json}.");
        }
        foreach (string literal in MossImagePromptTokenizer.TemplateSpecialTokens)
        {
            if (tokenizer.SpecialId(literal) is null)
                throw new InvalidDataException($"MOSS-Image needs the Qwen3.5 tokenizer, but the one found has no '{literal}' token.");
        }
        return tokenizer;
    }

    /// <summary>The FLUX.2 VAE: the caller's override, the pipeline folder's <c>vae/</c>, else the side model.</summary>
    private static string ResolveVaePath(RecipeContext context, string? root)
    {
        string? requested = context.Components?.Vae;
        string? folder = root is null ? null : Path.Combine(root, VaeFolder);
        if (string.IsNullOrWhiteSpace(requested) && folder is not null && Directory.Exists(folder))
            return HfCheckpointDirectory.WeightFiles(folder)[0];
        // OpenMOSS decodes with the small FLUX.2 decoder, so a single-file checkpoint falls back to that one rather than the
        // full-decoder FLUX.2 VAE, which decodes the same latents with the wrong widths.
        return ModelFileLocator.RequireOrEnsure(requested, SideModels.Flux2VaeSmall, "FLUX.2 small VAE", context.Cancel, "VAE", "vae");
    }
}
