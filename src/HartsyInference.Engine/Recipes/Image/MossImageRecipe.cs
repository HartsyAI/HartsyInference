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
    /// rather than an attention bias. There is no reference-image path: OpenMOSS lists reference generation as future work.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.Regional
        | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Lora;

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
                + $"VAE {Path.GetFileName(vaePath)}).");
            return new MossImageRecipePipeline(pipeline, new MossImagePromptTokenizer(tokenizer), config, variant, owned);
        }
        catch (Exception ex)
        {
            Logs.Error("[MossImageRecipe] Construction failed.", ex);
            new CompositeDisposable([.. Enumerable.Reverse(owned)]).Dispose();
            throw;
        }
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
        return ModelFileLocator.RequireOrEnsure(requested, SideModels.Qwen35_9B, "Qwen3.5-9B text encoder", context.Cancel,
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
        return ModelFileLocator.RequireOrEnsure(requested, SideModels.Flux2Vae, "FLUX.2 VAE", context.Cancel, "VAE", "vae");
    }
}
