using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>MiniMax-H3-Image recipe (text-to-image). The checkpoint is the audio-free H3 DiT; the Qwen3-VL-32B text
/// encoder and the single-frame image VAE are found next to it. Declares no composition features yet: editing (5b),
/// LoRA (5c), the prompt enhancer (5d) and the memory features (5e) are later phases.
/// <para>UNTESTED. Written without weights or a GPU. See <see cref="MiniMaxH3ImageRecipePipeline"/> for the TODO(gpu)
/// list.</para></summary>
public sealed class MiniMaxH3ImageRecipe : IArchitectureRecipe
{
    private const string FamilyId = "minimax-h3-image";
    private const string ImageVaeFile = "minimax_h3_image_vae_fp16.safetensors";

    /// <inheritdoc/>
    public string Name => FamilyId;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, FamilyId, StringComparison.OrdinalIgnoreCase);

    /// <summary>Text-to-image only for now. TODO(phase 5b): add RefEdit once reference conditioning is wired.
    /// TODO(phase 5c): add Lora once merges are verified on the bf16 base.</summary>
    public ImageFeatures Supports => ImageFeatures.None;

    /// <summary>The image build is tuned at 2048x2048 (ComfyUI's node default and the official minimum area). The 16 MP
    /// sizes are supported but are not the default.</summary>
    public ImageDefaults Defaults { get; } = new ImageDefaults { Steps = 6, CfgScale = 1.0f, Width = 2048, Height = 2048 };

    /// <inheritdoc/>
    /// <remarks>TODO(gpu): the video recipe declares CondScale for the same encoder. Declare it here once the image
    /// pipeline applies it and the ledger test pins the family.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting => Diffusion.Prompting.PromptWeightingMode.None;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        string checkpointDir = Path.GetDirectoryName(Path.GetFullPath(context.CheckpointPath))
            ?? throw new ArgumentException("The checkpoint path has no directory.", nameof(context));
        // TODO(gpu): honour context.Components for the text encoder and image VAE, as the video recipe does through
        // MiniMaxH3Assets. The lookup below assumes the Comfy-Org repack layout.
        string imageVae = FindFile(checkpointDir, ImageVaeFile, "vae")
            ?? throw new FileNotFoundException(
                $"MiniMax-H3-Image needs {ImageVaeFile} next to the checkpoint (vae/).");
        string textEncoder = FindFile(checkpointDir, "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders")
            ?? FindFile(checkpointDir, "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "text_encoders")
            ?? throw new FileNotFoundException(
                "MiniMax-H3-Image needs its Qwen3-VL-32B text encoder next to the checkpoint (text_encoders/).");

        List<IDisposable> loaders = new List<IDisposable>();
        bool handedOff = false;
        try
        {
            // TODO(gpu): context.CheckpointPath is assumed to be the DiT file. Confirm against a real request.
            MiniMaxH3Transformer transformer = MiniMaxH3Components.LoadTransformer(context.CheckpointPath, loaders,
                DType.F32, context.TransformerBackends, out MiniMaxH3Config config, out Dictionary<string, Tensor> _);
            MiniMaxH3TextEncoder encoder = MiniMaxH3Components.LoadTextEncoder(textEncoder, loaders);
            Qwen2Tokenizer tokenizer = MiniMaxH3Components.LoadTokenizer(null);
            MiniMaxH3VideoVaeDecoder decoder = LoadImageVae(imageVae, loaders);

            MiniMaxH3ImageRecipePipeline pipeline = new MiniMaxH3ImageRecipePipeline(transformer, config, decoder,
                encoder, tokenizer, context.Backend, context.TextEncoderBackendOrDefault,
                context.VaeBackendOrDefault, loaders);
            handedOff = true;
            return pipeline;
        }
        finally
        {
            // Until the pipeline owns them, release everything loaded so far.
            if (!handedOff)
            {
                foreach (IDisposable loader in loaders)
                {
                    loader.Dispose();
                }
            }
        }
    }

    /// <summary>The 2D decoder config is detected from the image VAE's shapes; the constants come from the image
    /// baseline. TODO(gpu): the detection path has not been run on the real file.</summary>
    private static MiniMaxH3VideoVaeDecoder LoadImageVae(string file, List<IDisposable> loaders)
    {
        CheckpointSource source = CheckpointSource.Open(file);
        loaders.Add(source);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(source.Weights, StringComparer.Ordinal);
        MiniMaxH3VideoVaeConfig config = MiniMaxH3VideoVaeConfig.Detect(weights);
        MiniMaxH3VideoVaeDecoder decoder = new MiniMaxH3VideoVaeDecoder(config);
        decoder.LoadWeights(weights);
        return decoder;
    }

    private static string? FindFile(string checkpointDir, string fileName, string subdir)
    {
        string[] candidates =
        [
            Path.Combine(checkpointDir, fileName),
            Path.Combine(checkpointDir, subdir, fileName),
            Path.Combine(checkpointDir, "..", subdir, fileName),
        ];
        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return Path.GetFullPath(candidate);
            }
        }
        return null;
    }
}
