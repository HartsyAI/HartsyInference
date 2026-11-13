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

    /// <inheritdoc/>
    public string Name => FamilyId;

    /// <inheritdoc/>
    public bool Matches(string familyId) =>
        string.Equals(familyId, FamilyId, StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "minimax-h3-image-bf16", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "minimax-h3-image-w4a8", StringComparison.OrdinalIgnoreCase);

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
        // The VAE and the text encoder are found from the DiT's location, or taken from the request's component overrides.
        MiniMaxH3ImageAssets assets = MiniMaxH3ImageAssets.Resolve(context.CheckpointPath, context.Components);

        List<IDisposable> loaders = new List<IDisposable>();
        bool handedOff = false;
        try
        {
            // TODO(gpu): context.CheckpointPath is assumed to be the DiT file. Confirm against a real request.
            MiniMaxH3Transformer transformer = MiniMaxH3Components.LoadTransformer(context.CheckpointPath, loaders,
                DType.F32, context.TransformerBackends, out MiniMaxH3Config config, out Dictionary<string, Tensor> ditWeights);
            // LoadTransformer converts the weights but does not bind them; the video recipe binds them the same way.
            transformer.LoadWeights(ditWeights);
            MiniMaxH3TextEncoder encoder = MiniMaxH3Components.LoadTextEncoder(assets.TextEncoder, loaders);
            Qwen2Tokenizer tokenizer = MiniMaxH3Components.LoadTokenizer(null);
            MiniMaxH3VideoVaeDecoder decoder = LoadImageVae(assets.ImageVae, loaders);

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
    internal static MiniMaxH3VideoVaeDecoder LoadImageVae(string file, List<IDisposable> loaders)
    {
        CheckpointSource source = CheckpointSource.Open(file);
        loaders.Add(source);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(source.Weights, StringComparer.Ordinal);
        MiniMaxH3VideoVaeConfig config = MiniMaxH3VideoVaeConfig.Detect(weights);
        MiniMaxH3VideoVaeDecoder decoder = new MiniMaxH3VideoVaeDecoder(config);
        decoder.LoadWeights(weights);
        return decoder;
    }
}
