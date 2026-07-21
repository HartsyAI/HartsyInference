using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Lumina-Image-2.0 recipe (Alpha-VLLM 2B NextDiT, Apache-2.0). Lifted from the SwarmUI backend's <c>Lumina2Loader</c>: the checkpoint is the diffusers-format transformer; the Gemma-2-2B text encoder (<see cref="SideModels.Gemma2_2B"/>) and the FLUX.1 16-channel VAE (<see cref="SideModels.FluxAe"/>) resolve as side models. Conditioning is a LIVE Gemma-2 encode (system-prompt-prefixed, <c>hidden_states[-2]</c>) driven inside <see cref="Lumina2RecipePipeline"/>; the pipeline itself owns the timestep inversion and velocity negation. Constructs and drives through <see cref="Lumina2RecipePipeline"/>.</summary>
public sealed class Lumina2Recipe : IArchitectureRecipe
{
    /// <summary>Lumina-Image-2.0's system prompt, verbatim from the official pipeline.</summary>
    private const string SystemPrompt =
        "You are an assistant designed to generate superior images with the superior degree of " +
        "image-text alignment based on textual prompts or user prompts.";

    /// <summary>Gemma-2 SentencePiece model (256k vocab). Ungated on the Lumina-Image-2.0 repo.</summary>
    private const string TokenizerRepo = "Alpha-VLLM/Lumina-Image-2.0";
    private const string TokenizerRepoPath = "tokenizer/tokenizer.model";
    private const string TokenizerSha256 = "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2";

    /// <inheritdoc/>
    public string Name => "lumina2";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "lumina2", StringComparison.OrdinalIgnoreCase);

    /// <summary>Lumina-Image-2.0's official sampling settings: 30 steps at guidance 4.0, 1024x1024 (diffusers <c>Lumina2Pipeline.__call__</c>, mirrored by <c>GenerationDefaults.Lumina2</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 30, CfgScale = 4.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): user overrides from ImageRequest.Components (SwarmUI read GemmaModel / VAE) + img2img/
        // inpaint are deferred — this ports the text-to-image core with canonical side models.
        string tevPath = ModelDownloader.EnsureSideModelAsync(SideModels.Gemma2_2B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            (Lumina2CheckpointConverter.ConvertedWeights converted, IReadOnlyList<SafeTensorsLoader> transformerLoaders) =
                Lumina2CheckpointConverter.LoadAndConvert(context.CheckpointPath);
            loaders.AddRange(transformerLoaders);
            Dictionary<string, Tensor> transformerWeights = CastWeightsToF32(converted.Transformer);

            Lumina2Config config = Lumina2Config.FromWeights(transformerWeights);
            Logs.Info($"[Lumina2Recipe] Building transformer (2B NextDiT).");
            Lumina2Transformer transformer = new Lumina2Transformer(config);
            transformer.LoadWeights(transformerWeights);

            SafeTensorsLoader teLoader = new SafeTensorsLoader();
            teLoader.Load(tevPath);
            loaders.Add(teLoader);
            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Gemma2_2B);
            textEncoder.LoadWeights(teLoader.GetAllTensors());

            (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
            loaders.Add(vaeLoader);
            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
            vae.LoadWeights(vaeWeights);

            string tokenizerPath = EnsureTokenizer(tevPath);
            GemmaTokenizer tokenizer = new GemmaTokenizer(tokenizerPath, maxLength: 512);

            Lumina2Pipeline pipeline = new Lumina2Pipeline(context.Backend, transformer, vae, config);
            Logs.Info("[Lumina2Recipe] Lumina-2 ready.");
            return new Lumina2RecipePipeline(pipeline, context.Backend, textEncoder, tokenizer, transformer, SystemPrompt, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[Lumina2Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dt = kvp.Value.DType;
            f32[kvp.Key] = (dt == DType.F16 || dt == DType.BF16) ? kvp.Value.CastTo(DType.F32) : kvp.Value;
        }
        return f32;
    }

    /// <summary>Locates the Gemma-2 SentencePiece model: next to the encoder checkpoint, then the canonical models path, else downloads it (hash-verified) under the models root.</summary>
    private static string EnsureTokenizer(string encoderPath)
    {
        string encoderDir = Path.GetDirectoryName(encoderPath) ?? "";
        foreach (string candidate in new[]
        {
            Path.Combine(encoderDir, "gemma2_tokenizer.model"),
            Path.Combine(encoderDir, "tokenizer.model"),
        })
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }
        string dir = Path.Combine(RepoPaths.ModelsRoot(), "text_encoders", "Gemma2");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "tokenizer.model");
        if (!File.Exists(path))
        {
            Logs.Info($"[Lumina2Recipe] Downloading Gemma-2 tokenizer.model ({TokenizerRepo})...");
            using HuggingFaceClient client = new HuggingFaceClient();
            client.DownloadFileAsync(TokenizerRepo, TokenizerRepoPath, path, progress: null, sha256: TokenizerSha256, CancellationToken.None)
                .GetAwaiter().GetResult();
        }
        return path;
    }
}
