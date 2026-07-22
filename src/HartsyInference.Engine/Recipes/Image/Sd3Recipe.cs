using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Stable Diffusion 3 / 3.5 recipe (MMDiT, triple text encoder): the all-in-one checkpoint bundles the transformer, CLIP-L, CLIP-G, an optional T5-XXL, and the VAE. Lifted from the SwarmUI backend's <c>Sd3Loader</c> (all-in-one branch). CLIP-L, CLIP-G, and VAE come from the checkpoint; when the checkpoint omits T5, the canonical <see cref="SideModels.T5XxlEnconly"/> is resolved as a side model. Constructs and drives through <see cref="Sd3RecipePipeline"/>.</summary>
public sealed class Sd3Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "sd3";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "sd3", StringComparison.OrdinalIgnoreCase);

    /// <summary>SD 3.5's official sampling settings: 28 steps at CFG 7.0, 1024x1024 (diffusers <c>StableDiffusion3Pipeline.__call__</c>, mirrored by <c>GenerationDefaults.Sd35</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 28, CfgScale = 7.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor split-file CLIP-L / CLIP-G / T5 / VAE overrides from ImageRequest.Components
        // (the SwarmUI loader read input.Get(T2IParamTypes.ClipLModel/ClipGModel/T5XXLModel/VAE) in split mode).
        // The recipe uses the all-in-one components bundled in the checkpoint.
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader mainLoader) = Sd3CheckpointConverter.LoadAndConvert(context.CheckpointPath);
        if (converted.Transformer.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException("SD3 checkpoint has no transformer weights.");
        }
        if (converted.ClipL.Count == 0 || converted.ClipG.Count == 0 || converted.Vae.Count == 0)
        {
            mainLoader.Dispose();
            throw new InvalidOperationException("SD3 all-in-one mode: this checkpoint is missing CLIP-L/CLIP-G/VAE components. Pick a complete SD3 file.");
        }

        int patchEmbedOutChannels = DetectPatchEmbedOutChannels(converted.Transformer);
        Sd3Config sd3Config = Sd3Config.FromWeightShape(patchEmbedOutChannels);
        Logs.Info($"[Sd3Recipe] Building MMDiT (patch_embed out_channels={patchEmbedOutChannels}).");
        Sd3Transformer transformer = new Sd3Transformer(sd3Config);
        transformer.LoadWeights(converted.Transformer);

        ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
        clipL.LoadWeights(converted.ClipL, prefix: "text_model");

        ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
        clipG.LoadWeights(converted.ClipG, prefix: "text_model");

        // T5-XXL: bundled when present, else resolve the canonical encoder-only side model.
        SafeTensorsLoader? t5Loader = null;
        T5TextEncoder? t5 = null;
        T5Tokenizer? t5Tokenizer = null;
        if (converted.T5.Count > 0)
        {
            Logs.Info("[Sd3Recipe] Building T5-XXL encoder (bundled).");
            t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(converted.T5);
            t5Tokenizer = new T5Tokenizer(maxLength: 256);
        }
        else
        {
            string t5Path = ModelDownloader.EnsureSideModelAsync(SideModels.T5XxlEnconly, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            t5Loader = new SafeTensorsLoader();
            t5Loader.Load(t5Path);
            Dictionary<string, Tensor> t5Weights = StripT5Prefix(t5Loader.GetAllTensors());
            if (t5Weights.Count == 0)
            {
                t5Loader.Dispose();
                mainLoader.Dispose();
                throw new InvalidOperationException($"T5 model file '{t5Path}' has no usable T5 tensors.");
            }
            Logs.Info("[Sd3Recipe] Building T5-XXL encoder (side model).");
            t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            t5.LoadWeights(t5Weights);
            t5Tokenizer = new T5Tokenizer(maxLength: 256);
        }

        VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sd3);
        vaeDecoder.LoadWeights(converted.Vae);
        VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Sd3);
        vaeEncoder.LoadWeights(converted.Vae);

        Sd3Pipeline pipeline = new Sd3Pipeline(context.Backend, clipL, clipG, t5, transformer, vaeDecoder, vaeEncoder);
        Logs.Info($"[Sd3Recipe] SD3 ready (T5={t5 is not null}).");
        return new Sd3RecipePipeline(pipeline, new ClipTokenizer(), t5Tokenizer, mainLoader, t5Loader);
    }

    /// <summary>SD3's MMDiT patch_embed.weight is [embedDim, 16, 2, 2]; the leading dim selects the arch (Medium 1536, 3.5 Large 2432). Reads the ComfyUI (<c>pos_embed.proj.weight</c>) or diffusers (<c>patch_embed.proj.weight</c>) key, falling back to Medium.</summary>
    private static int DetectPatchEmbedOutChannels(Dictionary<string, Tensor> transformer)
    {
        if (transformer.TryGetValue("pos_embed.proj.weight", out Tensor? t) && t.Shape.Rank >= 1)
        {
            return (int)t.Shape[0];
        }
        if (transformer.TryGetValue("patch_embed.proj.weight", out Tensor? t2) && t2.Shape.Rank >= 1)
        {
            return (int)t2.Shape[0];
        }
        return 1536;
    }

    /// <summary>Strips Comfy's <c>text_encoders.t5xxl.transformer.</c> prefix from a standalone T5-XXL safetensors file and drops the <c>position_ids</c> buffer the encoder doesn't consume.</summary>
    private static Dictionary<string, Tensor> StripT5Prefix(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        const string ComfyPrefix = "text_encoders.t5xxl.transformer.";
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string key = kv.Key.StartsWith(ComfyPrefix, StringComparison.Ordinal) ? kv.Key[ComfyPrefix.Length..] : kv.Key;
            if (!key.EndsWith("position_ids", StringComparison.Ordinal))
            {
                result[key] = kv.Value;
            }
        }
        return result;
    }
}
