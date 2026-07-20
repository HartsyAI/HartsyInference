using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>OmniGen 2 recipe (VectorSpaceLab, 4B MMDiT, Apache-2.0). The checkpoint is the fp16 transformer; the Qwen2.5-VL-3B text encoder (<see cref="SideModels.Qwen2_5_VL_3B"/>) and the FLUX.1 16-channel VAE (<see cref="SideModels.FluxAe"/>) resolve as side models. Lifted from the SwarmUI backend's <c>OmniGen2Loader</c>; the text encoder lives OUTSIDE the pipeline, so <see cref="OmniGen2RecipePipeline"/> live-encodes the caption and calls the embeddings overload.</summary>
public sealed class OmniGen2Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "omnigen2";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "omnigen2", StringComparison.OrdinalIgnoreCase);

    /// <summary>OmniGen2's official sampling settings: 28 steps at text-guidance 4.0, 1024x1024 (<c>GenerationDefaults.OmniGen2</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 28, CfgScale = 4.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): the reference-image edit path (Init Image / Prompt Images → VAE ref latents + dual
        // text/image guidance via OmniGen2Pipeline.EditFromEmbeddings) is deferred, as are user text-encoder / VAE
        // overrides from ImageRequest.Components (the loader read T2IParamTypes.QwenModel / T2IParamTypes.VAE).
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            (OmniGen2CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader transformerLoader) = OmniGen2CheckpointConverter.LoadAndConvert(context.CheckpointPath);
            loaders.Add(transformerLoader);
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"OmniGen2 checkpoint '{Path.GetFileName(context.CheckpointPath)}' contains no transformer weights.");
            }
            Logs.Info($"[OmniGen2Recipe] Parsed checkpoint: {converted.Transformer.Count} transformer tensors.");

            // BF16 (not F16, not F32): F16 overflows under CFG (NaN → all-black at cfg >= 5), F32 doubles the
            // footprint to ~16 GB. BF16 keeps 8 GB with F32's exponent range — the validated e2e recipe.
            OmniGen2Config config = OmniGen2Config.V1;
            OmniGen2Transformer transformer = new OmniGen2Transformer(config);
            transformer.LoadWeights(CastWeightsToBf16(converted.Transformer));

            string encoderPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen2_5_VL_3B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader teLoader = new SafeTensorsLoader();
            teLoader.Load(encoderPath);
            loaders.Add(teLoader);
            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_3B);
            textEncoder.LoadWeights(teLoader.GetAllTensors());

            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
            loaders.Add(vaeLoader);
            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
            vae.LoadWeights(vaeWeights);
            // The encoder half is what the deferred reference-image edit path needs; constructing with it keeps
            // OmniGen2Pipeline.EditFromEmbeddings reachable once E-IMG-4 lands.
            VaeEncoder vaeEncoder = new VaeEncoder(VaeConfig.Flux);
            vaeEncoder.LoadWeights(vaeWeights);

            OmniGen2Pipeline pipeline = new OmniGen2Pipeline(context.Backend, transformer, vae, vaeEncoder, config);
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 512);
            Logs.Info("[OmniGen2Recipe] OmniGen 2 ready (Qwen2.5-VL-3B live encode, FLUX.1 VAE).");
            return new OmniGen2RecipePipeline(pipeline, tokenizer, textEncoder, transformer, context.Backend, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[OmniGen2Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Casts F16/F32 weights to BF16 (others pass through) — same footprint as F16 with F32's exponent range, so CFG-amplified activations cannot overflow.</summary>
    private static Dictionary<string, Tensor> CastWeightsToBf16(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            DType dtype = kv.Value.DType;
            result[kv.Key] = (dtype == DType.F16 || dtype == DType.F32) ? kv.Value.CastTo(DType.BF16) : kv.Value;
        }
        return result;
    }
}
