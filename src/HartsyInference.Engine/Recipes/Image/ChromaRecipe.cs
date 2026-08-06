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
using HartsyInference.Engine.Placement;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Chroma recipe (Lodestone Rock's 8.9B Flux derivative: T5-only, no CLIP/pooled conditioning, joint-attention DiT, same VAE as Flux.1). Lifted from the SwarmUI backend's <c>ChromaLoader</c>; the checkpoint is the DiT, the T5-XXL text encoder and Flux VAE are resolved as side models. Constructs and drives through <see cref="ChromaRecipePipeline"/>.</summary>
public sealed class ChromaRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "chroma";


    /// <inheritdoc/>
    /// <remarks>Chroma reuses the Flux.1 VAE; the encoder half is built alongside the decoder and ChromaPipeline implements the packed-latent masked path.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "chroma", StringComparison.OrdinalIgnoreCase);

    /// <summary>Chroma's official sampling settings: 35 steps at guidance 5.0, 1024x1024 (<c>ChromaConfig.DefaultSteps</c>/<c>ChromaConfig.DefaultCfgScale</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 35, CfgScale = 5.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // Resolve the two side models synchronously (Construct is a sync seam): T5-XXL (encoder-only fp8) + Flux VAE.
        // TODO(E-IMG-4): honor a user-picked T5/VAE override from ImageRequest.Components/Extra instead of always
        // taking the canonical SideModels entry (the SwarmUI loader read input.Get(T2IParamTypes.T5XXLModel/VAE)).
        string t5Path = ModelDownloader.EnsureSideModelAsync(SideModels.T5XxlEnconly, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        // 1. Load + convert the Chroma transformer.
        (ChromaCheckpointConverter.ConvertedWeights zConv, SafeTensorsLoader zLoader) = ChromaCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        if (zConv.Transformer.Count == 0)
        {
            zLoader.Dispose();
            throw new InvalidOperationException("Chroma checkpoint has no recognized transformer weights after conversion.");
        }
        ChromaConfig config = ChromaConfig.V1;
        Logs.Info($"[ChromaRecipe] Building transformer ({config.HiddenSize} hidden, {config.Depth} double / {config.DepthSingleBlocks} single).");
        ChromaTransformer transformer = new ChromaTransformer(config);
        transformer.LoadWeights(zConv.Transformer);

        // DiT sharding split point — byte-weighted: Chroma's 19 double blocks are ~2× its 38 single blocks,
        // so a count-proportional split would misallocate by GBs. Computed post-load (needs live free VRAM).
        int ditShardSplitBlock = 0;
        if (context.DitShardBackend is not null)
        {
            long[] perBlockBytes = new long[transformer.BlockCount];
            for (int i = 0; i < perBlockBytes.Length; i++)
            {
                foreach (Tensor t in transformer.EnumerateBlockRangeWeights(i, i + 1))
                {
                    perBlockBytes[i] += t.DType.ComputeByteCount(t.ElementCount);
                }
            }
            long sharedWeightBytes = 0;
            foreach (Tensor t in transformer.EnumerateSharedWeights())
            {
                sharedWeightBytes += t.DType.ComputeByteCount(t.ElementCount);
            }
            (long freeA, _) = context.Backend.GetVramInfo();
            (long freeB, _) = context.DitShardBackend.GetVramInfo();
            ditShardSplitBlock = PlacementPlanner.DitSplitPlan([freeA, freeB], perBlockBytes, sharedWeightBytes)[0];
            Logs.Info($"[ChromaRecipe] DiT sharding enabled: blocks [0,{ditShardSplitBlock}) on the primary "
                + $"backend, [{ditShardSplitBlock},{transformer.BlockCount}) on the shard backend "
                + "(sequential dual-pass CFG; the step graph is disabled while sharded).");
        }

        // 2. Load T5-XXL + its embedded tokenizer.
        SafeTensorsLoader t5Loader = new SafeTensorsLoader();
        t5Loader.Load(t5Path);
        Dictionary<string, Tensor> t5Weights = LoadT5FromStandalone(t5Loader.GetAllTensors());
        if (t5Weights.Count == 0)
        {
            t5Loader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"T5 model file '{t5Path}' has no usable T5 tensors.");
        }
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        t5.LoadWeights(t5Weights);
        T5Tokenizer tokenizer = new T5Tokenizer(maxLength: 512);

        // 3. Load the Flux VAE (Chroma reuses it verbatim).
        (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
        if (vaeWeights.Count == 0)
        {
            vaeLoader.Dispose();
            t5Loader.Dispose();
            zLoader.Dispose();
            throw new InvalidOperationException($"VAE file '{vaePath}' has no usable VAE tensors.");
        }
        VaeDecoder vae = new VaeDecoder(VaeConfig.Chroma);
        vae.LoadWeights(vaeWeights);

                VaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.Chroma, vaeWeights, "ChromaRecipe");
        ChromaPipeline pipeline = new ChromaPipeline(context.Backend, t5, transformer, vae, vaeEncoder, config)
        {
            TextEncoderBackend = context.TextEncoderBackendOrDefault,
            VaeBackend = context.VaeBackendOrDefault,
            DitShardBackend = context.DitShardBackend,
            DitShardSplitBlock = ditShardSplitBlock,
        };
        Logs.Info("[ChromaRecipe] Chroma ready.");
        return new ChromaRecipePipeline(pipeline, tokenizer, zLoader, t5Loader, vaeLoader);
    }

    /// <summary>Strips Comfy's <c>text_encoders.t5xxl.transformer.</c> prefix from a standalone T5-XXL safetensors file (keys are otherwise stored as-is), so the encoder finds them.</summary>
    private static Dictionary<string, Tensor> LoadT5FromStandalone(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        const string ComfyPrefix = "text_encoders.t5xxl.transformer.";
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            if (kv.Key.StartsWith(ComfyPrefix, StringComparison.Ordinal))
            {
                result[kv.Key[ComfyPrefix.Length..]] = kv.Value;
            }
            else
            {
                result[kv.Key] = kv.Value;
            }
        }
        return result;
    }
}
