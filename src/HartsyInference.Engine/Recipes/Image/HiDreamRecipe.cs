using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Placement;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
namespace HartsyInference.Engine.Recipes.Image;

/// <summary>HiDream-I1 recipe (Full / Dev): an MMDiT with FOUR text encoders — CLIP-L, CLIP-G, T5-XXL, and Llama-3.1-8B (run as a multi-layer feature extractor). Lifted from the SwarmUI backend's <c>HiDreamLoader</c>: an all-in-one checkpoint's bundled components win, and anything it omits resolves from the canonical side models (<see cref="SideModels.HiDreamClipL"/>, <see cref="SideModels.HiDreamClipG"/>, <see cref="SideModels.T5XxlEnconly"/>, <see cref="SideModels.Llama31_8B"/>, <see cref="SideModels.FluxAe"/> — HiDream reuses the Flux VAE). Constructs and drives through <see cref="HiDreamRecipePipeline"/>.</summary>
public sealed class HiDreamRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "hidream";


    /// <inheritdoc/>
    /// <remarks>HiDream shares the Flux.1 VAE, so its encoder half rides the encode-parity gate that already
    /// covers that config; the img2img path in HiDreamPipeline was built for this.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "hidream", StringComparison.OrdinalIgnoreCase);

    /// <summary>HiDream-I1's official sampling settings: 28 steps at CFG 5.0, 1024x1024. Kept from the lifted loader rather than <c>GenerationDefaults.HiDreamFull</c> (50/5.0) because Full/Dev/Fast are not distinguishable from the weights and Dev is the shipped checkpoint.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 28, CfgScale = 5.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // The Llama-3.1 branch needs a real 128k-vocab BPE tokenizer, shipped as an embedded resource only when the
        // asset files were present at build time. Without it HiDream output is noise — refuse at load, as the loader did.
        if (!EmbeddedTokenizerResources.HasLlama3Assets)
        {
            throw new InvalidOperationException(
                "HiDream needs the Llama-3.1 tokenizer, which isn't embedded in this HartsyInference build. " +
                "Extract vocab.json + merges.txt from the Llama-3.1 tokenizer.json into " +
                "src/HartsyInference.ModelAssets.Tokenizers/Resources/ as llama3_vocab.json + llama3_merges.txt, then rebuild.");
        }

        // TODO(E-IMG-4): honor user-picked CLIP-L / CLIP-G / T5 / Llama / VAE overrides from ImageRequest.Components
        // (the SwarmUI loader read T2IParamTypes.ClipLModel/ClipGModel/T5XXLModel/LLaMAModel/VAE).
        (HiDreamCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader mainLoader) = HiDreamCheckpointConverter.LoadAndConvert(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader> { mainLoader };
        try
        {
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException(
                    $"HiDream checkpoint '{Path.GetFileName(context.CheckpointPath)}' contains no transformer weights " +
                    "(looked for caption_projection.* / double_stream_blocks.*).");
            }
            Logs.Info($"[HiDreamRecipe] Parsed checkpoint: {converted.Transformer.Count} transformer tensors, fp8_mix={converted.IsFp8Mix}.");

            HiDreamConfig config = HiDreamConfig.AutoDetect(converted.Transformer);
            HiDreamTransformer transformer = new HiDreamTransformer(config);
            transformer.LoadWeights(CastToF32(converted.Transformer));

            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            if (converted.ClipL.Count > 0)
            {
                clipL.LoadWeights(converted.ClipL, prefix: "text_model");
            }
            else
            {
                Dictionary<string, Tensor> w = LoadSideModel(SideModels.HiDreamClipL, "text_encoders.clip_l.transformer.", loaders);
                clipL.LoadWeights(w, prefix: "text_model");
            }

            ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
            if (converted.ClipG.Count > 0)
            {
                clipG.LoadWeights(converted.ClipG, prefix: "text_model");
            }
            else
            {
                Dictionary<string, Tensor> w = LoadSideModel(SideModels.HiDreamClipG, "text_encoders.clip_g.transformer.", loaders);
                clipG.LoadWeights(w, prefix: "text_model");
            }

            T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
            if (converted.T5.Count > 0)
            {
                t5.LoadWeights(converted.T5);
            }
            else
            {
                t5.LoadWeights(LoadSideModel(SideModels.T5XxlEnconly, "text_encoders.t5xxl.transformer.", loaders));
            }

            LlamaStyleEncoder llama = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Llama31_8B);
            if (converted.Llama.Count > 0)
            {
                llama.LoadWeights(converted.Llama);
            }
            else
            {
                llama.LoadWeights(LoadSideModel(SideModels.Llama31_8B, "text_encoders.llama.transformer.", loaders));
            }

            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
            Dictionary<string, Tensor> vaeWeights;
            if (converted.Vae.Count > 0)
            {
                vaeWeights = CastToF32(converted.Vae);
            }
            else
            {
                string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                (Dictionary<string, Tensor> standaloneVae, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
                loaders.Add(vaeLoader);
                // The canonical flux ae.safetensors is BFL-native LDM naming; VaeDecoder wants diffusers keys.
                vaeWeights = standaloneVae;
            }
            // BF16 on Ampere+ (F32-equivalent range, halves the full-res decode workspace), F32 otherwise —
            // the SDXL-VAE precision policy; both branches above force F32, this recovers BF16 where safe.
            vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeights, VaePrecisionHelper.PreferredVaeDtype(context.Backend));
            vae.LoadWeights(vaeWeights);
            VaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.Flux, vaeWeights, "HiDreamRecipe");

            // Phase 8 DiT sharding (VRAM pooling, not latency — see HiDreamPipeline's use of DitShardBackend):
            // the split point depends on the transformer's actual block sizes and free VRAM on the two
            // backends, both only known here (RecipeContext is built before weights load). HiDream's double
            // blocks are heavier than its single blocks (extra text-side attention + FFN weights) — a
            // count-proportional split would misallocate by GBs, so this uses the BYTE-WEIGHTED overload
            // (same class of heterogeneous-block DiT as Chroma/HunyuanImage/Flux).
            int ditShardSplitBlock = 0;
            if (context.DitShardBackend is not null)
            {
                long sharedWeightBytes = 0;
                foreach (Tensor t in transformer.EnumerateSharedWeights())
                {
                    sharedWeightBytes += t.DType.ComputeByteCount(t.ElementCount);
                }
                long[] perBlockBytes = new long[transformer.BlockCount];
                for (int i = 0; i < transformer.BlockCount; i++)
                {
                    long bytes = 0;
                    foreach (Tensor t in transformer.EnumerateBlockRangeWeights(i, i + 1))
                    {
                        bytes += t.DType.ComputeByteCount(t.ElementCount);
                    }
                    perBlockBytes[i] = bytes;
                }
                (long freeA, _) = context.Backend.GetVramInfo();
                (long freeB, _) = context.DitShardBackend.GetVramInfo();
                ditShardSplitBlock = PlacementPlanner.DitSplitPlan([freeA, freeB], perBlockBytes, sharedWeightBytes)[0];
                Logs.Info($"[HiDreamRecipe] DiT sharding enabled: blocks [0,{ditShardSplitBlock}) on the primary "
                    + $"backend, [{ditShardSplitBlock},{transformer.BlockCount}) on the shard backend.");
            }

            HiDreamPipeline pipeline = new HiDreamPipeline(context.Backend, clipL, clipG, t5, llama, transformer, vae, vaeEncoder, config)
            {
                DitShardBackend = context.DitShardBackend,
                DitShardSplitBlock = ditShardSplitBlock,
            };
            Logs.Info("[HiDreamRecipe] HiDream ready (4 text encoders; scheduler=flow-match).");
            return new HiDreamRecipePipeline(pipeline, new ClipTokenizer(), new T5Tokenizer(maxLength: 256), new LlamaTokenizer(maxLength: 256), t5, llama, transformer, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[HiDreamRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Resolves + loads one standalone side-model file, registering its loader for disposal and stripping the Comfy <paramref name="comfyPrefix"/> wrapper off every key.</summary>
    private static Dictionary<string, Tensor> LoadSideModel(ModelAsset asset, string comfyPrefix, List<SafeTensorsLoader> loaders)
    {
        string path = ModelDownloader.EnsureSideModelAsync(asset, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        loaders.Add(loader);
        Dictionary<string, Tensor> weights = StripStandalonePrefix(loader.GetAllTensors(), comfyPrefix);
        if (weights.Count == 0)
        {
            throw new InvalidOperationException($"Side model '{path}' has no usable tensors.");
        }
        return weights;
    }

    /// <summary>Casts F16/BF16 weights to F32 (the precision the HiDream transformer and the Flux VAE decoder are validated at), passing already-F32 tensors through.</summary>
    private static Dictionary<string, Tensor> CastToF32(Dictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            result[kv.Key] = (kv.Value.DType == DType.F16 || kv.Value.DType == DType.BF16) ? kv.Value.CastTo(DType.F32) : kv.Value;
        }
        return result;
    }

    /// <summary>Strips a Comfy <c>text_encoders.*.transformer.</c> wrapper prefix and drops the <c>position_ids</c> buffer the encoders don't consume.</summary>
    private static Dictionary<string, Tensor> StripStandalonePrefix(IReadOnlyDictionary<string, Tensor> raw, string comfyPrefix)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            if (kv.Key.StartsWith(comfyPrefix, StringComparison.Ordinal))
            {
                string rest = kv.Key[comfyPrefix.Length..];
                if (!rest.EndsWith("position_ids", StringComparison.Ordinal))
                {
                    result[rest] = kv.Value;
                }
            }
            else
            {
                result[kv.Key] = kv.Value;
            }
        }
        return result;
    }
}
