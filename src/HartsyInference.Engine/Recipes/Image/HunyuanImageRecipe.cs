using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Placement;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>HunyuanImage 2.1 recipe (Tencent, 17B MMDiT — 20 double + 40 single blocks, 32×/64-channel VAE). The checkpoint is the transformer; the Qwen2.5-VL-7B text stack (<see cref="SideModels.Qwen25Vl7BHunyuan"/>) and the HunyuanImage VAE (<see cref="SideModels.HunyuanImageVae"/>) resolve as side models. Lifted from the SwarmUI backend's <c>HunyuanImageLoader</c>; drives through <see cref="HunyuanImageRecipePipeline"/>.</summary>
public sealed class HunyuanImageRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "hunyuan-image";


    /// <inheritdoc/>
    /// <remarks>Img2img only. HunyuanImage integrates in token space after a one-time patchify, and the shared
    /// mask-blend helpers have no variant for that packing — a masked path would need a token-space blend that
    /// does not exist yet, so Inpaint is deliberately not declared.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Inpaint;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "hunyuan-image", StringComparison.OrdinalIgnoreCase);

    /// <summary>HunyuanImage 2.1's official sampling settings: 50 steps at CFG 3.5 and a native 2048x2048 (<c>GenerationDefaults.HunyuanImage</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 3.5f, Width = 2048, Height = 2048 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor user VAE / Qwen text-encoder overrides from ImageRequest.Components (the SwarmUI
        // loader read T2IParamTypes.QwenModel / T2IParamTypes.VAE) instead of always taking the SideModels entry.
        // The optional ByT5 glyph branch is not wired here either (it is optional at forward time upstream too).
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        IDisposable? ggufHandle = null;
        try
        {
            // GGUF repacks ship original-Tencent keys; the converter remaps them to diffusers naming and the quants
            // stay native (transient per-GEMM dequant). BF16 GGUF tensors MUST become F16 — the transient weight
            // path skips BF16 and yields a blank image.
            bool isGguf = context.CheckpointPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
            HunyuanImageCheckpointConverter.ConvertedWeights converted;
            if (isGguf)
            {
                GgufModelLoader.LoadedGgufModel gguf = GgufModelLoader.Load(context.CheckpointPath);
                ggufHandle = gguf;
                Dictionary<string, Tensor> relabeled = GgufModelLoader.RelabelRank2ToPyTorchOrder(gguf.Weights);
                Dictionary<string, Tensor> cast = new Dictionary<string, Tensor>(relabeled.Count);
                foreach (KeyValuePair<string, Tensor> kv in relabeled)
                {
                    cast[kv.Key] = kv.Value.DType == DType.BF16 ? kv.Value.CastTo(DType.F16) : kv.Value;
                }
                converted = HunyuanImageCheckpointConverter.Convert(cast);
            }
            else
            {
                (HunyuanImageCheckpointConverter.ConvertedWeights c, SafeTensorsLoader mainLoader) = HunyuanImageCheckpointConverter.LoadAndConvert(context.CheckpointPath);
                converted = c;
                loaders.Add(mainLoader);
            }
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"HunyuanImage checkpoint '{Path.GetFileName(context.CheckpointPath)}' contains no transformer weights.");
            }
            Logs.Info($"[HunyuanImageRecipe] Parsed checkpoint: {converted.Transformer.Count} transformer tensors (fp8={converted.IsFp8Mix}).");

            HunyuanImageConfig config = HunyuanImageConfig.V21;
            HunyuanImageTransformer transformer = new HunyuanImageTransformer(config);
            transformer.LoadWeights(converted.Transformer);

            // DiT sharding split point — byte-weighted: HunyuanImage's 20 double blocks are ~2× its 40 single
            // blocks, so a count-proportional split would misallocate by GBs. Computed post-load (needs live
            // free VRAM).
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
                Logs.Info($"[HunyuanImageRecipe] DiT sharding enabled: blocks [0,{ditShardSplitBlock}) on the "
                    + $"primary backend, [{ditShardSplitBlock},{transformer.BlockCount}) on the shard backend.");
            }

            string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen25Vl7BHunyuan, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
            qwenLoader.Load(qwenPath);
            loaders.Add(qwenLoader);
            LlamaStyleEncoder llama = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
            llama.LoadWeights(qwenLoader.GetAllTensors());
            HunyuanImageQwenTextEncoder qwenEncoder = new HunyuanImageQwenTextEncoder(llama);

            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.HunyuanImageVae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaePath);
            loaders.Add(vaeLoader);
            Dictionary<string, Tensor> vaeWeights = RemapVaeKeys(vaeLoader.GetAllTensors());
            HunyuanImageVaeDecoder vaeDecoder = new HunyuanImageVaeDecoder(VaeConfig.HunyuanImage);
            vaeDecoder.LoadWeights(vaeWeights);
            // The 2.1 VAE's encoder half doesn't fit the generic VaeEncoder (channel changes live in the
            // downsamplers, not the resnets) — HunyuanImageVaeEncoder is its bespoke mirror of the decoder,
            // and is what makes img2img/inpaint real for this family.
            HunyuanImageVaeEncoder hyVaeEncoder = new HunyuanImageVaeEncoder(VaeConfig.HunyuanImage);
            hyVaeEncoder.LoadWeights(vaeWeights);

            HunyuanImagePipeline pipeline = new HunyuanImagePipeline(context.Backend, qwenEncoder, transformer, vaeDecoder, config)
            {
                HyVaeEncoder = hyVaeEncoder,
                TextEncoderBackend = context.TextEncoderBackendOrDefault,
                VaeBackend = context.VaeBackendOrDefault,
                DitShardBackend = context.DitShardBackend,
                DitShardSplitBlock = ditShardSplitBlock,
            };
            Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
            Logs.Info("[HunyuanImageRecipe] HunyuanImage 2.1 ready.");
            return new HunyuanImageRecipePipeline(pipeline, tokenizer, llama, qwenEncoder, transformer, vaeDecoder, loaders, ggufHandle);
        }
        catch (Exception ex)
        {
            Logs.Error("[HunyuanImageRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            ggufHandle?.Dispose();
            throw;
        }
    }

    /// <summary>Remaps the HunyuanImage VAE file's old-LDM key naming to diffusers. Its <c>up.0</c> is the DEEPEST level (opposite of SD-LDM), so the usual index reversal is disabled and there are 6 up levels.</summary>
    private static Dictionary<string, Tensor> RemapVaeKeys(IReadOnlyDictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
        {
            string mapped = CheckpointConvertUtils.ConvertVaeKey(kv.Key, numUpLevels: 6, reverseUpIndices: false) ?? kv.Key;
            result[mapped] = kv.Value;
        }
        return result;
    }
}
