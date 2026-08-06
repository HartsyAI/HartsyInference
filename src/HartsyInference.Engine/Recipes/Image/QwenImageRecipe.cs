using System.Linq;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Placement;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Qwen-Image recipe (Alibaba, 20B MMDiT): the checkpoint is transformer-only in the common Comfy <c>diffusion_models</c> layout, so the Qwen2.5-VL-7B text encoder (<see cref="SideModels.Qwen2_5_VL_7B"/>) and the 16-channel Qwen-Image VAE (<see cref="SideModels.QwenImageVae"/>) resolve as side models; an all-in-one file's bundled copies win when present. Lifted from the SwarmUI backend's <c>QwenImageLoader</c>; drives through <see cref="QwenImageRecipePipeline"/>.</summary>
public sealed class QwenImageRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "qwen-image";

    /// <inheritdoc/>
    /// <remarks>The only family offering both init-image modes: classic strength-based img2img/inpaint through the
    /// packed-latent masked path, and Qwen-Image-Edit reference conditioning. <c>Img2Img.Mode</c> selects; Auto prefers
    /// classic. Edit fidelity is below the reference implementation until <c>editRefVisionImages</c> is wired — the VL
    /// branch cannot see the image without it.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.RefEdit;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "qwen-image", StringComparison.OrdinalIgnoreCase);

    /// <summary>Qwen-Image's official sampling settings: 50 steps at true-CFG 4.0, 1024x1024 (diffusers <c>QwenImagePipeline.__call__</c>, mirrored by <c>GenerationDefaults.QwenImage</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 4.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): Qwen-Image-Edit (2509 / 2511) is deferred — the SwarmUI loader additionally built the
        // Qwen2.5-VL vision tower + Qwen25VlMultimodalEncoder from the TE file's `visual.*` weights and passed the
        // init/prompt images as edit references. Text-to-image only here, so no vision tower is constructed.
        // TODO(E-IMG-4): honor user VAE / Qwen text-encoder overrides from ImageRequest.Components (the loader read
        // T2IParamTypes.QwenModel / T2IParamTypes.VAE) instead of always taking the canonical SideModels entry.
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        IDisposable? ggufHandle = null;
        try
        {
            // GGUF checkpoints (e.g. qwen-image Q5_K_M) stay quant-native: dequantizing a 20B Q4/Q5 file to F16 on
            // the host needs ~40 GB RAM. The Linear path dequantizes per-GEMM on the GPU instead. The rank-2
            // relabel converts ggml's [in, out] shape metadata to the [out, in] order the converters assume.
            bool isGguf = context.CheckpointPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase);
            QwenImageCheckpointConverter.ConvertedWeights converted;
            if (isGguf)
            {
                GgufModelLoader.LoadedGgufModel gguf = GgufModelLoader.Load(context.CheckpointPath);
                ggufHandle = gguf;
                converted = QwenImageCheckpointConverter.Convert(GgufModelLoader.RelabelRank2ToPyTorchOrder(gguf.Weights));
            }
            else
            {
                (QwenImageCheckpointConverter.ConvertedWeights c, SafeTensorsLoader mainLoader) = QwenImageCheckpointConverter.LoadAndConvert(context.CheckpointPath);
                converted = c;
                loaders.Add(mainLoader);
            }
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"Qwen-Image checkpoint '{Path.GetFileName(context.CheckpointPath)}' contains no transformer weights (looked for transformer_blocks.* / img_in.*).");
            }
            Logs.Info($"[QwenImageRecipe] Parsed checkpoint: {converted.Transformer.Count} transformer tensors.");

            // V1 is the released 20B Qwen-Image (depth=60, hidden=3072). The V2 presets are speculative
            // placeholders for unreleased weights, so there is no auto-detection into them.
            QwenImageConfig config = QwenImageConfig.V1;
            QwenImageTransformer transformer = new QwenImageTransformer(config);
            // Weights load as-is (fp8/fp16 kept for the quantized GEMM path) — the reference does NOT upcast the
            // transformer to F32.
            transformer.LoadWeights(converted.Transformer);

            // fp8 single-file checkpoints (the Edit-2511 fp8mixed) on hardware without native FP8 GEMM cast each
            // fp8 weight to F16 on first use; caching those casts roughly doubles VRAM — fatal for the sharded
            // 3060 half. Same rule as Flux1Recipe: native-FP8 cards keep the cache. GGUF-quantized checkpoints
            // (the usual Qwen-Image path) never hit this — quantized GEMM reads blocks directly.
            bool hasFp8 = false;
            foreach (Tensor t in converted.Transformer.Values)
            {
                if (t.DType.IsFp8) { hasFp8 = true; break; }
            }
            if (hasFp8)
            {
                RecipeBackendFlags.DisableCacheWeightCasts(context, "QwenImageRecipe", onlyWithoutNativeFp8Gemm: true);
            }

            // DiT sharding stages — computed here, not in InferenceEngine, because block count and live free VRAM
            // are only known once the transformer weights are loaded. Qwen's 60 blocks are homogeneous, so the
            // count-proportional plan is already byte-accurate. Qwen-Image is the one recipe wired for N-way
            // sharding (ROADMAP.md item 7); the other five sharded families still resolve a single split point
            // from context.DitShardBackend.
            List<DitShardStage>? ditShardStages = null;
            if (context.DitShardBackends is { Count: > 0 })
            {
                long sharedWeightBytes = 0;
                foreach (Tensor t in transformer.EnumerateSharedWeights())
                {
                    sharedWeightBytes += t.DType.ComputeByteCount(t.ElementCount);
                }
                List<IBackend> stageBackends = new(1 + context.DitShardBackends.Count) { context.Backend };
                stageBackends.AddRange(context.DitShardBackends);
                List<long> freeBytesPerStage = new(stageBackends.Count);
                foreach (IBackend stageBackend in stageBackends)
                {
                    (long free, _) = stageBackend.GetVramInfo();
                    freeBytesPerStage.Add(free);
                }
                IReadOnlyList<int> counts = PlacementPlanner.DitSplitPlan(freeBytesPerStage, transformer.BlockCount, sharedWeightBytes);
                ditShardStages = new List<DitShardStage>(stageBackends.Count);
                int start = 0;
                for (int i = 0; i < stageBackends.Count; i++)
                {
                    ditShardStages.Add(new DitShardStage(stageBackends[i], start, start + counts[i]));
                    start += counts[i];
                }
                Logs.Info($"[QwenImageRecipe] DiT sharding enabled across {ditShardStages.Count} stages: "
                    + string.Join(", ", ditShardStages.Select(s => $"[{s.StartBlock},{s.EndBlock})")));
            }

            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
            if (converted.TextEncoder.Count > 0)
            {
                textEncoder.LoadWeights(converted.TextEncoder);
            }
            else
            {
                string encoderPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen2_5_VL_7B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                SafeTensorsLoader encoderLoader = new SafeTensorsLoader();
                encoderLoader.Load(encoderPath);
                loaders.Add(encoderLoader);
                textEncoder.LoadWeights(encoderLoader.GetAllTensors());
                Logs.Info("[QwenImageRecipe] Qwen2.5-VL-7B resolved as side model.");
            }

            // Both VAE halves: the decoder for output, the encoder because it is what img2img / edit needs. A
            // decode-only VAE yields a null encoder, which the pipeline turns into a named refusal rather than a
            // silently text-to-image result.
            Dictionary<string, Tensor> vaeWeights;
            if (converted.Vae.Count > 0)
            {
                vaeWeights = CastToF32(converted.Vae);
            }
            else
            {
                string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.QwenImageVae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
                vaeLoader.Load(vaePath);
                loaders.Add(vaeLoader);
                vaeWeights = CastToF32(vaeLoader.GetAllTensors());
                Logs.Info("[QwenImageRecipe] Qwen-Image VAE resolved as side model.");
            }
            QwenImageVaeDecoder vae = new QwenImageVaeDecoder(VaeConfig.QwenImage);
            vae.LoadWeights(vaeWeights);
            QwenImageVaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildQwenEncoder(VaeConfig.QwenImage, vaeWeights, "QwenImageRecipe");

            QwenImagePipeline pipeline = new QwenImagePipeline(context.Backend, textEncoder, transformer, vae, vaeEncoder, config)
            {
                TextEncoderBackend = context.TextEncoderBackendOrDefault,
                VaeBackend = context.VaeBackendOrDefault,
                DitShardStages = ditShardStages,
            };
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 512);
            Logs.Info("[QwenImageRecipe] Qwen-Image ready (Qwen2.5-VL-7B encoder; flow-match Euler, dynamic shift).");
            return new QwenImageRecipePipeline(pipeline, tokenizer, textEncoder, transformer, vae, vaeEncoder, loaders, ggufHandle);
        }
        catch (Exception ex)
        {
            Logs.Error("[QwenImageRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            ggufHandle?.Dispose();
            throw;
        }
    }

    /// <summary>Upcasts 16-bit VAE weights to F32 (the Qwen-Image VAE runs on the F32 path); other dtypes pass through untouched.</summary>
    private static Dictionary<string, Tensor> CastToF32(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            DType dtype = kv.Value.DType;
            result[kv.Key] = (dtype == DType.F16 || dtype == DType.BF16) ? kv.Value.CastTo(DType.F32) : kv.Value;
        }
        return result;
    }
}
