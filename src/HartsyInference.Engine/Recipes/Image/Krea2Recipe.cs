using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Placement;
using HartsyInference.Vulkan;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Krea 2 recipe (Krea, 12.9B single-stream MMDiT flow-match T2I). The checkpoint is the transformer in Krea's raw key naming (<c>blocks.N.*</c>, <c>txtfusion.*</c>, <c>tmlp</c>, <c>tproj</c>, <c>last.*</c>), remapped by <see cref="Krea2CheckpointConverter.RemapTransformerKey"/>; the Qwen3-VL-4B text encoder (<see cref="SideModels.Qwen3VL_4B"/>) and the Qwen-Image VAE (<see cref="SideModels.QwenImageVae"/>) resolve as side models. Base vs Turbo is auto-detected from the checkpoint filename. Lifted from the SwarmUI backend's <c>Krea2Loader</c>; drives through <see cref="Krea2RecipePipeline"/>.</summary>
public sealed class Krea2Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "krea2";

    /// <inheritdoc/>
    /// <remarks>Krea 2 shares the Qwen-Image VAE; its encoder half is built alongside the decoder and
    /// <see cref="Krea2Pipeline"/> implements the packed-latent masked path.
    /// <para><see cref="ImageFeatures.Regional"/> added 2026-08-11 (Tier 3.7): <see cref="Krea2Attention"/> gained
    /// an <c>attnBias</c> slot (the GQA K/V repeat happens BEFORE the SDPA call, so the bias broadcasts the same
    /// way it does for non-GQA architectures — no special-casing needed), threaded through <see cref="Krea2Block"/>
    /// and <see cref="Krea2Transformer"/>'s eager forward path, wired via <see cref="Krea2Pipeline.GenerateFromTokens"/>
    /// and <see cref="Krea2RecipePipeline.BuildRegionalPlan"/> — real-weight verified with a two-region prompt.
    /// DiT sharding (<see cref="Krea2Transformer.ForwardPatchedSharded"/>) has no bias parameter, so a regional
    /// request forces the unsharded path for that generation with a logged warning, same precedent as Flux.1's
    /// own DiT-sharding exclusion for regional plans.</para>
    /// <para><see cref="ImageFeatures.Lora"/> added 2026-08-20. <see cref="Krea2Transformer"/> names its blocks
    /// <c>transformer_blocks.{i}</c> after <see cref="Krea2CheckpointConverter.RemapTransformerKey"/>, which is the
    /// canonical diffusers root, so both the <c>transformer.</c>-wrapped PEFT format and bare-root files map straight
    /// through <c>DiffusersFluxMapper</c> with no Krea2-specific mapper. Attention is loaded SPLIT
    /// (<see cref="Krea2Attention"/> reads <c>to_q</c>/<c>to_k</c>/<c>to_v</c>) even on the fp8-scaled production
    /// checkpoint — <see cref="Krea2CheckpointConverter.RemapTransformerKey"/> splits Krea's raw <c>attn.wq</c> during
    /// conversion — so <c>LoraStack</c>'s fused-sibling fallback is not on this family's path at all.</para></remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.Regional | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "krea2", StringComparison.OrdinalIgnoreCase);

    /// <summary>Krea 2 Base's official sampling settings: 28 steps at CFG 4.5, 1024x1024 (<c>Krea2Config.Base</c>); a Turbo/TDM checkpoint narrows this to 8 guidance-free steps via <see cref="Krea2RecipePipeline.VariantDefaults"/>.</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 28, CfgScale = 4.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <summary>Krea 2 Turbo/TDM's official sampling settings: 8 distilled steps, guidance-free (<c>Krea2Config.Turbo</c>).</summary>
    public static ImageDefaults TurboDefaults { get; } = new ImageDefaults { Steps = 8, CfgScale = 1.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor user text-encoder / VAE overrides from ImageRequest.Components (the SwarmUI loader
        // read T2IParamTypes.QwenModel / T2IParamTypes.VAE). NOTE: the "krea2" catalog entry lists its text-encoder
        // asset as text_encoders/qwen3vl_4b_bf16.safetensors while SideModels.Qwen3VL_4B points at the fp8_scaled
        // file under text_encoders/Krea2/ — same encoder, different quantization/location. The VAE asset
        // (VAE/QwenImage/qwen_image_vae.safetensors) matches SideModels.QwenImageVae exactly.
        string fileName = Path.GetFileName(context.CheckpointPath);
        bool isTurbo = IsTurbo(fileName);
        Krea2Config config = isTurbo ? Krea2Config.Turbo : Krea2Config.Base;
        Logs.Info($"[Krea2Recipe] Loading Krea 2 ({(isTurbo ? "Turbo/TDM" : "Base")}): {fileName}.");

        string encoderPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen3VL_4B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.QwenImageVae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();

        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            (Dictionary<string, Tensor> ditWeights, SafeTensorsLoader ditLoader) = ComponentLoader.Load(
                context.CheckpointPath, "Krea2Recipe", key => Krea2CheckpointConverter.RemapTransformerKey(CheckpointConvertUtils.StripTransformerPrefix(key)), applyFp8Dequant: true);
            loaders.Add(ditLoader);
            if (ditWeights.Count == 0)
            {
                throw new InvalidOperationException($"Krea 2 checkpoint '{fileName}' contains no transformer weights.");
            }

            (Dictionary<string, Tensor> teWeights, SafeTensorsLoader teLoader) = ComponentLoader.Load(encoderPath, "Krea2Recipe", CheckpointConvertUtils.RemapQwenLanguageKey, applyFp8Dequant: true);
            loaders.Add(teLoader);

            (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = ComponentLoader.Load(vaePath, "Krea2Recipe", keyTransform: null, applyFp8Dequant: false);
            loaders.Add(vaeLoader);

            // Krea 2's fp8-scaled transformer is ~13 GB; caching every fp8->F16/BF16 weight cast roughly doubles
            // that in VRAM (original fp8 + cached activation-dtype copy) and OOMs partway through text encoding —
            // same class of fix as Flux1Recipe/LtxVideoRecipe/HunyuanVideoRecipe above. Keep weights resident in
            // their checkpoint dtype with a transient per-GEMM dequant instead.
            RecipeBackendFlags.DisableCacheWeightCasts(context, "Krea2Recipe");

            Logs.Info("[Krea2Recipe] Building Krea 2 models (single-stream MMDiT, 28 blocks, text-fusion stage).");
            Krea2Transformer transformer = new Krea2Transformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: ditWeights);
            transformer.LoadWeights(ditWeights);

            // Phase 8 DiT sharding (VRAM pooling, not latency — see Krea2Pipeline's use of DitShardBackend): the
            // split point depends on the transformer's actual block count and free VRAM on the two backends, both
            // only known here (RecipeContext is built before weights load), so it's computed at construction time
            // rather than passed in from InferenceEngine.
            int ditShardSplitBlock = 0;
            if (context.DitShardBackend is not null)
            {
                ditShardSplitBlock = DitShardPlanner.SplitBlockByCount(
                    context.Backend, context.DitShardBackend, transformer.BlockCount, transformer.EnumerateSharedWeights());
                Logs.Info($"[Krea2Recipe] DiT sharding enabled: blocks [0,{ditShardSplitBlock}) on the primary "
                    + $"backend, [{ditShardSplitBlock},{transformer.BlockCount}) on the shard backend.");
            }

            LlamaStyleEncoder textEncoder = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_VL_4B);
            textEncoder.LoadWeights(teWeights);

            Dictionary<string, Tensor> vaeF32 = VaePrecisionHelper.CastWeights(vaeWeights, [DType.F16, DType.BF16], DType.F32);
            QwenImageVaeDecoder vae = new QwenImageVaeDecoder(VaeConfig.QwenImage);
            vae.LoadWeights(vaeF32);
            // Encoder half from the same staged dict — null when the checkpoint ships decode-only, which the pipeline
            // turns into a named refusal rather than a silently text-to-image result.
            QwenImageVaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildQwenEncoder(VaeConfig.QwenImage, vaeF32, "Krea2Recipe");

            Krea2Pipeline pipeline = new Krea2Pipeline(context.Backend, textEncoder, transformer, vae, vaeEncoder, config)
            {
                DitShardBackend = context.DitShardBackend,
                DitShardSplitBlock = ditShardSplitBlock,
            };
            Qwen3Tokenizer tokenizer = new Qwen3Tokenizer(maxLength: 512);
            Logs.Info("[Krea2Recipe] Krea 2 ready.");
            return new Krea2RecipePipeline(pipeline, tokenizer, textEncoder, transformer, vae, vaeEncoder, isTurbo, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[Krea2Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Whether the checkpoint filename marks a distilled (Turbo/TDM) variant — those pin the flow-match shift and run guidance-free.</summary>
    private static bool IsTurbo(string name)
    {
        if (string.IsNullOrEmpty(name))
        {
            return false;
        }
        string lower = name.ToLowerInvariant();
        return lower.Contains("turbo") || lower.Contains("tdm") || lower.Contains("distill");
    }
}
