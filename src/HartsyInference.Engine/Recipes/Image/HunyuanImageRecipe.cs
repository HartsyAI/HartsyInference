using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Placement;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Models;
using HartsyInference.ModelAssets.Checkpoints;
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
    /// does not exist yet, so Inpaint is deliberately not declared.
    /// <para><see cref="ImageFeatures.Lora"/> added 2026-08-20. <see cref="HartsyInference.Diffusion.Models.Denoisers.HunyuanImageTransformer"/> names its blocks <c>transformer_blocks.{i}</c> / <c>single_transformer_blocks.{i}</c>, both already-recognized canonical diffusers roots. NOTE: this family's production config is a Q4_K_M GGUF, and a K-quant weight cannot take a LoRA merge (requantizing a merged result back into block form is not implemented) — such a request refuses by name, telling the user to pick a safetensors/fp8 build. That refusal is the intended behaviour, not a regression.</para></remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Inpaint | ImageFeatures.Refiner | ImageFeatures.Lora;

    /// <inheritdoc/>
    /// <remarks>Ledger evidence in <c>PromptWeightingModeLedgerTests</c>: <c>supported_models.py:2109</c> →
    /// <c>hunyuan_image.HunyuanImageTokenizer</c>, a <c>QwenImageTokenizer</c> subclass, so the Qwen arm disables
    /// weights (<c>qwen_image.py:39</c>). Its byt5 arm keeps them but is populated only for QUOTED text, and
    /// SwarmUI's discriminator probes the unquoted literal <c>(x:2)</c>, so byt5 never enters the probe.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting =>
        Diffusion.Prompting.PromptWeightingMode.CondScale;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "hunyuan-image", StringComparison.OrdinalIgnoreCase);

    /// <summary>HunyuanImage 2.1's official sampling settings: 50 steps at CFG 3.5 and a native 2048x2048 (<c>GenerationDefaults.HunyuanImage</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 3.5f, Width = 2048, Height = 2048 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.BlockStreaming | MemoryCapabilities.DitSharding | MemoryCapabilities.ComponentPlacement;

    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor user VAE / Qwen text-encoder overrides from ImageRequest.Components (the SwarmUI
        // loader read T2IParamTypes.QwenModel / T2IParamTypes.VAE) instead of always taking the SideModels entry.
        // The optional ByT5 glyph branch is not wired here either (it is optional at forward time upstream too).
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        IDisposable? checkpoint = null;
        try
        {
            // GGUF repacks ship original-Tencent keys; the converter remaps them to diffusers naming either way and
            // the quants stay native (transient per-GEMM dequant).
            CheckpointSource source = CheckpointSource.Open(context.CheckpointPath);
            checkpoint = source;
            // A BF16 tensor passing through a GGUF must become F16 — the transient weight path skips BF16 and yields a
            // blank image. Kept keyed on the container rather than on the dtype alone: the safetensors builds run
            // without this cast today, and the real rule is a backend capability nobody has measured, so widening it
            // here would change a working path on a guess.
            bool castBf16 = source.Format == ModelFormat.Gguf;
            Dictionary<string, Tensor> raw = new Dictionary<string, Tensor>(source.Weights.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> kv in source.Weights)
            {
                raw[kv.Key] = castBf16 && kv.Value.DType == DType.BF16 ? kv.Value.CastTo(DType.F16) : kv.Value;
            }
            HunyuanImageCheckpointConverter.ConvertedWeights converted = HunyuanImageCheckpointConverter.Convert(raw);
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"HunyuanImage checkpoint '{Path.GetFileName(context.CheckpointPath)}' contains no transformer weights.");
            }
            Logs.Info($"[HunyuanImageRecipe] Parsed checkpoint: {converted.Transformer.Count} transformer tensors (fp8={converted.IsFp8Mix}).");

            HunyuanImageConfig config = HunyuanImageConfig.V21;
            HunyuanImageTransformer transformer = new HunyuanImageTransformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            // Any quant this backend has no packed-weight kernel for widens here rather than failing inside the
            // first GEMM, minutes into a generation.
            QuantizedWeightPolicy.PreparedWeights prepared =
                QuantizedWeightPolicy.PrepareForBackends(converted.Transformer, context.TransformerBackends);
            // Tracked immediately so a failure further down frees the widened copies rather than
            // leaving them to the finalizer.
            checkpoint = new CompositeDisposable(source, prepared);
            MergedLoraStack? loraStack = RecipeLoraMerge.Apply(
                context,
                new LoraMergeTargets { Transformer = converted.Transformer },
                "HunyuanImageRecipe");
            transformer.LoadWeights(converted.Transformer);

            // DiT sharding split point — byte-weighted: HunyuanImage's 20 double blocks are ~2× its 40 single
            // blocks, so a count-proportional split would misallocate by GBs. Computed post-load (needs live
            // free VRAM).
            int ditShardSplitBlock = 0;
            if (context.DitShardBackend is not null)
            {
                ditShardSplitBlock = DitShardPlanner.SplitBlockByBytes(
                    context.Backend, context.DitShardBackend, transformer.BlockCount,
                    transformer.EnumerateBlockRangeWeights, transformer.EnumerateSharedWeights());
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
            return new HunyuanImageRecipePipeline(pipeline, tokenizer, llama, qwenEncoder, transformer, vaeDecoder, loaders, checkpoint, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[HunyuanImageRecipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            checkpoint?.Dispose();
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
