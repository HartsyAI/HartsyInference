using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
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

/// <summary>Kandinsky 5.0 T2I-Lite recipe (kandinskylab, ~6B DiT): the checkpoint is the transformer — either a single repackaged safetensors or a diffusers <c>transformer/</c> shard directory — and the dual text stack (Qwen2.5-VL-7B sequence embeddings via <see cref="SideModels.Qwen2_5_VL_7B"/> + CLIP-L pooled via <see cref="SideModels.ClipL"/>) plus the 16-channel Flux VAE (<see cref="SideModels.FluxAe"/>) resolve as side models.</summary>
/// <remarks>Unlike the other lifted families this one has NO SwarmUI loader to port — the extension lists Kandinsky 5 as unsupported because <see cref="Kandinsky5Pipeline"/> only accepts PRE-COMPUTED embeddings. Construction here follows the pipeline's own ctor and the <c>Kandinsky5GenerationTests</c> wiring; the live encode in <see cref="Kandinsky5RecipePipeline"/> is ported from the diffusers reference (<c>pipeline_kandinsky_t2i.encode_prompt</c>) and is UNVERIFIED against real weights.</remarks>
public sealed class Kandinsky5Recipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "kandinsky5";


    /// <inheritdoc/>
    /// <remarks>Kandinsky 5 image-lite reuses the Flux.1 VAE; the encoder half is built alongside the decoder.
    /// <para><see cref="ImageFeatures.Lora"/> added 2026-08-20. <see cref="HartsyInference.Diffusion.Models.Denoisers.Kandinsky5Transformer"/> names its two block stacks <c>text_transformer_blocks.{i}</c> and <c>visual_transformer_blocks.{i}</c> — neither was a root the bare-root LoRA detector recognized before the same change.</para></remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.Inpaint | ImageFeatures.SeamlessTiling | ImageFeatures.VariationSeed | ImageFeatures.Refiner | ImageFeatures.Lora;
    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "kandinsky5", StringComparison.OrdinalIgnoreCase);

    /// <summary>Kandinsky-5's official image sampling settings: 50 steps at guidance 3.5, 1024x1024 (<c>GenerationDefaults.Kandinsky5Image</c>).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 50, CfgScale = 3.5f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4): honor user-picked Qwen2.5-VL / CLIP-L / VAE overrides from ImageRequest.Components.
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            Kandinsky5CheckpointConverter.ConvertedWeights converted;
            if (Directory.Exists(context.CheckpointPath))
            {
                Logs.Info($"[Kandinsky5Recipe] Loading transformer diffusers dir: {context.CheckpointPath}.");
                (Kandinsky5CheckpointConverter.ConvertedWeights c, List<SafeTensorsLoader> shardLoaders) = Kandinsky5CheckpointConverter.LoadDiffusersFolder(context.CheckpointPath);
                converted = c;
                loaders.AddRange(shardLoaders);
            }
            else
            {
                Logs.Info($"[Kandinsky5Recipe] Loading transformer: {Path.GetFileName(context.CheckpointPath)}.");
                (Kandinsky5CheckpointConverter.ConvertedWeights c, SafeTensorsLoader loader) = Kandinsky5CheckpointConverter.LoadAndConvert(context.CheckpointPath);
                converted = c;
                loaders.Add(loader);
            }
            if (converted.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"Kandinsky 5 checkpoint '{context.CheckpointPath}' has no transformer weights.");
            }

            // BF16 -> F16 (12 GB, native F16 GEMM). The transient BF16 dequant path is validated for fp8/GGUF, not
            // BF16, and produced a blank image in the generation test; F32 would be 24 GB.
            Kandinsky5Config config = Kandinsky5Config.Lite;
            Dictionary<string, Tensor> transformerWeights = new Dictionary<string, Tensor>(converted.Transformer.Count);
            foreach (KeyValuePair<string, Tensor> kv in converted.Transformer)
            {
                transformerWeights[kv.Key] = kv.Value.DType == DType.BF16 ? kv.Value.CastTo(DType.F16) : kv.Value;
            }
            Kandinsky5Transformer transformer = new Kandinsky5Transformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: transformerWeights);
            transformer.LoadWeights(transformerWeights);

            string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen2_5_VL_7B, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
            qwenLoader.Load(qwenPath);
            loaders.Add(qwenLoader);
            LlamaStyleEncoder qwen = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
            qwen.LoadWeights(qwenLoader.GetAllTensors());

            string clipPath = ModelDownloader.EnsureSideModelAsync(SideModels.ClipL, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            SafeTensorsLoader clipLoader = new SafeTensorsLoader();
            clipLoader.Load(clipPath);
            loaders.Add(clipLoader);
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(Kandinsky5TextEncoding.ConvertClipLFromStandalone(clipLoader.GetAllTensors()), prefix: "text_model");

            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.FluxAe, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
            (Dictionary<string, Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = LoaderVaeUtils.LoadFluxVaeF32(vaePath);
            loaders.Add(vaeLoader);
            // BF16 on Ampere+ (F32-equivalent range, halves the full-res decode workspace), F32 otherwise —
            // the SDXL-VAE precision policy; LoadFluxVaeF32 force-upcasts to F32, this recovers BF16 where safe.
            vaeWeights = VaePrecisionHelper.CastVaeWeights(vaeWeights, VaePrecisionHelper.PreferredVaeDtype(context.Backend));
            VaeDecoder vae = new VaeDecoder(VaeConfig.Flux);
            vae.LoadWeights(vaeWeights);

            // Scheduler shift 5.0 and the Flux VAE scale/shift are the Lite defaults baked into the ctor.
                        VaeEncoder? vaeEncoder = LoaderVaeUtils.TryBuildEncoder(VaeConfig.Flux, vaeWeights, "Kandinsky5Recipe");
            Kandinsky5Pipeline pipeline = new Kandinsky5Pipeline(context.Backend, transformer, vae, vaeEncoder, config);
            Logs.Info("[Kandinsky5Recipe] Kandinsky 5 T2I-Lite ready (Qwen2.5-VL + CLIP-L live encode).");
            return new Kandinsky5RecipePipeline(pipeline, qwen, clipL, new Qwen2Tokenizer(), new ClipTokenizer(), context.Backend, transformer, loaders, loraStack);
        }
        catch (Exception ex)
        {
            Logs.Error("[Kandinsky5Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

}
