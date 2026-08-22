using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Drives the classic SDXL-refiner checkpoint (<c>stable-diffusion-xl-v1-refiner</c>) as a standalone
/// model, matching ComfyUI's behavior of letting the refiner generate directly. The checkpoint carries ONE text
/// encoder (CLIP-G at <c>conditioner.embedders.0</c>) plus the standard SDXL VAE; conditioning is the refiner's
/// 5-scalar aesthetic-score ADM. Text-to-image runs the full schedule from noise (strength 1 over a mid-gray
/// source); Init Image gives the model its natural img2img/refine use.</summary>
public sealed class SdxlRefinerRecipe : IArchitectureRecipe
{
    /// <inheritdoc/>
    public string Name => "sdxl-refiner";

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "sdxl-refiner", StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    /// <remarks><see cref="ImageFeatures.Lora"/> is deliberately NOT declared, and this is the one image family
    /// left out of the 2026-08-20 LoRA sweep — recorded here so it reads as a decision rather than an oversight.
    /// The refiner UNet is <c>UNetConfig.SdxlRefiner</c>: four down-levels and a 1280-dim trunk against base
    /// SDXL's three, so a Kohya SDXL LoRA's <c>lora_unet_down_blocks_*</c> keys do not name anything in it. It
    /// would detect as <see cref="ModelAssets.Lora.LoraFormat.KohyaSdxl"/>, match zero weights, and hit
    /// <c>LoraApplier</c>'s zero-match refusal — trading the feature gate's accurate "this family does not
    /// support LoRA" for a merge-time error that reads like a broken file. Refiner-targeted LoRAs are not a
    /// thing the community trains. Declare it only alongside a real refiner LoRA to test against.</remarks>
    public ImageFeatures Supports => ImageFeatures.Img2Img | ImageFeatures.SeamlessTiling;

    /// <summary>The refiner's recommended settings mirror base SDXL's (diffusers img2img defaults).</summary>
    public static ImageDefaults FamilyDefaults { get; } = new ImageDefaults { Steps = 40, CfgScale = 5.0f, Width = 1024, Height = 1024 };

    /// <inheritdoc/>
    public ImageDefaults Defaults => FamilyDefaults;

    /// <inheritdoc/>
    public IRecipePipeline Construct(RecipeContext context)
    {
        SdxlRefinerEntry entry = SdxlRefinerLoader.Load(context.CheckpointPath);
        SafeTensorsLoader? auxLoader = null;
        try
        {
            (Dictionary<string, Tensor> clipGRaw, Dictionary<string, Tensor> vaeRaw, SafeTensorsLoader loader) =
                SdxlCheckpointConverter.LoadRefinerAuxiliary(context.CheckpointPath);
            auxLoader = loader;
            if (clipGRaw.Count == 0)
            {
                throw new InvalidOperationException(
                    $"'{context.CheckpointPath}' has no CLIP-G at conditioner.embedders.0 — not a standalone-loadable SDXL refiner checkpoint.");
            }
            // F32 like the whole SDXL family (SdxlRecipe's WeightStaging) — the fp16 SDXL VAE famously NaNs,
            // which came out as pure-black frames when these dicts stayed F16.
            Dictionary<string, Tensor> clipGWeights = WeightStaging.ToOwnedF32(clipGRaw);
            Dictionary<string, Tensor> vaeWeights = WeightStaging.ToOwnedF32(vaeRaw);
            ClipTextEncoder clipG = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGWeights, "text_model");
            VaeDecoder vaeDecoder = new VaeDecoder(VaeConfig.Sdxl);
            vaeDecoder.LoadWeights(vaeWeights);
            VaeEncoder vaeEncoder = LoaderVaeUtils.BuildEncoder(VaeConfig.Sdxl, vaeWeights, "SdxlRefinerRecipe");
            SdxlRefinerPipeline pipeline = new SdxlRefinerPipeline(context.Backend, clipG, entry.Unet, vaeEncoder, vaeDecoder);
            Logs.Info("[SdxlRefinerRecipe] SDXL refiner ready as a standalone model (CLIP-G + aesthetic-score ADM).");
            return new SdxlRefinerRecipePipeline(pipeline, clipG, entry, auxLoader);
        }
        catch
        {
            entry.Dispose();
            auxLoader?.Dispose();
            throw;
        }
    }
}

/// <summary>Thin driver over <see cref="SdxlRefinerPipeline.RefineFromTokens"/>: CLIP-G-only tokenization,
/// a mid-gray full-strength source for text-to-image, the request's own init image otherwise.</summary>
internal sealed class SdxlRefinerRecipePipeline : IRecipePipeline
{
    private readonly SdxlRefinerPipeline _pipeline;
    private readonly SdxlRefinerEntry _entry;
    private readonly SafeTensorsLoader _auxLoader;
    private readonly ClipTokenizer _tokenizer = new ClipTokenizer();

    public SdxlRefinerRecipePipeline(SdxlRefinerPipeline pipeline, ClipTextEncoder clipG, SdxlRefinerEntry entry, SafeTensorsLoader auxLoader)
    {
        _pipeline = pipeline;
        _ = clipG;
        _entry = entry;
        _auxLoader = auxLoader;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? SdxlRefinerRecipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? SdxlRefinerRecipe.FamilyDefaults.CfgScale;
        (int width, int height) = RecipeRequestMapper.Size(request);

        int[] tokensG = _tokenizer.Encode(prompt);
        int[] negG = _tokenizer.Encode(negative);
        int eosG = ClipTokenizer.FindEosPosition(tokensG);
        int negEosG = ClipTokenizer.FindEosPosition(negG);

        using Img2ImgResolver.Img2ImgSpec? initImage = RecipeImg2ImgBinder.Resolve(request, width, height);
        Tensor source;
        float strength;
        if (initImage is not null)
        {
            source = initImage.SourceTensor;
            strength = initImage.Strength;
        }
        else
        {
            // Text-to-image: strength 1 runs the whole schedule from noise, so the source content is never
            // consumed — mid-gray keeps the encode numerically tame.
            source = new Tensor(new Core.Tensors.TensorShape([1L, 3, height, width]), DType.F32);
            strength = 1f;
        }
        try
        {
            SdxlRefinerRequest inner = new SdxlRefinerRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                SourceImage = source,
                Strength = strength,
                Steps = steps,
                CfgScale = cfg,
                Width = width,
                Height = height,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                SeamlessTiling = request.SeamlessTiling,
                // The refiner runs the legacy SchedulerFactory path, which throws on a name it cannot build — so
                // routing through the resolver here is what turns a named sampler into a refusal instead of a drop.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            };
            Action<GenerationProgress>? bridge = progress is null ? null : p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress.Report(new StepPreview { Step = p.Step, TotalSteps = steps });
            };
            (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.RefineFromTokens(tokensG, negG, eosG, negEosG, inner, bridge);
            return new ImageResult
            {
                Rgb = rgb,
                Width = outW,
                Height = outH,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "sdxl-refiner",
                    ["size"] = $"{outW}x{outH}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                    ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            if (initImage is null)
            {
                source.Dispose();
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // ClipTextEncoder/ClipTokenizer are not IDisposable — their tensors belong to the aux loader.
        _pipeline.Dispose();
        _entry.Dispose();
        _auxLoader.Dispose();
    }
}
