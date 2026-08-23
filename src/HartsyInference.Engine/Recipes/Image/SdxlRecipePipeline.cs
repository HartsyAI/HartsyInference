using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Tokenizers;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed SDXL pipeline driven against the native <see cref="ImageRequest"/>. Encodes the prompt/negative through the embedded CLIP-L/CLIP-G vocab, resolves the request's composition objects through <see cref="UnetCompositionPlan"/> plus the refiner swap, and runs <see cref="SdxlPipeline.GenerateFromTokens"/>.</summary>
public sealed class SdxlRecipePipeline : IRecipePipeline
{
    private readonly SdxlPipeline _pipeline;
    private readonly ClipTokenizer _tokenizer = new ClipTokenizer();
    private readonly IBackend _backend;
    /// <summary>Backend for weighted CLIP conditioning — follows the pipeline's text-encoder placement so the weighted and unweighted encode paths agree on where the encoders live.</summary>
    private readonly IBackend _textBackend;
    private readonly ClipTextEncoder _clipL;
    private readonly ClipTextEncoder _clipG;
    private readonly MergedLoraStack? _loraStack;
    private readonly IpAdapterCache _ipAdapterCache = new IpAdapterCache();
    private readonly Dictionary<string, SdxlRefinerEntry> _refinerCache = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>SDXL's dual-CLIP conditioning is penultimate-layer by spec, so weighted encoding uses the same depth.</summary>
    private const int SdxlLayersFromEnd = 2;

    /// <summary>Wraps the constructed SDXL pipeline plus its text encoders and merged LoRA stack, taking ownership of every disposable.</summary>
    public SdxlRecipePipeline(
        SdxlPipeline pipeline,
        IBackend backend,
        ClipTextEncoder clipL,
        ClipTextEncoder clipG,
        MergedLoraStack? loraStack)
    {
        _pipeline = pipeline;
        _backend = backend;
        _textBackend = pipeline.TextEncoderBackend;
        _clipL = clipL;
        _clipG = clipG;
        _loraStack = loraStack;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancel.ThrowIfCancellationRequested();
        string negative = request.NegativePrompt ?? "";

        // Textual-inversion embed markers (\0swarmembed:NAME\0end — SwarmUI core's own rewrite of <embed:name>,
        // arriving on request.Prompt/NegativePrompt before this recipe ever sees them) resolve to a token plan with
        // placeholder ids past the CLIP vocab, encoded via the InlineMap ClipTextEncoder.Encode/EncodePenultimate
        // already accept. Null when a prompt carries no markers, so the plain path below is unchanged for every
        // other request. The negative plan's placeholder ids are offset well past the positive plan's range — both
        // maps get merged into one dictionary shared across the whole 2-row batch (BuildDualClipSchedule), so a
        // "bad-hands"-style embed in the negative prompt and a style embed in the positive one must not collide.
        // The plain tokensL/tokensG/eosG below always use the STRIPPED prompt: they still feed the pooled vector
        // (SdxlPipeline computes it internally regardless of conditioningSchedule — see EmbeddingResolver's doc),
        // and encoding the raw \0swarmembed control characters through the normal tokenizer would be garbage BPE.
        string strippedPrompt = EmbeddingResolver.StripMarkers(request.Prompt) ?? request.Prompt;
        string strippedNegative = EmbeddingResolver.StripMarkers(negative) ?? negative;
        using EmbeddingResolver.Plan? embedPlan = EmbeddingResolver.Resolve(request.Prompt, _tokenizer, [768, 1280]);
        using EmbeddingResolver.Plan? negEmbedPlan = EmbeddingResolver.Resolve(negative, _tokenizer, [768, 1280], startPlaceholderId: ClipTokenizer.VocabSize + 1_000_000);

        int[] tokensL = _tokenizer.Encode(strippedPrompt);
        int[] negL = _tokenizer.Encode(strippedNegative);
        int[] tokensG = _tokenizer.Encode(strippedPrompt);
        int[] negG = _tokenizer.Encode(strippedNegative);
        int eosG = ClipTokenizer.FindEosPosition(tokensG);
        int negEosG = ClipTokenizer.FindEosPosition(negG);

        using UnetCompositionPlan plan = UnetCompositionPlan.Build(
            request,
            _backend,
            UNetConfig.SdxlBase,
            IpAdapterBaseModel.Sdxl,
            embedPlan is not null || negEmbedPlan is not null
                // Embeds and weighted-prompt syntax ((word:1.2)) both build the same dual-CLIP conditioning
                // schedule slot; combining them would need weight parsing to skip over embed placeholder runs,
                // which WeightedConditioning doesn't do — an embed-bearing prompt wins outright for now.
                ? () => EmbeddingResolver.BuildDualClipSchedule(_textBackend, _clipL, _clipG, _tokenizer, embedPlan, strippedPrompt, strippedNegative, SdxlLayersFromEnd, negEmbedPlan)
                : () => WeightedConditioning.BuildDualClip(_textBackend, _clipL, _clipG, _tokenizer, strippedPrompt, negative, SdxlLayersFromEnd),
            _ipAdapterCache.Lookup,
            _ipAdapterCache.Cache,
            cancel);

        // Two hand-off methods, resolved once: "StepSwap" stays a single GenerateFromTokens call (base+refiner
        // share one in-flight latent, same resolution throughout); anything else — "PostApply" is the resolver's
        // own default — runs the base pass alone here, then ApplyPostRefiner does a full pixel-space roundtrip
        // afterward, which is what lets it change resolution (Tier 3.1 hires-fix).
        // Internal handling covers ONLY the classic SDXL / SDXL-refiner pair (this is what carries the
        // aesthetic-score conditioning and true mid-loop StepSwap). Any other refiner model — Flux over SDXL,
        // whatever — is the generic RefinerStage's job after this pipeline returns; consuming it here would try
        // to load that checkpoint as an SDXL refiner UNet. RefinerStage.IsSdxlInternalPair is the shared decision.
        RefinerResolver.RefinerSpec? refinerSpec = RefinerStage.IsSdxlInternalPair(request, "sdxl")
            ? RefinerResolver.Resolve(
                request.Refiner, request.Steps ?? SdxlRecipe.FamilyDefaults.Steps, request.CfgScale ?? SdxlRecipe.FamilyDefaults.CfgScale)
            : null;
        bool postApply = refinerSpec is not null && !string.Equals(refinerSpec.Method, "StepSwap", StringComparison.OrdinalIgnoreCase);
        RefinerSwapConfig? stepSwapRefiner = postApply ? null : BuildStepSwapConfig(refinerSpec);
        TextToImageRequest inner = BuildInner(request, negative, plan);

        int totalSteps = request.Steps ?? SdxlRecipe.FamilyDefaults.Steps;
        (byte[] rgb, int width, int height, int usedSeed) = _pipeline.GenerateFromTokens(
            tokensL, negL, tokensG, negG, eosG, negEosG, inner,
            p => progress?.Report(new StepPreview { Step = p.Step, TotalSteps = totalSteps }),
            controlNets: plan.ControlNets?.Conditionings,
            refiner: stepSwapRefiner,
            ipAdapters: plan.IpAdapters?.Conditionings,
            conditioningSchedule: plan.Conditioning);

        Dictionary<string, string> meta = new(StringComparer.OrdinalIgnoreCase)
        {
            ["arch"] = "sdxl",
            ["size"] = $"{width}x{height}",
            ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
            ["steps"] = totalSteps.ToString(CultureInfo.InvariantCulture),
        };

        if (postApply && refinerSpec is not null)
        {
            (rgb, width, height) = ApplyPostRefiner(refinerSpec, rgb, width, height, strippedPrompt, strippedNegative, tokensG, negG, eosG, negEosG, usedSeed, request.SeamlessTiling);
            meta["refiner_post_apply"] = refinerSpec.Model;
            meta["size"] = $"{width}x{height}";
        }

        return new ImageResult
        {
            Rgb = rgb,
            Width = width,
            Height = height,
            Seed = usedSeed,
            Meta = meta,
        };
    }

    /// <summary>Builds the inner diffusion request — an <see cref="ImageToImageRequest"/> when the plan resolved an init image, else plain text-to-image with any variation-seed noise injected.</summary>
    private static TextToImageRequest BuildInner(ImageRequest request, string negative, UnetCompositionPlan plan)
    {
        TextToImageRequest inner = RecipeRequestMapper.ToTextToImage(request, negative) with
        {
            // request.Sampler is what the "HartsyInference Sampler" param actually populates (euler/ddim/
            // dpm++2m/lcm/tcd — SchedulerFactory's own domain despite the field name); request.Scheduler is
            // always null from the extension today ("the Engine resolves the family's canonical schedule").
            // Reading only Scheduler here made the Sampler param silently inert — found 2026-08-10 chasing an
            // unrelated "tcd" dropdown fix.
            // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name
            // here — before the checkpoint loads — instead of deep inside SchedulerFactory.
            Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            // The plan only resolves variation noise on the text-to-image path, so this is null under img2img.
        };
        return RecipeImg2ImgBinder.Apply(inner, plan.Img2Img);
    }

    /// <summary>Builds the mid-loop StepSwap config for a "StepSwap"-method refiner spec, loading (and caching) the refiner UNet. Returns null for a null spec (no refiner) — the PostApply case never reaches here, its spec is handled by <see cref="ApplyPostRefiner"/> after the base pass instead.</summary>
    private RefinerSwapConfig? BuildStepSwapConfig(RefinerResolver.RefinerSpec? spec)
    {
        if (spec is null)
        {
            return null;
        }
        if (spec.Upscale is > 1f)
        {
            Logs.Warning($"[SdxlRecipePipeline] Refiner upscale {spec.Upscale} is ignored — StepSwap keeps the base latent resolution. Use RefinerMethod=PostApply for hires-fix.");
        }
        SdxlRefinerEntry entry = LoadRefinerEntry(spec.Model);
        return new RefinerSwapConfig { RefinerUnet = entry.Unet, Strength = spec.Strength };
    }

    /// <summary>Tier 3.1 hires-fix: the PostApply hand-off. Resizes the base pass's decoded pixels (only when <see cref="RefinerResolver.RefinerSpec.Upscale"/> != 1 — same tensor, no-op resize otherwise), VAE-encodes the result through the SAME untiled encoder ordinary img2img already uses (SDXL's VAE runs F32 — outside the BF16 cuDNN fast-path the tiled encoder's segfault requires, see ROADMAP.md), and redenoises the WHOLE refiner schedule (<see cref="SdxlRefinerPipeline"/>, <c>Strength</c> = the resolved Control fraction) with the refiner UNet's own CLIP-G-only/aesthetic-score conditioning. Reuses the base pass's own CLIP-G tokens (no separate <c>&lt;refiner&gt;</c> prompt support in this first slice — StepSwap's <see cref="RefinerSwapConfig.RefinerConditioning"/> override has no PostApply equivalent yet).</summary>
    private (byte[] rgb, int width, int height) ApplyPostRefiner(
        RefinerResolver.RefinerSpec spec, byte[] baseRgb, int baseWidth, int baseHeight,
        string strippedPrompt, string strippedNegative, int[] tokensG, int[] negG, int eosG, int negEosG, int seed,
        string? seamlessTiling)
    {
        SdxlRefinerEntry entry = LoadRefinerEntry(spec.Model);
        int newWidth = spec.Upscale is > 0f and not 1f ? RoundToMultipleOf8(baseWidth * spec.Upscale) : baseWidth;
        int newHeight = spec.Upscale is > 0f and not 1f ? RoundToMultipleOf8(baseHeight * spec.Upscale) : baseHeight;
        Tensor resized = ResizeRgbToSourceTensor(baseRgb, baseWidth, baseHeight, newWidth, newHeight);

        // VaeEncoder is guaranteed non-null: SdxlRecipe always constructs the base pipeline with one (SDXL img2img
        // has been supported since Tier 0), so every SdxlRecipePipeline instance has it.
        SdxlRefinerPipeline refinerPipeline = new SdxlRefinerPipeline(
            _backend, _clipG, entry.Unet, _pipeline.VaeEncoder!, _pipeline.VaeDecoder);
        SdxlRefinerRequest refinerRequest = new SdxlRefinerRequest
        {
            Prompt = strippedPrompt,
            NegativePrompt = strippedNegative,
            SourceImage = resized,
            Strength = spec.Strength,
            Steps = spec.Steps,
            CfgScale = spec.CfgScale,
            Width = newWidth,
            Height = newHeight,
            Seed = seed,
            // Must match the base stage: the refiner denoises over the base output, and square padding here
            // would re-introduce exactly the seam the base pass avoided.
            SeamlessTiling = seamlessTiling,
        };
        Logs.Info($"[SdxlRecipePipeline] PostApply refiner: {baseWidth}x{baseHeight} -> {newWidth}x{newHeight}, strength={spec.Strength:F2}.");
        (byte[] rgb, int width, int height, int _) = refinerPipeline.RefineFromTokens(tokensG, negG, eosG, negEosG, refinerRequest);
        resized.Dispose();
        return (rgb, width, height);
    }

    /// <summary>Loads (and caches) the refiner UNet for <paramref name="model"/> — shared by both hand-off methods.</summary>
    private SdxlRefinerEntry LoadRefinerEntry(string model)
    {
        if (!_refinerCache.TryGetValue(model, out SdxlRefinerEntry? entry))
        {
            entry = SdxlRefinerLoader.Load(model);
            _refinerCache[model] = entry;
        }
        return entry;
    }

    /// <summary>Nearest multiple of 8, minimum 8 — SDXL's VAE downsamples by 8x, so both dimensions must divide evenly for the latent shape to be exact.</summary>
    private static int RoundToMultipleOf8(double v) => Math.Max(8, (int)(Math.Round(v / 8.0) * 8.0));

    /// <summary>Resizes RGB bytes to a new resolution and converts to a VAE-input tensor <c>[1,3,dstH,dstW]</c> in <c>[-1,1]</c>. Same-size case skips the resize pass entirely (byte-identical to <see cref="Diffusion.Utilities.ImagePostProcessor.RgbBytesToTensor"/>). The resize itself mirrors <c>SeedVr2Preprocess</c>'s own <c>TorchResize.BicubicAntialiasChw</c> usage: build [0,1] CHW planes, resize, then map to [-1,1] — this hardcodes one resize algorithm rather than reading a user-selectable one (Comfy's <c>refinerupscalemethod</c> stays unconsumed for that reason, see <c>HonoredComfyParams</c>'s own comment in the extension repo).</summary>
    private static Tensor ResizeRgbToSourceTensor(byte[] rgb, int srcW, int srcH, int dstW, int dstH)
    {
        if (srcW == dstW && srcH == dstH)
        {
            return ImagePostProcessor.RgbBytesToTensor(rgb, srcW, srcH);
        }
        float[] srcPlanes = new float[3 * srcH * srcW];
        for (int y = 0; y < srcH; y++)
        {
            int rowBase = y * srcW;
            for (int x = 0; x < srcW; x++)
            {
                int px = (rowBase + x) * 3;
                int planeIdx = rowBase + x;
                srcPlanes[planeIdx] = rgb[px] * (1f / 255f);
                srcPlanes[srcH * srcW + planeIdx] = rgb[px + 1] * (1f / 255f);
                srcPlanes[2 * srcH * srcW + planeIdx] = rgb[px + 2] * (1f / 255f);
            }
        }
        float[] dstPlanes = new float[3 * dstH * dstW];
        TorchResize.BicubicAntialiasChw(srcPlanes, 3, srcH, srcW, dstPlanes, dstH, dstW);

        Tensor image = new Tensor(new TensorShape(1, 3, dstH, dstW), DType.F32);
        Span<float> dst = image.AsSpan<float>();
        for (int i = 0; i < dstPlanes.Length; i++)
        {
            // Clamp then map [0,1] -> [-1,1], matching ImagePostProcessor.RgbBytesToTensor's convention. Bicubic
            // overshoot is real (TorchResize's own doc: PyTorch keeps it) — clamp here the same way the source
            // byte path implicitly does (a byte is already clamped to [0,255] before this method ever sees it).
            dst[i] = Math.Clamp(dstPlanes[i], 0f, 1f) * 2f - 1f;
        }
        return image;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _ipAdapterCache.Dispose();
        foreach (SdxlRefinerEntry entry in _refinerCache.Values)
        {
            entry.Dispose();
        }
        _refinerCache.Clear();
        _pipeline.Dispose();
        _tokenizer.Dispose();
        // The LoRA stack owns the merged tensors the components reference, so it outlives them by exactly this much.
        _loraStack?.Dispose();
        Logs.Verbose("[SdxlRecipePipeline] Disposed.");
    }
}
