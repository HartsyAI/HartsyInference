using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>Fibo 2.0 text-to-image. Flow-matching Euler over the FLUX.2 32-channel latent with the static 1.15 shift, classifier-free
/// guidance against the null caption, and the FLUX.2 VAE decode with its BatchNorm denormalisation. The text encoder, denoiser and VAE
/// are borrowed from the caller; the pipeline owns no weights. With a VAE encoder it also runs img2img and mask inpaint
/// (see <c>Fibo2Pipeline.Img2Img.cs</c>).</summary>
public sealed unsafe partial class Fibo2Pipeline : DiffusionPipelineBase
{
    /// <summary>Width of the VAE's spatial compression; the denoiser's patch size adds a further factor of two.</summary>
    private const int VaeSpatialScale = 8;

    private static readonly int[] TapLayers = [.. Fibo2TextBundles.TapLayers];

    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Fibo2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly Tensor _bnMean;
    private readonly Tensor _bnVar;
    private readonly float _bnEps;
    private readonly Fibo2Config _config;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly ResidentWeights _textEncoderWeights;
    private readonly ResidentWeights _denoiserWeights;

    /// <summary>Creates a Fibo 2.0 pipeline over borrowed components. <paramref name="bnMean"/> and <paramref name="bnVar"/> are the
    /// FLUX.2 VAE's F32 BatchNorm statistics over the patchified 128-channel latent.</summary>
    public Fibo2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder, Fibo2Transformer transformer, VaeDecoder vaeDecoder,
        Tensor bnMean, Tensor bnVar, Fibo2Config config, float bnEps = 1e-4f, VaeEncoder? vaeEncoder = null)
        : base(backend)
    {
        ArgumentNullException.ThrowIfNull(textEncoder);
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(vaeDecoder);
        ArgumentNullException.ThrowIfNull(bnMean);
        ArgumentNullException.ThrowIfNull(bnVar);
        ArgumentNullException.ThrowIfNull(config);
        int patchVolume = config.InChannels * config.PatchSize * config.PatchSize;
        if (bnMean.Shape.Rank != 1 || bnMean.Shape[0] != patchVolume || bnVar.Shape.Rank != 1 || bnVar.Shape[0] != patchVolume)
            throw new ArgumentException($"Fibo 2.0 BatchNorm statistics must be [{patchVolume}], got {bnMean.Shape} and {bnVar.Shape}.");
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _bnMean = bnMean;
        _bnVar = bnVar;
        _bnEps = bnEps;
        _config = config;
        _vaeEncoder = vaeEncoder;
        _textEncoderWeights = new ResidentWeights(backend, textEncoder.EnumerateWeights);
        _denoiserWeights = new ResidentWeights(backend, transformer.EnumerateWeights);
    }

    /// <summary>Generates one image from the conditional caption tokens and the null-caption tokens (the CFG branch). Both sequences come
    /// from <see cref="Fibo2PromptTemplate"/> and are encoded at their own unpadded length, because the denoiser derives the gist count
    /// from it.</summary>
    /// <param name="conditionalTokens">Template-wrapped caption ids.</param>
    /// <param name="unconditionalTokens">Template-wrapped null-caption ids; encoded only when guidance is active.</param>
    /// <param name="width">Output width in pixels; a multiple of 16.</param>
    /// <param name="height">Output height in pixels; a multiple of 16.</param>
    /// <param name="steps">Euler steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale; at or below 1 the unconditional branch is skipped.</param>
    /// <param name="sigmaShift">The scheduler's <c>s</c> in <c>s·t/(1 + (s − 1)·t)</c>; ComfyUI's Fibo 2.0 shift 1.15 is
    /// <c>s = e^1.15</c>.</param>
    /// <param name="seed">Noise seed; null draws one.</param>
    /// <param name="onProgress">Optional per-step callback; img2img reports only the steps it runs.</param>
    /// <param name="img2img">Init image, strength and optional mask (1 = regenerate); null for text-to-image. Its size must match
    /// <paramref name="width"/> and <paramref name="height"/>. Needs the VAE encoder.</param>
    /// <returns>Interleaved RGB bytes and the actual size and seed.</returns>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(int[] conditionalTokens, int[] unconditionalTokens,
        int width, int height, int steps, float guidanceScale, float sigmaShift, int? seed, Action<GenerationProgress>? onProgress = null, ImageToImageRequest? img2img = null)
    {
        ThrowIfDisposed();
        ValidateRequest(conditionalTokens, unconditionalTokens, width, height, steps, guidanceScale, sigmaShift);
        int resolvedSeed = seed ?? SeedGenerator.RandomSeed();
        int latentHeight = height / VaeSpatialScale;
        int latentWidth = width / VaeSpatialScale;

        Img2ImgSetup.Plan plan = PrepareInit(img2img, height, width, steps);
        if (plan.PassThrough)
            return (ImagePostProcessor.TensorToRgbBytes(img2img!.SourceImage), width, height, resolvedSeed);

        bool guided = CfgHelper.IsGuidanceActive(guidanceScale);
        Tensor? conditional = null;
        Tensor? unconditional = null;
        Tensor? latent = null;
        Tensor? sourceLatent = null;
        Tensor? latentMask = null;
        try
        {
            // The text encoder leaves the device before the denoiser loads: on a 12 GB card the two do not fit together. A denoiser kept
            // resident from an earlier generation is freed here unless the encoder fits beside it (FreeDenoiserForEncoder).
            FreeDenoiserForEncoder(() => _textEncoder.EnumerateWeights());
            _textEncoderWeights.Preload();
            try
            {
                conditional = EncodeBundles(conditionalTokens);
                unconditional = guided ? EncodeBundles(unconditionalTokens) : null;
            }
            finally
            {
                _textEncoderWeights.Release(force: true);
            }
            AfterWeightRelease(() => _textEncoder.EnumerateWeights(), "text encoder");
            FlowMatchEulerDiscreteScheduler scheduler = new(sigmaShift);
            scheduler.SetTimesteps(steps);
            TensorShape latentShape = new(1, _config.InChannels, latentHeight, latentWidth);
            if (img2img is not null)
                sourceLatent = EncodeInitLatent(img2img.SourceImage);
            latent = BuildStartLatent(scheduler, SavedNoise.Create(EngineKnobs.Fibo2NoiseNpy, latentShape, resolvedSeed, "Fibo2"), sourceLatent,
                plan.StartStep);
            LatentInpaint? inpaint = null;
            if (plan.MaskPixel is not null)
            {
                latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(plan.MaskPixel, latentHeight, latentWidth);
                inpaint = new LatentInpaint(scheduler, sourceLatent!, latentMask, resolvedSeed);
            }

            _denoiserWeights.Preload();
            try
            {
                SampleFlow(scheduler, ref latent, plan.StartStep, steps,
                    (current, sigma) => Denoise(current, sigma, conditional!, unconditional, guidanceScale), inpaint, onProgress);
            }
            finally
            {
                _denoiserWeights.Release();
            }
            AfterWeightRelease(() => _transformer.EnumerateWeights(), "denoiser");

            DeviceMemoryTrace.Record(Backend, "fibo2: before VAE decode");
            using Tensor decoded = DecodeLatent(latent!);
            DeviceMemoryTrace.Record(Backend, "fibo2: after VAE decode");
            // Suppresses the VAE round trip in the kept region; the host recomposites itself when it owns the mask.
            if (plan.MaskPixel is not null && img2img!.RecompositeAtEnd)
                MaskBlendUtilities.BlendChannelsInPlace(decoded, img2img.SourceImage, plan.MaskPixel);
            return (ImagePostProcessor.TensorToRgbBytes(decoded), (int)decoded.Shape[3], (int)decoded.Shape[2], resolvedSeed);
        }
        finally
        {
            latent?.Dispose();
            sourceLatent?.Dispose();
            latentMask?.Dispose();
            conditional?.Dispose();
            unconditional?.Dispose();
        }
    }

    /// <summary>After a weight release: trims the device pool so freed blocks leave the reservation, records the device usage at this
    /// stage, and logs how many of the released tensors are still resident. A nonzero count means the release did not leave the GPU.</summary>
    private void AfterWeightRelease(Func<IEnumerable<Tensor>> released, string label)
    {
        Backend.TrimMemoryPool();
        DeviceMemoryTrace.Record(Backend, $"fibo2: {label} released");
        int total = 0, resident = 0;
        foreach (Tensor tensor in released())
        {
            total++;
            if (Backend.IsWeightResident(tensor))
                resident++;
        }
        string outcome = resident == 0 ? "freed" : "kept resident (the VRAM policy kept them, or the free did not reach the device)";
        Logs.Info($"[Fibo2][Residency] {label}: {resident} of {total} tensors resident; {outcome}");
    }

    /// <summary>Whether <paramref name="needBytes"/> fit in the device memory free now, with <see cref="HeadroomBytes"/> to spare. A backend
    /// that reports no free memory (0) is never a fit, so the caller takes the release path.</summary>
    internal static bool Fits(long freeBytes, long needBytes, long headroomBytes) =>
        freeBytes > 0 && needBytes >= 0 && freeBytes >= needBytes + headroomBytes;

    /// <summary>The smallest headroom a fit keeps. The fit uses the larger of this and <c>vram.preloadHeadroomMb</c>, the room
    /// <c>PreloadWeights</c> itself keeps, so a fit cannot make the encoder preload skip tensors.</summary>
    internal const long HeadroomFloorBytes = 1L << 30;

    /// <summary>Frees the denoiser before an encoder preloads, unless the encoder fits next to it. The pool is trimmed first, so the free figure
    /// counts memory the allocator is holding for nothing. The denoiser is already on the device when it is kept, so its bytes are not in the
    /// free figure; a fit therefore keeps both, and the next generation skips the re-upload. Whether the keep path is cheaper on a large card
    /// is not measured.</summary>
    private void FreeDenoiserForEncoder(Func<IEnumerable<Tensor>> encoderWeights)
    {
        long encoderBytes = 0;
        foreach (Tensor tensor in encoderWeights())
            encoderBytes += tensor.DType.ComputeByteCount(tensor.ElementCount);
        Backend.TrimMemoryPool();
        long headroom = Math.Max(HeadroomFloorBytes, EngineKnobs.PreloadHeadroomMb.Value << 20);
        if (Fits(Backend.FreeMemoryBytes(), encoderBytes, headroom))
            return;
        _denoiserWeights.Release(force: true);
    }

    /// <summary>One guided velocity: <c>uncond + g · (cond − uncond)</c> when CFG is active, otherwise the conditional velocity alone.
    /// The caller owns the returned tensor.</summary>
    private Tensor Denoise(Tensor latent, float sigma, Tensor conditional, Tensor? unconditional, float guidanceScale)
    {
        if (DeviceMemoryTrace.Enabled)
            DeviceMemoryTrace.Record(Backend, "fibo2: denoise", $"sigma={sigma:F4}");
        Tensor conditionalVelocity = _transformer.Forward(Backend, latent, sigma, conditional);
        if (unconditional is null)
            return conditionalVelocity;

        Tensor? unconditionalVelocity = null;
        try
        {
            unconditionalVelocity = _transformer.Forward(Backend, latent, sigma, unconditional);
            return CfgHelper.ApplyCfg(unconditionalVelocity, conditionalVelocity, guidanceScale);
        }
        finally
        {
            unconditionalVelocity?.Dispose();
            conditionalVelocity.Dispose();
        }
    }

    /// <summary>Encodes one caption and returns its F32 <c>[1, S, 6 · 3 · 2560]</c> text bundles.</summary>
    private Tensor EncodeBundles(int[] tokens)
    {
        Tensor taps = _textEncoder.EncodeMultiLayer(Backend, [tokens], TapLayers, interleavedLayout: false);
        try
        {
            int hidden = (int)(taps.Shape[2] / TapLayers.Length);
            return Fibo2TextBundles.Build(taps, hidden);
        }
        finally
        {
            taps.Dispose();
        }
    }

    /// <summary>Normalised 32-channel latent to pixels: 2×2 patchify, BatchNorm denormalisation on the 128 channels, unpatchify, then the
    /// VAE decoder.</summary>
    private Tensor DecodeLatent(Tensor latent)
    {
        using Tensor vaeLatent = Flux2LatentOps.UnnormalizePatchified(Backend, latent, _bnMean, _bnVar, _bnEps, _config.PatchSize);
        return _vaeDecoder.DecodeTiled(Backend, vaeLatent);
    }

    private void ValidateRequest(int[] conditionalTokens, int[] unconditionalTokens, int width, int height, int steps,
        float guidanceScale, float sigmaShift)
    {
        ArgumentNullException.ThrowIfNull(conditionalTokens);
        ArgumentNullException.ThrowIfNull(unconditionalTokens);
        if (conditionalTokens.Length == 0 || unconditionalTokens.Length == 0)
            throw new ArgumentException("Fibo 2.0 needs non-empty conditional and unconditional token sequences.");
        int multiple = VaeSpatialScale * _config.PatchSize;
        if (width <= 0 || height <= 0 || width % multiple != 0 || height % multiple != 0)
            throw new ArgumentException($"Fibo 2.0 size {width}x{height} must be a positive multiple of {multiple} on both sides.");
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "Fibo 2.0 needs at least one step.");
        if (!float.IsFinite(guidanceScale) || guidanceScale <= 0f)
            throw new ArgumentOutOfRangeException(nameof(guidanceScale), guidanceScale, "Guidance scale must be finite and above 0.");
        if (!float.IsFinite(sigmaShift) || sigmaShift <= 0f)
            throw new ArgumentOutOfRangeException(nameof(sigmaShift), sigmaShift, "Sigma shift must be finite and above 0.");
    }

    /// <summary>Scales <paramref name="tensor"/> through its host pointer, so it must be host-resident F32.</summary>
    private static void ScaleInPlace(Tensor tensor, float factor)
    {
        float* values = (float*)tensor.DataPointer;
        long count = tensor.ElementCount;
        for (long i = 0; i < count; i++)
            values[i] *= factor;
    }
}
