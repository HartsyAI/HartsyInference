using System.Diagnostics;
using HartsyInference.Core.Backends;
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
/// are borrowed from the caller; the pipeline owns no weights.</summary>
public sealed unsafe class Fibo2Pipeline : DiffusionPipelineBase
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

    /// <summary>Creates a Fibo 2.0 pipeline over borrowed components. <paramref name="bnMean"/> and <paramref name="bnVar"/> are the
    /// FLUX.2 VAE's F32 BatchNorm statistics over the patchified 128-channel latent.</summary>
    public Fibo2Pipeline(IBackend backend, LlamaStyleEncoder textEncoder, Fibo2Transformer transformer, VaeDecoder vaeDecoder,
        Tensor bnMean, Tensor bnVar, Fibo2Config config, float bnEps = 1e-5f)
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
    /// <param name="onProgress">Optional per-step callback.</param>
    /// <returns>Interleaved RGB bytes and the actual size and seed.</returns>
    public (byte[] rgbData, int width, int height, int seed) GenerateFromTokens(int[] conditionalTokens, int[] unconditionalTokens,
        int width, int height, int steps, float guidanceScale, float sigmaShift, int? seed, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ValidateRequest(conditionalTokens, unconditionalTokens, width, height, steps, guidanceScale, sigmaShift);
        int resolvedSeed = seed ?? SeedGenerator.RandomSeed();
        int latentHeight = height / VaeSpatialScale;
        int latentWidth = width / VaeSpatialScale;

        bool guided = CfgHelper.IsGuidanceActive(guidanceScale);
        Tensor conditional = EncodeBundles(conditionalTokens);
        Tensor? unconditional = guided ? EncodeBundles(unconditionalTokens) : null;
        Tensor? latent = null;
        try
        {
            FlowMatchEulerDiscreteScheduler scheduler = new(sigmaShift);
            scheduler.SetTimesteps(steps);
            latent = SeedGenerator.CreateNoise(new TensorShape(1, _config.InChannels, latentHeight, latentWidth), resolvedSeed);
            float initialSigma = scheduler.InitialNoiseSigma;
            if (MathF.Abs(initialSigma - 1f) > 1e-6f)
                ScaleInPlace(latent, initialSigma);

            for (int i = 0; i < steps; i++)
            {
                Stopwatch stepWatch = Stopwatch.StartNew();
                Tensor velocity = Denoise(latent, scheduler.SigmaAt(i), conditional, unconditional, guidanceScale);
                Tensor next = new(latent.Shape, DType.F32);
                scheduler.Step(next, velocity, latent, i);
                velocity.Dispose();
                latent.Dispose();
                latent = next;
                stepWatch.Stop();
                onProgress?.Invoke(new GenerationProgress(i + 1, steps, stepWatch.Elapsed.TotalMilliseconds));
            }

            using Tensor decoded = DecodeLatent(latent);
            return (ImagePostProcessor.TensorToRgbBytes(decoded), (int)decoded.Shape[3], (int)decoded.Shape[2], resolvedSeed);
        }
        finally
        {
            latent?.Dispose();
            conditional.Dispose();
            unconditional?.Dispose();
        }
    }

    /// <summary>One guided velocity: <c>uncond + g · (cond − uncond)</c> when CFG is active, otherwise the conditional velocity alone.
    /// The caller owns the returned tensor.</summary>
    private Tensor Denoise(Tensor latent, float sigma, Tensor conditional, Tensor? unconditional, float guidanceScale)
    {
        Tensor conditionalVelocity = _transformer.Forward(Backend, latent, sigma, conditional);
        if (unconditional is null)
            return conditionalVelocity;

        Tensor unconditionalVelocity = _transformer.Forward(Backend, latent, sigma, unconditional);
        Tensor guided = CfgHelper.ApplyCfg(unconditionalVelocity, conditionalVelocity, guidanceScale);
        unconditionalVelocity.Dispose();
        conditionalVelocity.Dispose();
        return guided;
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
        using Tensor patched = Flux2LatentOps.Patchify(latent, _config.InChannels * _config.PatchSize * _config.PatchSize, _config.PatchSize);
        using Tensor unnormalized = Flux2LatentOps.BnUnNormalize(patched, _bnMean, _bnVar, _bnEps);
        using Tensor vaeLatent = Flux2LatentOps.Unpatchify(unnormalized, _config.InChannels, _config.PatchSize);
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
