using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>MOSS-Image text-to-image, image-to-image and inpaint. The prompt is encoded by Qwen3.5-9B
/// (<see cref="IFinalHiddenStateEncoder"/>, the final normalized hidden state of every token, no padding), the image
/// lives in FLUX.2 VAE latents as BN-normalized packed tokens, and <see cref="MossImageTransformer"/> denoises on
/// OpenMOSS's schedule: a shifted grid snapped to the 1000-step training grid (<see cref="TrainingGridSchedule"/>).
/// Guidance follows the reference: true CFG whenever the scale is not 1, with the unconditional branch skipping its
/// <see cref="MossImageGuidance.UncondSkipLayers"/> and an optional three-pass SLG branch. The text side of every
/// branch is built once per generation, and on the default Euler sampler the whole branch set of a step replays as one
/// captured step graph where the backend supports it.</summary>
public sealed unsafe class MossImagePipeline : DiffusionPipelineBase
{
    private const string Family = "MOSS-Image";
    private const int CondSlot = 0;
    private const int UncondSlot = 1;

    private readonly IFinalHiddenStateEncoder _textEncoder;
    private readonly MossImageTransformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly Tensor _bnMean;
    private readonly Tensor _bnVar;
    private readonly float _bnEps;
    private readonly MossImageConfig _config;
    private readonly EncodedPromptCache _promptCache = new(2);
    private bool _ditResident;

    /// <summary>Creates the pipeline; <paramref name="vaeEncoder"/> is required for img2img and inpaint.</summary>
    public MossImagePipeline(IBackend backend, IFinalHiddenStateEncoder textEncoder, MossImageTransformer transformer,
        VaeDecoder vaeDecoder, VaeEncoder? vaeEncoder, Tensor bnMean, Tensor bnVar, float bnEps = 1e-5f)
        : base(backend)
    {
        ArgumentNullException.ThrowIfNull(textEncoder);
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(vaeDecoder);
        ArgumentNullException.ThrowIfNull(bnMean);
        ArgumentNullException.ThrowIfNull(bnVar);
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _bnMean = bnMean;
        _bnVar = bnVar;
        _bnEps = bnEps;
        _config = transformer.Config;
        if (textEncoder.HiddenSize != _config.TextDim)
        {
            throw new HartsyInferenceException(
                $"{Family}: the text encoder's hidden size {textEncoder.HiddenSize} does not match the transformer's text width {_config.TextDim}.");
        }
    }

    /// <summary>Keeps the DiT resident across generations unless the VRAM profile says otherwise.</summary>
    private bool KeepModelsResident => VramLevers.KeepResident(Backend);

    /// <summary>Generates an image. <paramref name="negative"/> is encoded whenever the CFG scale is not 1 (the reference
    /// runs both branches even below 1); an absent negative is the empty prompt. <see cref="TextToImageRequest.SigmaShift"/>
    /// is the schedule's time shift (reference 5). With <paramref name="references"/> this is an edit: the prompt's
    /// placeholder run takes the vision copies through the encoder, and the VAE copies become the DiT's reference tokens.
    /// The Ostris layouts also send the vision copies with the negative; the inline layout's negative encodes no image.</summary>
    public (byte[] rgbData, int width, int height, int seed) Generate(WeightedTokenSequence prompt, WeightedTokenSequence negative,
        TextToImageRequest request, MossImageGuidance guidance, Action<GenerationProgress>? onProgress = null,
        ReferenceImageInputs? references = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(negative);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(guidance);
        guidance.Validate(_config.NumLayers);
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        if (request is ImageToImageRequest && _vaeEncoder is null)
            throw new InvalidOperationException($"{Family} image-to-image and inpaint need the FLUX.2 VAE encoder.");
        ValidateReferences(references);
        bool encodes = references is not null || request is ImageToImageRequest;

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int requestedSteps = request.Steps;
        float cfg = request.CfgScale;
        int reqWidth = request.Width;
        int reqHeight = request.Height;
        int unit = _config.VaeDownscaleFactor;
        int imgH = Math.Max(unit, reqHeight / unit * unit);
        int imgW = Math.Max(unit, reqWidth / unit * unit);
        int patH = imgH / unit;
        int patW = imgW / unit;
        float shift = request.SigmaShift ?? MossImageConfig.ReferenceShift;
        // Everything that can refuse the request resolves before anything is allocated.
        FlowMatchEulerDiscreteScheduler scheduler = new(shift: 1f);
        scheduler.SetSigmas(TrainingGridSchedule.Sigmas(requestedSteps, shift));
        int steps = scheduler.NumInferenceSteps;
        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, imgH, imgW, steps);
        if (plan.PassThrough)
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), imgW, imgH, seed);
        bool guided = MathF.Abs(cfg - 1f) > 1e-6f;
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, Family, startsFromNoisedInit: plan.StartStep > 0);
        bool masked = plan.MaskPixel is not null;
        Logs.Info($"{Family}: {imgW}x{imgH}, {steps} steps (shift {shift}), cfg={cfg}, uncond skip=[{string.Join(",", guidance.UncondSkipLayers)}]"
            + (guidance.HasSlgBranch ? $", slg {guidance.SlgScale} on [{string.Join(",", guidance.SlgLayers)}]" : "")
            + $", window {guidance.Start}..{guidance.Stop}, seed={seed}"
            + (references is null ? "" : $", {references.Count} reference image(s), {references.Options.Mode}"));
        Stopwatch sw = Stopwatch.StartNew();

        Tensor? latent = null;
        Tensor? sourceTokens = null;
        Tensor? packedMask = null;
        Tensor? weightedCond = null;
        Tensor? weightedUncond = null;
        MossImageConditioning? cond = null;
        MossImageConditioning? uncond = null;
        ReferenceLatentTokens[]? referenceTokens = null;
        bool graph = false;
        try
        {
            IReadOnlyList<Tensor> vision = references?.Vision ?? [];
            // The inline layout's negative is a plain encode of the prompt with no image, as the colorize node runs it.
            IReadOnlyList<Tensor> negativeVision = references?.Options.Mode == ReferenceAttentionMode.Inline ? [] : vision;
            Tensor condText = EncodePrompt(prompt, CondSlot, vision);
            Tensor? uncondText = guided ? EncodePrompt(negative, UncondSlot, negativeVision) : null;

            if (encodes)
                Backend.PreloadWeights(_vaeEncoder!.EnumerateWeights());
            try
            {
                if (references is not null)
                {
                    referenceTokens = Flux2ReferenceEncoder.Encode(Backend, _vaeEncoder!, references.Latent, _bnMean, _bnVar, _bnEps,
                        _config.PatchSize, _config.VaeDownscaleFactor, Family);
                }
                (latent, sourceTokens) = BuildInitialLatent(request, scheduler, patH, patW, seed, plan.StartStep, masked);
            }
            finally
            {
                if (encodes)
                {
                    Backend.Sync();
                    Backend.FreeWeights(_vaeEncoder!.EnumerateWeights());
                    Backend.FreeActivations();
                }
            }
            if (masked)
            {
                int latH = imgH / 8, latW = imgW / 8;
                using Tensor latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(plan.MaskPixel!, latH, latW);
                packedMask = MaskBlendUtilities.PackLatentMask2x2(latentMask, latH, latW);
            }

            weightedCond = prompt.IsUniformlyUnweighted ? null : CondTokenWeights.Apply(Backend, condText, null, prompt).Cond;
            weightedUncond = guided && !negative.IsUniformlyUnweighted ? CondTokenWeights.Apply(Backend, uncondText!, null, negative).Cond : null;
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            _ditResident = true;
            ReferenceAttentionMode mode = references?.Options.Mode ?? ReferenceAttentionMode.CachedKv;
            cond = _transformer.PrepareConditioning(Backend, weightedCond ?? condText, patH, patW, referenceTokens, mode);
            uncond = guided ? _transformer.PrepareConditioning(Backend, weightedUncond ?? uncondText!, patH, patW, referenceTokens, mode) : null;
            if (references is not null)
                WarmReferenceCaches(references.Options, cond, uncond, guided, guidance, plan.StartStep, steps);

            // The step graph needs the latent at a fixed address that only the in-place Euler update writes, so a
            // second-order sampler (extra forwards, scratch latents) and the inpaint re-blend (a host write) run eagerly.
            graph = !masked && !FlowMatchSampling.IsNonDefault(request.Scheduler) && _transformer.GraphEligible(Backend);
            if (graph)
            {
                Tensor fresh = latent;
                latent = _transformer.PrepareGraphLatent(Backend, fresh);
                fresh.Dispose();
            }
            float[] timesteps = scheduler.Timesteps.ToArray();
            DelegateDenoisePredictor predictor = new(PredictionType.FlowVelocity, (x, sigma, stepIndex) =>
            {
                float t = stepIndex < steps && sigma == scheduler.SigmaAt(stepIndex) ? timesteps[stepIndex] : sigma * 1000f;
                return Predict(x, t, stepIndex, steps, cond, uncond, guided, cfg, guidance);
            });

            for (int i = plan.StartStep; i < steps; i++)
            {
                Stopwatch stepSw = Stopwatch.StartNew();
                if (i == plan.StartStep)
                    sampler.Reset(latent.Shape);
                sampler.Step(Backend, latent, predictor, i);
                if (masked)
                    BlendSource(latent, sourceTokens!, packedMask!, sampler.Sigmas[i + 1], seed + i + 1, patH, patW);
                stepSw.Stop();
                Logs.Debug($"{Family} step {i + 1}/{steps} done in {stepSw.ElapsedMilliseconds}ms");
                if (onProgress is not null)
                {
                    using Tensor? snapshot = graph ? _transformer.SnapshotGraphLatent(Backend) : null;
                    using Tensor preview = Flux2LatentOps.FromPackedTokens(snapshot ?? latent, patH, patW, _bnMean, _bnVar, _bnEps,
                        _config.VaeLatentChannels, _config.PatchSize);
                    onProgress(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                    {
                        Latent = preview,
                        LatentArch = LatentArchitecture.Flux2,
                    });
                }
            }
            if (graph)
                latent = _transformer.SnapshotGraphLatent(Backend);
        }
        catch
        {
            if (!graph) latent?.Dispose();
            ReclaimAfterFailure();
            throw;
        }
        finally
        {
            // A captured graph bakes this generation's conditioning addresses; never let it outlive them.
            _transformer.InvalidateStepGraph(Backend);
            cond?.Dispose();
            uncond?.Dispose();
            if (referenceTokens is not null)
            {
                foreach (ReferenceLatentTokens reference in referenceTokens) reference.Tokens.Dispose();
            }
            weightedCond?.Dispose();
            weightedUncond?.Dispose();
            sourceTokens?.Dispose();
            packedMask?.Dispose();
        }

        if (!KeepModelsResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        byte[] rgb;
        try
        {
            Tensor vaeLatent;
            using (Tensor finalLatent = latent)
            {
                vaeLatent = Flux2LatentOps.FromPackedTokens(finalLatent, patH, patW, _bnMean, _bnVar, _bnEps,
                    _config.VaeLatentChannels, _config.PatchSize);
            }
            Tensor image;
            using (vaeLatent)
            {
                image = _vaeDecoder.DecodeTiled(Backend, vaeLatent);
            }
            using (image)
            {
                if (masked && ((ImageToImageRequest)request).RecompositeAtEnd)
                    MaskBlendUtilities.BlendChannelsInPlace(image, ((ImageToImageRequest)request).SourceImage, plan.MaskPixel!);
                rgb = ImagePostProcessor.TensorToRgbBytes(image);
            }
        }
        catch
        {
            ReclaimAfterFailure();
            throw;
        }
        Backend.FreeActivations();
        Logs.Info($"{Family} generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
        return (rgb, imgW, imgH, seed);
    }

    /// <inheritdoc/>
    protected override void DisposeCore() => _promptCache.Dispose();

    /// <summary>One step's velocity pair. The step's branches (<see cref="MossImageGuidance.BranchesFor"/>) run together,
    /// as one captured graph when eligible. Plain CFG returns the pair for the sampler's fused combine; an SLG step
    /// combines here (<see cref="MossImageGuidance.CombineSlg"/>).</summary>
    private DenoisePrediction Predict(Tensor x, float timestep, int stepIndex, int steps, MossImageConditioning cond,
        MossImageConditioning? uncond, bool guided, float cfg, MossImageGuidance guidance)
    {
        List<MossImageBranch> branches = guidance.BranchesFor(cond, uncond, guided, stepIndex, steps);
        (Tensor[] velocities, bool callerOwns) = _transformer.ForwardBranches(Backend, x, timestep, branches);
        bool slg = velocities.Length > (guided ? 2 : 1);
        if (!slg)
        {
            return guided
                ? new DenoisePrediction(velocities[0], velocities[1], cfg, callerOwns)
                : new DenoisePrediction(velocities[0], velocities[0], 1f, callerOwns);
        }
        try
        {
            Tensor combined = guidance.CombineSlg(Backend, velocities[0], guided ? velocities[1] : null, velocities[^1], cfg);
            return new DenoisePrediction(combined, combined, 1f);
        }
        finally
        {
            if (callerOwns)
            {
                foreach (Tensor velocity in velocities) velocity.Dispose();
            }
        }
    }

    /// <summary>After a stage throws mid-op, the temporaries of whatever op was running are still in the backend's
    /// activation cache; reclaiming them here lets a retry start from the memory a fresh request would. Best effort.</summary>
    private void ReclaimAfterFailure()
    {
        try
        {
            Backend.FreeActivations();
        }
        catch (Exception cleanup)
        {
            Logs.Warning($"{Family}: reclaiming activations after a failed generation also failed: {cleanup.Message}");
        }
    }

    /// <summary>Encodes one prompt's final Qwen3.5 hidden states, reusing the last result for the same ids and images.
    /// On a shared backend the DiT is evicted first, since the 9B encoder and a resident DiT need not fit together.</summary>
    private Tensor EncodePrompt(WeightedTokenSequence prompt, int slot, IReadOnlyList<Tensor> images)
    {
        int discriminator = 0;
        if (images.Count > 0)
        {
            long signature = ImageSignature.CombineList(images);
            discriminator = unchecked((int)(signature ^ (signature >> 32))) | 1;
        }
        if (_promptCache.TryGet(slot, prompt.Tokens, discriminator, out Tensor? cached))
            return cached!;
        IBackend encoderBackend = TextEncoderBackend;
        if (_ditResident && ReferenceEquals(encoderBackend, Backend))
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        Tensor hidden;
        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            hidden = images.Count == 0
                ? _textEncoder.Encode(encoderBackend, prompt.Tokens)
                : ((IMultimodalFinalHiddenStateEncoder)_textEncoder).Encode(encoderBackend, prompt.Tokens, images);
        }
        finally
        {
            encoderBackend.FreeWeights(_textEncoder.EnumerateWeights());
            if (images.Count > 0)
                encoderBackend.FreeWeights(((IMultimodalFinalHiddenStateEncoder)_textEncoder).EnumerateVisionWeights());
            encoderBackend.FreeActivations();
        }
        Logs.Info($"{Family}: encoded {prompt.Tokens.Length} prompt tokens ({images.Count} image(s)) in {sw.ElapsedMilliseconds}ms");
        _promptCache.Store(slot, prompt.Tokens, discriminator, hidden);
        return hidden;
    }

    /// <summary>Refuses references the pipeline cannot use before anything is allocated.</summary>
    private void ValidateReferences(ReferenceImageInputs? references)
    {
        if (references is null)
            return;
        references.Validate(Family);
        if (references.Count > MossImageConfig.MaxReferenceImages)
            throw new ArgumentException($"{Family} conditions on at most {MossImageConfig.MaxReferenceImages} reference images, "
                + $"got {references.Count}.");
        if (references.Options.Mode == ReferenceAttentionMode.Inline && references.Count != 1)
            throw new ArgumentException($"{Family}'s colorize layout conditions on exactly one reference, got {references.Count}.");
        if (_vaeEncoder is null)
            throw new InvalidOperationException($"{Family} reference images need the FLUX.2 VAE encoder.");
        if (references.Vision.Count > 0 && _textEncoder is not IMultimodalFinalHiddenStateEncoder { HasVision: true })
            throw new InvalidOperationException($"{Family} reference images need a Qwen3.5 encoder with its vision tower.");
    }

    /// <summary>Checks the reference caches against the device budget and builds the first step's caches, so a failed
    /// prepass surfaces before the denoise loop starts.</summary>
    private void WarmReferenceCaches(ReferenceConditioningOptions options, MossImageConditioning cond, MossImageConditioning? uncond,
        bool guided, MossImageGuidance guidance, int startStep, int steps)
    {
        if (options.Mode != ReferenceAttentionMode.CachedKv)
            return;
        DType act = Backend.SupportsF16Activations ? DitDtype.Act : DType.F32;
        long perCache = ReferenceKvCache.EstimateBytes(_config.NumLayers, cond.References!.TokenCount, _config.HiddenSize, act);
        ReferenceCachePolicy.Check(Backend, options, perCache, guidance.DistinctBranchCaches(guided), Family);
        _transformer.EnsureReferenceCaches(Backend, guidance.BranchesFor(cond, uncond, guided, startStep, steps));
    }

    /// <summary>Initial packed latent: noise for text-to-image; for img2img the encoded source noised at
    /// <c>sigma[startStep]</c>, also returning the clean source tokens when a masked inpaint must keep blending.</summary>
    private (Tensor Latent, Tensor? Source) BuildInitialLatent(TextToImageRequest request, FlowMatchEulerDiscreteScheduler scheduler,
        int patH, int patW, int seed, int startStep, bool keepSource)
    {
        using Tensor noise = TakeOrCreateNoise(request, new TensorShape(1, _config.LatentChannels, patH, patW), seed);
        Tensor packedNoise = Flux2LatentOps.Pack(noise);
        if (request is not ImageToImageRequest img2img)
            return (packedNoise, null);
        using Tensor vaeLatent = _vaeEncoder!.Encode(Backend, img2img.SourceImage);
        Tensor source = Flux2LatentOps.ToPackedTokens(vaeLatent, _bnMean, _bnVar, _bnEps, _config.PatchSize);
        Tensor noised = new(packedNoise.Shape, DType.F32);
        scheduler.AddNoise(noised, source, packedNoise, startStep);
        packedNoise.Dispose();
        if (keepSource)
            return (noised, source);
        source.Dispose();
        return (noised, null);
    }

    /// <summary>Holds the unmasked region on the source's own trajectory after a sampler step: the source is re-noised at
    /// <paramref name="sigma"/> (clean after the last step) and blended in under the mask.</summary>
    private void BlendSource(Tensor latent, Tensor source, Tensor packedMask, float sigma, int noiseSeed, int patH, int patW)
    {
        Backend.OffloadActivation(latent);
        if (sigma <= 0f)
        {
            MaskBlendUtilities.BlendPackedInPlace(latent, source, packedMask);
            return;
        }
        using Tensor freshNoise = SeedGenerator.CreateNoise(new TensorShape(1, _config.LatentChannels, patH, patW), noiseSeed);
        using Tensor freshPacked = Flux2LatentOps.Pack(freshNoise);
        using Tensor noisedSource = new(latent.Shape, DType.F32);
        FlowMatchEulerDiscreteScheduler.AddNoiseAtSigma(noisedSource, source, freshPacked, sigma);
        MaskBlendUtilities.BlendPackedInPlace(latent, noisedSource, packedMask);
    }
}
