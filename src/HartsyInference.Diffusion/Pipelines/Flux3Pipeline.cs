using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>FLUX.3 Klein text-to-image, image-to-image, inpaint and reference editing. The prompt is encoded by
/// Qwen3-VL-4B (eight hidden-state taps, padding masked and parked at RoPE position 1), reference images go through
/// the Flux.2 VAE into BN-normalized packed tokens that join both CFG branches, and <see cref="Flux3Transformer"/>
/// denoises under a static flow-match shift. The full build runs true CFG with the conditioning streams computed once;
/// the step-distilled flow-map build runs one pass per step and feeds it the next sigma of the family schedule.</summary>
public sealed unsafe class Flux3Pipeline : DiffusionPipelineBase
{
    private const string Family = "FLUX.3";

    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Flux3Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder? _vaeEncoder;
    private readonly Tensor _bnMean;
    private readonly Tensor _bnVar;
    private readonly float _bnEps;
    private readonly Flux3Config _config;
    private readonly PromptCache _promptCache = new PromptCache();
    private bool _ditResident;

    /// <summary>Creates the pipeline; <paramref name="vaeEncoder"/> is required for img2img, inpaint and references.</summary>
    public Flux3Pipeline(IBackend backend, LlamaStyleEncoder textEncoder, Flux3Transformer transformer, VaeDecoder vaeDecoder,
        VaeEncoder? vaeEncoder, Tensor bnMean, Tensor bnVar, float bnEps = 1e-5f)
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
    }

    /// <summary>The official defaults of the loaded build.</summary>
    public GenerationDefaults Defaults => _config.FlowMap ? GenerationDefaults.Flux3Distilled : GenerationDefaults.Flux3;

    /// <summary>Keeps the DiT resident across generations unless the VRAM profile says otherwise.</summary>
    private bool KeepModelsResident => VramLevers.KeepResident(Backend);

    /// <summary>Generates an image. <paramref name="negative"/> is encoded only when the full build runs guided
    /// (<c>cfg &gt; 1</c>). <paramref name="referenceImages"/> are <c>[1, 3, H, W]</c> RGB in [-1, 1] with sides that are
    /// multiples of 16, in <c>&lt;ref_image_N&gt;</c> order.</summary>
    public (byte[] rgbData, int width, int height, int seed) Generate(Flux3PromptTokens prompt, Flux3PromptTokens? negative,
        TextToImageRequest request, IReadOnlyList<Tensor>? referenceImages = null, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(request);
        using IDisposable seamlessScope = BeginSeamlessTiling(request.SeamlessTiling);
        bool hasReferences = referenceImages is { Count: > 0 };
        if ((request is ImageToImageRequest || hasReferences) && _vaeEncoder is null)
            throw new InvalidOperationException("FLUX.3 image-to-image and reference editing need the Flux.2 VAE encoder.");
        if (hasReferences && referenceImages!.Count > _config.MaxReferenceImages)
            throw new ArgumentException($"FLUX.3 takes at most {_config.MaxReferenceImages} reference images.", nameof(referenceImages));

        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        (int steps, float cfg, int reqWidth, int reqHeight) = Defaults.Resolve(request);
        int imgH = reqHeight / _config.VaeDownscaleFactor * _config.VaeDownscaleFactor;
        int imgW = reqWidth / _config.VaeDownscaleFactor * _config.VaeDownscaleFactor;
        int patH = imgH / _config.VaeDownscaleFactor;
        int patW = imgW / _config.VaeDownscaleFactor;
        Img2ImgSetup.Plan plan = Img2ImgSetup.Prepare(request, imgH, imgW, steps);
        if (plan.PassThrough)
            return (ImagePostProcessor.TensorToRgbBytes(((ImageToImageRequest)request).SourceImage), imgW, imgH, seed);
        bool guided = !_config.FlowMap && CfgHelper.IsGuidanceActive(cfg);
        if (guided && negative is null)
            throw new ArgumentException("Guided FLUX.3 sampling needs the negative prompt's tokens.", nameof(negative));
        // Everything that can refuse the request resolves before anything is allocated.
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(
            MathF.Exp(request.SigmaShift ?? Flux3Config.ReferenceTimeShiftMu));
        scheduler.SetTimesteps(steps);
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, Family, startsFromNoisedInit: plan.StartStep > 0);
        Logs.Info($"{Family} Klein{(_config.FlowMap ? " distilled" : "")}: {imgW}x{imgH}, {steps} steps, "
            + $"cfg={(guided ? cfg : 1f)}, refs={referenceImages?.Count ?? 0}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        bool masked = plan.MaskPixel is not null;
        Tensor? latent = null;
        Tensor? sourceTokens = null;
        Tensor? packedMask = null;
        Tensor? referenceTokens = null;
        Tensor? weightedCond = null;
        Tensor? weightedUncond = null;
        Flux3ConditionStates? condStates = null;
        Flux3ConditionStates? uncondStates = null;
        try
        {
            Tensor condText = EncodePrompt(prompt, PromptCache.Cond);
            Tensor? uncondText = guided ? EncodePrompt(negative!, PromptCache.Uncond) : null;

            // VAE-encoder phase: references and the init image under one preload, released before the DiT loads. Every
            // output is read back by the host-side packing ops, so the activation reclaim drops nothing still in use.
            bool encodes = hasReferences || request is ImageToImageRequest;
            List<(int Height, int Width)> referenceShapes;
            if (encodes)
                Backend.PreloadWeights(_vaeEncoder!.EnumerateWeights());
            try
            {
                (referenceTokens, referenceShapes) = EncodeReferences(referenceImages);
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

            weightedCond = prompt.IsWeighted ? CondTokenWeights.Apply(Backend, condText, null, prompt.Tokens).Cond : null;
            weightedUncond = guided && negative!.IsWeighted ? CondTokenWeights.Apply(Backend, uncondText!, null, negative.Tokens).Cond : null;
            Flux3ConditionInput condInput = new Flux3ConditionInput(weightedCond ?? condText, referenceTokens, referenceShapes);
            Flux3ConditionInput? uncondInput = guided
                ? new Flux3ConditionInput(weightedUncond ?? uncondText!, referenceTokens, referenceShapes) : null;
            if (masked)
            {
                int latH = imgH / 8, latW = imgW / 8;
                using Tensor latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(plan.MaskPixel!, latH, latW);
                packedMask = MaskBlendUtilities.PackLatentMask2x2(latentMask, latH, latW);
            }

            Backend.PreloadWeights(_transformer.EnumerateWeights());
            _ditResident = true;
            condStates = _config.FlowMap ? null : _transformer.PrecomputeConditioning(Backend, condInput);
            uncondStates = _config.FlowMap || !guided ? null : _transformer.PrecomputeConditioning(Backend, uncondInput!);
            float[] timestepTable = scheduler.Timesteps.ToArray();
            DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(PredictionType.FlowVelocity, (x, sigma, stepIndex) =>
            {
                float t = stepIndex < steps && sigma == scheduler.SigmaAt(stepIndex) ? timestepTable[stepIndex] / 1000f : sigma;
                float? target = _config.FlowMap ? NextScheduleSigma(sampler.Sigmas, sigma) : null;
                Tensor cond = _transformer.Forward(Backend, x, patH, patW, t, condInput, condStates, target);
                if (!guided)
                    return new DenoisePrediction(cond, cond);
                Tensor uncond;
                try
                {
                    uncond = _transformer.Forward(Backend, x, patH, patW, t, uncondInput!, uncondStates, target);
                }
                catch
                {
                    cond.Dispose();
                    throw;
                }
                return new DenoisePrediction(cond, uncond, cfg);
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
                    using Tensor preview = Flux2LatentOps.FromPackedTokens(latent, patH, patW, _bnMean, _bnVar, _bnEps,
                        _config.VaeLatentChannels, _config.PatchSize);
                    onProgress(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                    {
                        Latent = preview,
                        LatentArch = LatentArchitecture.Flux2,
                    });
                }
            }
        }
        catch
        {
            latent?.Dispose();
            ReclaimAfterFailure();
            throw;
        }
        finally
        {
            condStates?.Dispose();
            uncondStates?.Dispose();
            weightedCond?.Dispose();
            weightedUncond?.Dispose();
            referenceTokens?.Dispose();
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

    /// <summary>After a stage throws mid-op (an OOM, a cancellation, a device error), the temporaries of whatever op was
    /// running are still in the backend's activation cache. Reclaiming them here covers every block and stage at once,
    /// so a retry starts from the memory a fresh request would. Best effort: a sticky device error must not mask the
    /// failure being reported.</summary>
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

    /// <summary>Encodes one prompt through the eight Qwen3-VL taps, reusing the last result for the same ids. The DiT
    /// is evicted first because the encoder and a resident DiT need not fit together.</summary>
    private Tensor EncodePrompt(Flux3PromptTokens prompt, int slot)
    {
        if (_promptCache.TryGet(slot, prompt, out Tensor? cached))
            return cached!;
        if (_ditResident)
        {
            Backend.FreeWeights(_transformer.EnumerateWeights());
            _ditResident = false;
        }
        Tensor hidden;
        try
        {
            hidden = _textEncoder.EncodeMultiLayer(Backend, [prompt.Tokens.Tokens], _config.TextEncoderLayers,
                interleavedLayout: false, TextEncoderPadding.Transformers(prompt.ValidLength));
        }
        finally
        {
            Backend.FreeWeights(_textEncoder.EnumerateWeights());
        }
        _ = hidden.DataPointer;
        Backend.FreeActivations();
        if (hidden.Shape[2] != _config.ContextInDim)
            throw new HartsyInferenceException($"FLUX.3 text features are {hidden.Shape[2]} wide; the transformer expects {_config.ContextInDim}.");
        _promptCache.Store(slot, prompt, hidden);
        return hidden;
    }

    /// <summary>VAE-encodes each reference and concatenates their packed tokens in order.</summary>
    private (Tensor? Tokens, List<(int Height, int Width)> Shapes) EncodeReferences(IReadOnlyList<Tensor>? images)
    {
        List<(int Height, int Width)> shapes = [];
        if (images is not { Count: > 0 })
            return (null, shapes);
        List<Tensor> parts = new List<Tensor>(images.Count);
        try
        {
            foreach (Tensor image in images)
            {
                int h = (int)image.Shape[2], w = (int)image.Shape[3];
                if (h % _config.VaeDownscaleFactor != 0 || w % _config.VaeDownscaleFactor != 0)
                {
                    throw new ArgumentException(
                        $"FLUX.3 reference images need sides that are multiples of {_config.VaeDownscaleFactor}; got {w}x{h}.");
                }
                using Tensor vaeLatent = _vaeEncoder!.Encode(Backend, image);
                parts.Add(Flux2LatentOps.ToPackedTokens(vaeLatent, _bnMean, _bnVar, _bnEps, _config.PatchSize));
                shapes.Add((h / _config.VaeDownscaleFactor, w / _config.VaeDownscaleFactor));
            }
            if (parts.Count == 1)
            {
                Tensor only = parts[0];
                parts.Clear();
                return (only, shapes);
            }
            long total = 0;
            foreach (Tensor part in parts) total += part.Shape[1];
            Tensor joined = new Tensor(new TensorShape(1, total, _config.InChannels), DType.F32);
            Backend.Concat(joined, parts.ToArray(), 1);
            _ = joined.DataPointer;
            return (joined, shapes);
        }
        finally
        {
            foreach (Tensor part in parts) part.Dispose();
        }
    }

    /// <summary>Initial packed latent: noise for text-to-image; for img2img the encoded source noised at
    /// <c>sigma[startStep]</c>, also returning the clean source tokens when a masked inpaint must keep blending.</summary>
    private (Tensor Latent, Tensor? Source) BuildInitialLatent(TextToImageRequest request, FlowMatchEulerDiscreteScheduler scheduler,
        int patH, int patW, int seed, int startStep, bool keepSource)
    {
        using Tensor noise = TakeOrCreateNoise(request, new TensorShape(1, _config.InChannels, patH, patW), seed);
        Tensor packedNoise = Flux2LatentOps.Pack(noise);
        if (request is not ImageToImageRequest img2img)
            return (packedNoise, null);
        using Tensor vaeLatent = _vaeEncoder!.Encode(Backend, img2img.SourceImage);
        Tensor source = Flux2LatentOps.ToPackedTokens(vaeLatent, _bnMean, _bnVar, _bnEps, _config.PatchSize);
        Tensor noised = new Tensor(packedNoise.Shape, DType.F32);
        scheduler.AddNoise(noised, source, packedNoise, startStep);
        packedNoise.Dispose();
        if (keepSource)
            return (noised, source);
        source.Dispose();
        return (noised, null);
    }

    /// <summary>Holds the unmasked region on the source's own trajectory after a sampler step: the source is re-noised
    /// at <paramref name="sigma"/>, the level the sampler just reached (clean after the last step), and blended in under
    /// the mask. The latent comes to the host first, so the next forward uploads the blended values rather than the
    /// step's device copy.</summary>
    private void BlendSource(Tensor latent, Tensor source, Tensor packedMask, float sigma, int noiseSeed, int patH, int patW)
    {
        Backend.OffloadActivation(latent);
        if (sigma <= 0f)
        {
            MaskBlendUtilities.BlendPackedInPlace(latent, source, packedMask);
            return;
        }
        using Tensor freshNoise = SeedGenerator.CreateNoise(new TensorShape(1, _config.InChannels, patH, patW), noiseSeed);
        using Tensor freshPacked = Flux2LatentOps.Pack(freshNoise);
        using Tensor noisedSource = new Tensor(latent.Shape, DType.F32);
        FlowMatchEulerDiscreteScheduler.AddNoiseAtSigma(noisedSource, source, freshPacked, sigma);
        MaskBlendUtilities.BlendPackedInPlace(latent, noisedSource, packedMask);
    }

    /// <summary>The flow map's jump target: the sigma after the schedule entry nearest <paramref name="sigma"/>. This is
    /// ComfyUI's rule over <c>sample_sigmas</c>, the schedule the sampler actually runs, so a sub-step evaluation
    /// (Heun's corrector, a midpoint) and a re-spaced schedule both get the target ComfyUI gives them.</summary>
    public static float NextScheduleSigma(IReadOnlyList<float> schedule, float sigma)
    {
        ArgumentNullException.ThrowIfNull(schedule);
        int nearest = 0;
        float best = float.MaxValue;
        for (int i = 0; i < schedule.Count; i++)
        {
            float distance = MathF.Abs(schedule[i] - sigma);
            if (distance < best)
            {
                best = distance;
                nearest = i;
            }
        }
        return schedule[Math.Min(nearest + 1, schedule.Count - 1)];
    }

    /// <summary>The last encoded conditional and unconditional prompts, keyed by their ids and how many of them are
    /// real: the padding mask is part of the encoding. Weights are applied after encoding, so they are not.</summary>
    private sealed class PromptCache : IDisposable
    {
        public const int Cond = 0;
        public const int Uncond = 1;
        private readonly int[]?[] _keys = new int[]?[2];
        private readonly int[] _validLengths = new int[2];
        private readonly Tensor?[] _values = new Tensor?[2];

        public bool TryGet(int slot, Flux3PromptTokens prompt, out Tensor? value)
        {
            value = _keys[slot] is int[] cached && _validLengths[slot] == prompt.ValidLength
                && cached.AsSpan().SequenceEqual(prompt.Tokens.Tokens) ? _values[slot] : null;
            return value is not null;
        }

        public void Store(int slot, Flux3PromptTokens prompt, Tensor value)
        {
            _values[slot]?.Dispose();
            _keys[slot] = (int[])prompt.Tokens.Tokens.Clone();
            _validLengths[slot] = prompt.ValidLength;
            _values[slot] = value;
        }

        public void Dispose()
        {
            for (int i = 0; i < _values.Length; i++)
            {
                _values[i]?.Dispose();
                _values[i] = null;
                _keys[i] = null;
            }
        }
    }
}
