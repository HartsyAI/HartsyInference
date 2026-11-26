using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Pipelines;

/// <summary>FLUX.2 Klein reference-image editing, the way diffusers' <c>Flux2KleinPipeline</c> runs it with an input image:
/// the reference is VAE-encoded to packed tokens, appended to the target tokens, shares the timestep embedding and sits at
/// its own RoPE T coordinate (10, 20, ... for the references, 0 for the target). Guidance is true CFG against a negative
/// prompt (the empty prompt in diffusers), run as two batch-1 passes, and only the target tokens' velocity is stepped.
/// Open Vision Banana runs on this: its prompts are tasks (segmentation, depth, normals) and its answer is a painted image.
/// Without references it is plain Klein text-to-image with CFG, which the base (non-distilled) builds use.</summary>
public sealed unsafe class Flux2EditPipeline : DiffusionPipelineBase
{
    /// <summary>The T coordinate spacing between the target and each reference (diffusers' <c>scale</c>).</summary>
    public const float ReferenceTimeScale = 10f;

    private const string Family = "FLUX.2 reference edit";
    private const int CondSlot = 0;
    private const int UncondSlot = 1;

    private readonly Flux2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder _vaeEncoder;
    private readonly Tensor _bnMean;
    private readonly Tensor _bnVar;
    private readonly float _bnEps;
    private readonly Flux2Config _config;
    private readonly CachedPromptTaps _prompts;
    private readonly ResidentWeights _denoiser;

    /// <summary>Creates the pipeline. Every component is borrowed; the caller owns and disposes them.</summary>
    public Flux2EditPipeline(IBackend backend, LlamaStyleEncoder textEncoder, Flux2Transformer transformer, VaeDecoder vaeDecoder,
        VaeEncoder vaeEncoder, Tensor bnMean, Tensor bnVar, Flux2Config config, int[] textEncoderLayers, float bnEps = 1e-4f)
        : base(backend)
    {
        ArgumentNullException.ThrowIfNull(textEncoder);
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(vaeDecoder);
        ArgumentNullException.ThrowIfNull(vaeEncoder);
        ArgumentNullException.ThrowIfNull(bnMean);
        ArgumentNullException.ThrowIfNull(bnVar);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(textEncoderLayers);
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _bnMean = bnMean;
        _bnVar = bnVar;
        _bnEps = bnEps;
        _config = config;
        _prompts = new CachedPromptTaps(backend, textEncoder, textEncoderLayers, config.ContextInDim, Family);
        _denoiser = new ResidentWeights(backend, transformer.EnumerateWeights);
    }

    /// <summary>The T coordinate of every grid: the target at 0, then reference <c>i</c> at <c>10 + 10·i</c>.</summary>
    public static float[] GridTimes(int referenceCount)
    {
        float[] times = new float[referenceCount + 1];
        for (int i = 0; i < referenceCount; i++)
            times[i + 1] = ReferenceTimeScale + ReferenceTimeScale * i;
        return times;
    }

    /// <summary>Generates the target image. <paramref name="references"/> are <c>[1, 3, H, W]</c> RGB in [-1, 1] and must be
    /// the target's size (<paramref name="request"/> width and height, rounded down to the VAE's multiple); none runs
    /// text-to-image. A <paramref name="negative"/> prompt is required when guidance is active (<c>CfgScale &gt; 1</c>).
    /// Honours <see cref="TextToImageRequest.InitialNoise"/>, <see cref="TextToImageRequest.Seed"/>,
    /// <see cref="TextToImageRequest.Scheduler"/> and <see cref="TextToImageRequest.SigmaShift"/> (the schedule's mu).</summary>
    public (byte[] Rgb, int Width, int Height, int Seed) Edit(IrisZPrompt prompt, IrisZPrompt? negative,
        IReadOnlyList<Tensor> references, TextToImageRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(request);
        bool guided = CfgHelper.IsGuidanceActive(request.CfgScale);
        if (guided && negative is null)
            throw new ArgumentException($"{Family} needs a negative prompt when CfgScale is {request.CfgScale} (above 1).", nameof(negative));
        // Text first: the encoder cannot sit beside a resident transformer. The cache owns both tensors. A failure after the encode
        // reclaims in EditFromTokens; one inside the encode reclaims here.
        Tensor cond;
        Tensor? uncond = null;
        try
        {
            cond = _prompts.Encode(CondSlot, prompt.Tokens, prompt.Padding, _denoiser);
            if (guided)
                uncond = _prompts.Encode(UncondSlot, negative!.Tokens, negative.Padding, _denoiser);
        }
        catch
        {
            ReclaimAfterFailure(Family);
            throw;
        }
        return EditEncoded(cond, uncond, references, request, onProgress);
    }

    /// <summary><see cref="Edit"/> from already-encoded conditioning: <paramref name="cond"/> and
    /// <paramref name="uncond"/> are <c>[1, L, context]</c> and stay the caller's. <paramref name="uncond"/> is required
    /// when guidance is active.</summary>
    public (byte[] Rgb, int Width, int Height, int Seed) EditEncoded(Tensor cond, Tensor? uncond,
        IReadOnlyList<Tensor> references, TextToImageRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(references);
        ArgumentNullException.ThrowIfNull(request);
        int factor = _config.VaeDownscaleFactor;
        int patW = request.Width / factor, patH = request.Height / factor;
        if (patW < 1 || patH < 1)
            throw new ArgumentException($"{Family} needs at least {factor} pixels on a side; got {request.Width}x{request.Height}.");
        int width = patW * factor, height = patH * factor;
        foreach (Tensor reference in references)
        {
            ArgumentNullException.ThrowIfNull(reference, nameof(references));
            if (reference.Shape.Rank != 4 || reference.Shape[0] != 1 || reference.Shape[1] != 3
                || reference.Shape[2] != height || reference.Shape[3] != width)
            {
                throw new ArgumentException(
                    $"{Family} reference images must be [1, 3, {height}, {width}], the target's size; got {reference.Shape}.",
                    nameof(references));
            }
        }
        if (CfgHelper.IsGuidanceActive(request.CfgScale) && uncond is null)
        {
            throw new ArgumentException(
                $"{Family} needs unconditional conditioning when CfgScale is {request.CfgScale} (above 1).", nameof(uncond));
        }
        ReferenceLatentTokens[]? referenceTokens = null;
        try
        {
            if (references.Count > 0)
            {
                Backend.PreloadWeights(_vaeEncoder.EnumerateWeights());
                try
                {
                    referenceTokens = Flux2ReferenceEncoder.Encode(Backend, _vaeEncoder, references, _bnMean, _bnVar, _bnEps,
                        _config.PatchSize, factor, Family);
                }
                finally
                {
                    Backend.Sync();
                    Backend.FreeWeights(_vaeEncoder.EnumerateWeights());
                    Backend.FreeActivations();
                }
            }
            return EditFromTokens(cond, uncond, referenceTokens ?? [], request, onProgress);
        }
        finally
        {
            if (referenceTokens is not null)
            {
                foreach (ReferenceLatentTokens reference in referenceTokens) reference.Tokens.Dispose();
            }
        }
    }

    /// <summary><see cref="EditEncoded"/> from reference tokens that are already packed and BatchNorm-normalized
    /// (<see cref="Flux2ReferenceEncoder"/>'s output): each is <c>[1, H·W, in_channels]</c> on the target's patch grid. They
    /// stay the caller's.</summary>
    public (byte[] Rgb, int Width, int Height, int Seed) EditFromTokens(Tensor cond, Tensor? uncond,
        IReadOnlyList<ReferenceLatentTokens> referenceTokens, TextToImageRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(referenceTokens);
        ArgumentNullException.ThrowIfNull(request);
        int factor = _config.VaeDownscaleFactor;
        int patW = request.Width / factor, patH = request.Height / factor;
        if (patW < 1 || patH < 1)
            throw new ArgumentException($"{Family} needs at least {factor} pixels on a side; got {request.Width}x{request.Height}.");
        int width = patW * factor, height = patH * factor;
        foreach (ReferenceLatentTokens reference in referenceTokens)
        {
            reference.Validate(_config.InChannels, Family);
            if (reference.Height != patH || reference.Width != patW)
            {
                throw new ArgumentException(
                    $"{Family} reference tokens must cover the target's {patH}x{patW} patch grid; got {reference.Height}x{reference.Width}.",
                    nameof(referenceTokens));
            }
        }
        int steps = request.Steps;
        float cfg = request.CfgScale;
        bool guided = CfgHelper.IsGuidanceActive(cfg);
        if (guided && uncond is null)
            throw new ArgumentException($"{Family} needs unconditional conditioning when CfgScale is {cfg} (above 1).", nameof(uncond));
        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int gridTokens = patH * patW;
        Logs.Info($"{Family}: {width}x{height}, {referenceTokens.Count} reference(s), {steps} steps, cfg={cfg}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();

        Tensor? latent = null;
        try
        {
            DenoiserProbe.Condition("cond", cond);
            DenoiserProbe.Condition("uncond", uncond);

            TensorShape noiseShape = new TensorShape(1, _config.InChannels, patH, patW);
            using (Tensor noise = TakeOrCreateNoise(request, noiseShape, seed))
                latent = Flux2LatentOps.PackHost(noise);

            // An explicit shift is the schedule's mu, as for FLUX.2; otherwise diffusers' empirical mu on the TARGET tokens.
            float mu = request.SigmaShift ?? FlowMatchEulerDiscreteScheduler.EmpiricalMu(gridTokens, steps);
            FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(MathF.Exp(mu));
            scheduler.SetTimesteps(steps);
            ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, Family);

            float[] gridTimes = GridTimes(referenceTokens.Count);
            Tensor[] referenceParts = referenceTokens.Select(r => r.Tokens).ToArray();
            _denoiser.Preload();
            TwoPassCfgPredictor<Tensor> predictor = new TwoPassCfgPredictor<Tensor>(PredictionType.FlowVelocity,
                (x, sigma, _, text) => Predict(x, referenceParts, sigma, text, gridTimes, patH, patW), cond, guided ? uncond : null, cfg);
            for (int i = 0; i < steps; i++)
            {
                Stopwatch stepSw = Stopwatch.StartNew();
                if (i == 0)
                    sampler.Reset(latent.Shape);
                sampler.Step(Backend, latent, predictor, i);
                stepSw.Stop();
                Logs.Debug($"{Family} step {i + 1}/{steps} done in {stepSw.ElapsedMilliseconds}ms");
                if (onProgress is not null)
                {
                    using Tensor preview = Flux2LatentOps.FromPackedTokens(Backend, latent, patH, patW, _bnMean, _bnVar, _bnEps,
                        _config.VaeLatentChannels, _config.PatchSize);
                    onProgress(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                    {
                        Latent = preview,
                        LatentArch = LatentArchitecture.Flux2,
                    });
                }
            }
            _denoiser.Release();

            using Tensor vaeLatent = Flux2LatentOps.FromPackedTokens(Backend, latent, patH, patW, _bnMean, _bnVar, _bnEps,
                _config.VaeLatentChannels, _config.PatchSize);
            using Tensor image = _vaeDecoder.DecodeTiled(Backend, vaeLatent);
            byte[] rgb = ImagePostProcessor.TensorToRgbBytes(image);
            Logs.Info($"{Family} complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
            return (rgb, width, height, seed);
        }
        catch
        {
            ReclaimAfterFailure(Family);
            throw;
        }
        finally
        {
            latent?.Dispose();
        }
    }

    /// <inheritdoc/>
    protected override void DisposeCore() => _prompts.Dispose();

    /// <summary>The target tokens' velocity for one prompt: <c>[x, references...]</c> through the transformer, then the
    /// first grid's rows.</summary>
    private Tensor Predict(Tensor x, Tensor[] referenceParts, float sigma, Tensor text, float[] gridTimes, int patH, int patW)
    {
        int gridTokens = patH * patW;
        if (referenceParts.Length == 0)
            return _transformer.ForwardWithReferences(Backend, x, text, sigma, patH, patW, gridTimes);
        using Tensor tokens = new Tensor(new TensorShape(1, (long)gridTimes.Length * gridTokens, _config.InChannels), DType.F32);
        Tensor[] parts = new Tensor[referenceParts.Length + 1];
        parts[0] = x;
        Array.Copy(referenceParts, 0, parts, 1, referenceParts.Length);
        Backend.Concat(tokens, parts, 1);
        using Tensor velocity = _transformer.ForwardWithReferences(Backend, tokens, text, sigma, patH, patW, gridTimes);
        Tensor target = new Tensor(new TensorShape(1, gridTokens, velocity.Shape[2]), DType.F32);
        Backend.SliceRows(target, velocity, 0);
        return target;
    }
}
