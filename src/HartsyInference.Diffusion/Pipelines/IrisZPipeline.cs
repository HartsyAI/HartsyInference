using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory.Vram;
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

/// <summary>Iris-Z (speridlabs/iris-z): a FLUX.2 fine-tune that denoises an RGB grid and a depth grid on the same patch
/// positions. Each task fixes one grid or neither: prompt to image (the depth grid stays noise), prompt to image and
/// depth (both denoised), image to depth (the photo is given), and depth to image (the depth map is given). Depth goes
/// through the FLUX.2 VAE as an image, see <see cref="IrisZDepth"/>. Guidance is true CFG against the empty prompt,
/// run as two batch-1 passes.</summary>
public sealed unsafe class IrisZPipeline : DiffusionPipelineBase
{
    /// <summary>Sampling steps the reference estimates depth with.</summary>
    public const int DepthSteps = 4;

    /// <summary>How many final clean-depth estimates the reference averages.</summary>
    public const int DepthAverageLast = 2;

    private const string Family = "Iris-Z";
    private const int RgbModality = 0;
    private const int DepthModality = 1;
    private const int GivenRgbModality = 2;
    private const int GivenDepthModality = 3;
    private const float DepthTimeCoordinate = 1f;
    private const int DepthNoiseSalt = 0x3D0F1A2B;
    private const int CondSlot = 0;
    private const int UncondSlot = 1;

    private readonly LlamaStyleEncoder _textEncoder;
    private readonly Flux2Transformer _transformer;
    private readonly VaeDecoder _vaeDecoder;
    private readonly VaeEncoder _vaeEncoder;
    private readonly Tensor _bnMean;
    private readonly Tensor _bnVar;
    private readonly float _bnEps;
    private readonly Flux2Config _config;
    private readonly Tensor _emptyPrompt;
    private readonly float _skyCeiling;
    private readonly int[] _textEncoderLayers;
    private readonly EncodedPromptCache _promptCache = new(2);
    private bool _ditResident;

    /// <summary>Creates the pipeline. <paramref name="emptyPrompt"/> is the release's <c>empty_prompt</c> tensor as F32
    /// <c>[1, L, context]</c>, borrowed; it conditions every run without a prompt, so those never load the text encoder.</summary>
    public IrisZPipeline(IBackend backend, LlamaStyleEncoder textEncoder, Flux2Transformer transformer, VaeDecoder vaeDecoder,
        VaeEncoder vaeEncoder, Tensor bnMean, Tensor bnVar, Flux2Config config, Tensor emptyPrompt, float skyCeiling,
        int[] textEncoderLayers, float bnEps = 1e-4f)
        : base(backend)
    {
        ArgumentNullException.ThrowIfNull(textEncoder);
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(vaeDecoder);
        ArgumentNullException.ThrowIfNull(vaeEncoder);
        ArgumentNullException.ThrowIfNull(bnMean);
        ArgumentNullException.ThrowIfNull(bnVar);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(emptyPrompt);
        ArgumentNullException.ThrowIfNull(textEncoderLayers);
        if (config.ModalityCount <= GivenDepthModality)
            throw new ArgumentException($"Iris-Z needs four modality rows; the configuration has {config.ModalityCount}.", nameof(config));
        if (emptyPrompt.DType != DType.F32 || emptyPrompt.Shape.Rank != 3 || emptyPrompt.Shape[0] != 1
            || emptyPrompt.Shape[2] != config.ContextInDim)
        {
            throw new ArgumentException(
                $"The empty prompt must be F32 [1, L, {config.ContextInDim}], got {emptyPrompt.DType} {emptyPrompt.Shape}.", nameof(emptyPrompt));
        }
        _textEncoder = textEncoder;
        _transformer = transformer;
        _vaeDecoder = vaeDecoder;
        _vaeEncoder = vaeEncoder;
        _bnMean = bnMean;
        _bnVar = bnVar;
        _bnEps = bnEps;
        _config = config;
        _emptyPrompt = emptyPrompt;
        _skyCeiling = skyCeiling;
        _textEncoderLayers = textEncoderLayers;
    }

    private bool KeepModelsResident => VramLevers.KeepResident(Backend);

    /// <summary>Prompt to image, optionally with its depth. A null <paramref name="prompt"/> is the empty prompt and runs
    /// unguided. <paramref name="emptyPrompt"/> is the tokenized empty prompt for the unconditional branch; null uses
    /// the release's tensor.</summary>
    public IrisZImage TextToImage(IrisZPrompt? prompt, IrisZPrompt? emptyPrompt, TextToImageRequest request,
        IrisZDepthOutput depth, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);
        (int width, int height, int patW, int patH) = Geometry(request.Width, request.Height);
        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int gridTokens = patH * patW;
        bool joint = depth != IrisZDepthOutput.None;
        Logs.Info($"{Family} prompt to image{(joint ? " + depth" : "")}: {width}x{height}, {request.Steps} steps, cfg={request.CfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();
        TensorShape noiseShape = new TensorShape(1, _config.InChannels, patH, patW);
        Tensor? latent = null;
        Tensor? depthNoise = null;
        try
        {
            latent = PackedNoise(TakeOrCreateNoise(request, noiseShape, seed));
            Tensor packedDepthNoise = PackedNoise(SeedGenerator.CreateNoise(noiseShape, seed ^ DepthNoiseSalt));
            depthNoise = packedDepthNoise;
            if (joint)
            {
                Tensor both = new Tensor(new TensorShape(1, 2L * gridTokens, _config.InChannels), DType.F32);
                using (Tensor rgbNoise = latent)
                {
                    latent = both;
                    Backend.Concat(both, [rgbNoise, packedDepthNoise], 1);
                }
                packedDepthNoise.Dispose();
                depthNoise = null;
            }
            SampleRgb(latent, depthNoise, 1f, DepthModality, prompt, emptyPrompt, request, seed, patH, patW, onProgress);

            byte[] rgb;
            float[]? logDepth = null;
            if (joint)
            {
                using Tensor rgbTokens = SliceGrid(latent, 0, gridTokens);
                rgb = DecodeRgb(rgbTokens, patH, patW);
                if (depth == IrisZDepthOutput.Joint)
                {
                    using Tensor depthTokens = SliceGrid(latent, 1, gridTokens);
                    logDepth = DecodeLogDepth(depthTokens, patH, patW);
                }
            }
            else
            {
                rgb = DecodeRgb(latent, patH, patW);
            }
            latent.Dispose();
            latent = null;
            Backend.FreeActivations();
            if (depth == IrisZDepthOutput.Estimated)
            {
                // The reference estimates from the saved 8-bit image, so the estimate sees the same quantized pixels.
                using Tensor image = ImagePostProcessor.RgbBytesToTensor(rgb, width, height);
                logDepth = ImageToDepth(image, DepthSteps, DepthAverageLast, [seed]);
            }
            Logs.Info($"{Family} generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
            return new IrisZImage(rgb, logDepth, width, height, seed);
        }
        catch
        {
            ReclaimAfterFailure();
            throw;
        }
        finally
        {
            latent?.Dispose();
            depthNoise?.Dispose();
        }
    }

    /// <summary>Depth to image. <paramref name="encodedDepth"/> is the depth map as the model reads it
    /// (<see cref="IrisZDepth.EncodeLogDepth"/>), row-major at the request's width and height.</summary>
    public IrisZImage DepthToImage(ReadOnlySpan<float> encodedDepth, IrisZPrompt? prompt, IrisZPrompt? emptyPrompt,
        TextToImageRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);
        (int width, int height, int patW, int patH) = Geometry(request.Width, request.Height);
        if (width != request.Width || height != request.Height)
            throw new ArgumentException($"A depth map needs sides that are multiples of {_config.VaeDownscaleFactor}; got {request.Width}x{request.Height}.");
        if (encodedDepth.Length != width * height)
            throw new ArgumentException($"A {encodedDepth.Length}-pixel depth map is not {width}x{height}.", nameof(encodedDepth));
        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        Logs.Info($"{Family} depth to image: {width}x{height}, {request.Steps} steps, cfg={request.CfgScale}, seed={seed}");
        Stopwatch sw = Stopwatch.StartNew();
        Tensor? latent = null;
        Tensor? depthTokens = null;
        try
        {
            using (Tensor depthImage = new Tensor(new TensorShape(1, 3, height, width), DType.F32))
            {
                Span<float> pixels = depthImage.AsSpan<float>();
                for (int channel = 0; channel < 3; channel++) encodedDepth.CopyTo(pixels.Slice(channel * encodedDepth.Length));
                depthTokens = EncodeImage(depthImage);
            }
            latent = PackedNoise(TakeOrCreateNoise(request, new TensorShape(1, _config.InChannels, patH, patW), seed));
            SampleRgb(latent, depthTokens, 0f, GivenDepthModality, prompt, emptyPrompt, request, seed, patH, patW, onProgress);
            byte[] rgb = DecodeRgb(latent, patH, patW);
            Backend.FreeActivations();
            Logs.Info($"{Family} generation complete in {sw.ElapsedMilliseconds}ms (seed={seed})");
            return new IrisZImage(rgb, null, width, height, seed);
        }
        catch
        {
            ReclaimAfterFailure();
            throw;
        }
        finally
        {
            latent?.Dispose();
            depthTokens?.Dispose();
        }
    }

    /// <summary>Image to depth: relative log depth per pixel of <paramref name="image"/> (<c>[1, 3, H, W]</c> RGB in
    /// [-1, 1], sides multiples of 16). The last <paramref name="averageLast"/> clean-depth estimates of each seed are
    /// aligned and their median kept (0 keeps the final sample); several <paramref name="seeds"/> are combined the same way.</summary>
    public float[] ImageToDepth(Tensor image, int steps, int averageLast, IReadOnlyList<int> seeds,
        Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(seeds);
        if (image.Shape.Rank != 4 || image.Shape[0] != 1 || image.Shape[1] != 3)
            throw new ArgumentException($"The image must be [1, 3, H, W], got {image.Shape}.", nameof(image));
        int height = (int)image.Shape[2], width = (int)image.Shape[3];
        (int alignedWidth, int alignedHeight, int patW, int patH) = Geometry(width, height);
        if (alignedWidth != width || alignedHeight != height)
            throw new ArgumentException($"The image needs sides that are multiples of {_config.VaeDownscaleFactor}; got {width}x{height}.", nameof(image));
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), steps, "At least one step is required.");
        if (seeds.Count == 0)
            throw new ArgumentException("At least one seed is required.", nameof(seeds));
        averageLast = Math.Clamp(averageLast, 0, steps);
        Logs.Info($"{Family} image to depth: {width}x{height}, {steps} steps, {seeds.Count} seed(s)");
        Stopwatch sw = Stopwatch.StartNew();
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(
            MathF.Exp(FlowMatchEulerDiscreteScheduler.EmpiricalMu(patH * patW, steps)));
        scheduler.SetTimesteps(steps);
        float[] sigmas = scheduler.Sigmas();
        TensorShape noiseShape = new TensorShape(1, _config.InChannels, patH, patW);
        List<List<Tensor>> estimates = new List<List<Tensor>>(seeds.Count);
        Tensor? rgbTokens = null;
        try
        {
            rgbTokens = EncodeImage(image);
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            _ditResident = true;
            int done = 0;
            foreach (int seed in seeds)
            {
                List<Tensor> seedEstimates = new List<Tensor>(Math.Max(averageLast, 1));
                estimates.Add(seedEstimates);
                Tensor? depth = PackedNoise(SeedGenerator.CreateNoise(noiseShape, seed));
                try
                {
                    for (int step = 0; step < steps; step++)
                    {
                        Stopwatch stepSw = Stopwatch.StartNew();
                        float sigma = sigmas[step];
                        using Tensor velocity = PredictDepth(rgbTokens, depth, sigma, patH, patW);
                        if (averageLast > 0 && step >= steps - averageLast)
                        {
                            Tensor estimate = new Tensor(depth.Shape, DType.F32);
                            seedEstimates.Add(estimate);
                            Backend.CopyInto(estimate, depth);
                            Backend.GuidedEulerStep(estimate, velocity, velocity, GuidedEulerSpec.Lerp(1f, -sigma));
                        }
                        Backend.GuidedEulerStep(depth, velocity, velocity, GuidedEulerSpec.Lerp(1f, sigmas[step + 1] - sigma));
                        onProgress?.Invoke(new GenerationProgress(++done, steps * seeds.Count, stepSw.Elapsed.TotalMilliseconds));
                    }
                    if (averageLast == 0)
                    {
                        seedEstimates.Add(depth);
                        depth = null;
                    }
                }
                finally
                {
                    depth?.Dispose();
                }
            }
            ReleaseTransformer();

            List<float[]> members = new List<float[]>(seeds.Count);
            foreach (List<Tensor> seedEstimates in estimates)
            {
                List<float[]> decoded = new List<float[]>(seedEstimates.Count);
                foreach (Tensor estimate in seedEstimates) decoded.Add(DecodeLogDepth(estimate, patH, patW));
                members.Add(IrisZDepth.Ensemble(decoded, decoded.Count - 1));
            }
            Backend.FreeActivations();
            Logs.Info($"{Family} depth complete in {sw.ElapsedMilliseconds}ms");
            return IrisZDepth.Ensemble(members, 0);
        }
        catch
        {
            ReclaimAfterFailure();
            throw;
        }
        finally
        {
            rgbTokens?.Dispose();
            foreach (List<Tensor> seedEstimates in estimates)
                foreach (Tensor estimate in seedEstimates) estimate.Dispose();
        }
    }

    /// <inheritdoc/>
    protected override void DisposeCore() => _promptCache.Dispose();

    /// <summary>Denoises <paramref name="latent"/> in place: the RGB grid alone when <paramref name="fixedDepth"/> holds
    /// the depth grid's tokens (at <paramref name="fixedDepthSigma"/>, never stepped), or both grids when it is null.</summary>
    private void SampleRgb(Tensor latent, Tensor? fixedDepth, float fixedDepthSigma, int depthModality, IrisZPrompt? prompt,
        IrisZPrompt? emptyPrompt, TextToImageRequest request, int seed, int patH, int patW, Action<GenerationProgress>? onProgress)
    {
        int steps = request.Steps;
        float cfg = request.CfgScale;
        int gridTokens = patH * patW;
        // An explicit shift is the schedule's mu, as for FLUX.2.
        float mu = request.SigmaShift ?? FlowMatchEulerDiscreteScheduler.EmpiricalMu(gridTokens, steps);
        FlowMatchEulerDiscreteScheduler scheduler = new FlowMatchEulerDiscreteScheduler(MathF.Exp(mu));
        scheduler.SetTimesteps(steps);
        ISampler sampler = FlowMatchSampling.Resolve(request.Scheduler, scheduler, seed, Family);

        bool guided = prompt is not null && CfgHelper.IsGuidanceActive(cfg);
        Tensor cond = prompt is null ? _emptyPrompt : EncodePrompt(prompt, CondSlot);
        Tensor? uncond = !guided ? null : emptyPrompt is null ? _emptyPrompt : EncodePrompt(emptyPrompt, UncondSlot);
        Tensor? weighted = null;
        try
        {
            if (prompt?.Weights is { } weights)
            {
                weighted = CondTokenWeights.Apply(Backend, cond, null, weights).Cond
                    ?? throw new HartsyInferenceException("Weighted conditioning returned no tensor.");
            }
            Tensor condText = weighted ?? cond;
            DenoiserProbe.Condition("cond", condText);
            DenoiserProbe.Condition("uncond", uncond);
            Backend.PreloadWeights(_transformer.EnumerateWeights());
            _ditResident = true;
            DelegateDenoisePredictor predictor = new DelegateDenoisePredictor(PredictionType.FlowVelocity, (x, sigma, _) =>
            {
                Tensor condVelocity = PredictRgb(x, fixedDepth, sigma, fixedDepthSigma, depthModality, condText, patH, patW);
                if (!guided)
                    return new DenoisePrediction(condVelocity, condVelocity);
                Tensor uncondVelocity;
                try
                {
                    uncondVelocity = PredictRgb(x, fixedDepth, sigma, fixedDepthSigma, depthModality, uncond!, patH, patW);
                }
                catch
                {
                    condVelocity.Dispose();
                    throw;
                }
                return new DenoisePrediction(condVelocity, uncondVelocity, cfg);
            });
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
                    using Tensor? rgbSlice = fixedDepth is null ? SliceGrid(latent, 0, gridTokens) : null;
                    using Tensor preview = Flux2LatentOps.FromPackedTokens(rgbSlice ?? latent, patH, patW, _bnMean, _bnVar, _bnEps,
                        _config.VaeLatentChannels, _config.PatchSize);
                    onProgress(new GenerationProgress(i + 1, steps, stepSw.Elapsed.TotalMilliseconds)
                    {
                        Latent = preview,
                        LatentArch = LatentArchitecture.Flux2,
                    });
                }
            }
        }
        finally
        {
            weighted?.Dispose();
        }
        ReleaseTransformer();
    }

    /// <summary>The RGB grid's velocity, or both grids' when the depth grid is being denoised too.</summary>
    private Tensor PredictRgb(Tensor x, Tensor? fixedDepth, float sigma, float fixedDepthSigma, int depthModality, Tensor text,
        int patH, int patW)
    {
        int gridTokens = patH * patW;
        Span<Flux2Grid> grids =
        [
            new Flux2Grid(sigma, RgbModality, 0f),
            new Flux2Grid(fixedDepth is null ? sigma : fixedDepthSigma, depthModality, DepthTimeCoordinate),
        ];
        if (fixedDepth is null)
            return _transformer.ForwardGrids(Backend, x, text, sigma, grids, patH, patW);
        using Tensor tokens = new Tensor(new TensorShape(1, 2L * gridTokens, _config.InChannels), DType.F32);
        Backend.Concat(tokens, [x, fixedDepth], 1);
        using Tensor velocity = _transformer.ForwardGrids(Backend, tokens, text, sigma, grids, patH, patW);
        return SliceGrid(velocity, 0, gridTokens);
    }

    /// <summary>The depth grid's velocity given the clean photo's tokens, under the empty prompt.</summary>
    private Tensor PredictDepth(Tensor rgbTokens, Tensor depth, float sigma, int patH, int patW)
    {
        int gridTokens = patH * patW;
        Span<Flux2Grid> grids =
        [
            new Flux2Grid(0f, GivenRgbModality, 0f),
            new Flux2Grid(sigma, DepthModality, DepthTimeCoordinate),
        ];
        using Tensor tokens = new Tensor(new TensorShape(1, 2L * gridTokens, _config.InChannels), DType.F32);
        Backend.Concat(tokens, [rgbTokens, depth], 1);
        using Tensor velocity = _transformer.ForwardGrids(Backend, tokens, _emptyPrompt, sigma, grids, patH, patW);
        return SliceGrid(velocity, 1, gridTokens);
    }

    private Tensor SliceGrid(Tensor tokens, int grid, int gridTokens)
    {
        Tensor slice = new Tensor(new TensorShape(1, gridTokens, tokens.Shape[2]), DType.F32);
        Backend.SliceRows(slice, tokens, grid * gridTokens);
        return slice;
    }

    /// <summary>Encodes a prompt through the variant's taps, reusing the last result for the same ids. The transformer is
    /// evicted first: the encoder and a resident transformer need not fit together.</summary>
    private Tensor EncodePrompt(IrisZPrompt prompt, int slot)
    {
        if (_promptCache.TryGet(slot, prompt.Tokens, prompt.ValidLength, out Tensor? cached))
            return cached!;
        ReleaseTransformer(force: true);
        Tensor hidden;
        try
        {
            hidden = prompt.ValidLength > 0
                ? _textEncoder.EncodeMultiLayer(Backend, [prompt.Tokens], _textEncoderLayers, interleavedLayout: false,
                    TextEncoderPadding.Transformers(prompt.ValidLength))
                : _textEncoder.EncodeMultiLayer(Backend, [prompt.Tokens], _textEncoderLayers);
        }
        finally
        {
            Backend.FreeWeights(_textEncoder.EnumerateWeights());
        }
        _ = hidden.DataPointer;
        Backend.FreeActivations();
        if (hidden.Shape[2] != _config.ContextInDim)
        {
            long got = hidden.Shape[2];
            hidden.Dispose();
            throw new HartsyInferenceException($"{Family} text features are {got} wide; the transformer expects {_config.ContextInDim}.");
        }
        _promptCache.Store(slot, prompt.Tokens, prompt.ValidLength, hidden);
        return hidden;
    }

    /// <summary>VAE-encodes an image into BN-normalized packed tokens, releasing the encoder afterwards.</summary>
    private Tensor EncodeImage(Tensor image)
    {
        Backend.PreloadWeights(_vaeEncoder.EnumerateWeights());
        try
        {
            using Tensor vaeLatent = _vaeEncoder.Encode(Backend, image);
            return Flux2LatentOps.ToPackedTokens(vaeLatent, _bnMean, _bnVar, _bnEps, _config.PatchSize);
        }
        finally
        {
            Backend.Sync();
            Backend.FreeWeights(_vaeEncoder.EnumerateWeights());
            Backend.FreeActivations();
        }
    }

    private Tensor DecodeTokens(Tensor tokens, int patH, int patW)
    {
        using Tensor vaeLatent = Flux2LatentOps.FromPackedTokens(tokens, patH, patW, _bnMean, _bnVar, _bnEps,
            _config.VaeLatentChannels, _config.PatchSize);
        return _vaeDecoder.DecodeTiled(Backend, vaeLatent);
    }

    private byte[] DecodeRgb(Tensor tokens, int patH, int patW)
    {
        using Tensor image = DecodeTokens(tokens, patH, patW);
        return ImagePostProcessor.TensorToRgbBytes(image);
    }

    /// <summary>Decodes depth tokens: the mean of the decoded image's three channels, mapped back to log depth.</summary>
    private float[] DecodeLogDepth(Tensor tokens, int patH, int patW)
    {
        using Tensor image = DecodeTokens(tokens, patH, patW);
        ReadOnlySpan<float> pixels = image.AsReadOnlySpan<float>();
        int plane = pixels.Length / 3;
        float[] logDepth = new float[plane];
        for (int i = 0; i < plane; i++)
            logDepth[i] = Math.Clamp((pixels[i] + pixels[plane + i] + pixels[2 * plane + i]) / 3f, -1f, 1f);
        IrisZDepth.UnsqueezeFromCeiling(logDepth, _skyCeiling);
        return logDepth;
    }

    private static Tensor PackedNoise(Tensor noise)
    {
        using (noise) return Flux2LatentOps.Pack(noise);
    }

    private (int Width, int Height, int PatW, int PatH) Geometry(int width, int height)
    {
        int factor = _config.VaeDownscaleFactor;
        int patW = width / factor, patH = height / factor;
        if (patW < 1 || patH < 1)
            throw new ArgumentException($"{Family} needs at least {factor} pixels on a side; got {width}x{height}.");
        return (patW * factor, patH * factor, patW, patH);
    }

    /// <summary>Frees the transformer's device weights unless the VRAM profile keeps models resident.</summary>
    private void ReleaseTransformer(bool force = false)
    {
        if (!_ditResident || (!force && KeepModelsResident))
            return;
        Backend.FreeWeights(_transformer.EnumerateWeights());
        _ditResident = false;
    }

    /// <summary>Best effort: a failed stage leaves its temporaries in the activation cache, and a sticky device error
    /// must not mask the failure being reported.</summary>
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
}
