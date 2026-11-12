using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Features;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.MiniMaxH3;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>MiniMax-H3-Image text-to-image pipeline: Qwen3-VL text encoding, six stacked-head denoise steps on the
/// checkpoint's own sigma grid, and the 2D single-frame VAE decode.
/// <para>CPU parity tests cover the schedule, noise and packing; the DiT steps and VAE decode have device parity tests
/// against ComfyUI that need real weights and a GPU. Each remaining TODO(gpu) is still open.</para></summary>
internal sealed class MiniMaxH3ImageRecipePipeline : IRecipePipeline
{
    /// <summary>The image build's head grid is distilled on six steps; the official server refuses any other count.</summary>
    internal const int RequiredSteps = 6;

    private readonly MiniMaxH3Transformer _transformer;
    private readonly MiniMaxH3Config _config;
    private readonly MiniMaxH3VideoVaeDecoder _decoder;
    private readonly MiniMaxH3TextEncoder _encoder;
    private readonly Qwen2Tokenizer _tokenizer;
    private readonly IBackend _backend;
    private readonly IBackend _textBackend;
    private readonly IBackend _vaeBackend;
    private readonly IReadOnlyList<IDisposable> _loaders;
    private readonly StackedOutputHeads _heads;
    private bool _disposed;

    internal MiniMaxH3ImageRecipePipeline(MiniMaxH3Transformer transformer, MiniMaxH3Config config,
        MiniMaxH3VideoVaeDecoder decoder, MiniMaxH3TextEncoder encoder, Qwen2Tokenizer tokenizer,
        IBackend backend, IBackend textBackend, IBackend vaeBackend, IReadOnlyList<IDisposable> loaders)
    {
        _transformer = transformer ?? throw new ArgumentNullException(nameof(transformer));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _textBackend = textBackend ?? throw new ArgumentNullException(nameof(textBackend));
        _vaeBackend = vaeBackend ?? throw new ArgumentNullException(nameof(vaeBackend));
        _loaders = loaders ?? throw new ArgumentNullException(nameof(loaders));
        if (config.OutputHeadSigmas is null)
        {
            throw new ArgumentException("MiniMax-H3-Image config carries no output-head sigma grid.", nameof(config));
        }
        _heads = new StackedOutputHeads(config.OutputHeadSigmas);
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, ImageEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(settings);
        cancel.ThrowIfCancellationRequested();
        RefuseUnsupported(request, settings);

        int width = settings.Width, height = settings.Height;
        // TODO(owner): snap-or-refuse for sides that are not multiples of 32 is an open question; refusing for now.
        if (width % 32 != 0 || height % 32 != 0)
        {
            throw new NotSupportedException(
                $"MiniMax-H3-Image needs width and height that are multiples of 32; got {width}x{height}.");
        }

        int latentH = height / 16, latentW = width / 16;
        int seed = RecipeRequestMapper.MapSeed(request.Seed) ?? SeedGenerator.RandomSeed();
        // torch.manual_seed keeps the low 32 bits, so a request seed from 2^31 up needs its own bit 31 to match ComfyUI.
        long noiseSeed = request.Seed >= 1L << 31 ? request.Seed & 0xFFFFFFFFL : seed;
        double[] sigmas = MiniMaxH3Schedule.ImageSigmas(_config.OutputHeadSigmas!, RequiredSteps);

        // TODO(gpu): the picture label. The image path uses "Picture N: " with no brackets (nodes_minimax_h3.py); this
        // text-only path has no references, so no label is emitted yet. Phase 5b must add it.
        MiniMaxH3TextEncoder.Result encoded = EncodePrompt(request.Prompt);
        Tensor? x = null;
        Tensor? noise = null;
        try
        {
            int textLen = (int)encoded.HiddenStates.Shape[0];
            MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(textLen, 1, latentH, latentW, 0);

            // The same seed gives the same noise as ComfyUI's torch.randn (see TorchCompatibleNoise), so a request is comparable
            // with the reference image, not just with itself.
            noise = TorchCompatibleNoise.Randn(new TensorShape([1L, _config.LatentsDim, 1L, latentH, latentW]), noiseSeed);
            x = MiniMaxH3Latents.PackVideo(noise, _config);
            noise.Dispose();
            noise = null;

            (Tensor cos, Tensor sin) = RopeTables(layout);
            try
            {
                (Tensor stackedWeight, Tensor stackedBias) = StackedHead();
                // x is passed by ref so this frame always holds the live latent, even if a later step is cancelled.
                RunSteps(ref x, sigmas, (current, sigma, sigmaNext) => DenoiseStep(current, layout, encoded, cos, sin,
                    stackedWeight, stackedBias, sigma, sigmaNext), progress, cancel);
                // TODO(gpu): live latent previews (LatentArchitecture.MiniMaxH3) are not wired yet; progress ticks are.
            }
            finally
            {
                cos.Dispose();
                sin.Dispose();
            }

            return Decode(x, latentH, latentW, width, height, seed);
        }
        finally
        {
            noise?.Dispose();
            x?.Dispose();
            encoded.HiddenStates.Dispose();
        }
    }

    /// <summary>Refuses every request the image build cannot honour, by name. Features the recipe does not declare are
    /// refused upstream by the engine; these are the settings that reach the pipeline.</summary>
    internal static void RefuseUnsupported(ImageRequest request, ImageEffectiveSettings settings)
    {
        if (request.Batch != 1)
        {
            throw new NotSupportedException(
                $"MiniMax-H3-Image generates one image per request; got batch {request.Batch}. Send one request per image.");
        }

        if (settings.Steps != RequiredSteps)
        {
            throw new NotSupportedException(
                $"MiniMax-H3-Image runs exactly {RequiredSteps} steps; the stacked output heads are distilled on that grid. Got {settings.Steps}.");
        }

        if (settings.CfgScale != 1f)
        {
            throw new NotSupportedException(
                $"MiniMax-H3-Image takes no guidance: CFG must be 1. Got {settings.CfgScale}.");
        }

        if (!string.IsNullOrEmpty(request.NegativePrompt))
        {
            throw new NotSupportedException("MiniMax-H3-Image does not take a negative prompt.");
        }

        // The family's only schedule is simple. SamplingParamResolver treats normal and default as identity, so an
        // explicit normal or default is accepted as that default; any other scheduler is refused by name.
        string? selection = SamplingParamResolver.ResolveSchedulerName(request);
        if (selection is not null && !IsSolverOwnSelection(selection))
        {
            throw new NotSupportedException(
                $"MiniMax-H3-Image runs euler over the simple sigma schedule; sampler '{request.Sampler}' and "
                + $"schedule '{request.Scheduler}' are not supported.");
        }
    }

    /// <summary>True for the selections the image build's own solver already is: euler, or euler over the simple schedule.</summary>
    internal static bool IsSolverOwnSelection(string selection) =>
        selection.Trim().ToLowerInvariant() is "euler" or "euler_simple";

    /// <summary>The prompt's token presentation for the image build. Plain text in, the way ComfyUI's image node presents
    /// it: the Qwen byte-level BPE of the string as written, with no template and no special tokens.</summary>
    internal static MiniMaxH3TextEncoding.Encoded PresentPrompt(Qwen2Tokenizer tokenizer, string prompt) =>
        MiniMaxH3TextEncoding.Build(tokenizer, prompt, conditions: null, weightedPrompt: false);

    private MiniMaxH3TextEncoder.Result EncodePrompt(string prompt)
    {
        // TODO(gpu): the encoder's preload and release calls are not wired here; the video pipeline frees the encoder
        // weights after encoding, which matters once the image path shares a device with the DiT.
        return _encoder.Encode(_textBackend, PresentPrompt(_tokenizer, prompt), null);
    }

    /// <summary>The text-side context the body receives, for device checks against ComfyUI's preprocessed context.</summary>
    internal Tensor RefinedText(IBackend backend, Tensor textStates) => _transformer.RefinedText(backend, textStates);

    /// <summary>The checkpoint-derived config the pipeline runs with.</summary>
    internal MiniMaxH3Config Config => _config;

    /// <summary>The rotary tables for one packed layout, as <see cref="Generate"/> builds them once per request.</summary>
    internal (Tensor Cos, Tensor Sin) RopeTables(MiniMaxH3PackedLayout layout) =>
        MiniMaxH3Rope.BuildTables(layout.PositionIds, _transformer.RopeInvFreq(), _config.AttentionHeadDim);

    /// <summary>The checkpoint's stacked video head, which <see cref="DenoiseStep"/> combines per step.</summary>
    internal (Tensor Weight, Tensor Bias) StackedHead() => _transformer.StackedVideoHead();

    /// <summary>Runs the sampler's steps over <paramref name="sigmas"/>: one denoise call per adjacent sigma pair, in order.
    /// Cancellation is checked before each step, and each finished step reports one progress tick.</summary>
    /// <remarks>The latent is passed by reference and replaced after every step, so the caller always holds the live tensor.
    /// Each denoise call takes ownership of its input and returns the next latent; if a later step is cancelled, the caller
    /// disposes the latent it holds, which is the one the last completed step returned.</remarks>
    internal static void RunSteps(ref Tensor x, double[] sigmas, Func<Tensor, double, double, Tensor> denoise,
        IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        int steps = sigmas.Length - 1;
        for (int step = 0; step < steps; step++)
        {
            cancel.ThrowIfCancellationRequested();
            x = denoise(x, sigmas[step], sigmas[step + 1]);
            progress?.Report(new StepPreview { Step = step + 1, TotalSteps = steps });
        }
    }

    /// <summary>One sampler step: resolve the step's stacked head, run the image forward, and take the Euler step.</summary>
    internal Tensor DenoiseStep(Tensor x, MiniMaxH3PackedLayout layout, MiniMaxH3TextEncoder.Result encoded,
        Tensor cos, Tensor sin, Tensor stackedWeight, Tensor stackedBias, double sigma, double sigmaNext)
    {
        // ComfyUI hands the DiT float32(sigma * 1000) and runs on t = 1 - that / 1000. Only sigma * 1000 is rounded to float
        // here; the rest, including dt below, is double, so t and dt can differ from ComfyUI's float32 by about one ulp.
        float tVideo = (float)(1.0 - (double)(float)(sigma * 1000.0) / 1000.0);
        (float[] timesteps, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(layout, tVideo, 0f, MiniMaxH3Schedule.VisualCondTimestep,
                MiniMaxH3Schedule.AudioCondTimestep);

        // The span comes from the sigma pair, the path ComfyUI's FinalLayer takes; the step-velocity device tests check it.
        OutputHeadStep span = _heads.Resolve(sigma, sigmaNext);
        (Tensor headWeight, Tensor headBias) = StackedOutputHeadWeights.Combine(_backend, stackedWeight, stackedBias,
            _config.VideoPatchDim, span);
        using PddFusedHeads heads = new PddFusedHeads(headWeight, headBias, null, null);

        Tensor raw = _transformer.ForwardImage(_backend, layout, x, encoded.HiddenStates, cos, sin, timesteps, rowOf,
            heads, encoded.TagRuns);
        try
        {
            // ComfyUI's MiniMaxH3ImageModel returns -out as the model output, so the head output is negated here.
            Tensor velocity = new Tensor(raw.Shape, DType.F32);
            Tensor scaled = new Tensor(raw.Shape, DType.F32);
            Tensor? next = new Tensor(raw.Shape, DType.F32);
            try
            {
                _backend.Scale(velocity, raw, -1f);
                _backend.Scale(scaled, velocity, (float)(sigmaNext - sigma));
                _backend.Add(next, x, scaled);
                x.Dispose();
                Tensor result = next;
                next = null;
                return result;
            }
            finally
            {
                velocity.Dispose();
                scaled.Dispose();
                next?.Dispose();
            }
        }
        finally
        {
            raw.Dispose();
        }
    }

    internal ImageResult Decode(Tensor x, int latentH, int latentW, int width, int height, int seed)
    {
        using Tensor latent = MiniMaxH3Latents.UnpackVideo(x, 1, latentH, latentW, _config);
        using Tensor frame = _decoder.Decode(_vaeBackend, latent);
        // The 2D decoder returns [1, 3, 1, H, W]; the RGB converter takes [1, 3, H, W].
        using Tensor image = frame.Reshape(new TensorShape(1, 3, height, width));
        byte[] rgb = ImagePostProcessor.TensorToRgbBytes(image);

        return new ImageResult
        {
            Rgb = rgb,
            Width = width,
            Height = height,
            Seed = seed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "minimax-h3-image",
                ["size"] = $"{width}x{height}",
                ["seed"] = seed.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["steps"] = RequiredSteps.ToString(System.Globalization.CultureInfo.InvariantCulture),
            },
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _transformer.Dispose();
        _decoder.Dispose();
        _encoder.Dispose();
        foreach (IDisposable loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
