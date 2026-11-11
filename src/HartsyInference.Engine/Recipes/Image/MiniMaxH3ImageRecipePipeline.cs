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
/// <para>UNTESTED. Written without weights, a GPU or a parity reference. Every TODO(gpu) marks a step that still needs a
/// real-weight run before it can be trusted. The first suspects, in order: the velocity sign, the head selected per step,
/// the picture-label string, the latent statistics and the 2D RoPE axis order.</para></summary>
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
        double[] sigmas = MiniMaxH3Schedule.ImageSigmas(_config.OutputHeadSigmas!);

        // TODO(gpu): the picture label. The image path uses "Picture N: " with no brackets (nodes_minimax_h3.py); this
        // text-only path has no references, so no label is emitted yet. Phase 5b must add it.
        MiniMaxH3TextEncoder.Result encoded = EncodePrompt(request.Prompt);
        Tensor? x = null;
        Tensor? noise = null;
        try
        {
            int textLen = (int)encoded.HiddenStates.Shape[0];
            MiniMaxH3PackedLayout layout = new MiniMaxH3PackedLayout(textLen, 1, latentH, latentW, 0);

            // TODO(gpu): ComfyUI draws noise with its own generator. Matching the seed bit-for-bit is not required; the
            // parity test feeds saved noise instead.
            noise = SeedGenerator.CreateNoise(new TensorShape(new long[] { 1, _config.LatentsDim, 1, latentH, latentW }), seed);
            x = MiniMaxH3Latents.PackVideo(noise, _config);

            (Tensor cos, Tensor sin) = MiniMaxH3Rope.BuildTables(layout.PositionIds, _transformer.RopeInvFreq(),
                _config.AttentionHeadDim);
            try
            {
                (Tensor stackedWeight, Tensor stackedBias) = _transformer.StackedVideoHead();
                for (int step = 0; step < RequiredSteps; step++)
                {
                    cancel.ThrowIfCancellationRequested();
                    x = DenoiseStep(x, layout, encoded, cos, sin, stackedWeight, stackedBias, sigmas[step],
                        sigmas[step + 1]);
                    // TODO(gpu): step progress and live previews through RecipeProgressAdapter and
                    // LatentArchitecture.MiniMaxH3 are not wired yet.
                }
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

        if (SamplingParamResolver.ResolveSchedulerName(request) is not null)
        {
            throw new NotSupportedException(
                "MiniMax-H3-Image picks its own sigma schedule; sampler and scheduler selection is not supported.");
        }
    }

    private MiniMaxH3TextEncoder.Result EncodePrompt(string prompt)
    {
        // TODO(gpu): the encoder's preload and release calls are not wired here; the video pipeline frees the encoder
        // weights after encoding, which matters once the image path shares a device with the DiT.
        return _encoder.Encode(_textBackend, _tokenizer, prompt, null, null);
    }

    /// <summary>One sampler step: resolve the step's stacked head, run the image forward, and take the Euler step.</summary>
    private Tensor DenoiseStep(Tensor x, MiniMaxH3PackedLayout layout, MiniMaxH3TextEncoder.Result encoded,
        Tensor cos, Tensor sin, Tensor stackedWeight, Tensor stackedBias, double sigma, double sigmaNext)
    {
        float tVideo = (float)(1.0 - sigma);
        (float[] timesteps, IReadOnlyDictionary<MiniMaxH3SegmentKind, int> rowOf) =
            MiniMaxH3Conditioning.BuildTimestepRows(layout, tVideo, 0f, MiniMaxH3Schedule.VisualCondTimestep,
                MiniMaxH3Schedule.AudioCondTimestep);

        // TODO(gpu): the head span is resolved from the sigma pair, the same path ComfyUI's FinalLayer takes. Verify the
        // span against the reference on a real step; the plan flags the block-0 weighting as a likely error.
        OutputHeadStep span = _heads.Resolve(sigma, sigmaNext);
        (Tensor headWeight, Tensor headBias) = StackedOutputHeadWeights.Combine(_backend, stackedWeight, stackedBias,
            _config.VideoPatchDim, span);
        using PddFusedHeads heads = new PddFusedHeads(headWeight, headBias, null, null);

        Tensor raw = _transformer.ForwardImage(_backend, layout, x, encoded.HiddenStates, cos, sin, timesteps, rowOf,
            heads, encoded.TagRuns);
        try
        {
            // TODO(gpu): ComfyUI's MiniMaxH3ImageModel returns -out as the model output. This negates the head output
            // to match. If every image comes out as noise, check this sign first.
            Tensor velocity = new Tensor(raw.Shape, DType.F32);
            Tensor scaled = new Tensor(raw.Shape, DType.F32);
            Tensor next = new Tensor(raw.Shape, DType.F32);
            try
            {
                _backend.Scale(velocity, raw, -1f);
                _backend.Scale(scaled, velocity, (float)(sigmaNext - sigma));
                _backend.Add(next, x, scaled);
            }
            finally
            {
                velocity.Dispose();
                scaled.Dispose();
            }

            x.Dispose();
            return next;
        }
        finally
        {
            raw.Dispose();
        }
    }

    private ImageResult Decode(Tensor x, int latentH, int latentW, int width, int height, int seed)
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
