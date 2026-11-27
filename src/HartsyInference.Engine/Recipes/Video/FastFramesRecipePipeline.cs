using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed FastFrames pipeline driven from a native <see cref="VideoRequest"/>: one umT5-XXL pass over the prompt,
/// then the frame-wise rollout, which yields each latent frame as it decodes. Owns the generator, the refiner, the decoder and the
/// text encoder, and disposes them with the loaders it was given.</summary>
public sealed class FastFramesRecipePipeline : IVideoRecipePipeline
{
    private readonly IBackend _backend;
    private readonly IBackend _textBackend;
    private readonly FastFramesPipeline _pipeline;
    private readonly FastFramesConfig _config;
    private readonly T5TextEncoder _umt5;
    private readonly T5Tokenizer _tokenizer;
    private readonly List<IDisposable> _loaders;
    private bool _disposed;

    /// <summary>Takes ownership of <paramref name="loaders"/>, which must hold every component the pipeline uses.</summary>
    public FastFramesRecipePipeline(IBackend backend, IBackend textBackend, FastFramesPipeline pipeline, FastFramesConfig config, T5TextEncoder umt5,
        T5Tokenizer tokenizer, List<IDisposable> loaders)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _textBackend = textBackend ?? throw new ArgumentNullException(nameof(textBackend));
        _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _umt5 = umt5 ?? throw new ArgumentNullException(nameof(umt5));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _loaders = loaders ?? throw new ArgumentNullException(nameof(loaders));
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, VideoEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(settings);
        cancel.ThrowIfCancellationRequested();
        ObjectDisposedException.ThrowIf(_disposed, this);
        CheckSettings(settings);
        int latentFrames = (settings.Frames - 1) / 4 + 1;
        int seed = (int)(settings.Seed & 0x7FFF_FFFF);
        IFastFramesNoise noise = new FastFramesSeededNoise(seed, _config);

        Tensor promptEmbeds = EncodePrompt(request.Prompt);
        try
        {
            // Progress counts latent frames done out of latentFrames, not denoising steps.
            Action<int, int>? report = progress is null
                ? null
                : (done, total) => progress.Report(new StepPreview { Step = done, TotalSteps = total });
            List<byte[]> frames = new List<byte[]>(latentFrames * 4);
            int width = FastFramesRecipe.OutputWidth, height = FastFramesRecipe.OutputHeight;
            foreach (HartsyInference.Video.VideoFrame frame in _pipeline.GenerateFrames(promptEmbeds, noise, latentFrames, report, cancel))
            {
                frames.Add(frame.Rgb);
                width = frame.Width;
                height = frame.Height;
            }
            if (frames.Count != settings.Frames)
            {
                throw new InvalidOperationException($"FastFrames produced {frames.Count} frames; the request asked for {settings.Frames}.");
            }
            Logs.Info($"[FastFramesRecipePipeline] {frames.Count} frames {width}x{height} at {FastFramesRecipe.OutputFps} fps.");
            return VideoRecipeUtils.ToResult(frames.ToArray(), width, height, request, fps: FastFramesRecipe.OutputFps);
        }
        finally
        {
            promptEmbeds.Dispose();
        }
    }

    /// <summary>Refuses a size or frame count the released models cannot produce, before the text encoder runs.</summary>
    /// <remarks>Steps and CFG are not tunables here: the generator's schedule and its lack of guidance are the released ones.
    /// A different value is logged and ignored rather than failing, since the planner fills both from the recipe's defaults.</remarks>
    internal static void CheckSettings(VideoEffectiveSettings settings)
    {
        if (settings.Width != FastFramesRecipe.OutputWidth || settings.Height != FastFramesRecipe.OutputHeight)
        {
            throw new NotSupportedException(
                $"FastFrames generates {FastFramesRecipe.OutputWidth}x{FastFramesRecipe.OutputHeight} only; the request asked for {settings.Width}x{settings.Height}.");
        }
        if (settings.Frames < 1 || settings.Frames > FastFramesRecipe.MaxFrames || (settings.Frames - 1) % 4 != 0)
        {
            throw new NotSupportedException(
                $"FastFrames generates 4n+1 frames up to {FastFramesRecipe.MaxFrames}; the request asked for {settings.Frames}.");
        }
        if (settings.Steps != FastFramesRecipe.FirstFrameSteps || settings.CfgScale != FastFramesRecipe.NoGuidance)
        {
            Logs.Warning($"[FastFramesRecipePipeline] Steps {settings.Steps} and CFG {settings.CfgScale} are ignored: FastFrames runs its released schedule without guidance.");
        }
    }

    /// <summary>The umT5 embedding of <paramref name="prompt"/>, zero-padded to 512 rows as upstream's <c>embed_text</c> does.</summary>
    private Tensor EncodePrompt(string prompt)
    {
        int[] tokens = _tokenizer.Encode(PromptTagFlattening.Flatten(prompt));
        _textBackend.PreloadWeights(_umt5.EnumerateWeights());
        Tensor[] embeds = VideoRecipeUtils.EncodeWanPromptBatch(_textBackend, _umt5, FastFramesRecipe.TextDim, tokens);
        return embeds[0];
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        // The pipeline first: it frees the generator's device weights, and the loaders then release the host objects. The loaders
        // hold the transformer, the refiner, the decoder and the umT5 encoder. Each one's Tensor.Dispose runs its GPU dispose callback,
        // so their device copies go with it (not verified on a device). A recipe dropped without Dispose frees nothing.
        // CompositeDisposable releases each part even when an earlier one throws, then rethrows the failures together.
        IDisposable?[] parts = [_pipeline, .. _loaders];
        _loaders.Clear();
        new CompositeDisposable(parts).Dispose();
    }
}
