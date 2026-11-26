using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed MovieRoll pipeline driven against the native <see cref="VideoRequest"/>. The prompt's Qwen features are the reference's chat
/// template with the crop measured from the tokenised prompt; its glyph text, if any, goes through Glyph-ByT5-small (loaded on the first quoted prompt).
/// Each request runs the rolling sampler over the latent grid and decodes the result. Takes ownership of every disposable it is given.</summary>
public sealed class MovieRollRecipePipeline(IBackend backend, MovieRollPipeline pipeline, HunyuanVideo15Dit dit, LlamaStyleEncoder llama,
    HunyuanImageQwenTextEncoder qwen, Qwen2Tokenizer tokenizer,
    Func<CancellationToken, (T5TextEncoder Encoder, IDisposable Loader)> glyphFactory, List<IDisposable> loaders) : IVideoRecipePipeline
{
    private readonly object _glyphLock = new();
    private T5TextEncoder? _glyph;
    private IDisposable? _glyphLoader;

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, VideoEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        (int width, int height) = VideoRecipeUtils.ResolveResolution(settings, multiple: 16);
        int frames = settings.Frames;
        // Frames are 16n+1 (the constraints snap them); the sampler emits n chunks: a first of five latent frames and n-1 of four.
        if (frames < 17 || (frames - 1) % 16 != 0)
            throw new ArgumentException($"MovieRoll generates 16n+1 frames; got {frames}.", nameof(settings));
        int chunks = (frames - 1) / 16;
        int latentHeight = height / 16;
        int latentWidth = width / 16;

        MovieRollSamplerOptions options = new()
        {
            ChunkFrames = MovieRollRecipe.ChunkFrames,
            FirstChunkFrames = MovieRollRecipe.FirstChunkFrames,
            NumChunks = chunks,
            FullHistoryBlocks = MovieRollRecipe.FullHistoryBlocks,
            ContextNoise = MovieRollRecipe.ContextNoise,
            KvRecomputeEnabled = true,
            KvRecomputeThreshold = MovieRollRecipe.KvRecomputeThreshold,
            KvRecomputeRecentK = MovieRollRecipe.KvRecomputeRecentK,
            // The distilled chain has no classifier-free guidance: a null Cfg is the no-CFG path.
            Cfg = null,
        };

        // An unset seed draws a fresh one, and the log line records it so the run can be repeated.
        long seed = RecipeRequestMapper.MapSeed(request.Seed) ?? Random.Shared.Next();
        Tensor? text = null;
        Tensor? glyphFeatures = null;
        try
        {
            string prompt = request.Prompt;
            (int[] ids, int crop) = HunyuanVideo15Prompt.Tokenize(tokenizer, prompt);
            backend.PreloadWeights(llama.EnumerateWeights());
            text = qwen.EncodeCropped(backend, ids, crop);
            backend.Sync();
            backend.FreeWeights(llama.EnumerateWeights());

            glyphFeatures = EncodeGlyphText(prompt, cancel);

            Func<int, Tensor> noise = SeededNoise(seed, HunyuanVideo15Config.Full.LatentChannels, latentHeight, latentWidth);
            using Tensor rgb = pipeline.Generate(options, text, glyphFeatures, null, null, latentHeight, latentWidth, noise);

            int f = (int)rgb.Shape[2];
            byte[][] decoded = new byte[f][];
            for (int i = 0; i < f; i++) decoded[i] = VideoRgbFrames.ExtractFrame(rgb, i);
            int w = (int)rgb.Shape[4], h = (int)rgb.Shape[3];
            Logs.Info($"[MovieRollRecipePipeline] Generated {decoded.Length} frames {w}x{h} (seed {seed}).");
            return VideoRecipeUtils.ToResult(decoded, w, h, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[MovieRollRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            text?.Dispose();
            glyphFeatures?.Dispose();
        }
    }

    /// <summary>The glyph text's ByT5 features, or null when the prompt quotes nothing (the DiT then sees no byT5 rows, as the reference).</summary>
    private Tensor? EncodeGlyphText(string prompt, CancellationToken cancel)
    {
        string? glyphText = HunyuanVideo15Prompt.FormatGlyphText(prompt);
        if (glyphText is null) return null;
        lock (_glyphLock)
        {
            if (_glyph is null)
            {
                (T5TextEncoder encoder, IDisposable loader) = glyphFactory(cancel);
                _glyph = encoder;
                _glyphLoader = loader;
            }
        }
        int[] tokens = ByT5Tokenizer.Encode(glyphText);
        if (tokens.Length > HunyuanVideo15Prompt.ByT5MaxTokens)
            tokens = tokens[..HunyuanVideo15Prompt.ByT5MaxTokens];
        backend.PreloadWeights(_glyph!.EnumerateWeights());
        Tensor features = _glyph.Encode(backend, [tokens]);
        backend.Sync();
        backend.FreeWeights(_glyph.EnumerateWeights());
        return features;
    }

    /// <summary>Draws the sampler's noise from one seed: each call takes the next seed, so a request is reproducible.</summary>
    private static Func<int, Tensor> SeededNoise(long seed, int channels, int height, int width)
    {
        long next = seed;
        return frames => TorchCompatibleNoise.Randn(new TensorShape([1L, channels, frames, height, width]), next++);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        pipeline.Dispose();
        dit.Dispose();
        llama.Dispose();
        _glyph?.Dispose();
        _glyphLoader?.Dispose();
        foreach (IDisposable loader in loaders)
            loader.Dispose();
    }
}
