using System.Linq;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed LTX-Video pipeline driven against the native <see cref="VideoRequest"/>. Encodes the prompt
/// pair with T5-XXL (zeroing the rows past the real tokens — the engine's LTX path has no context attention mask) and
/// feeds <see cref="VideoRequest.Fps"/> to the pipeline itself, which uses it for RoPE interpolation the same way
/// Comfy injects <c>LTXVConditioning.frame_rate</c>. Mirrors the SwarmUI backend's <c>LtxVideoLoader.Generate</c>.</summary>
public sealed class LtxVideoRecipePipeline : IVideoRecipePipeline
{
    private readonly IBackend _backend;
    private readonly LtxVideoPipeline _pipeline;
    private readonly LtxVideoConfig _config;
    private readonly T5Tokenizer _tokenizer;
    private readonly T5TextEncoder _t5;
    private readonly LtxVideoTransformer _transformer;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed LTX-Video pipeline plus its text encoder, taking ownership of every disposable.</summary>
    public LtxVideoRecipePipeline(IBackend backend, LtxVideoPipeline pipeline, LtxVideoConfig config, T5Tokenizer tokenizer,
        T5TextEncoder t5, LtxVideoTransformer transformer, List<SafeTensorsLoader> loaders)
    {
        _backend = backend;
        _pipeline = pipeline;
        _config = config;
        _tokenizer = tokenizer;
        _t5 = t5;
        _transformer = transformer;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? _config.NumInferenceSteps;
        (int width, int height) = VideoRecipeUtils.ResolveResolution(request, _config.VaeSpatialCompression);
        int numFrames = VideoRecipeUtils.ResolveFrames(request, modelDefault: 97, step: _config.VaeTemporalCompression);
        int frameRate = request.Fps ?? 24;
        float cfgScale = request.CfgScale ?? _config.GuidanceScale;

        int[] promptTokens = _tokenizer.Encode(prompt);
        int[] negTokens = _tokenizer.Encode(negative);
        int[] promptMask = T5Tokenizer.CreateAttentionMask(promptTokens);
        int[] negMask = T5Tokenizer.CreateAttentionMask(negTokens);
        Tensor batch = _t5.Encode(_backend, [promptTokens, negTokens], [promptMask, negMask]);
        // Drop right-padding: feed cross-attention only the real (non-pad) T5 tokens — attending the pad rows
        // unmasked dilutes the caption (LtxVideoGenerationTests' proven fix; zeroing the pad rows in place, the
        // Engine's earlier approach, still attends them and was NOT the fix that made LTX-Video coherent).
        int promptLen = promptMask.Sum(), negLen = negMask.Sum();
        Tensor promptEmbeds = CfgHelper.SliceBatchElementPrefix(batch, 0, promptTokens.Length, promptLen, _config.CaptionChannels);
        Tensor negEmbeds = CfgHelper.SliceBatchElementPrefix(batch, 1, negTokens.Length, negLen, _config.CaptionChannels);
        batch.Dispose();
        _backend.Sync();
        _backend.FreeWeights(_t5.EnumerateWeights());

        // TODO(E-IMG-4/5): image-to-video conditioning (InitImage / VideoEndFrame) is not wired for LTX-Video — the
        // extension's loader drives text-to-video only.
        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
        };

        Action<GenerationProgress> bridge = p =>
        {
            cancel.ThrowIfCancellationRequested();
            progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
        };

        try
        {
            (byte[][] frames, int outW, int outH, int _) = _pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, inner, numFrames, frameRate, bridge);
            Logs.Info($"[LtxVideoRecipePipeline] Pipeline returned {frames.Length} frames {outW}x{outH}.");
            return VideoRecipeUtils.ToResult(frames, outW, outH, request);
        }
        catch (Exception ex)
        {
            Logs.Error("[LtxVideoRecipePipeline] Generation failed.", ex);
            throw;
        }
        finally
        {
            promptEmbeds.Dispose();
            negEmbeds.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _tokenizer.Dispose();
        _t5.Dispose();
        _transformer.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
