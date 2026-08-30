using HartsyInference.Engine.Features;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Linq;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed LTX-Video pipeline driven against the native <see cref="VideoRequest"/>. Encodes the prompt pair with T5-XXL (zeroing the rows past the real tokens — the engine's LTX path has no context attention mask) and feeds <see cref="VideoRequest.Fps"/> to the pipeline itself, which uses it for RoPE interpolation the same way Comfy injects <c>LTXVConditioning.frame_rate</c>. Mirrors the SwarmUI backend's <c>LtxVideoLoader.Generate</c>.</summary>
public sealed class LtxVideoRecipePipeline : IVideoRecipePipeline
{
    private readonly IBackend _backend;
    /// <summary>Backend the T5-XXL prompt encoder runs on — separable from <see cref="_backend"/> because the embeddings are host-materialized (SliceBatchElementPrefix is a host loop) before the denoiser consumes them, so moving the ~5 GB encoder off the denoiser GPU costs nothing.</summary>
    private readonly IBackend _textBackend;
    /// <summary>Backend <see cref="_vaeEncoder"/> (Tier 3.4 I2V) runs on — same device the VAE decoder uses.</summary>
    private readonly IBackend _vaeBackend;
    private readonly LtxVideoPipeline _pipeline;
    private readonly LtxVideoConfig _config;
    private readonly T5Tokenizer _tokenizer;
    private readonly T5TextEncoder _t5;
    private readonly LtxVideoTransformer _transformer;
    /// <summary>Null for the 0.9.5/13B variants (Tier 3.4's encoder was built and verified against base 0.9 only — see <see cref="LtxVideoRecipe.SupportsFor"/>); plan-driven feature validation refuses an init image before this pipeline is reached.</summary>
    private readonly LtxVideoVaeEncoder? _vaeEncoder;
    private readonly List<SafeTensorsLoader> _loaders;
    private readonly MergedLoraStack? _loraStack;

    /// <summary>Wraps the constructed LTX-Video pipeline plus its text encoder, taking ownership of every disposable. <paramref name="textBackend"/>/<paramref name="vaeBackend"/> may equal <paramref name="backend"/> (single-device default).</summary>
    public LtxVideoRecipePipeline(IBackend backend, IBackend textBackend, IBackend vaeBackend, LtxVideoPipeline pipeline, LtxVideoConfig config,
        T5Tokenizer tokenizer, T5TextEncoder t5, LtxVideoTransformer transformer, LtxVideoVaeEncoder? vaeEncoder, List<SafeTensorsLoader> loaders, MergedLoraStack? loraStack = null)
    {
        _loraStack = loraStack;
        _backend = backend;
        _textBackend = textBackend;
        _vaeBackend = vaeBackend;
        _pipeline = pipeline;
        _config = config;
        _tokenizer = tokenizer;
        _t5 = t5;
        _transformer = transformer;
        _vaeEncoder = vaeEncoder;
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
        // T5 runs on the (possibly separate) text backend; the host-side SliceBatchElementPrefix passes below ARE
        // the cross-device boundary — they force the embeddings to host, so the denoiser's backend re-uploads from
        // there. Load-bearing for TextEncoderDevice placement: keep them host-side.
        Tensor batch = _t5.Encode(_textBackend, [promptTokens, negTokens], [promptMask, negMask]);
        // Drop right-padding: feed cross-attention only the real (non-pad) T5 tokens — attending the pad rows
        // unmasked dilutes the caption (LtxVideoGenerationTests' proven fix; zeroing the pad rows in place, the
        // Engine's earlier approach, still attends them and was NOT the fix that made LTX-Video coherent).
        int promptLen = promptMask.Sum(), negLen = negMask.Sum();
        Tensor promptEmbeds = CfgHelper.SliceBatchElementPrefix(batch, 0, promptTokens.Length, promptLen, _config.CaptionChannels);
        Tensor negEmbeds = CfgHelper.SliceBatchElementPrefix(batch, 1, negTokens.Length, negLen, _config.CaptionChannels);
        batch.Dispose();
        _textBackend.Sync();
        _textBackend.FreeWeights(_t5.EnumerateWeights());

        // Tier 3.4 (I2V, base 0.9 only — see LtxVideoRecipe.SupportsFor): VAE-encode the init image into a
        // single-frame latent, the same firstFrameLatent-in/firstFrameLatent-out shape Wan's I2V path uses.
        // VideoRequest.VideoEndFrame is still not wired here — LTX's per-token conditioning mechanism (Tier 3.4's
        // transformer/pipeline change) conditions on a SINGLE mask row (frame 0); extending it to a second,
        // end-of-clip conditioned token range is unbuilt.
        Tensor? firstFrameLatent = null;
        if (request.InitImage is not null)
        {
            if (_vaeEncoder is null)
            {
                throw new NotSupportedException("This LTX-Video checkpoint's VAE encoder was not built for this variant (0.9.5/13B) — image-to-video is only wired for base 0.9.");
            }
            byte[] frameRgb = VideoRecipeUtils.ResizeRgb24(request.InitImage, width, height);
            _vaeBackend.PreloadWeights(_vaeEncoder.EnumerateWeights());
            firstFrameLatent = _vaeEncoder.EncodeRgbFrame(_vaeBackend, frameRgb, width, height);
            _vaeBackend.Sync();
            _vaeBackend.FreeWeights(_vaeEncoder.EnumerateWeights());
        }

        TextToImageRequest inner = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = steps,
            CfgScale = cfgScale,
            Seed = RecipeRequestMapper.MapSeed(request.Seed),
            // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
            // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
            Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
        };

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

        try
        {
            (byte[][] frames, int outW, int outH, int _) = _pipeline.GenerateFromEmbeddings(promptEmbeds, negEmbeds, inner, numFrames, frameRate, bridge, firstFrameLatent);
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
            firstFrameLatent?.Dispose();
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
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
