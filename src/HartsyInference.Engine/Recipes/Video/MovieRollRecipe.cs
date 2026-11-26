using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.TextEncoders;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>MovieRoll (Tencent, distilled text-to-video) recipe. The checkpoint is the distilled T2V DiT of <c>yunyangge/MovieRoll</c> (HunyuanVideo-1.5's
/// 8.3B architecture, streamed block by block through MovieRoll's rolling sampler). Its text side is the reference's: Qwen2.5-VL-7B through the video
/// chat template with the crop measured from the tokenised prompt, and Glyph-ByT5-small on the quoted text only. The VAE is the HunyuanVideo-1.5 VAE the
/// reference loads. The distilled chain runs five steps with no classifier-free guidance, context noise 0.15, and the full-history block count 6.
/// Construction is not validated against real weights yet: the DiT is gated, so its catalog entry is <c>ValidationPending</c> and carries no pin.</summary>
public sealed class MovieRollRecipe : IVideoRecipe
{
    /// <summary>Latent frames of each emitted chunk after the first, and the first chunk's <c>1 + ChunkFrames</c> (upstream <c>check_and_get_chunk_sizes</c>).</summary>
    internal const int ChunkFrames = 4;
    internal const int FirstChunkFrames = 5;

    /// <summary>Full-history block count (upstream <c>full_history_block_num</c>): the first six blocks keep the whole clip.</summary>
    internal const int FullHistoryBlocks = 6;

    /// <summary>Context noise of the distilled chain (<c>context_noise</c> in the reference's <c>DISTILLED</c> profile).</summary>
    internal const float ContextNoise = 0.15f;

    /// <summary>KV recompute trigger and recent-chunk count (the reference's <c>COMMON</c> profile).</summary>
    internal const int KvRecomputeThreshold = 18;
    internal const int KvRecomputeRecentK = 3;

    /// <summary>The distilled chain's denoising steps (<c>denoising_step_list</c>) and timestep shift; the window is one chunk per step.</summary>
    internal static readonly int[] DenoisingSteps = [1000, 800, 600, 400, 200];
    internal const double TimestepShift = 5.0;

    /// <inheritdoc/>
    public string Name => "movieroll";

    /// <inheritdoc/>
    public PromptProfile? Prompt { get; } = new()
    {
        Kind = "video", Encoders = ["vlm", "t5"], Format = "text", Negative = "none", Supports = [],
    };

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "movieroll", StringComparison.OrdinalIgnoreCase);

    /// <summary>The distilled chain's settings: five steps, no guidance, 480 x 864, 81 frames (five chunks) at 24 fps. The size and frame count are the
    /// reference's <c>COMMON</c> profile (<c>_HEIGHT</c>, <c>_WIDTH</c>, <c>video_fps</c>); the frame count is one of the 16n+1 counts the sampler emits.</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Revision = 1, Steps = DenoisingSteps.Length, CfgScale = 1f, Width = 864, Height = 480, Frames = 81, Fps = 24 };

    /// <summary>Frames are 16n+1: four pixel frames per latent frame after the first, and a first chunk of five latent frames, so n emitted chunks give
    /// <c>4·(4n+1−1)+1 = 16n+1</c> pixel frames.</summary>
    /// <remarks>The grid starts at 17 (the smallest count with a chunk), so a request below it snaps up instead of failing in the pipeline.</remarks>
    public VideoConstraints Constraints { get; } = new VideoConstraints { Frames = new FrameGrid(4 * ChunkFrames, 4 * ChunkFrames + 1, FrameNormalization.SnapNearest) };

    /// <inheritdoc/>
    public VideoFeatures Supports => VideoFeatures.None;

    /// <inheritdoc/>
    /// <remarks>The reference applies no weighting to the prompt tokens; the byT5 stream reads only quoted text.</remarks>
    public Diffusion.Prompting.PromptWeightingMode PromptWeighting => Diffusion.Prompting.PromptWeightingMode.None;

    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.BlockStreaming;

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // MovieRollPipeline decodes on the DiT's backend. A separate VAE backend would need a host hop the pipeline does not do yet, so it is
        // refused here by name rather than failing on the first generation.
        if (!ReferenceEquals(context.VaeBackendOrDefault, context.Backend))
            throw new NotSupportedException("MovieRoll decodes on the DiT backend only; a separate VAE backend is not wired.");

        List<IDisposable> loaders = new List<IDisposable>();
        HunyuanVideo15Dit? dit = null;
        HunyuanVideo15Vae? vae = null;
        LlamaStyleEncoder? llama = null;
        try
        {
            Logs.Info($"[MovieRollRecipe] Loading + converting DiT: {Path.GetFileName(context.CheckpointPath)}.");
            // The converted DiT weights can alias the container's memory, so the container lives as long as the pipeline.
            CheckpointSource ditSource = CheckpointSource.Open(context.CheckpointPath);
            loaders.Add(ditSource);
            HunyuanVideo15Config config = HunyuanVideo15Config.Full;
            Dictionary<string, Tensor> ditWeights = HunyuanVideo15CheckpointConverter.Convert(
                new Dictionary<string, Tensor>(ditSource.Weights, StringComparer.Ordinal), config.NumBlocks, config.HiddenSize);
            if (ditWeights.Count == 0)
                throw new InvalidOperationException($"MovieRoll checkpoint '{context.CheckpointPath}' has no recognized DiT weights after conversion.");
            // DisableCacheWeightCasts turns off the device-side cache of per-dtype weight casts: each cached cast would be a second device copy of a
            // block the DiT streams. The BF16 to F16 cast below is made once on the host, and that copy is what the DiT keeps.
            RecipeBackendFlags.DisableCacheWeightCasts(context, "MovieRollRecipe");
            // BF16 -> F16 for the native F16 GEMM, as the HunyuanVideo recipe does.
            foreach (string k in ditWeights.Keys.ToList())
            {
                if (ditWeights[k].DType == DType.BF16)
                    ditWeights[k] = ditWeights[k].CastTo(DType.F16);
            }

            dit = new HunyuanVideo15Dit(config);
            dit.LoadWeights(ditWeights);

            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.MovieRollVae, onProgress: null, context.Cancel).GetAwaiter().GetResult();
            SafeTensorsLoader vaeLoader = new SafeTensorsLoader();
            vaeLoader.Load(vaePath);
            loaders.Add(vaeLoader);
            // The VAE stays fp32. Its layers make their own F32 copies of the weights, so an F16 cast would save nothing and only lose precision.
            vae = new HunyuanVideo15Vae(HunyuanVideo15VaeConfig.Default);
            vae.LoadWeights(vaeLoader.GetAllTensors());

            string qwenPath = ModelDownloader.EnsureSideModelAsync(SideModels.Qwen25Vl7BHunyuan, onProgress: null, context.Cancel).GetAwaiter().GetResult();
            SafeTensorsLoader qwenLoader = new SafeTensorsLoader();
            qwenLoader.Load(qwenPath);
            loaders.Add(qwenLoader);
            llama = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen2_5_VL_7B);
            llama.LoadWeights(qwenLoader.GetAllTensors());
            HunyuanImageQwenTextEncoder qwen = new HunyuanImageQwenTextEncoder(llama);

            // The glyph encoder loads on the first prompt that quotes text, as the HunyuanImage recipe does.
            Func<CancellationToken, (T5TextEncoder, IDisposable)> glyphFactory = cancel =>
            {
                string byt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.ByT5SmallGlyphXl, onProgress: null, cancel).GetAwaiter().GetResult();
                SafeTensorsLoader byt5Loader = new SafeTensorsLoader();
                try
                {
                    byt5Loader.Load(byt5Path);
                    T5TextEncoder byt5 = new T5TextEncoder(T5TextEncoderConfig.ByT5SmallGlyph);
                    byt5.LoadWeights(byt5Loader.GetAllTensors());
                    Logs.Info("[MovieRollRecipe] Glyph-ByT5-small loaded for quoted prompt text.");
                    return (byt5, byt5Loader);
                }
                catch
                {
                    // A failed load must not leave the container mapped: the pipeline never receives it to dispose.
                    byt5Loader.Dispose();
                    throw;
                }
            };

            // The window is one chunk per denoising step (the DMD rollout requires window == len(denoising_step_list)).
            MovieRollPipeline pipeline = new MovieRollPipeline(context.Backend, dit!, vae!,
                MovieRollSchedule.Distilled(DenoisingSteps, windowSize: DenoisingSteps.Length, TimestepShift));
            Logs.Info("[MovieRollRecipe] MovieRoll ready (distilled text-to-video).");
            return new MovieRollRecipePipeline(context.Backend, pipeline, dit!, vae!, llama!, qwen, new Qwen2Tokenizer(), glyphFactory, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[MovieRollRecipe] Construction failed.", ex);
            // Everything built so far, in the order it was built, so a failed construction leaves no weights resident.
            llama?.Dispose();
            vae?.Dispose();
            dit?.Dispose();
            foreach (IDisposable loader in loaders)
                loader.Dispose();
            throw;
        }
    }
}
