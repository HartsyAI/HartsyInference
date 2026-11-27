using HartsyInference.Core.Backends;
using HartsyInference.Core.Runtime;
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
public sealed class MovieRollRecipe : IVideoRecipe, IVideoDeviceCheck
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
    public bool RefusesUnhonouredPlacements => true;

    /// <summary>Why a backend cannot hold the released DiT at <paramref name="weightBytes"/>, or null. A backend with a streaming cache moves the blocks
    /// through the device and is not refused here. One without keeps every block on the device, and the DiT is held at its load dtype: FP32 off CUDA
    /// (the F16 cast is CUDA only), so a file larger than the usable device memory cannot run there.</summary>
    public static string? WeightsRefusal(bool streams, long totalBytes, long weightBytes, string backendName)
    {
        if (streams || totalBytes <= 0) return null;
        long usable = Math.Max(0, totalBytes - HartsyInference.Engine.Placement.PlacementPlanner.PerDeviceReserveBytes);
        if (weightBytes <= usable) return null;
        return $"MovieRoll cannot run on {backendName}: this backend has no weight streaming, so every DiT block stays on the device, and the DiT is held at its "
            + $"FP32 load dtype (the F16 cast is CUDA only), {HartsyInference.Core.Memory.Vram.ByteFormat.GbF1(weightBytes)}. The device has "
            + $"{HartsyInference.Core.Memory.Vram.ByteFormat.GbF1(usable)} usable of {HartsyInference.Core.Memory.Vram.ByteFormat.GbF1(totalBytes)}. "
            + "The CUDA backend streams the blocks.";
    }

    /// <inheritdoc/>
    string? IVideoDeviceCheck.DeviceRefusal(IBackend backend, string checkpointPath)
    {
        if (!File.Exists(checkpointPath)) return null;
        long bytes = new FileInfo(checkpointPath).Length;
        return WeightsRefusal(backend.StreamingCache is not null, backend.GetVramInfo().TotalBytes, bytes, backend.Capabilities.Name);
    }

    /// <summary>The source dtypes the DiT casts to F16 at load on this backend. BF16 is cast on every backend, as before; FP32 (the released file's
    /// dtype) is cast on CUDA only, because the F16 path for FP32 weights is proven there and not on the CPU backend, which keeps them.</summary>
    public static IReadOnlyList<DType> DitHalfCastSources(IBackend backend) =>
        backend is HartsyInference.Cuda.CudaBackend ? [DType.BF16, DType.F32] : [DType.BF16];

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // A text-encoder or VAE placement on another device is reported and run on the primary backend, as every recipe that does not honour it does.
        UnhonouredBackends.Refuse(context, "MovieRoll", warn: false);

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
            // block the DiT streams. The BF16 to F16 copies below are made once on the host, and they are what the DiT keeps.
            RecipeBackendFlags.DisableCacheWeightCasts(context, "MovieRollRecipe");
            // F16 for the native F16 GEMM: BF16 on every backend, FP32 (the released file) on CUDA only. Only the cast copies are owned here; the rest
            // are the container's.
            Dictionary<string, Tensor> ditF16 = VaePrecisionHelper.CastWeights(ditWeights, DitHalfCastSources(context.Backend), DType.F16);
            HashSet<Tensor> converted = new(ditWeights.Values, ReferenceEqualityComparer.Instance);
            foreach (Tensor copy in ditF16.Values)
            {
                if (!converted.Contains(copy)) loaders.Add(copy);
            }

            dit = new HunyuanVideo15Dit(config);
            dit.LoadWeights(ditF16);

            string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.MovieRollVae, onProgress: null, context.Cancel).GetAwaiter().GetResult();
            // The VAE stays fp32. Its layers make their own F32 copies of the weights, so an F16 cast would save nothing and only lose precision.
            // The container is released as soon as the layers have copied it: the VAE holds no reference into the mapped file after LoadWeights.
            using (SafeTensorsLoader vaeLoader = new SafeTensorsLoader())
            {
                vaeLoader.Load(vaePath);
                vae = new HunyuanVideo15Vae(HunyuanVideo15VaeConfig.Default);
                vae.LoadWeights(vaeLoader.GetAllTensors());
            }

            HunyuanImageQwenTextEncoder qwen;
            (llama, qwen) = HunyuanTextEncoders.LoadQwen25Vl(context, loaders);

            // The glyph encoder loads on the first prompt that quotes text, as the HunyuanImage recipe does.
            Func<CancellationToken, (T5TextEncoder, IDisposable)> glyphFactory = HunyuanTextEncoders.LoadGlyphByt5;

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
            new CompositeDisposable([llama, vae, dit, .. loaders]).Dispose();
            throw;
        }
    }
}
