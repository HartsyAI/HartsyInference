using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>LongCat-Video, the base family: text-to-video and image-to-video on the 13.6B DiT, flow-match Euler with CFG-Zero* guidance
/// (<see cref="LongCatVideoPipeline"/>). The distilled sibling is <see cref="LongCatVideoDistilledRecipe"/>. Long video, video continuation and
/// the 720p refinement are not wired here yet.</summary>
public sealed class LongCatVideoRecipe : IVideoRecipe, IVideoDeviceCheck
{
    /// <summary>Refuses the CFG-parallel, DiT-shard and context-parallel placements before the memory report (see IArchitectureRecipe).</summary>
    public bool RefusesUnhonouredPlacements => true;

    private readonly LongCatSideModels? _sides;

    /// <summary>The production recipe: umT5 and the VAE come from the engine's side-model downloads.</summary>
    public LongCatVideoRecipe() : this(null)
    {
    }

    /// <summary>A recipe whose umT5 and VAE come from <paramref name="sides"/> (null for the downloads); the tests inject stand-ins here.</summary>
    internal LongCatVideoRecipe(LongCatSideModels? sides)
    {
        _sides = sides;
    }

    /// <inheritdoc/>
    public string Name => "longcat-video";

    /// <inheritdoc/>
    public PromptProfile? Prompt { get; } = new()
    {
        Kind = "video",
        Encoders = ["t5"],
    };

    /// <inheritdoc/>
    /// <remarks>The init image becomes the first latent frame (<see cref="LongCatConditioningMode.Image"/>).</remarks>
    public VideoFeatures Supports => VideoFeatures.InitImage;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "longcat-video", StringComparison.OrdinalIgnoreCase);

    /// <summary>The base family's sampling settings, the pinned demos' defaults: 50 steps, guidance 4.0, 832x480, 93 frames, 15 fps
    /// (run_demo_text_to_video.py:79-90, run_demo_image_to_video.py:86-98 @ 58eb336), shift 12 (LongCatRecipeSupport.BaseFlowShift).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults
    {
        Steps = 50, CfgScale = 4.0f, Width = 832, Height = 480, Frames = 93, Fps = 15, FlowShift = LongCatRecipeSupport.BaseFlowShift,
    };

    /// <inheritdoc/>
    /// <remarks>The VAE compresses time 4x, so a clip is 1 + 4k frames.</remarks>
    public VideoConstraints Constraints { get; } = new VideoConstraints
    {
        Frames = FrameGrid.StepPlusOne(4, FrameNormalization.SnapNearest),
    };

    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.BlockStreaming;

    /// <inheritdoc/>
    string? IVideoDeviceCheck.DeviceRefusal(IBackend backend, string checkpointPath) =>
        LongCatWeightFit.Refusal(backend, LongCatRecipeSupport.CheckpointDirectory(checkpointPath));

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        List<IDisposable> loaders = new List<IDisposable>();
        try
        {
            LongCatParts parts = LongCatRecipeSupport.LoadParts(context, loaders, _sides, attachDmd: false);
            LongCatVideoPipeline pipeline = new LongCatVideoPipeline(context.Backend, parts.Dit);
            Logs.Info("[LongCatVideoRecipe] LongCat-Video ready (text and image to video).");
            return new LongCatVideoRecipePipeline(context.Backend, pipeline, parts, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[LongCatVideoRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
                loader.Dispose();
            throw;
        }
    }
}
