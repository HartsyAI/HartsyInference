using HartsyInference.Engine.Planning;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>LongCat-Video distilled: the base DiT with the cfg_step LoRA, text-to-video and image-to-video in 16 steps at guidance 1.0
/// (<see cref="LongCatVideoPipeline"/>, distilled schedule). The undistilled base is <see cref="LongCatVideoRecipe"/>.</summary>
public sealed class LongCatVideoDistilledRecipe : IVideoRecipe, IVideoDeviceCheck
{
    /// <summary>Refuses the CFG-parallel, DiT-shard and context-parallel placements before the memory report (see IArchitectureRecipe).</summary>
    public bool RefusesUnhonouredPlacements => true;

    private readonly LongCatSideModels? _sides;

    /// <summary>The production recipe: umT5 and the VAE come from the engine's side-model downloads.</summary>
    public LongCatVideoDistilledRecipe() : this(null)
    {
    }

    /// <summary>A recipe whose umT5 and VAE come from <paramref name="sides"/> (null for the downloads); the tests inject stand-ins here.</summary>
    internal LongCatVideoDistilledRecipe(LongCatSideModels? sides)
    {
        _sides = sides;
    }

    /// <summary>The catalog row and family id of the distilled build.</summary>
    public const string DistilledFamilyId = "longcat-video-distilled";

    /// <inheritdoc/>
    public string Name => DistilledFamilyId;

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
    public bool Matches(string familyId) => string.Equals(familyId, DistilledFamilyId, StringComparison.OrdinalIgnoreCase);

    /// <summary>The distilled sampling settings of the demo's <c>use_distill</c> path: 16 steps with guidance 1.0 (no CFG), the cfg_step LoRA
    /// on (run_demo_text_to_video.py, i2v and vc demos; LongCat-Video @ 58eb336 avatar facts). Size, frames and fps follow the base family.</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults
    {
        Steps = 16, CfgScale = 1.0f, Width = 832, Height = 480, Frames = 93, Fps = 15, FlowShift = LongCatRecipeSupport.BaseFlowShift,
        LockedFields = VideoLockedFields.Steps | VideoLockedFields.CfgScale,
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
        // Refused before any weight is read: the distilled build without its cfg_step LoRA would run the undistilled base as the distilled model.
        LongCatRecipeSupport.RequireCfgStepLora(context.CheckpointPath);
        List<IDisposable> loaders = new List<IDisposable>();
        try
        {
            LongCatParts parts = LongCatRecipeSupport.LoadParts(context, loaders, _sides, attachDmd: false, attachCfgStep: true);
            LongCatVideoPipeline pipeline = new LongCatVideoPipeline(context.Backend, parts.Dit);
            Logs.Info("[LongCatVideoDistilledRecipe] LongCat-Video distilled ready (16 steps, cfg_step LoRA).");
            return new LongCatVideoRecipePipeline(context.Backend, pipeline, parts, loaders, distilled: true);
        }
        catch (Exception ex)
        {
            Logs.Error("[LongCatVideoDistilledRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
                loader.Dispose();
            throw;
        }
    }
}
