using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>LongCat-Video, the base family: text-to-video and image-to-video on the 13.6B DiT, flow-match Euler with CFG-Zero* guidance
/// (<see cref="LongCatVideoPipeline"/>). Long video, video continuation and the 720p refinement are not wired here yet.</summary>
public sealed class LongCatVideoRecipe : IVideoRecipe
{
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

    /// <summary>The base family's sampling settings: 50 steps, shift 12 (research plan, sampler variants). The guidance scale and the
    /// output frame rate are not recorded in the research plan yet; they are confirmed against the pinned reference at the real-weight run (B8).</summary>
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
