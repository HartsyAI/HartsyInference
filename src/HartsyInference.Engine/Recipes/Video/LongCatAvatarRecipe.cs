using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Variants;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>LongCat-Video-Avatar, the 1.5 and 1.6 layout: speech-driven video from one portrait (<see cref="LongCatAvatarMode.Ai2v"/>) or from
/// speech alone (<see cref="LongCatAvatarMode.At2v"/>). The driving speech goes through Whisper-large-v3 and the avatar audio projection in
/// every block. One 93-frame segment at 25 fps; the segment loop, the 8-step distilled path and multi-speaker are not wired here yet.</summary>
/// <remarks>Likeness warning: the output is a generated talking person. Use it only with the consent of the person depicted.</remarks>
public sealed class LongCatAvatarRecipe : IVideoRecipe
{
    /// <summary>Upper bound of one segment; the avatar DiT and its audio windows are built for it.</summary>
    internal const int SegmentFrames = 93;

    /// <summary>The frame rate the Whisper windows are aligned to (the reference's 25 fps).</summary>
    internal const int OutputFps = 25;

    /// <inheritdoc/>
    public string Name => "longcat-avatar";

    /// <inheritdoc/>
    public PromptProfile? Prompt { get; } = new()
    {
        Kind = "video",
        Encoders = ["t5"],
    };

    /// <inheritdoc/>
    /// <remarks>The init image is the identity reference (the first latent frame). The driving speech is <c>VideoRequest.VideoAudioReference</c>.</remarks>
    public VideoFeatures Supports => VideoFeatures.InitImage | VideoFeatures.DrivingAudio;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "longcat-avatar", StringComparison.OrdinalIgnoreCase);

    /// <summary>The 1.6 undistilled sampler: 50 steps, shift 7, text guidance 4 (research plan, avatar sampler). The output frame rate is 25 fps.
    /// Whether the official default guidance differs is confirmed against the pinned reference at the real-weight run (B8).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults
    {
        Steps = 50, CfgScale = 4.0f, Width = 832, Height = 480, Frames = SegmentFrames, Fps = OutputFps,
        FlowShift = LongCatAvatarPipelineSettings.Shift,
    };

    /// <inheritdoc/>
    /// <remarks>One segment: 1 + 4k frames up to <see cref="SegmentFrames"/>.</remarks>
    public VideoConstraints Constraints { get; } = new VideoConstraints
    {
        Frames = FrameGrid.StepPlusOne(LongCatRecipeSupport.TemporalCompressionFactor, FrameNormalization.SnapNearest, SegmentFrames),
    };

    /// <summary>The 1.5 distilled build: 8 steps, guidance off (1.0), the 1.5 scheduler's shift 7. Steps and guidance are fixed by the
    /// distillation, so the planner locks them.</summary>
    public static VideoDefaults DistilledDefaults { get; } = new VideoDefaults
    {
        Steps = 8, CfgScale = 1.0f, Width = 832, Height = 480, Frames = SegmentFrames, Fps = OutputFps,
        FlowShift = LongCatAvatarPipelineSettings.Shift,
        LockedFields = VideoLockedFields.Steps | VideoLockedFields.CfgScale,
    };

    /// <inheritdoc/>
    public ModelVariantCatalog? Variants => LongCatAvatarVariants.Catalog;

    /// <inheritdoc/>
    /// <remarks>Only a named build (a caller hint) imposes the 1.5 settings; a name-only or unnamed request keeps the 1.6 numbers.</remarks>
    public VideoDefaults DefaultsFor(ResolvedModelVariant? variant) =>
        variant is not null && variant.Is(LongCatAvatarVariants.Avatar15) && variant.IsDefinitive ? DistilledDefaults : Defaults;

    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.BlockStreaming;

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        List<IDisposable> loaders = new List<IDisposable>();
        try
        {
            LongCatParts parts = LongCatRecipeSupport.LoadParts(context, loaders, sides: null);
            if (!parts.HasAudio)
            {
                throw new InvalidOperationException(
                    $"'{context.CheckpointPath}' has no audio_proj weights, so it is not a LongCat-Video-Avatar checkpoint. Point longcat-avatar at the avatar DiT.");
            }

            string whisperPath = ModelDownloader.EnsureSideModelAsync(SideModels.WhisperLargeV3, onProgress: null, context.Cancel)
                .GetAwaiter().GetResult();
            SafeTensorsLoader whisperLoader = new SafeTensorsLoader();
            whisperLoader.Load(whisperPath);
            loaders.Add(whisperLoader);
            WhisperEncoder whisper = new WhisperEncoder(WhisperConfig.LargeV3);
            whisper.LoadWeights(whisperLoader.GetAllTensors());
            loaders.Add(whisper);
            MelSpectrogramExtractor mel = new MelSpectrogramExtractor(MelSpectrogramExtractor.WhisperConfig(WhisperConfig.LargeV3.NumMelBins));

            LongCatAvatarPipeline pipeline = new LongCatAvatarPipeline(context.Backend, parts.Dit);
            Logs.Info("[LongCatAvatarRecipe] LongCat-Video-Avatar ready (speech and image to video, one segment).");
            ResolvedModelVariant variant = context.ResolveVariant(LongCatAvatarVariants.Catalog);
            bool distilled = variant.Is(LongCatAvatarVariants.Avatar15) && variant.IsDefinitive;
            return new LongCatAvatarRecipePipeline(context.Backend, pipeline, parts, whisper, mel, loaders, distilled);
        }
        catch (Exception ex)
        {
            Logs.Error("[LongCatAvatarRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
                loader.Dispose();
            throw;
        }
    }
}

/// <summary>The avatar sampler's fixed values, shared by the recipe's defaults and its pipeline.</summary>
internal static class LongCatAvatarPipelineSettings
{
    /// <summary>Scheduler shift of the 1.5 and 1.6 checkpoints.</summary>
    internal const float Shift = 7f;

    /// <summary>Audio guidance of the three-pass CFG, the reference's default (the text guidance comes from the request).</summary>
    internal const float AudioGuidance = 4f;
}
