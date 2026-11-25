using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Variants;
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

    /// <summary>The audio front end: Whisper-large-v3 from its upstream repo, at the revision the catalog row pins (the 1.5 repo bundles the same file).</summary>
    internal const string WhisperRepo = "openai/whisper-large-v3";

    /// <summary>The Whisper revision (<c>openai/whisper-large-v3</c> @ 06f233fe, its model.safetensors sha256 a8e94b85…).</summary>
    internal const string WhisperRevision = "06f233fe06e710322aca913c1bc4249a0d71fce1";

    /// <inheritdoc/>
    public ModelVariantCatalog? Variants => LongCatAvatarVariants.Catalog;

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
    public VideoFeatures Supports => VideoFeatures.InitImage | VideoFeatures.DrivingAudio | VideoFeatures.LongFormChain;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, "longcat-avatar", StringComparison.OrdinalIgnoreCase);

    /// <summary>The default build's settings (1.5, the catalog row's weights): the distilled path, 8 steps, guidance off (1.0), shift 7. Steps
    /// and guidance are fixed by the distillation, so the planner locks them.</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults
    {
        Steps = 8, CfgScale = 1.0f, Width = 832, Height = 480, Frames = SegmentFrames, Fps = OutputFps,
        FlowShift = LongCatAvatarPipelineSettings.Shift,
        LockedFields = VideoLockedFields.Steps | VideoLockedFields.CfgScale,
    };

    /// <summary>The 1.6 undistilled settings: 50 steps, shift 7, text guidance 4 (research plan). They apply only once a 1.6 INT8 catalog row
    /// exists; the recipe refuses the 1.6 build until then.</summary>
    public static VideoDefaults Undistilled16Defaults { get; } = new VideoDefaults
    {
        Steps = 50, CfgScale = 4.0f, Width = 832, Height = 480, Frames = SegmentFrames, Fps = OutputFps,
        FlowShift = LongCatAvatarPipelineSettings.Shift,
    };

    /// <inheritdoc/>
    /// <remarks>Only a named 1.6 build takes the 1.6 settings; the default and a named 1.5 build take the distilled ones.</remarks>
    public VideoDefaults DefaultsFor(ResolvedModelVariant? variant) =>
        variant is not null && variant.Is(LongCatAvatarVariants.Avatar16) ? Undistilled16Defaults : Defaults;

    /// <inheritdoc/>
    /// <remarks>One segment: 1 + 4k frames up to <see cref="SegmentFrames"/>.</remarks>
    public VideoConstraints Constraints { get; } = new VideoConstraints
    {
        Frames = FrameGrid.StepPlusOne(LongCatRecipeSupport.TemporalCompressionFactor, FrameNormalization.SnapNearest, SegmentFrames),
    };

    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.BlockStreaming;

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // The 1.6 settings apply only with a 1.6 INT8 catalog row, which does not exist yet; a 1.6 request is refused before any weight is read.
        // The default and a named 1.5 build run the distilled path with the 1.5 settings.
        ResolvedModelVariant variant = context.ResolveVariant(LongCatAvatarVariants.Catalog);
        if (variant.Is(LongCatAvatarVariants.Avatar16))
        {
            throw new NotSupportedException("longcat-avatar's 1.6 build runs only once a 1.6 INT8 catalog row exists; none does yet. "
                + "The 1.5 build (the default) runs instead.");
        }
        bool distilled = variant.Is(LongCatAvatarVariants.Avatar15);
        // The distilled build is the base plus the DMD LoRA gated on. Without that file the request would run the bare base at 8 steps, so it
        // is refused here, before the checkpoint is opened.
        if (distilled && !File.Exists(LongCatRecipeSupport.DmdLoraPath(context.CheckpointPath)))
        {
            throw new NotSupportedException($"longcat-avatar's distilled build needs its distillation LoRA at "
                + $"'{LongCatRecipeSupport.DmdLoraPath(context.CheckpointPath)}'. Refusing to run the bare base at 8 steps.");
        }

        List<IDisposable> loaders = new List<IDisposable>();
        try
        {
            LongCatParts parts = LongCatRecipeSupport.LoadParts(context, loaders, sides: null, attachDmd: distilled);
            if (!parts.HasAudio)
            {
                throw new InvalidOperationException(
                    $"'{context.CheckpointPath}' has no audio_proj weights, so it is not a LongCat-Video-Avatar checkpoint. Point longcat-avatar at the avatar DiT.");
            }

            // The Whisper-large-v3 copy the audio stack already caches (stt/openai--whisper-large-v3), at the pinned upstream revision.
            IReadOnlyDictionary<string, string> fetched = AudioModelCache.FetchAllAsync(WhisperRepo, [new AudioModelFile("model.safetensors")],
                category: "stt", revision: WhisperRevision, ct: context.Cancel).GetAwaiter().GetResult();
            SafeTensorsLoader whisperLoader = new SafeTensorsLoader();
            whisperLoader.Load(fetched["model.safetensors"]);
            loaders.Add(whisperLoader);
            // Only the encoder's tensors go to the encoder; the decoder's are never read.
            Dictionary<string, Tensor> encoderWeights = whisperLoader.GetAllTensors()
                .Where(pair => pair.Key.StartsWith("model.encoder.", StringComparison.Ordinal))
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
            WhisperEncoder whisper = new WhisperEncoder(WhisperConfig.LargeV3);
            whisper.LoadWeights(encoderWeights);
            loaders.Add(whisper);
            MelSpectrogramExtractor mel = new MelSpectrogramExtractor(MelSpectrogramExtractor.WhisperConfig(WhisperConfig.LargeV3.NumMelBins));

            LongCatAvatarPipeline pipeline = new LongCatAvatarPipeline(context.Backend, parts.Dit);
            Logs.Info("[LongCatAvatarRecipe] LongCat-Video-Avatar ready (speech and image to video, one segment).");
            return new LongCatAvatarRecipePipeline(context.Backend, pipeline, parts, whisper, mel, loaders, distilled, parts.DmdGate);
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
