using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.LongCat;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>The driving features of 16 kHz mono speech at a video frame rate, <c>[frames, 5, 1280]</c> (Whisper-large-v3's pooled states).</summary>
internal delegate float[] LongCatDrivingFeatures(float[] speech16k, int fps);

/// <summary>LongCat-Video-Avatar, the 1.5 and 1.6 layout: speech-driven video from one portrait (<see cref="LongCatAvatarMode.Ai2v"/>) or from
/// speech alone (<see cref="LongCatAvatarMode.At2v"/>). The driving speech goes through Whisper-large-v3 and the avatar audio projection in
/// every block. One 93-frame segment at 25 fps; a longer request runs as a chain of segments. With <c>VideoRequest.Speakers</c>, two people in
/// the portrait are each driven by their own speech (<see cref="LongCatSpeakers"/>), on the same weights: 1.5 and 1.6 ship no separate
/// multi-speaker checkpoint. Avatar 1.0, whose multi-speaker weights are the separate <c>avatar_multi</c> folder, is not built.</summary>
/// <remarks>Likeness warning: the output is a generated talking person. Use it only with the consent of the person depicted.</remarks>
public sealed class LongCatAvatarRecipe : IVideoRecipe, IVideoDeviceCheck
{
    /// <summary>Refuses the CFG-parallel, DiT-shard and context-parallel placements before the memory report (see IArchitectureRecipe).</summary>
    public bool RefusesUnhonouredPlacements => true;

    private readonly LongCatSideModels? _sides;
    private readonly LongCatDrivingFeatures? _drivingFeatures;

    /// <summary>The production recipe: umT5, the VAE and Whisper come from the engine's downloads.</summary>
    public LongCatAvatarRecipe() : this(null, null)
    {
    }

    /// <summary>A recipe whose umT5 and VAE come from <paramref name="sides"/> and whose audio features come from
    /// <paramref name="drivingFeatures"/> (null for the downloads); the tests inject stand-ins here.</summary>
    internal LongCatAvatarRecipe(LongCatSideModels? sides, LongCatDrivingFeatures? drivingFeatures)
    {
        _sides = sides;
        _drivingFeatures = drivingFeatures;
    }

    /// <summary>Upper bound of one segment; the avatar DiT and its audio windows are built for it.</summary>
    internal const int SegmentFrames = 93;

    /// <summary>The frame rate the Whisper windows are aligned to (the reference's 25 fps).</summary>
    public const int OutputFps = 25;

    /// <summary>The longest chain the recipe accepts at 832x480, in frames. Each segment is decoded once, on its own, and its output frames are kept as
    /// bytes until the clip is written: at this size the kept frames are about 288 MB, and one segment's decode peaks near 445 MB of F32 RGB. The budget
    /// is in pixels, so other sizes get proportionally fewer frames. The decoded segments cover whole segments, so the clip can run a little past the
    /// request before it is trimmed. Kept at 241 for now: no measurement says a larger kept-frame set is safe on the target machines.</summary>
    internal const int MaxChainFrames = 241;

    /// <summary>The checks a request is refused on before any weight is read, each with its stable code.</summary>
    internal static void ValidateChainRequest(int? chainTotal, int frames, int width, int height, int fps)
    {
        if (ChainIssue(chainTotal, frames, width, height, fps) is { } issue)
            throw new VideoChainException(issue.Code, issue.Message, issue.Field);
    }

    /// <summary>The first chain problem of a request, or null. Planning reports it before any weight is read; the pipeline throws it.</summary>
    internal static (string Code, string Message, string Field)? ChainIssue(int? chainTotal, int frames, int width, int height, int fps)
    {
        if (chainTotal is int requested && !ChainWithinBudget(requested, width, height))
            return ("video.chain.budget",
                $"A LongCat-Video-Avatar chain of {requested} frames at {width}x{height} is over the budget of {MaxChainFrames} frames at 832x480 "
                + "(the same pixel count at other sizes).",
                "chainTotal");
        if (frames > SegmentFrames)
            return ("video.chain.segment_frames",
                $"One LongCat-Video-Avatar segment of {frames} frames is over the limit of {SegmentFrames}.", "frames");
        if (fps != OutputFps)
            return ("video.chain.fps",
                $"LongCat-Video-Avatar's audio windows are built for {OutputFps} fps; the request asked for {fps}.", "fps");
        return null;
    }

    /// <summary>Planning refusals: a request that can only fail after the weights are loaded (no driving speech; a bad chain).</summary>
    /// <inheritdoc/>
    public string? RequestRefusal(VideoRequest request, VideoEffectiveSettings settings)
    {
        if (request.Speakers is { } speakers)
        {
            if (LongCatAvatarSpeakers.Refusal(request, speakers) is { } refusal)
                return refusal;
        }
        else if (request.VideoAudioReference is null)
        {
            return "LongCat-Video-Avatar needs driving speech in VideoRequest.VideoAudioReference (or two speakers in VideoRequest.Speakers); "
                + "the avatar has no speech-free path.";
        }
        return ChainIssue(request.ChainTotalFrames, settings.Frames, settings.Width, settings.Height, settings.Fps) is { } issue
            ? issue.Message
            : null;
    }

    /// <summary>True when the driving audio is shorter than the clip: the rest of the clip is encoded as silence.</summary>
    internal static bool DrivingAudioIsShort(double seconds, int frames, int fps) => seconds + 1e-6 < frames / (double)fps;

    /// <summary>Whether a chain of <paramref name="totalFrames"/> frames at <paramref name="width"/>×<paramref name="height"/> fits the decode budget.</summary>
    internal static bool ChainWithinBudget(int totalFrames, int width, int height) =>
        (long)totalFrames * width * height <= (long)MaxChainFrames * 832 * 480;

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
    /// <remarks>The init image is the identity reference (the first latent frame). The driving speech is <c>VideoRequest.VideoAudioReference</c>,
    /// or per speaker in <c>VideoRequest.Speakers</c>.</remarks>
    public VideoFeatures Supports => VideoFeatures.InitImage | VideoFeatures.DrivingAudio | VideoFeatures.LongFormChain | VideoFeatures.MultiSpeaker;

    /// <inheritdoc/>
    public bool Matches(string familyId) => LongCatAvatarVariants.RowIds.Contains(familyId, StringComparer.OrdinalIgnoreCase);

    /// <summary>The default build's settings (1.5, the catalog row's weights): the distilled path, 8 steps, guidance off (1.0), shift 7. Steps
    /// and guidance are fixed by the distillation, so the planner locks them.</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults
    {
        Steps = 8, CfgScale = 1.0f, Width = 832, Height = 480, Frames = SegmentFrames, Fps = OutputFps,
        FlowShift = LongCatAvatarPipelineSettings.Shift,
        LockedFields = VideoLockedFields.Steps | VideoLockedFields.CfgScale,
    };

    /// <inheritdoc/>
    /// <remarks>The 1.6 build is the same 8-step distilled path as 1.5 (upstream <c>--use_distill</c>: 8 steps, guidance 1.0), with its two LoRAs
    /// mixed per step. It has no undistilled settings: the upstream documents none for 1.6.</remarks>
    public VideoDefaults DefaultsFor(ResolvedModelVariant? variant) => Defaults;

    /// <inheritdoc/>
    /// <remarks>One segment: 1 + 4k frames up to <see cref="SegmentFrames"/>.</remarks>
    public VideoConstraints Constraints { get; } = new VideoConstraints
    {
        Frames = FrameGrid.StepPlusOne(LongCatRecipeSupport.TemporalCompressionFactor, FrameNormalization.SnapNearest, SegmentFrames),
    };

    /// <inheritdoc/>
    public MemoryCapabilities MemorySupports => MemoryCapabilities.BlockStreaming;

    /// <inheritdoc/>
    string? IVideoDeviceCheck.DeviceRefusal(IBackend backend, string checkpointPath) =>
        LongCatWeightFit.Refusal(backend, LongCatRecipeSupport.CheckpointDirectory(checkpointPath));

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // Both builds run the 8-step distilled path. The 1.6 build mixes its DMD (loraA) and OPSD (loraB) LoRAs per step (upstream enable_lora_mix).
        ResolvedModelVariant variant = context.ResolveVariant(LongCatAvatarVariants.Catalog);
        bool is16 = variant.Is(LongCatAvatarVariants.Avatar16);
        // The distilled build is the base plus the DMD LoRA gated on. Without that file the request would run the bare base at 8 steps, so it
        // is refused here, before the checkpoint is opened.
        // A checkpoint with the 1.6 layout (loraA/loraB) is refused by the 1.5 build: it would run 1.6's weights with 1.5's settings.
        if (!is16 && File.Exists(LongCatRecipeSupport.Avatar16LoraFiles(context.CheckpointPath).DmdPath))
        {
            throw new NotSupportedException($"'{context.CheckpointPath}' has the 1.6 layout (loraA and loraB). Run it as longcat-avatar-1.6 "
                + "(longcat-avatar:1.6 from the CLI), not as the 1.5 build.");
        }
        if (!is16 && !File.Exists(LongCatRecipeSupport.DmdLoraPath(context.CheckpointPath)))
        {
            throw new NotSupportedException($"longcat-avatar's distilled build needs its distillation LoRA at "
                + $"'{LongCatRecipeSupport.DmdLoraPath(context.CheckpointPath)}'. Refusing to run the bare base at 8 steps.");
        }

        List<IDisposable> loaders = new List<IDisposable>();
        try
        {
            LongCatRecipeSupport.LongCatLoraFiles? mixed = is16 ? LongCatRecipeSupport.Avatar16LoraFiles(context.CheckpointPath) : null;
            LongCatParts parts = LongCatRecipeSupport.LoadParts(context, loaders, _sides, attachDmd: !is16, mixedLoras: mixed);
            LongCatAvatarLoraSchedule? loraMix = is16
                ? new LongCatAvatarLoraSchedule(parts.DmdGate!, parts.OpsdGate!, LongCatAvatarVariants.Avatar16MixAlpha, LongCatAvatarVariants.Avatar16MixSteps)
                : null;
            if (!parts.HasAudio)
            {
                throw new InvalidOperationException(
                    $"'{context.CheckpointPath}' has no audio_proj weights, so it is not a LongCat-Video-Avatar checkpoint. Point longcat-avatar at the avatar DiT.");
            }

            LongCatDrivingFeatures drivingFeatures = _drivingFeatures ?? LoadWhisperFeatures(context, loaders);

            LongCatAvatarPipeline pipeline = new LongCatAvatarPipeline(context.Backend, parts.Dit);
            Logs.Info("[LongCatAvatarRecipe] LongCat-Video-Avatar ready (speech and image to video, one segment).");
            // The reference normalises the speech's loudness on 1.5 and leaves it alone on 1.6.
            return new LongCatAvatarRecipePipeline(context.Backend, pipeline, parts, drivingFeatures, normalizeSpeechLoudness: !is16, loaders,
                distilled: true, is16 ? null : parts.DmdGate, loraMix);
        }
        catch (Exception ex)
        {
            Logs.Error("[LongCatAvatarRecipe] Construction failed.", ex);
            foreach (IDisposable loader in loaders)
                loader.Dispose();
            throw;
        }
    }

    /// <summary>The production features: the Whisper-large-v3 copy the audio stack already caches (stt/openai--whisper-large-v3), at the pinned
    /// upstream revision. What it loads joins <paramref name="loaders"/>.</summary>
    private static LongCatDrivingFeatures LoadWhisperFeatures(RecipeContext context, List<IDisposable> loaders)
    {
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
        IBackend backend = context.Backend;
        return (speech16k, fps) =>
        {
            // Whisper is only needed for these features. Its device copies are released as soon as they are read, so they do not sit in VRAM
            // through the denoise (the pipeline is cached, and the next request uploads them again on demand).
            try
            {
                return LongCatWhisperFeatures.Extract(backend, whisper, mel, speech16k, fps);
            }
            finally
            {
                backend.FreeWeights(whisper.EnumerateWeights());
            }
        };
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
