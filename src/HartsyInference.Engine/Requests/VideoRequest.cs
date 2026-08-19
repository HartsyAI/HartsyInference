namespace HartsyInference.Engine.Requests;

/// <summary>Native, transport-agnostic text/image-to-video request. Carries the common generation props plus the
/// video-specific model selections, framing, trimming, and audio-track inputs the backend reads.</summary>
public sealed record VideoRequest
{
    /// <summary>The positive prompt.</summary>
    public required string Prompt { get; init; }

    /// <summary>The negative prompt, or null/empty for none.</summary>
    public string? NegativePrompt { get; init; }

    /// <summary>Output width in pixels; null uses the family's native training width.</summary>
    public int? Width { get; init; }

    /// <summary>Output height in pixels; null uses the family's native training height.</summary>
    public int? Height { get; init; }

    /// <summary>Number of denoising steps; null uses the family's officially recommended step count.</summary>
    public int? Steps { get; init; }

    /// <summary>Classifier-free guidance scale; null uses the family's officially recommended scale.</summary>
    public float? CfgScale { get; init; }

    /// <summary>RNG seed; negative means a random seed is chosen per request.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>Optional start/init image for image-to-video.</summary>
    public ImageData? InitImage { get; init; }

    /// <summary>Primary video model id or local path; null uses the loaded model.</summary>
    public string? VideoModel { get; init; }

    /// <summary>Second (low-noise) expert model for dual-expert families (Wan 2.2 A14B); null for none.</summary>
    public string? VideoSwapModel { get; init; }

    /// <summary>Fraction (0..1) of steps run by the swap model (schedule tail); null uses the family's official boundary.</summary>
    public double? VideoSwapPercent { get; init; }

    /// <summary>Target output resolution label (e.g. "720p"); null uses width/height.</summary>
    public string? VideoResolution { get; init; }

    /// <summary>Output frames per second; null uses the family's native frame rate.</summary>
    public int? Fps { get; init; }

    /// <summary>Container/codec format label for encoding (e.g. "mp4", "webp"); null uses the default.</summary>
    public string? VideoFormat { get; init; }

    /// <summary>Whether to boomerang (forward then reverse) the output.</summary>
    public bool VideoBoomerang { get; init; }

    /// <summary>Optional explicit end frame the sequence should land on.</summary>
    public ImageData? VideoEndFrame { get; init; }

    /// <summary>Optional audio track (encoded bytes) to mux into the output.</summary>
    public AudioClip? VideoAudioInput { get; init; }

    /// <summary>Optional reference audio driving audio-conditioned video (e.g. speech-to-video).</summary>
    public AudioClip? VideoAudioReference { get; init; }

    /// <summary>Reference images a reference-conditioned family should carry identity/style from; null for none.
    /// Distinct from <see cref="InitImage"/>, which pins an actual frame rather than describing subject matter.</summary>
    public IReadOnlyList<ImageData>? ReferenceImages { get; init; }

    /// <summary>Reference clips, each with its own optional soundtrack; null for none.</summary>
    public IReadOnlyList<ReferenceVideo>? ReferenceVideos { get; init; }

    /// <summary>Standalone reference audio clips, not tied to any reference video; null for none.</summary>
    public IReadOnlyList<AudioClip>? ReferenceAudios { get; init; }

    /// <summary>Driving motion video for character-animation families (Wan-Animate); null falls back to tiling
    /// <see cref="InitImage"/> across frames.</summary>
    public VideoClip? DrivingVideo { get; init; }

    /// <summary>Pre-rendered pose/skeleton driving video; overrides auto-preprocessing for the pose branch.</summary>
    public VideoClip? DrivingPoseVideo { get; init; }

    /// <summary>Pre-cropped face-square driving video; overrides auto-preprocessing for the face branch.</summary>
    public VideoClip? DrivingFaceVideo { get; init; }

    /// <summary>Wan-Animate replacement mode: the background clip the character is composited into. The concat
    /// conditioning's generated frames carry this video instead of the mid-gray placeholder (ComfyUI
    /// <c>WanAnimateToVideo.background_video</c>).</summary>
    public VideoClip? DrivingBackgroundVideo { get; init; }

    /// <summary>Wan-Animate replacement mode: per-frame character mask (white = generate the character there,
    /// black = keep the background). A single-frame clip repeats over the whole video (ComfyUI
    /// <c>WanAnimateToVideo.character_mask</c>).</summary>
    public VideoClip? DrivingMaskVideo { get; init; }

    /// <summary>Auto-derive the pose skeleton and face crop from <see cref="DrivingVideo"/> (the format the
    /// checkpoint was trained on); off passes the raw clip to both branches.</summary>
    public bool DrivingAutoPreprocess { get; init; } = true;

    /// <summary>Total frames to generate for text-to-video; null uses the family's native frame count.</summary>
    public int? Frames { get; init; }

    /// <summary>Frames to trim from the start of the generated sequence.</summary>
    public int TrimVideoStartFrames { get; init; }

    /// <summary>Frames to trim from the end of the generated sequence.</summary>
    public int TrimVideoEndFrames { get; init; }

    /// <summary>Per-request swappable-component overrides; null keeps recipe defaults.</summary>
    public ComponentOverrides? Components { get; init; }

    /// <summary>LoRA stack to fuse; null for none.</summary>
    public LoraStack? Loras { get; init; }

    /// <summary>Arch-specific or host-registered params not named by the flat contract.</summary>
    public IReadOnlyDictionary<string, object> Extra { get; init; } = new Dictionary<string, object>();
}
