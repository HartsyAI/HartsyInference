namespace HartsyInference.Engine.Requests;

/// <summary>Native, transport-agnostic text/image-to-video request. Carries the common generation props plus the
/// video-specific model selections, framing, trimming, and audio-track inputs the backend reads.</summary>
public sealed record VideoRequest
{
    /// <summary>The positive prompt.</summary>
    public required string Prompt { get; init; }

    /// <summary>The negative prompt, or null/empty for none.</summary>
    public string? NegativePrompt { get; init; }

    /// <summary>Output width in pixels.</summary>
    public int Width { get; init; } = 704;

    /// <summary>Output height in pixels.</summary>
    public int Height { get; init; } = 480;

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

    /// <summary>Model used for a mid-sequence swap pass; null for none.</summary>
    public string? VideoSwapModel { get; init; }

    /// <summary>Fraction (0..1) of the sequence at which the swap model takes over.</summary>
    public double? VideoSwapPercent { get; init; }

    /// <summary>Model used to extend an existing clip; null for none.</summary>
    public string? VideoExtendModel { get; init; }

    /// <summary>Target output resolution label (e.g. "720p"); null uses width/height.</summary>
    public string? VideoResolution { get; init; }

    /// <summary>Output frames per second.</summary>
    public int Fps { get; init; } = 25;

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

    /// <summary>Total frames to generate for text-to-video.</summary>
    public int Frames { get; init; } = 25;

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
