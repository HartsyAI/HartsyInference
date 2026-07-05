namespace HartsyInference.Cli.Infra;

/// <summary>Parsing and display helpers for <see cref="Modality"/>, bridging enum values and their CLI names.</summary>
public static class Modalities
{
    /// <summary>The CLI token for a modality, e.g. <see cref="Modality.ThreeD"/> renders as "3d".</summary>
    public static string ToCliName(Modality modality) => modality switch
    {
        Modality.Image => "image",
        Modality.Text => "text",
        Modality.Speech => "speech",
        Modality.Music => "music",
        Modality.Transcribe => "transcribe",
        Modality.Vision => "vision",
        Modality.Video => "video",
        Modality.ThreeD => "3d",
        Modality.Interactive => "interactive",
        _ => modality.ToString().ToLowerInvariant(),
    };

    /// <summary>Parses a CLI modality token (case-insensitive; accepts "3d" and "threed").</summary>
    public static bool TryParse(string? token, out Modality modality)
    {
        modality = default;
        if (string.IsNullOrWhiteSpace(token))
            return false;
        switch (token.Trim().ToLowerInvariant())
        {
            case "image": modality = Modality.Image; return true;
            case "text": modality = Modality.Text; return true;
            case "speech":
            case "speak": modality = Modality.Speech; return true;
            case "music": modality = Modality.Music; return true;
            case "transcribe": modality = Modality.Transcribe; return true;
            case "vision": modality = Modality.Vision; return true;
            case "video": modality = Modality.Video; return true;
            case "3d":
            case "threed": modality = Modality.ThreeD; return true;
            case "interactive":
            case "world": modality = Modality.Interactive; return true;
            default: return false;
        }
    }

    /// <summary>All modalities in display order.</summary>
    public static IReadOnlyList<Modality> All { get; } = new[]
    {
        Modality.Image, Modality.Text, Modality.Speech, Modality.Music, Modality.Transcribe,
        Modality.Vision, Modality.Video, Modality.ThreeD, Modality.Interactive,
    };
}
