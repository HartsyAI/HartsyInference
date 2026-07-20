namespace HartsyInference.Engine.Features;

/// <summary>The architecture-compat-class allow list a host app uses to decide whether the engine can take a given
/// checkpoint, plus a human-readable explanation when it cannot. The ids are the same compat-class strings SwarmUI's
/// <c>T2IModelClassSorter</c> assigns, so a host can dispatch straight off its own model metadata.</summary>
public static class ModelSupport
{
    /// <summary>Architectures with a fully wired recipe.</summary>
    private static readonly HashSet<string> _supportedArchs = new HashSet<string>(StringComparer.Ordinal)
    {
        "stable-diffusion-v1",
        "stable-diffusion-xl-v1",
        "stable-diffusion-v3-medium",
        "stable-diffusion-v3.5-medium",
        "stable-diffusion-v3.5-large",
        "flux-1",
        "flux-2",
        "flux-2-klein-4b",
        "flux-2-klein-9b",
        "chroma",
        "chroma-radiance",
        "zeta-chroma",
        "auraflow-v1",
        "f-lite",
        "ideogram-4",
        "boogu",
        "ernie-image",
        "lumina-2",
        "hunyuan-image-2_1",
        "omnigen-2",
        "z-image",
        "anima",
        "hidream-i1",
        "qwen-image",
        "wan-22-5b",
        "wan-21-1_3b",
        "wan-21-14b",
        "lightricks-ltx-video",
        "lightricks-ltx-video-2",
        "ace-step-1_5",
        "lance",
        "lance-video",
        "lens",
        "krea-2",
    };

    /// <summary>Architectures the engine has a pipeline for but refuses today, mapped to the human-readable blocker.
    /// Refusing one of these is "the glue is a TODO", not "this can't work" — the message should say so.</summary>
    private static readonly Dictionary<string, string> _pendingArchs = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        // The pipeline exists and passes structural tests, but its conditioning encoders are not faithfully implemented:
        // the upstream E2E tests feed PRE-COMPUTED embeddings, so wiring a recipe with guessed templates would produce
        // semantically-wrong conditioning.
        ["kandinsky5-imglite"] = "Kandinsky 5 Image Lite (pipeline needs pre-computed Qwen2.5-VL + CLIP-L embeddings — live encode path unverified)",
        ["yue"] = "YuE (no YuE mm tokenizer yet — lyrics can't be encoded)",
        ["musicgen"] = "MusicGen (missing EnCodec-32kHz / T5-Base presets and the converter's bundled-text-encoder path)",
    };

    /// <summary>Every compat class with a wired recipe.</summary>
    public static IReadOnlyCollection<string> SupportedArchitectures => _supportedArchs;

    /// <summary>Compat classes the engine has a pipeline for but does not accept yet, mapped to the blocker reason.</summary>
    public static IReadOnlyDictionary<string, string> PendingArchitectures => _pendingArchs;

    /// <summary>True when <paramref name="compatClass"/> has a wired recipe.</summary>
    public static bool IsArchitectureSupported(string? compatClass)
        => !string.IsNullOrEmpty(compatClass) && _supportedArchs.Contains(compatClass);

    /// <summary>Explains why a compat class isn't accepted, distinguishing "pipeline exists, glue pending" from "not implemented".</summary>
    public static string WhyNotSupported(string? compatClass)
    {
        if (string.IsNullOrEmpty(compatClass))
        {
            return "Model has no architecture compat class set — the engine can't dispatch.";
        }
        if (_pendingArchs.TryGetValue(compatClass, out string? friendlyName))
        {
            return $"{friendlyName} ('{compatClass}') is not yet wired into a recipe. The pipeline + checkpoint converter "
                + "exist, but the per-architecture glue (text-encoder selection, side-model download, tokenizer setup) is a TODO.";
        }
        return $"Architecture '{compatClass}' is not implemented. Supported today: {string.Join(", ", _supportedArchs)}.";
    }
}
