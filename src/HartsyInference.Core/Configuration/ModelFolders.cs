using HartsyInference.Core.IO;

namespace HartsyInference.Core.Configuration;

/// <summary>A top-level folder under the models root: its catalog name, its default name on disk (the name SwarmUI
/// downloads into), and the settings override that replaces the default.</summary>
public sealed record ModelFolder(string CatalogName, string DefaultName, Func<string?> Override);

/// <summary>Maps the engine's logical model folders to the names on disk, so one settings file moves any of them and
/// the catalog, the resolver, the scanner, the downloads and SwarmUI all agree on the same folder.</summary>
/// <remarks>The catalog keeps the logical names (<c>LLM/…</c>, <c>Audio/…</c>); only the first path segment is mapped.
/// Defaults are the names SwarmUI's download roots use, so an unset override agrees with SwarmUI out of the box. Lookups
/// try <see cref="Spellings"/> in order, so a folder that already holds models is still found under the name the catalog
/// uses.</remarks>
public static class ModelFolders
{
    public static ModelFolder Llm => new("LLM", "llm", () => EngineKnobs.ModelFolderLlm.Value);
    public static ModelFolder Audio => new("Audio", "audio", () => EngineKnobs.ModelFolderAudio.Value);
    public static ModelFolder StableDiffusion => new("Stable-Diffusion", "Stable-Diffusion", () => EngineKnobs.ModelFolderStableDiffusion.Value);
    public static ModelFolder Vae => new("VAE", "VAE", () => EngineKnobs.ModelFolderVae.Value);
    public static ModelFolder TextEncoders => new("text_encoders", "text_encoders", () => EngineKnobs.ModelFolderTextEncoders.Value);
    public static ModelFolder ClipVision => new("clip_vision", "clip_vision", () => EngineKnobs.ModelFolderClipVision.Value);
    public static ModelFolder ControlNet => new("controlnet", "controlnet", () => EngineKnobs.ModelFolderControlNet.Value);
    public static ModelFolder Lora => new("Lora", "Lora", () => EngineKnobs.ModelFolderLora.Value);
    public static ModelFolder Embedding => new("Embedding", "Embedding", () => EngineKnobs.ModelFolderEmbedding.Value);
    public static ModelFolder Ipadapter => new("ipadapter", "ipadapter", () => EngineKnobs.ModelFolderIpAdapter.Value);
    public static ModelFolder Vision => new("Vision", "Vision", () => EngineKnobs.ModelFolderVision.Value);
    public static ModelFolder Decision => new("Decision", "Decision", () => EngineKnobs.ModelFolderDecision.Value);
    public static ModelFolder Video => new("Video", "Video", () => EngineKnobs.ModelFolderVideo.Value);
    public static ModelFolder DiffusionModels => new("diffusion_models", "diffusion_models", () => EngineKnobs.ModelFolderDiffusionModels.Value);
    public static ModelFolder GroundingDino => new("grounding-dino", "grounding-dino", () => EngineKnobs.ModelFolderGroundingDino.Value);
    public static ModelFolder Sam2 => new("sam2", "sam2", () => EngineKnobs.ModelFolderSam2.Value);
    public static ModelFolder Rtdetr => new("rtdetr", "rtdetr", () => EngineKnobs.ModelFolderRtdetr.Value);
    public static ModelFolder Clipseg => new("clipseg", "clipseg", () => EngineKnobs.ModelFolderClipseg.Value);

    private static ModelFolder[] Known =>
    [
        Llm, Audio, StableDiffusion, Vae, TextEncoders, ClipVision, ControlNet, Lora, Embedding, Ipadapter, Vision,
        Decision, Video, DiffusionModels, GroundingDino, Sam2, Rtdetr, Clipseg,
    ];

    /// <summary>The name a folder has on disk right now: its override when set, else its default.</summary>
    public static string Physical(ModelFolder folder) =>
        Override(folder) ?? folder.DefaultName;

    /// <summary>Maps one folder name (a catalog name, a default name, or any spelling of either, ignoring case) to its
    /// name on disk; a name that is not a model folder comes back unchanged.</summary>
    public static string MapName(string name) =>
        Find(name) is { } folder ? Physical(folder) : name;

    /// <summary>Maps the first segment of a catalog-style relative path (<c>LLM/nanbeige43</c>) to the folder's name on
    /// disk and keeps the rest; a path whose first segment is not a model folder comes back unchanged. Used to create a
    /// folder, so it names the folder the settings choose.</summary>
    public static string MapSubdir(string relative)
    {
        (string head, string? rest) = Split(relative);
        if (Find(head) is not { } folder)
            return relative;
        return rest is null ? Physical(folder) : Path.Combine(Physical(folder), rest);
    }

    /// <summary>The folder paths a lookup tries, in order, for a catalog-style relative path. With a setting for the
    /// folder, the setting comes first and the catalog's spelling second. Without one, the catalog's spelling comes first
    /// (a folder already holding models keeps resolving) and the default name second. A path that is not a model folder
    /// comes back alone.</summary>
    public static IReadOnlyList<string> Spellings(string relative)
    {
        (string head, string? rest) = Split(relative);
        if (Find(head) is not { } folder)
            return [relative];
        string spelled = rest is null ? head : Path.Combine(head, rest);
        string named = Physical(folder);
        string onDisk = rest is null ? named : Path.Combine(named, rest);
        return Override(folder) is null
            ? Distinct(spelled, onDisk)
            : Distinct(onDisk, spelled);
    }

    /// <summary>The path of a model folder under the models root, plus any further segments. A folder that already exists
    /// spelled in another case is used, as the downloads and lookups use it, so a helper never starts a second folder.</summary>
    public static string Under(string modelsRoot, ModelFolder folder, params string[] rest) =>
        Path.Combine([CaseInsensitivePath.ResolveDirectory(modelsRoot, Physical(folder)), .. rest]);

    private static string? Override(ModelFolder folder) =>
        folder.Override() is { Length: > 0 } name ? name : null;

    private static (string Head, string? Tail) Split(string relative)
    {
        int split = relative.IndexOfAny(['/', '\\']);
        return split < 0 ? (relative, null) : (relative[..split], relative[(split + 1)..]);
    }

    private static IReadOnlyList<string> Distinct(string first, string second) =>
        string.Equals(first, second, StringComparison.Ordinal) ? [first] : [first, second];

    private static ModelFolder? Find(string name) =>
        Array.Find(Known, f => string.Equals(f.CatalogName, name, StringComparison.OrdinalIgnoreCase)
            || string.Equals(f.DefaultName, name, StringComparison.OrdinalIgnoreCase));
}
