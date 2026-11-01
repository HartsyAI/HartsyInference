namespace HartsyInference.Core.Configuration;

/// <summary>A top-level folder under the models root: its catalog name, its default name on disk (the name SwarmUI
/// downloads into), and the settings override that replaces the default.</summary>
public sealed record ModelFolder(string CatalogName, string DefaultName, Func<string?> Override);

/// <summary>Maps the engine's logical model folders to the names on disk, so one settings file moves any of them and
/// the catalog, the resolver, the scanner, the downloads and SwarmUI all agree on the same folder.</summary>
/// <remarks>The catalog keeps the logical names (<c>LLM/…</c>, <c>Audio/…</c>); only the first path segment is mapped.
/// Defaults are the names SwarmUI's download roots use, so an unset override agrees with SwarmUI out of the box.</remarks>
public static class ModelFolders
{
    public static readonly ModelFolder Llm = new("LLM", "llm", () => EngineKnobs.ModelFolderLlm.Value);
    public static readonly ModelFolder Audio = new("Audio", "audio", () => EngineKnobs.ModelFolderAudio.Value);
    public static readonly ModelFolder StableDiffusion = new("Stable-Diffusion", "Stable-Diffusion", () => EngineKnobs.ModelFolderStableDiffusion.Value);
    public static readonly ModelFolder Vae = new("VAE", "VAE", () => EngineKnobs.ModelFolderVae.Value);
    public static readonly ModelFolder TextEncoders = new("text_encoders", "text_encoders", () => EngineKnobs.ModelFolderTextEncoders.Value);
    public static readonly ModelFolder ClipVision = new("clip_vision", "clip_vision", () => EngineKnobs.ModelFolderClipVision.Value);
    public static readonly ModelFolder ControlNet = new("controlnet", "controlnet", () => EngineKnobs.ModelFolderControlNet.Value);
    public static readonly ModelFolder Lora = new("Lora", "Lora", () => EngineKnobs.ModelFolderLora.Value);
    public static readonly ModelFolder Embedding = new("Embedding", "Embedding", () => EngineKnobs.ModelFolderEmbedding.Value);
    public static readonly ModelFolder Ipadapter = new("ipadapter", "ipadapter", () => EngineKnobs.ModelFolderIpAdapter.Value);
    public static readonly ModelFolder Vision = new("Vision", "Vision", () => EngineKnobs.ModelFolderVision.Value);
    public static readonly ModelFolder Decision = new("Decision", "Decision", () => EngineKnobs.ModelFolderDecision.Value);
    public static readonly ModelFolder Video = new("Video", "Video", () => EngineKnobs.ModelFolderVideo.Value);
    public static readonly ModelFolder DiffusionModels = new("diffusion_models", "diffusion_models", () => EngineKnobs.ModelFolderDiffusionModels.Value);
    public static readonly ModelFolder GroundingDino = new("grounding-dino", "grounding-dino", () => EngineKnobs.ModelFolderGroundingDino.Value);
    public static readonly ModelFolder Sam2 = new("sam2", "sam2", () => EngineKnobs.ModelFolderSam2.Value);
    public static readonly ModelFolder Rtdetr = new("rtdetr", "rtdetr", () => EngineKnobs.ModelFolderRtdetr.Value);
    public static readonly ModelFolder Clipseg = new("clipseg", "clipseg", () => EngineKnobs.ModelFolderClipseg.Value);

    private static readonly ModelFolder[] Known =
    [
        Llm, Audio, StableDiffusion, Vae, TextEncoders, ClipVision, ControlNet, Lora, Embedding, Ipadapter, Vision,
        Decision, Video, DiffusionModels, GroundingDino, Sam2, Rtdetr, Clipseg,
    ];

    /// <summary>The name a folder has on disk right now: its override when set, else its default.</summary>
    public static string Physical(ModelFolder folder) =>
        folder.Override() is { Length: > 0 } name ? name : folder.DefaultName;

    /// <summary>Maps one folder name (a catalog name, a default name, or any spelling of either, ignoring case) to its
    /// name on disk; a name that is not a model folder comes back unchanged.</summary>
    public static string MapName(string name) =>
        Find(name) is { } folder ? Physical(folder) : name;

    /// <summary>Maps the first segment of a catalog-style relative path (<c>LLM/nanbeige43</c>) to the folder's name on
    /// disk and keeps the rest; a path whose first segment is not a model folder comes back unchanged.</summary>
    public static string MapSubdir(string relative)
    {
        int split = relative.IndexOfAny(['/', '\\']);
        string head = split < 0 ? relative : relative[..split];
        if (Find(head) is not { } folder)
            return relative;
        return split < 0 ? Physical(folder) : Path.Combine(Physical(folder), relative[(split + 1)..]);
    }

    /// <summary>The path of a model folder under the models root, plus any further segments.</summary>
    public static string Under(string modelsRoot, ModelFolder folder, params string[] rest) =>
        Path.Combine([modelsRoot, Physical(folder), .. rest]);

    private static ModelFolder? Find(string name) =>
        Array.Find(Known, f => string.Equals(f.CatalogName, name, StringComparison.OrdinalIgnoreCase)
            || string.Equals(f.DefaultName, name, StringComparison.OrdinalIgnoreCase));
}
