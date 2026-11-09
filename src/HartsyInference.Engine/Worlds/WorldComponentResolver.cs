using HartsyInference.Engine.Dispatch;

namespace HartsyInference.Engine.Worlds;

/// <summary>The one place a world model's checkpoint files are named. The CLI and the HTTP route both reach it through
/// <c>WorldService.Open</c>, and it runs before any weight is read.</summary>
public static class WorldComponentResolver
{
    /// <summary>Aux key that points at the model's VAE checkpoint (Oasis's ViT-VAE; Hunyuan-GameCraft's 3D causal VAE).</summary>
    public const string VaeAuxKey = "vae-path";

    /// <summary>Aux key that points at Hunyuan-GameCraft's Llava-Llama-3-8B text-encoder checkpoint.</summary>
    public const string LlavaAuxKey = "llava-path";

    /// <summary>Aux key that points at Hunyuan-GameCraft's CLIP-L text-encoder checkpoint.</summary>
    public const string ClipAuxKey = "clip-path";

    /// <summary>The architecture a catalog id loads with. An id the service does not know keeps the Oasis loader it always had.</summary>
    public static WorldArchitecture ArchitectureOf(string? catalogId) => (catalogId ?? "oasis").ToLowerInvariant() switch
    {
        "diamond" => WorldArchitecture.Diamond,
        "hunyuan-gamecraft" => WorldArchitecture.HunyuanGameCraft,
        _ => WorldArchitecture.Oasis,
    };

    /// <summary>Resolves the checkpoint files <paramref name="spec"/> needs for its architecture.</summary>
    /// <exception cref="FileNotFoundException">The spec carries no local checkpoint path.</exception>
    /// <exception cref="ArgumentException">A companion checkpoint the architecture needs is missing from the spec's aux paths.</exception>
    public static WorldComponents Resolve(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        string backbone = spec.LocalPath ?? throw new FileNotFoundException(
            $"No checkpoint found for world model '{spec.Requested}'. Supply its local path.");
        WorldArchitecture architecture = ArchitectureOf(spec.Catalog?.Id);
        switch (architecture)
        {
            case WorldArchitecture.Diamond:
                return new WorldComponents { Architecture = architecture, Backbone = backbone };
            case WorldArchitecture.HunyuanGameCraft:
                return new WorldComponents
                {
                    Architecture = architecture,
                    Backbone = backbone,
                    Vae = RequireAux(spec, VaeAuxKey, "Hunyuan-GameCraft", "3D-VAE"),
                    TextEncoder = RequireAux(spec, LlavaAuxKey, "Hunyuan-GameCraft", "Llava-Llama-3-8B"),
                    ClipEncoder = RequireAux(spec, ClipAuxKey, "Hunyuan-GameCraft", "CLIP-L"),
                };
            default:
                return new WorldComponents
                {
                    Architecture = architecture,
                    Backbone = backbone,
                    Vae = RequireAux(spec, VaeAuxKey, "Oasis", "ViT-VAE"),
                };
        }
    }

    private static string RequireAux(ModelSpec spec, string key, string model, string role)
    {
        if (!spec.Aux.TryGetValue(key, out string? path))
        {
            throw new ArgumentException($"{model} needs its {role} checkpoint via the '{key}' aux path.", nameof(spec));
        }
        return path;
    }
}
