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

    /// <summary>Aux key that points at SANA-WM's Gemma-2-2B text-encoder checkpoint.</summary>
    public const string TextEncoderAuxKey = "text-encoder-path";

    /// <summary>Aux key that points at SANA-WM's Gemma <c>tokenizer.model</c>.</summary>
    public const string TokenizerAuxKey = "tokenizer-path";

    /// <summary>Aux key that points at SANA-WM streaming's refiner transformer checkpoint.</summary>
    public const string RefinerAuxKey = "refiner-path";

    /// <summary>Aux key that points at the refiner's text connectors checkpoint.</summary>
    public const string RefinerConnectorsAuxKey = "refiner-connectors-path";

    /// <summary>Aux key that points at the refiner's Gemma-3-12B text encoder: a safetensors file or a folder of shards.</summary>
    public const string RefinerTextEncoderAuxKey = "refiner-text-encoder-path";

    /// <summary>Aux key that points at the refiner text encoder's <c>tokenizer.model</c>.</summary>
    public const string RefinerTokenizerAuxKey = "refiner-tokenizer-path";

    // SANA-WM's release layout: the DiT under dit/, with vae/ and text_encoder/gemma-2-2b-it/ beside it.
    private const string SanaWmVaeRelative = "../vae/diffusion_pytorch_model.safetensors";
    private const string SanaWmTextEncoderRelative = "../text_encoder/gemma-2-2b-it/gemma-2-2b-it.safetensors";
    private const string GemmaTokenizerFile = "tokenizer.model";

    // SANA-WM streaming's layout: the converted DiT under sana_dit/, the causal VAE under ltx2_causal_vae/ beside it, and the
    // stage-1 release's Gemma-2-2B, shared rather than downloaded twice.
    private const string SanaWmCausalVaeRelative = "../ltx2_causal_vae/diffusion_pytorch_model.safetensors";
    private const string SanaWmSharedTextEncoderRelative = "../../SanaWm/text_encoder/gemma-2-2b-it/gemma-2-2b-it.safetensors";

    // The streaming release's refiner: the diffusers LTX-2 transformer and text connectors under refiner_diffusers/, and
    // Gemma-3-12B as a folder of shards with its tokenizer.
    private const string SanaWmRefinerTransformerRelative = "../refiner_diffusers/transformer/diffusion_pytorch_model.safetensors";
    private const string SanaWmRefinerConnectorsRelative = "../refiner_diffusers/connectors/diffusion_pytorch_model.safetensors";
    private const string SanaWmRefinerTextEncoderRelative = "../gemma3_12b";

    /// <summary>The world models the service can load, by catalog id.</summary>
    public static IReadOnlyList<string> KnownIds { get; } = ["oasis", "diamond", "hunyuan-gamecraft", "sana-wm", "sana-wm-streaming"];

    /// <summary>The architecture a catalog id loads with.</summary>
    /// <exception cref="NotSupportedException">The id is not a world model this service loads; a catalog-less spec has no id
    /// and is refused too.</exception>
    public static WorldArchitecture ArchitectureOf(string? catalogId) => catalogId?.ToLowerInvariant() switch
    {
        "oasis" => WorldArchitecture.Oasis,
        "diamond" => WorldArchitecture.Diamond,
        "hunyuan-gamecraft" => WorldArchitecture.HunyuanGameCraft,
        "sana-wm" => WorldArchitecture.SanaWm,
        "sana-wm-streaming" => WorldArchitecture.SanaWmStreaming,
        _ => throw new NotSupportedException(
            $"'{catalogId ?? "(no catalog id)"}' is not a world model this engine can load; known: {string.Join(", ", KnownIds)}."),
    };

    /// <summary>Resolves the checkpoint files <paramref name="spec"/> needs for its architecture.</summary>
    /// <exception cref="FileNotFoundException">The spec carries no local checkpoint path.</exception>
    /// <exception cref="ArgumentException">A companion checkpoint the architecture needs is missing from the spec's aux paths.</exception>
    public static WorldComponents Resolve(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        string backbone = spec.LocalPath ?? throw new FileNotFoundException(
            $"No checkpoint found for world model '{spec.Requested}'. Supply its local path.");
        WorldArchitecture architecture = ArchitectureOf(spec.Catalog?.Id ?? spec.Requested);
        switch (architecture)
        {
            case WorldArchitecture.Diamond:
                return new WorldComponents { Architecture = architecture, Backbone = backbone };
            case WorldArchitecture.SanaWm:
                string textEncoder = AuxOrSibling(spec, TextEncoderAuxKey, backbone, SanaWmTextEncoderRelative, "Gemma-2-2B text encoder");
                return new WorldComponents
                {
                    Architecture = architecture,
                    Backbone = backbone,
                    Vae = AuxOrSibling(spec, VaeAuxKey, backbone, SanaWmVaeRelative, "LTX-2 VAE"),
                    TextEncoder = textEncoder,
                    Tokenizer = AuxOrSibling(spec, TokenizerAuxKey, textEncoder, GemmaTokenizerFile, "Gemma tokenizer"),
                };
            case WorldArchitecture.SanaWmStreaming:
                string sharedTextEncoder = AuxOrSibling(spec, TextEncoderAuxKey, backbone, SanaWmSharedTextEncoderRelative,
                    "Gemma-2-2B text encoder");
                return new WorldComponents
                {
                    Architecture = architecture,
                    Backbone = backbone,
                    Vae = AuxOrSibling(spec, VaeAuxKey, backbone, SanaWmCausalVaeRelative, "causal LTX-2 VAE"),
                    TextEncoder = sharedTextEncoder,
                    Tokenizer = AuxOrSibling(spec, TokenizerAuxKey, sharedTextEncoder, GemmaTokenizerFile, "Gemma tokenizer"),
                    Refiner = SanaWmRefiner(spec, backbone),
                };
            case WorldArchitecture.HunyuanGameCraft:
                return new WorldComponents
                {
                    Architecture = architecture,
                    Backbone = backbone,
                    Vae = RequireAux(spec, VaeAuxKey, "Hunyuan-GameCraft", "3D-VAE"),
                    TextEncoder = RequireAux(spec, LlavaAuxKey, "Hunyuan-GameCraft", "Llava-Llama-3-8B"),
                    ClipEncoder = RequireAux(spec, ClipAuxKey, "Hunyuan-GameCraft", "CLIP-L"),
                };
            case WorldArchitecture.Oasis:
                return new WorldComponents
                {
                    Architecture = architecture,
                    Backbone = backbone,
                    Vae = RequireAux(spec, VaeAuxKey, "Oasis", "ViT-VAE"),
                };
            default:
                throw new NotSupportedException($"The {architecture} architecture has no component resolution.");
        }
    }

    // Names the refiner's files without requiring them: a stream without the refiner never reads them.
    private static WorldRefinerFiles SanaWmRefiner(ModelSpec spec, string backbone)
    {
        string textEncoder = AuxOrRelative(spec, RefinerTextEncoderAuxKey, backbone, SanaWmRefinerTextEncoderRelative);
        // A single-file encoder keeps its tokenizer beside it; a folder of shards keeps it inside.
        bool singleFile = textEncoder.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase);
        string tokenizerFolder = singleFile ? Path.GetDirectoryName(textEncoder) ?? "." : textEncoder;
        return new WorldRefinerFiles
        {
            Transformer = AuxOrRelative(spec, RefinerAuxKey, backbone, SanaWmRefinerTransformerRelative),
            Connectors = AuxOrRelative(spec, RefinerConnectorsAuxKey, backbone, SanaWmRefinerConnectorsRelative),
            TextEncoder = textEncoder,
            Tokenizer = spec.Aux.TryGetValue(RefinerTokenizerAuxKey, out string? tokenizer)
                ? tokenizer
                : Path.Combine(tokenizerFolder, GemmaTokenizerFile),
        };
    }

    private static string AuxOrRelative(ModelSpec spec, string key, string anchor, string relative) =>
        spec.Aux.TryGetValue(key, out string? aux)
            ? aux
            : Path.GetFullPath(Path.Combine(Path.GetDirectoryName(anchor) ?? ".", relative));

    /// <summary>The aux path for <paramref name="key"/>, else <paramref name="relative"/> resolved against the folder of
    /// <paramref name="anchor"/>. The file must exist.</summary>
    private static string AuxOrSibling(ModelSpec spec, string key, string anchor, string relative, string role)
    {
        string path = spec.Aux.TryGetValue(key, out string? aux)
            ? aux
            : Path.GetFullPath(Path.Combine(Path.GetDirectoryName(anchor) ?? ".", relative));
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"SANA-WM needs its {role} at {path}; place it there (the release layout) or pass it via the '{key}' aux path.", path);
        }
        return path;
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
