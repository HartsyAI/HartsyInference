using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Locates the two components MiniMax-H3-Image loads beside its DiT: the single-frame image VAE and the Qwen3-VL-32B
/// text encoder. Both are found the way the video recipe finds its flat-layout components: walk up from the DiT and search
/// the component folders beneath each ancestor. That covers the Comfy-Org repack layout (flat/diffusion_models, flat/vae,
/// flat/text_encoders) and the models-root layout (diffusion_models/MiniMaxH3Image, VAE/MiniMaxH3, text_encoders). A request's
/// <c>Vae</c> and <c>Qwen</c> overrides replace the search.</summary>
internal sealed record MiniMaxH3ImageAssets
{
    /// <summary>The file the image build's decoder is named after. Used in the not-found message.</summary>
    internal const string ImageVaeFile = "minimax_h3_image_vae_fp16.safetensors";

    private static readonly string[] VaeFolders = ["vae", "VAE"];
    private static readonly string[] TextEncoderFolders = ["text_encoders", "clip", "llm"];

    // Preference order for the encoder: the nvfp4 build first, then int8 ConvRot. Both are safetensors and the same tower.
    private static readonly string[] TextEncoderPreference = ["nvfp4_awq", "int8_convrot"];

    /// <summary>The single-frame image VAE, dense FP16 as shipped.</summary>
    public required string ImageVae { get; init; }

    /// <summary>The Qwen3-VL-32B text encoder (quantized build).</summary>
    public required string TextEncoder { get; init; }

    /// <summary>Resolves both components for the DiT at <paramref name="ditFile"/>.</summary>
    public static MiniMaxH3ImageAssets Resolve(string ditFile, ComponentOverrides? components)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ditFile);
        List<string> roots = MiniMaxH3Assets.FlatSearchRoots(ditFile);

        string imageVae = components?.Vae
            ?? MiniMaxH3Assets.FindFlat(roots, VaeFolders, ["image_vae"], preferQuantized: false)
            ?? throw new FileNotFoundException(
                $"MiniMax-H3-Image needs {ImageVaeFile} near '{ditFile}' (a vae/ or VAE/ folder beside the checkpoint or "
                + "under its models root).");

        string textEncoder = components?.Qwen
            ?? FirstMatch(roots, TextEncoderPreference)
            ?? throw new FileNotFoundException(
                $"MiniMax-H3-Image needs its Qwen3-VL-32B text encoder near '{ditFile}' (a text_encoders/ folder beside "
                + "the checkpoint or under its models root). Expected qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors or "
                + "qwen3vl_32b_minimax_h3_int8_convrot.safetensors.");

        return new MiniMaxH3ImageAssets { ImageVae = imageVae, TextEncoder = textEncoder };
    }

    private static string? FirstMatch(List<string> roots, string[] hints)
    {
        // Hint-major on purpose: an nvfp4_awq encoder up to three ancestors away beats an int8_convrot one beside the DiT,
        // because the ComfyUI reference uses nvfp4_awq.
        foreach (string hint in hints)
        {
            string? found = MiniMaxH3Assets.FindFlat(roots, TextEncoderFolders, [hint]);
            if (found is not null)
            {
                return found;
            }
        }
        return null;
    }
}
