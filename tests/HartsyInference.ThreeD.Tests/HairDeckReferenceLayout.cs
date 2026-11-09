namespace HartsyInference.ThreeD.Tests;

/// <summary>Where the HairDeck real-weight inputs and the reference dump live: <c>HARTSY_HAIRDECK_WEIGHTS</c> (the released
/// model folder with <c>ckpt/</c> and <c>data/</c>) and <c>HARTSY_HAIRDECK_REF</c> (the folder of
/// <c>tools/parity/dump_hairdeck_core.py</c>), with the local defaults.</summary>
internal static class HairDeckReferenceLayout
{
    public const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";
    public const string WeightsEnv = "HARTSY_HAIRDECK_WEIGHTS";
    public const string ReferenceEnv = "HARTSY_HAIRDECK_REF";
    private const string DefaultWeights = @"C:\Users\AI Overlord\Models\3D\HairDeck";
    private const string DefaultReference = @"C:\Users\AIOVER~1\AppData\Local\Temp\claude\2bb7b6dccac17437\5400be18-00bc-4979-a3e9-f8c1a77647ce\scratchpad\hairdeck_core\dump";

    public static string Weights => Environment.GetEnvironmentVariable(WeightsEnv) ?? DefaultWeights;

    public static string Reference => Environment.GetEnvironmentVariable(ReferenceEnv) ?? DefaultReference;

    public static string ImageModel => Path.Combine(Weights, "ckpt", "image_model", "model.safetensors");

    public static string ColorModel => Path.Combine(Weights, "ckpt", "color_model", "model.safetensors");

    public static string ChainModel => Path.Combine(Weights, "ckpt", "chain_model", "model.safetensors");

    public static string CardVae => Path.Combine(Weights, "ckpt", "card_vae", "model.safetensors");

    public static string CardVaeStats => Path.Combine(Weights, "ckpt", "card_vae", "stats.json");

    public static string LatentStats => Path.Combine(Weights, "ckpt", "card_vae", "latent_stats_head.json");

    public static string TokenStats => Path.Combine(Weights, "data", "token_stats_head.json");

    public static string ChainStats => Path.Combine(Weights, "data", "spring", "chain_stats_head.json");

    public static string Manifest => Path.Combine(Reference, "manifest.json");

    /// <summary>The required files that are absent, in a fixed order.</summary>
    public static IEnumerable<string> MissingInputs()
    {
        string[] required =
        [
            ImageModel, ColorModel, ChainModel, CardVae, CardVaeStats, LatentStats, TokenStats, ChainStats, Manifest,
        ];
        return required.Where(p => !File.Exists(p));
    }
}
