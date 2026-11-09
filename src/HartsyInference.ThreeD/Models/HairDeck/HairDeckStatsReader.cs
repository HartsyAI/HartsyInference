using System.Text.Json;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>Reads the HairDeck standardisation files: the head-frame token and latent statistics, the card decoder's output
/// statistics and the chain statistics.</summary>
internal static class HairDeckStatsReader
{
    /// <summary>Mean and deviation vectors of one statistics file.</summary>
    internal sealed record Vectors(float[] Mu, float[] Sd);

    /// <summary>Root mean and deviation of the head-frame token statistics (<c>token_stats_head.json</c>).</summary>
    public static (float[] RootMu, float[] RootSd) ReadRoot(string path)
    {
        using JsonDocument doc = Open(path);
        return (Floats(doc, "root_mu", 3, path), Floats(doc, "root_sd", 3, path));
    }

    /// <summary>Mean and deviation of the card latent, <c>latent_stats_head.json</c> (32 channels).</summary>
    public static Vectors ReadLatent(string path, int channels) => ReadVectors(path, channels);

    /// <summary>Mean and deviation of the decoder outputs, <c>stats.json</c> (xyz, width, thickness).</summary>
    public static Vectors ReadOutput(string path, int channels) => ReadVectors(path, channels);

    /// <summary>Mean and deviation of the chain tokens, <c>chain_stats_head.json</c>.</summary>
    public static Vectors ReadChain(string path, int channels) => ReadVectors(path, channels);

    private static Vectors ReadVectors(string path, int channels)
    {
        using JsonDocument doc = Open(path);
        return new Vectors(Floats(doc, "mu", channels, path), Floats(doc, "sd", channels, path));
    }

    private static JsonDocument Open(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"HairDeck statistics file not found: {path}", path);
        return JsonDocument.Parse(File.ReadAllBytes(path));
    }

    private static float[] Floats(JsonDocument doc, string name, int count, string path)
    {
        if (!doc.RootElement.TryGetProperty(name, out JsonElement array) || array.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException($"HairDeck statistics '{path}' has no array '{name}'.");
        if (array.GetArrayLength() != count)
            throw new InvalidDataException($"HairDeck statistics '{path}' array '{name}' has {array.GetArrayLength()} values, expected {count}.");
        float[] values = new float[count];
        int i = 0;
        foreach (JsonElement e in array.EnumerateArray()) values[i++] = (float)e.GetDouble();
        return values;
    }
}
