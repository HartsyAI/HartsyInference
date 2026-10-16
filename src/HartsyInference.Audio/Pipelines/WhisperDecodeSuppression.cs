using System.Text.Json;

namespace HartsyInference.Audio.Pipelines;

/// <summary>The checkpoint's own decode suppression lists, read from its <c>generation_config.json</c>.
/// Hugging Face applies <c>suppress_tokens</c> at every step and <c>begin_suppress_tokens</c> at the first generated
/// one. Without them greedy decoding can settle on a music-note symbol (<c>♪</c>) over a sung passage and emit
/// nothing else: the same clip that transcribes to its words under the HF defaults comes back as <c>♪♪♪</c>.</summary>
internal sealed class WhisperDecodeSuppression
{
    /// <summary>No suppression: used when a checkpoint ships no <c>generation_config.json</c>.</summary>
    public static readonly WhisperDecodeSuppression None = new([], []);

    public WhisperDecodeSuppression(int[] suppressTokens, int[] beginSuppressTokens)
    {
        SuppressTokens = suppressTokens;
        BeginSuppressTokens = beginSuppressTokens;
    }

    /// <summary>Ids masked at every decoding step.</summary>
    public int[] SuppressTokens { get; }

    /// <summary>Ids masked at the first generated step only (the space and end-of-text, so a decode cannot open empty).</summary>
    public int[] BeginSuppressTokens { get; }

    /// <summary>Reads both lists. A missing file or key yields an empty list for that key, never an error, so a
    /// hand-built checkpoint directory still loads exactly as it did before this file was consulted.</summary>
    public static WhisperDecodeSuppression Load(string generationConfigPath)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(generationConfigPath));
        return new WhisperDecodeSuppression(
            ReadIds(document.RootElement, "suppress_tokens"),
            ReadIds(document.RootElement, "begin_suppress_tokens"));
    }

    /// <summary>Masks the suppressed ids in <paramref name="logits"/> to negative infinity, in place. Ids outside the
    /// vocabulary are skipped.</summary>
    public void Apply(Span<float> logits, bool firstStep)
    {
        foreach (int id in SuppressTokens)
        {
            if ((uint)id < (uint)logits.Length) logits[id] = float.NegativeInfinity;
        }
        if (!firstStep) return;
        foreach (int id in BeginSuppressTokens)
        {
            if ((uint)id < (uint)logits.Length) logits[id] = float.NegativeInfinity;
        }
    }

    private static int[] ReadIds(JsonElement root, string key)
    {
        if (!root.TryGetProperty(key, out JsonElement list) || list.ValueKind != JsonValueKind.Array) return [];
        List<int> ids = new(list.GetArrayLength());
        foreach (JsonElement item in list.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.Number && item.TryGetInt32(out int id)) ids.Add(id);
        }
        return ids.ToArray();
    }
}
