using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>The end-of-sequence token ids a Hugging Face checkpoint stops on: the config's end id plus <c>generation_config.json</c>'s
/// <c>eos_token_id</c>, which may be one int or a list.</summary>
public static class HfStopIds
{
    /// <param name="directory">The checkpoint folder.</param>
    /// <param name="configEos">The end id in <c>config.json</c>, or null when it has none.</param>
    /// <exception cref="HartsyInferenceException"><c>generation_config.json</c> is not valid JSON, or names an id that is not a 32-bit integer.</exception>
    public static List<int> Read(string directory, int? configEos)
    {
        List<int> ids = [];
        if (configEos is { } eos) ids.Add(eos);
        string path = Path.Combine(directory, "generation_config.json");
        if (!File.Exists(path)) return ids;
        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            if (!document.RootElement.TryGetProperty("eos_token_id", out JsonElement value)) return ids;
            if (value.ValueKind == JsonValueKind.Number) ids.Add(Id(value, path));
            else if (value.ValueKind == JsonValueKind.Array)
                foreach (JsonElement item in value.EnumerateArray())
                    if (item.ValueKind == JsonValueKind.Number) ids.Add(Id(item, path));
        }
        catch (JsonException ex)
        {
            throw new HartsyInferenceException($"'{path}' is not valid JSON: {ex.Message}", ex);
        }
        return [.. ids.Distinct()];
    }

    private static int Id(JsonElement value, string path)
    {
        if (value.TryGetInt32(out int id)) return id;
        throw new HartsyInferenceException($"'{path}' names eos_token_id {value.GetRawText()}, which is not a 32-bit token id.");
    }
}
