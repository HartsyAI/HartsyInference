using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>The byte-level BPE tokenizer and Jinja chat template of a Hugging Face directory (<c>tokenizer.json</c>, <c>chat_template.jinja</c>).</summary>
public static class HfChatAssets
{
    /// <summary>The begin and end literals the directory's <c>tokenizer_config.json</c> names (<c>bos_token</c>, <c>eos_token</c>), each null when it
    /// names none or the file is absent.</summary>
    /// <exception cref="HartsyInferenceException">The file is not valid JSON.</exception>
    public static (string? Bos, string? Eos) SpecialTokenLiterals(string directory)
    {
        string path = Path.Combine(directory, "tokenizer_config.json");
        if (!File.Exists(path)) return (null, null);
        try
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            return (Literal(document.RootElement, "bos_token"), Literal(document.RootElement, "eos_token"));
        }
        catch (JsonException ex)
        {
            throw new HartsyInferenceException($"'{path}' is not valid JSON: {ex.Message}", ex);
        }
    }

    private static string? Literal(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out JsonElement value)) return null;
        return value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            JsonValueKind.Object when value.TryGetProperty("content", out JsonElement content) && content.ValueKind == JsonValueKind.String => content.GetString(),
            _ => null,
        };
    }

    /// <summary>Loads the directory's tokenizer with <paramref name="bosToken"/> and <paramref name="eosToken"/> as its begin and end ids, and its chat template.</summary>
    /// <param name="stopIds">Ids that end generation, in addition to the end token.</param>
    public static (ILlmTokenizer Tokenizer, IChatTemplate Template) Load(string directory, IReadOnlyList<int> stopIds, string? bosToken, string? eosToken)
    {
        using FileStream tokenizerJson = File.OpenRead(Path.Combine(directory, "tokenizer.json"));
        ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerJson, stopIds, bosToken, eosToken);
        return (tokenizer, new JinjaChatTemplate(File.ReadAllText(Path.Combine(directory, "chat_template.jinja"))));
    }
}
