using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>The byte-level BPE tokenizer and Jinja chat template of a Hugging Face directory (<c>tokenizer.json</c>, <c>chat_template.jinja</c>).</summary>
public static class HfChatAssets
{
    /// <summary>Loads the directory's tokenizer with <paramref name="bosToken"/> and <paramref name="eosToken"/> as its begin and end ids, and its chat template.</summary>
    /// <param name="stopIds">Ids that end generation, in addition to the end token.</param>
    public static (ILlmTokenizer Tokenizer, IChatTemplate Template) Load(string directory, IReadOnlyList<int> stopIds, string bosToken, string eosToken)
    {
        using FileStream tokenizerJson = File.OpenRead(Path.Combine(directory, "tokenizer.json"));
        ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerJson, stopIds, bosToken, eosToken);
        return (tokenizer, new JinjaChatTemplate(File.ReadAllText(Path.Combine(directory, "chat_template.jinja"))));
    }
}
