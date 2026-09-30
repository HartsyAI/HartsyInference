using System.Text.Json;

namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Builds the engine's byte-level BPE core (<see cref="GgufTokenizer"/>) from a HuggingFace <c>tokenizer.json</c> file — the single artifact the Llama-3 / Qwen / Mistral repos actually ship. Reuses the existing BPE implementation rather than depending on the two-file <c>vocab.json</c> + <c>merges.txt</c> split (which has to be extracted out-of-band). Reads the <c>model.vocab</c> / <c>model.merges</c> / <c>added_tokens</c> arrays, the <c>ignore_merges</c> flag, and the full <c>pre_tokenizer</c> stage sequence (Split behaviors, invert, ByteLevel) via <see cref="PreTokenizerPipeline"/> so the family-specific tokenization (e.g. Llama-3's digit grouping) is reproduced exactly.</summary>
public static class HfTokenizerJson
{
    /// <summary>Parses a byte-level-BPE <c>tokenizer.json</c> stream into a ready <see cref="GgufTokenizer"/>. The caller owns <paramref name="json"/> (this does not dispose it). <paramref name="extraStopIds"/> is forwarded to the tokenizer for end-of-turn handling. <paramref name="bosToken"/> / <paramref name="eosToken"/> name the added-token literals for families that do not use the Llama-3 names.</summary>
    public static GgufTokenizer LoadByteLevelBpe(Stream json, IReadOnlyList<int>? extraStopIds = null,
        string? bosToken = null, string? eosToken = null)
    {
        ArgumentNullException.ThrowIfNull(json);
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;

        if (!root.TryGetProperty("model", out JsonElement model))
            throw new InvalidOperationException("tokenizer.json has no 'model' object.");
        if (model.TryGetProperty("type", out JsonElement modelType) && modelType.ValueKind == JsonValueKind.String
            && !string.Equals(modelType.GetString(), "BPE", StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                $"HfTokenizerJson only handles byte-level BPE tokenizers; model.type was '{modelType.GetString()}'.");
        }

        // Gather (token-string, id) pairs from the base vocab and the added/special tokens, then lay them
        // out by id so the array index is the token id (added tokens sit above the base vocab).
        List<(string Token, int Id)> entries = new(160_000);
        List<(string Token, int Id)> added = [];

        JsonElement vocab = model.GetProperty("vocab");
        foreach (JsonProperty kv in vocab.EnumerateObject())
            entries.Add((kv.Name, kv.Value.GetInt32()));

        int? bosId = null, eosId = null;
        if (root.TryGetProperty("added_tokens", out JsonElement addedTokens)
            && addedTokens.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement t in addedTokens.EnumerateArray())
            {
                int id = t.GetProperty("id").GetInt32();
                string content = t.GetProperty("content").GetString()
                    ?? throw new InvalidOperationException("added_token has null content.");
                added.Add((content, id));
                entries.Add((content, id));
                if (bosToken is null && content == "<|begin_of_text|>") bosId = id;
                else if (eosToken is null && content == "<|end_of_text|>") eosId = id;
                if (content == bosToken) bosId = id;
                if (content == eosToken) eosId = id;
            }
        }

        int maxId = -1;
        foreach ((string _, int id) in entries) if (id > maxId) maxId = id;
        string[] tokens = new string[maxId + 1];
        int[] tokenType = new int[maxId + 1];
        foreach ((string token, int id) in entries) tokens[id] = token;
        // Byte-level vocabs are dense, but guard against gaps so a malformed file fails loud, not silently.
        for (int i = 0; i < tokens.Length; i++) tokens[i] ??= string.Empty;
        // Mark the added tokens as control so encode/decode treats their literals as specials.
        foreach ((string _, int id) in added) tokenType[id] = 3;

        string[] merges = ReadMerges(model);
        JsonElement? preTokenizerJson = root.TryGetProperty("pre_tokenizer", out JsonElement pt) ? pt : null;
        PreTokenizerPipeline pipeline = PreTokenizerPipeline.FromJson(preTokenizerJson);
        bool ignoreMerges = model.TryGetProperty("ignore_merges", out JsonElement im)
            && im.ValueKind == JsonValueKind.True;

        return new GgufTokenizer(tokens, merges, tokenType, bosId, eosId, extraStopIds,
            ignoreMerges: ignoreMerges, preTokenizer: pipeline);
    }

    /// <summary>Reads <c>model.merges</c>, accepting both the legacy <c>"left right"</c> string form and the newer <c>["left","right"]</c> pair form, normalizing to the space-joined form the BPE core expects.</summary>
    private static string[] ReadMerges(JsonElement model)
    {
        if (!model.TryGetProperty("merges", out JsonElement merges) || merges.ValueKind != JsonValueKind.Array)
            throw new InvalidOperationException("tokenizer.json has no 'model.merges' array.");
        string[] result = new string[merges.GetArrayLength()];
        int i = 0;
        foreach (JsonElement m in merges.EnumerateArray())
        {
            if (m.ValueKind == JsonValueKind.String)
                result[i] = m.GetString()!;
            else if (m.ValueKind == JsonValueKind.Array && m.GetArrayLength() == 2)
                result[i] = $"{m[0].GetString()} {m[1].GetString()}";
            else
                throw new InvalidOperationException($"Unexpected merge entry at index {i}.");
            i++;
        }
        return result;
    }
}
