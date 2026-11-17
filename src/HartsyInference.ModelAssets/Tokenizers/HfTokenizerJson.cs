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

    /// <summary>The special-token ids the <c>post_processor</c> wraps a single sequence in, which Hugging Face adds when it encodes with
    /// <c>add_special_tokens=True</c> (the default, and what a serving <c>/v1/completions</c> endpoint does with a raw prompt). A missing or null
    /// post-processor and a <c>ByteLevel</c> one add none; a <c>Sequence</c> applies its processors in order, each wrapping the last one's output.
    /// The caller owns <paramref name="json"/> (this does not dispose it).</summary>
    /// <exception cref="NotSupportedException">The post-processor is a kind this reader does not apply (Bert, Roberta), or its single template
    /// has no <c>$A</c> or names <c>$B</c>.</exception>
    /// <exception cref="InvalidOperationException">The template names a special token its <c>special_tokens</c> map does not define, or the
    /// post-processor lacks a key its type requires.</exception>
    public static (int[] Prefix, int[] Suffix) ReadSingleSequenceSpecials(Stream json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using JsonDocument doc = JsonDocument.Parse(json);
        List<int> prefix = [], suffix = [];
        if (doc.RootElement.TryGetProperty("post_processor", out JsonElement processor))
            ApplyPostProcessor(processor, prefix, suffix);
        return ([.. prefix], [.. suffix]);
    }

    // Wraps the current prefix and suffix in what this processor adds around its input.
    private static void ApplyPostProcessor(JsonElement processor, List<int> prefix, List<int> suffix)
    {
        if (processor.ValueKind == JsonValueKind.Null) return;
        string? type = processor.TryGetProperty("type", out JsonElement t) ? t.GetString() : null;
        switch (type)
        {
            case "ByteLevel":
                return;
            case "Sequence":
                foreach (JsonElement inner in Required(processor, "processors", "Sequence post_processor").EnumerateArray())
                    ApplyPostProcessor(inner, prefix, suffix);
                return;
            case "TemplateProcessing":
                break;
            default:
                throw new NotSupportedException($"tokenizer.json has a '{type ?? "untyped"}' post_processor, whose special tokens this reader does not apply.");
        }

        JsonElement specials = Required(processor, "special_tokens", "TemplateProcessing post_processor");
        List<int> before = [], after = [];
        bool sawA = false;
        foreach (JsonElement piece in Required(processor, "single", "TemplateProcessing post_processor").EnumerateArray())
        {
            if (piece.TryGetProperty("Sequence", out JsonElement sequence))
            {
                string? id = Required(sequence, "id", "template Sequence piece").GetString();
                if (id != "A" || sawA)
                    throw new NotSupportedException($"tokenizer.json's single-sequence template names sequence '{id}', where only one '$A' is expected.");
                sawA = true;
                continue;
            }
            string literal = Required(Required(piece, "SpecialToken", "template piece"), "id", "template SpecialToken piece").GetString()
                ?? throw new InvalidOperationException("tokenizer.json's single-sequence template has a special token with no id.");
            if (!specials.TryGetProperty(literal, out JsonElement special))
                throw new InvalidOperationException($"tokenizer.json's single-sequence template names '{literal}', which its special_tokens map does not define.");
            foreach (JsonElement id in Required(special, "ids", $"special token '{literal}'").EnumerateArray())
                (sawA ? after : before).Add(id.GetInt32());
        }
        if (!sawA) throw new NotSupportedException("tokenizer.json's single-sequence template has no '$A', so it does not say where the text goes.");
        prefix.InsertRange(0, before);
        suffix.AddRange(after);
    }

    // A key the post-processor shape requires; a missing one is a malformed file, named for the caller's load error.
    private static JsonElement Required(JsonElement element, string key, string where)
        => element.ValueKind == JsonValueKind.Object && element.TryGetProperty(key, out JsonElement value) ? value
            : throw new InvalidOperationException($"tokenizer.json's {where} has no '{key}'.");

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
