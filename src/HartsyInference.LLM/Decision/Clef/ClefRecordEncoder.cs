using System.Text;
using System.Text.Json;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Decision.Clef;

/// <summary>Port of <c>encode_record</c> from Clef's <c>joint_schema_model.py</c>: lays the state and the typed schema out
/// as one prompt and records where each question and option lands. Text only; images and videos are not supported.</summary>
public sealed class ClefRecordEncoder
{
    public const string SystemPrompt = "Read the complete state and schema. Decide every field jointly. "
        + "Each answer must be exactly one of that field's allowed options.";

    public const int DefaultMaxLength = 16_384;

    private static readonly string[] NoulKeys = ["true", "false"];

    private readonly GgufTokenizer _tokenizer;

    public ClefRecordEncoder(GgufTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _tokenizer = tokenizer;
    }

    /// <summary>Encodes a request body (<c>state</c> plus <c>questions</c>).</summary>
    public ClefEncodedRecord Encode(JsonElement record, int maxLength = DefaultMaxLength, int? maxStateTokens = null)
    {
        if (record.TryGetProperty("images", out JsonElement images) && images.ValueKind == JsonValueKind.Array && images.GetArrayLength() > 0
            || record.TryGetProperty("videos", out JsonElement videos) && videos.ValueKind == JsonValueKind.Array && videos.GetArrayLength() > 0)
        {
            throw new NotSupportedException("Clef image and video input is not supported by this runtime yet.");
        }
        if (!record.TryGetProperty("state", out JsonElement state))
        {
            throw new ArgumentException("A record needs a 'state'.", nameof(record));
        }
        if (!record.TryGetProperty("questions", out JsonElement questions) || questions.ValueKind != JsonValueKind.Object)
        {
            throw new ArgumentException("A record needs a 'questions' object.", nameof(record));
        }

        List<int> schema = Tokens("\n\nSCHEMA FIELDS:\n");
        List<ClefQuestionSpans> spans = [];
        int index = 0;
        foreach (JsonProperty question in questions.EnumerateObject())
        {
            string type = question.Value.GetProperty("type").GetString() ?? string.Empty;
            int typeId = type switch
            {
                "noul" => 0,
                "choice" => 1,
                "score" => 2,
                _ => throw new ArgumentException($"{question.Name}: type must be noul, choice, or score."),
            };
            schema.AddRange(Tokens($"\nFIELD {index + 1}\nID: {question.Name}\nTYPE: {type}\nINSTRUCTION: "));
            int questionStart = schema.Count;
            string instructions = question.Value.TryGetProperty("instructions", out JsonElement ins)
                && ins.ValueKind != JsonValueKind.Null && !(ins.ValueKind == JsonValueKind.String && ins.GetString() == string.Empty)
                ? ClefJson.Render(ins) : question.Name;
            schema.AddRange(Tokens(instructions));
            int questionEnd = schema.Count;
            schema.AddRange(Tokens("\nALLOWED OPTIONS:\n"));

            List<(int, int)> optionSpans = [];
            List<string> optionIds = [];
            int optionIndex = 0;
            foreach ((string optionId, JsonElement? description) in Options(question.Value, type))
            {
                schema.AddRange(Tokens($"OPTION {optionIndex + 1}: "));
                int optionStart = schema.Count;
                schema.AddRange(Tokens(RenderOption(optionId, description)));
                optionSpans.Add((optionStart, schema.Count));
                optionIds.Add(optionId);
                schema.AddRange(Tokens("\n"));
                optionIndex++;
            }
            schema.AddRange(Tokens("END FIELD\n"));
            spans.Add(new ClefQuestionSpans
            {
                QuestionId = question.Name, QuestionType = typeId, QuestionSpan = (questionStart, questionEnd),
                OptionSpans = [.. optionSpans], OptionIds = [.. optionIds],
            });
            index++;
        }

        List<int> prefix = Tokens($"<|im_start|>system\n{SystemPrompt}<|im_end|>\n<|im_start|>user\nSTATE:\n");
        List<int> suffix = Tokens("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
        List<int> stateIds = Tokens(ClefJson.Render(state));
        if (maxStateTokens is int cap && stateIds.Count > cap)
        {
            stateIds = stateIds.GetRange(0, cap);
        }
        int fixedLength = prefix.Count + schema.Count + suffix.Count;
        if (fixedLength > maxLength)
        {
            throw new ArgumentException($"schema requires {fixedLength} tokens before state; maximum is {maxLength}");
        }
        if (stateIds.Count > maxLength - fixedLength)
        {
            stateIds = stateIds.GetRange(0, maxLength - fixedLength);
        }
        if (spans.Count == 0)
        {
            throw new ArgumentException("record produced no model input or questions");
        }
        int offset = prefix.Count + stateIds.Count;
        ClefQuestionSpans[] shifted = [.. spans.Select(q => q with
        {
            QuestionSpan = (q.QuestionSpan.Start + offset, q.QuestionSpan.End + offset),
            OptionSpans = [.. q.OptionSpans.Select(s => (s.Start + offset, s.End + offset))],
        })];
        return new ClefEncodedRecord { InputIds = [.. prefix, .. stateIds, .. schema, .. suffix], Questions = shifted };
    }

    private static IEnumerable<(string Id, JsonElement? Description)> Options(JsonElement question, string type)
    {
        question.TryGetProperty("criteria", out JsonElement criteria);
        if (type == "noul")
        {
            Dictionary<string, JsonElement?> merged = new(StringComparer.Ordinal)
            {
                ["true"] = JsonDocument.Parse("\"The proposition is true or the answer is yes.\"").RootElement,
                ["false"] = JsonDocument.Parse("\"The proposition is false or the answer is no.\"").RootElement,
            };
            if (criteria.ValueKind == JsonValueKind.Object)
            {
                foreach (JsonProperty p in criteria.EnumerateObject())
                {
                    merged[p.Name] = p.Value;
                }
            }
            return NoulKeys.Select(k => (k, merged[k]));
        }
        if (type == "choice")
        {
            if (criteria.ValueKind != JsonValueKind.Object)
            {
                throw new ArgumentException("A choice question needs a 'criteria' object.");
            }
            List<(string, JsonElement?)> items = [.. criteria.EnumerateObject().Select(p => (p.Name, (JsonElement?)p.Value))];
            items.Sort((a, b) => string.CompareOrdinal(a.Item1, b.Item1));
            return items;
        }
        if (criteria.ValueKind != JsonValueKind.Array)
        {
            throw new ArgumentException("A score question needs a 'criteria' array.");
        }
        return criteria.EnumerateArray().Select((e, i) => (i.ToString(System.Globalization.CultureInfo.InvariantCulture), (JsonElement?)e));
    }

    private static string RenderOption(string optionId, JsonElement? description)
    {
        StringBuilder sb = new("{");
        bool hasDescription = description is { ValueKind: not JsonValueKind.Null };
        if (hasDescription)
        {
            sb.Append("\"description\":");
            sb.Append(DescriptionJson(description!.Value));
            sb.Append(',');
        }
        sb.Append("\"option_id\":");
        ClefJson.WriteString(sb, optionId);
        sb.Append('}');
        return sb.ToString();
    }

    private static string DescriptionJson(JsonElement value)
    {
        StringBuilder sb = new();
        if (value.ValueKind == JsonValueKind.String)
        {
            ClefJson.WriteString(sb, value.GetString()!);
            return sb.ToString();
        }
        sb.Append(ClefJson.Render(value));
        return sb.ToString();
    }

    private List<int> Tokens(string text) => [.. _tokenizer.Encode(text.Normalize(NormalizationForm.FormC), addSpecial: true)];
}
