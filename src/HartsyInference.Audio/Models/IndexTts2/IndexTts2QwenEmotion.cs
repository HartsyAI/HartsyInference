using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>IndexTTS-2's bundled free-text emotion classifier: a Qwen3-0.6B fine-tune (<c>qwen0.6bemo4-merge</c>)
/// that turns a sentence describing the desired emotion ("speak very angrily") into the 8-dim vector
/// <see cref="IndexTts2EmotionVectorLookup"/> consumes. Ports the real <c>QwenEmotion</c> class verbatim
/// (<c>indextts/infer_v2_5.py</c>, read in full) — same fixed system prompt, Chinese label vocabulary, score
/// clamp, all-zero→calm fallback, and the documented sad/melancholic workaround (QwenEmotion's text analysis
/// alone cannot tell "sad" from "melancholic" apart; a fixed keyword list in the INPUT text forces the swap).
/// Only the glue is new — the actual generation runs through the existing, already-complete
/// <see cref="TextGenerationPipeline"/> (chat template → tokenize → prefill → decode → detokenize), with
/// <see cref="SamplingOptions.JsonMode"/> auto-wiring grammar-constrained JSON output, same as every other
/// JSON-mode caller in this codebase.</summary>
public sealed class IndexTts2QwenEmotion
{
    private const string SystemPrompt = "文本情感分类";
    private const float MinScore = 0f, MaxScore = 1.2f;

    /// <summary>Real <c>desired_vector_order</c> / <c>cn_key_to_en</c> — fixed category order and mapping.
    /// "calm" is last, matching <see cref="IndexTts2EmotionVectorLookup.EmoNum"/>'s own order.</summary>
    private static readonly (string Cn, string En)[] Categories =
    [
        ("高兴", "happy"), ("愤怒", "angry"), ("悲伤", "sad"), ("恐惧", "afraid"),
        ("反感", "disgusted"), ("低落", "melancholic"), ("惊讶", "surprised"), ("自然", "calm"),
    ];

    /// <summary>Real <c>melancholic_words</c>: phrases in the INPUT text that force the "sad"/"melancholic"
    /// vectors to swap, working around the model's inability to tell them apart from text alone.</summary>
    private static readonly string[] MelancholicWords = ["低落", "melancholy", "melancholic", "depression", "depressed", "gloomy"];

    private static readonly string[] AliasKeys = ["emotion", "emotion_label", "label", "情感", "情绪"];

    private readonly TextGenerationPipeline _pipeline;

    public IndexTts2QwenEmotion(TextGenerationPipeline pipeline) => _pipeline = pipeline;

    /// <summary>Real category order (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm).</summary>
    public static IReadOnlyList<string> EnglishCategoryOrder { get; } = [.. Categories.Select(c => c.En)];

    /// <summary>Runs the classifier on free text and returns an 8-dim emotion vector in
    /// <see cref="EnglishCategoryOrder"/>'s order, ready for <see cref="IndexTts2EmotionVectorLookup.ComputeEmoVecMat"/>.</summary>
    public float[] Infer(string text)
    {
        GenerationRequest request = new()
        {
            Messages = [new ChatMessage("system", SystemPrompt), new ChatMessage("user", text)],
            EnableThinking = false,
            // The real class caps at 32768 new tokens, but generation always stops on EOS first for this
            // one-shot classification task in practice; capping lower here only guards against a pathological
            // non-stopping run, it does not change normal behavior.
            MaxTokens = 1024,
            Sampling = SamplingOptions.Default with { JsonMode = true },
        };
        string content = _pipeline.Generate(request).Text;
        return ParseAndConvert(content, text);
    }

    /// <summary>The pure parsing/normalization/conversion logic, split out from <see cref="Infer"/> so it is
    /// testable without a real model: <paramref name="modelOutput"/> is the classifier's raw generated text,
    /// <paramref name="inputText"/> is the original user text (only consulted for the melancholic-word
    /// workaround).</summary>
    internal static float[] ParseAndConvert(string modelOutput, string inputText)
    {
        string content = modelOutput;

        // Real source defensively strips a leading `<think>...</think>` block even with thinking disabled
        // (a reasoning model's chat template can still emit one). String search here, not the real class's
        // token-id search (151668) — equivalent in practice since this path is a no-op for this model
        // (enable_thinking=false), and avoids re-decoding from TokenIds for a defensive-only branch.
        int thinkClose = content.LastIndexOf("</think>", StringComparison.Ordinal);
        if (thinkClose >= 0) content = content[(thinkClose + "</think>".Length)..];

        Dictionary<string, double> scores;
        try
        {
            using JsonDocument doc = JsonDocument.Parse(content);
            scores = NormalizeFromJson(doc.RootElement);
        }
        catch (JsonException)
        {
            scores = FallbackRegexParse(content);
        }

        if (ContainsMelancholicWord(inputText))
        {
            (double sad, double melancholic) = (scores.GetValueOrDefault("悲伤"), scores.GetValueOrDefault("低落"));
            scores["悲伤"] = melancholic;
            scores["低落"] = sad;
        }

        return Convert(scores);
    }

    /// <summary>Real <c>normalize_content</c>: handles a bare string label, alias keys
    /// (<see cref="AliasKeys"/>), and per-category string values, converting any of them into a one-hot
    /// Chinese-key score. A plain numeric object (the common case under JSON-mode grammar) passes through
    /// via the first loop and nothing else fires.</summary>
    private static Dictionary<string, double> NormalizeFromJson(JsonElement root)
    {
        Dictionary<string, double> result = [];
        if (root.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty p in root.EnumerateObject())
                if (p.Value.ValueKind == JsonValueKind.Number && p.Value.TryGetDouble(out double v))
                    result[p.Name] = v;
        }

        string? detected = root.ValueKind == JsonValueKind.String ? LabelToCnKey(root.GetString()) : null;
        if (detected is null && root.ValueKind == JsonValueKind.Object)
        {
            foreach (string alias in AliasKeys)
            {
                if (root.TryGetProperty(alias, out JsonElement aliasVal) && aliasVal.ValueKind == JsonValueKind.String)
                {
                    detected = LabelToCnKey(aliasVal.GetString());
                    if (detected is not null) break;
                }
            }
        }
        if (detected is not null && Categories.All(c => !result.ContainsKey(c.Cn))) result[detected] = 1.0;

        if (root.ValueKind == JsonValueKind.Object)
        {
            foreach ((string cn, _) in Categories)
            {
                if (root.TryGetProperty(cn, out JsonElement v) && v.ValueKind == JsonValueKind.String)
                {
                    string? d = LabelToCnKey(v.GetString());
                    if (d is not null)
                    {
                        result[cn] = d == cn ? 1.0 : 0.0;
                        if (d != cn) result[d] = 1.0;
                    }
                }
            }
        }
        return result;
    }

    private static string? LabelToCnKey(string? value)
    {
        if (value is null) return null;
        value = value.Trim();
        foreach ((string cn, _) in Categories) if (value == cn) return cn;
        string lower = value.ToLowerInvariant();
        foreach ((string cn, string en) in Categories) if (lower == en) return cn;
        return null;
    }

    /// <summary>Real fallback when <c>json.loads</c> fails: <c>re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)')</c>.</summary>
    private static Dictionary<string, double> FallbackRegexParse(string content)
    {
        Dictionary<string, double> dict = [];
        foreach (Match m in Regex.Matches(content, "([^\\s\":.,]+?)\"?\\s*:\\s*([\\d.]+)"))
            if (double.TryParse(m.Groups[2].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double v))
                dict[m.Groups[1].Value] = v;
        return dict;
    }

    private static bool ContainsMelancholicWord(string text)
    {
        string lower = text.ToLowerInvariant();
        foreach (string w in MelancholicWords) if (lower.Contains(w, StringComparison.Ordinal)) return true;
        return false;
    }

    /// <summary>Real <c>convert</c>: Chinese→English order, clamp to [0, 1.2], default to calm if all zero.</summary>
    private static float[] Convert(Dictionary<string, double> scores)
    {
        float[] result = new float[Categories.Length];
        for (int i = 0; i < Categories.Length; i++)
            result[i] = (float)Math.Clamp(scores.GetValueOrDefault(Categories[i].Cn, 0.0), MinScore, MaxScore);
        if (Array.TrueForAll(result, v => v <= 0f)) result[^1] = 1f;   // "calm" is last.
        return result;
    }
}
