using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>The text side of HunyuanVideo-1.5 and MovieRoll prompts, as the MovieRoll reference builds them
/// (<c>movieroll/models/text_encoders/__init__.py</c>, <c>glyph_text.py</c>, <c>byT5/format_prompt.py</c>, <c>utils/load.py</c> at 0b5f73e):
/// the Qwen2.5-VL chat prompt and its crop point, and the glyph text the ByT5 stream reads.
/// <para>Qwen features are tapped at <c>hidden_states[-3]</c> (<c>hidden_state_skip_layer=2</c>) without the final norm, with the attention mask on,
/// then the template prefix is cropped. A prompt without quoted text has no ByT5 tokens at all: the reference feeds zeros under an all-zero mask
/// and the mask drops them, so the DiT sees no byT5 rows.</para></summary>
public static partial class HunyuanVideo15Prompt
{
    /// <summary>The video encode template's system message, byte for byte. The reference writes it as a Python string with backslash line
    /// continuations, so each of the five numbered items is preceded by a space plus eight spaces of source indentation.</summary>
    public const string SystemPrompt =
        "You are a helpful assistant. Describe the video by detailing the following aspects:         " +
        "1. The main content and theme of the video.         " +
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         " +
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         " +
        "4. background environment, light, style and atmosphere.         " +
        "5. camera angles, movements, and transitions used in the video.";

    /// <summary>Prompt tokens kept after the template prefix is cropped (<c>text_len</c> 1000 in the shipped profiles).</summary>
    public const int MaxTextTokens = 1000;

    /// <summary>Counted back from the last layer: 2 skips the last two blocks, so hidden state <c>NumLayers - 2</c> in HuggingFace indexing.</summary>
    public const int HiddenStateSkipLayer = 2;

    /// <summary>ByT5 token window (<c>byt5_max_length</c> 256).</summary>
    public const int ByT5MaxTokens = 256;

    private static readonly Dictionary<string, int> ChatSpecialTokens = new()
    {
        ["<|im_start|>"] = Qwen2Tokenizer.ImStartId,
        ["<|im_end|>"] = Qwen2Tokenizer.ImEndId,
    };

    /// <summary>Chat-template token ids for <paramref name="prompt"/> (system message, user turn, generation prompt), truncated on the right to
    /// <see cref="MaxTextTokens"/> plus the template prefix, with the crop point: the number of leading template tokens to drop from the hidden
    /// states. Mirrors <c>text2tokens</c>: <c>apply_chat_template(add_generation_prompt=True)</c>, <c>max_length = text_len + crop_start</c>.
    /// <para>Encoded with <see cref="Qwen2Tokenizer.EncodeExact"/>, which matches the HuggingFace tokenizer (every newline is token 198).
    /// <see cref="Qwen2Tokenizer.EncodeChat"/> goes through the legacy path that drops bare newlines and is not used here. Like the reference's
    /// <c>truncation=True</c>, a prompt longer than the window loses the end of the template, the generation prompt included.</para></summary>
    public static (int[] Ids, int CropStart) Tokenize(Qwen2Tokenizer tokenizer, string prompt)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(prompt);
        string text = $"<|im_start|>system\n{SystemPrompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
        int[] full = tokenizer.EncodeExact(text, ChatSpecialTokens);
        int crop = FindCropStart(full, tokenizer);
        int length = Math.Min(full.Length, MaxTextTokens + crop);
        return (full[..length], crop);
    }

    /// <summary>The index just past the first <c>&lt;|im_start|&gt;user\n</c> marker (three ids: start, <c>user</c>, newline), or 0 when the
    /// marker is absent (the reference then falls back to a tokenizer special token naming the user role, which Qwen does not have, so it
    /// crops nothing).</summary>
    public static int FindCropStart(ReadOnlySpan<int> ids, Qwen2Tokenizer tokenizer)
    {
        int[] userTokens = tokenizer.EncodeExact("user\n");
        int markerLength = userTokens.Length + 1;
        for (int i = 0; i + markerLength <= ids.Length; i++)
        {
            if (ids[i] != Qwen2Tokenizer.ImStartId) continue;
            bool match = true;
            for (int j = 0; j < userTokens.Length && match; j++) match = ids[i + 1 + j] == userTokens[j];
            if (match) return i + markerLength;
        }
        return 0;
    }

    /// <summary>The quoted strings of <paramref name="prompt"/> in order of appearance (ASCII <c>"…"</c> and curly <c>“…”</c> quotes, one pattern),
    /// with repeats dropped when there is more than one. Mirrors <c>extract_glyph_texts</c>; unlike <see cref="HunyuanImageGlyphText"/> it does
    /// not read <c>‘…’</c> and does not collect ASCII spans first.</summary>
    public static IReadOnlyList<string> ExtractGlyphTexts(string? prompt)
    {
        if (string.IsNullOrEmpty(prompt)) return [];
        List<string> texts = [];
        foreach (Match match in QuotedSpan().Matches(prompt))
            texts.Add(match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value);
        if (texts.Count <= 1) return texts;
        // dict.fromkeys in the reference: first occurrence wins, order of appearance kept.
        HashSet<string> seen = new(StringComparer.Ordinal);
        List<string> unique = [];
        foreach (string text in texts)
            if (seen.Add(text)) unique.Add(text);
        return unique;
    }

    /// <summary>The ByT5 input for <paramref name="prompt"/>: <c>Text "…". </c> per glyph text (no color or font tokens, the reference passes
    /// none), or null when the prompt quotes nothing and the byT5 stream stays empty.</summary>
    public static string? FormatGlyphText(string? prompt)
    {
        IReadOnlyList<string> texts = ExtractGlyphTexts(prompt);
        if (texts.Count == 0) return null;
        StringBuilder text = new();
        foreach (string span in texts) text.Append("Text \"").Append(span).Append("\". ");
        return text.ToString();
    }

    // Python's re: '.' stops at a newline and '*?' is lazy; findall returns group 1 or group 2 depending on the alternative that matched.
    [GeneratedRegex("\"(.*?)\"|“(.*?)”")]
    private static partial Regex QuotedSpan();
}
