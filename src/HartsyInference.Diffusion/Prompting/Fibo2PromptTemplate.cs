using System.Text;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Fibo 2.0 prompt tokens. The caption sits in a Qwen3 user turn with no think block, rendered as
/// <c>&lt;|im_start|&gt;user\n{caption}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>; an empty prompt and the unconditional
/// branch use <see cref="NullCaption"/>. A JSON caption is re-serialised first as ComfyUI does (<see cref="Normalize"/>). Glyph text
/// images need a glyph rasteriser the engine does not have yet, so <see cref="TokenizePrompt"/> refuses a caption that asks for one.</summary>
public static class Fibo2PromptTemplate
{
    /// <summary>ComfyUI's null caption: the empty structured prompt that stands in for an empty prompt and for CFG.</summary>
    public const string NullCaption =
        "{\"short_description\":\"\",\"objects\":[],\"background_setting\":\"\",\"lighting\":{},\"aesthetics\":{},"
        + "\"photographic_characteristics\":{},\"style_medium\":\"\",\"context\":\"\",\"artistic_style\":\"\"}";

    /// <summary>ComfyUI's <c>VISION_MARKER</c>: where a glyph image sits in the caption.</summary>
    public const string VisionMarker = "<|vision_start|><|image_pad|><|vision_end|>";

    /// <summary>The sigma at and below which a generation shows Qwen3-VL the glyph images (ComfyUI's <c>TEXT_IMAGES_START</c> at shift 1.15).</summary>
    public const double TextImagesSigma = 0.9;

    /// <summary>Qwen3-VL's <c>&lt;|vision_start|&gt;</c> id.</summary>
    public const int VisionStartId = 151652;

    /// <summary>Qwen3-VL's <c>&lt;|vision_end|&gt;</c> id.</summary>
    public const int VisionEndId = 151653;

    /// <summary>Qwen3-VL's <c>&lt;|image_pad|&gt;</c> id, once per vision token.</summary>
    public const int ImagePadId = 151655;

    private const string ImageTokensKey = "image_tokens";

    /// <summary>The caption the conditional branch encodes: the prompt, or <see cref="NullCaption"/> when it is empty.</summary>
    public static string CaptionFor(string prompt) => string.IsNullOrWhiteSpace(prompt) ? NullCaption : prompt;

    /// <summary>True when a generation at <paramref name="sigma"/> encodes the caption with glyph images (ComfyUI's two-pass gate).</summary>
    public static bool UsesGlyphImages(double sigma) => sigma <= TextImagesSigma;

    /// <summary>ComfyUI's caption handling before templating. An empty prompt is <see cref="NullCaption"/> untouched. Otherwise the
    /// prompt is parsed as JSON: a non-object stays as written, an object is cleaned and re-serialised compactly (a caption with
    /// nothing left is <c>{}</c>). Each <c>text_render</c> entry with a non-empty <c>text</c> gains an <c>image_tokens</c> key holding
    /// <see cref="VisionMarker"/> when <paramref name="textImages"/> is set and an empty string otherwise.</summary>
    public static Fibo2NormalizedPrompt Normalize(string prompt, bool textImages)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        if (prompt.Trim().Length == 0)
            return new Fibo2NormalizedPrompt(NullCaption, []);

        object? caption = Fibo2CaptionCleaner.Parse(prompt);
        List<string> glyphTexts = [];
        if (caption is PyDict dict && dict.Get("text_render") is List<object?> entries)
        {
            foreach (object? item in entries)
            {
                if (item is PyDict entry && entry.Get("text") is string text && text.Length > 0)
                {
                    entry.Set(ImageTokensKey, textImages ? VisionMarker : string.Empty);
                    glyphTexts.Add(text);
                }
            }
        }
        string normalized = caption is string ? prompt : PythonJson.Dumps(caption ?? new PyDict());
        return new Fibo2NormalizedPrompt(normalized, glyphTexts);
    }

    /// <summary>Tokenizes <paramref name="caption"/> inside the Fibo 2.0 template as given (no normalisation), unpadded and untruncated.</summary>
    public static int[] Tokenize(Qwen3Tokenizer tokenizer, string caption)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(caption);
        return tokenizer.EncodeChatUnpadded(caption, systemPrompt: null, includeThinkBlock: false);
    }

    /// <summary>Tokenizes a user prompt the way ComfyUI does for a text-only generation.</summary>
    /// <exception cref="NotSupportedException">The caption has a <c>text_render</c> entry with text, which needs a glyph image.</exception>
    public static int[] TokenizePrompt(Qwen3Tokenizer tokenizer, string prompt)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        Fibo2NormalizedPrompt normalized = Normalize(prompt, textImages: false);
        if (normalized.GlyphTexts.Count > 0)
            throw new NotSupportedException("glyph text images are not implemented yet for Fibo 2.0 (text_render entries): the engine has no glyph rasteriser");
        return Tokenize(tokenizer, normalized.Text);
    }

    /// <summary>The ids of the pass used above sigma <see cref="TextImagesSigma"/>: glyph entries stay, their markers empty, no images.</summary>
    public static int[] TokenizeWithoutImages(Qwen3Tokenizer tokenizer, string prompt)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        return Tokenize(tokenizer, Normalize(prompt, textImages: false).Text);
    }

    /// <summary>The ids of the pass used at or below sigma <see cref="TextImagesSigma"/>: each glyph marker becomes
    /// <c>&lt;|vision_start|&gt;</c>, <paramref name="padCounts"/>[i] image pads and <c>&lt;|vision_end|&gt;</c>.</summary>
    public static int[] TokenizeWithImages(Qwen3Tokenizer tokenizer, string prompt, IReadOnlyList<int> padCounts)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(padCounts);
        Fibo2NormalizedPrompt normalized = Normalize(prompt, textImages: true);
        if (padCounts.Count != normalized.GlyphTexts.Count)
            throw new ArgumentException($"The caption has {normalized.GlyphTexts.Count} glyph images, got {padCounts.Count} pad counts.", nameof(padCounts));
        if (normalized.GlyphTexts.Count == 0)
            return Tokenize(tokenizer, normalized.Text);

        string[] segments = normalized.Text.Split(VisionMarker);
        List<int> ids = [Qwen3Tokenizer.ImStartId];
        for (int i = 0; i < segments.Length; i++)
        {
            string segment = i == 0 ? string.Concat("user\n", segments[0]) : segments[i];
            if (i > 0)
            {
                if (padCounts[i - 1] < 1)
                    throw new ArgumentOutOfRangeException(nameof(padCounts), $"Glyph image {i} needs at least one vision token.");
                ids.Add(VisionStartId);
                for (int p = 0; p < padCounts[i - 1]; p++)
                    ids.Add(ImagePadId);
                ids.Add(VisionEndId);
            }
            ids.AddRange(tokenizer.EncodeRaw(segment));
        }
        ids.Add(Qwen3Tokenizer.ImEndId);
        ids.AddRange(tokenizer.EncodeRaw("\n"));
        ids.Add(Qwen3Tokenizer.ImStartId);
        ids.AddRange(tokenizer.EncodeRaw("assistant\n"));
        return [.. ids];
    }
}
