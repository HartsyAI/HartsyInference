using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Fibo 2.0 prompt tokens. The caption sits in a Qwen3 user turn with no think block, rendered as
/// <c>&lt;|im_start|&gt;user\n{caption}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>; an empty prompt and the unconditional
/// branch use <see cref="NullCaption"/>. Glyph text images are not implemented, so a caption that asks for one is refused.</summary>
public static class Fibo2PromptTemplate
{
    /// <summary>ComfyUI's null caption: the empty structured prompt that stands in for an empty prompt and for CFG.</summary>
    public const string NullCaption =
        "{\"short_description\":\"\",\"objects\":[],\"background_setting\":\"\",\"lighting\":{},\"aesthetics\":{},"
        + "\"photographic_characteristics\":{},\"style_medium\":\"\",\"context\":\"\",\"artistic_style\":\"\"}";

    /// <summary>The <c>text_render</c> entry that asks for a glyph text image.</summary>
    private const string GlyphMarker = "text_render";

    /// <summary>The caption the conditional branch encodes: the prompt, or <see cref="NullCaption"/> when it is empty.</summary>
    public static string CaptionFor(string prompt) => string.IsNullOrWhiteSpace(prompt) ? NullCaption : prompt;

    /// <summary>Tokenizes <paramref name="caption"/> inside the Fibo 2.0 template, unpadded and untruncated.</summary>
    /// <exception cref="NotSupportedException">The caption contains a <c>text_render</c> entry.</exception>
    public static int[] Tokenize(Qwen3Tokenizer tokenizer, string caption)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(caption);
        if (caption.Contains(GlyphMarker, StringComparison.Ordinal))
            throw new NotSupportedException("glyph text images are not implemented yet for Fibo 2.0 (text_render entries)");
        return tokenizer.EncodeChatUnpadded(caption, systemPrompt: null, includeThinkBlock: false);
    }
}
