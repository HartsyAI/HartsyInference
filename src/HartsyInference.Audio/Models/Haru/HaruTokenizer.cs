using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Haru's text tokenizer: the Qwen3.5 byte-level BPE, in which the PAD and EOP tokens of the text stream are the added tokens
/// <c>&lt;|audio_pad|&gt;</c> and <c>&lt;|audio_end|&gt;</c>.</summary>
internal static class HaruTokenizer
{
    internal const string PadLiteral = "<|audio_pad|>";
    internal const string EopLiteral = "<|audio_end|>";
    internal const string TurnStartLiteral = "<|im_start|>";
    internal const string TurnEndLiteral = "<|im_end|>";
    internal const string EndOfTextLiteral = "<|endoftext|>";

    /// <summary>Reads <c>tokenizer.json</c> and checks that the tokens the frame loop and the prompt rely on sit at the ids the
    /// config names: a tokenizer from another build would put PAD and EOP on the wrong tokens and the model would never speak. The
    /// caller owns <paramref name="tokenizerJson"/>.</summary>
    internal static GgufTokenizer Load(Stream tokenizerJson, HaruConfig config)
    {
        ArgumentNullException.ThrowIfNull(tokenizerJson);
        ArgumentNullException.ThrowIfNull(config);
        GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerJson);
        Expect(tokenizer, PadLiteral, config.PadTokenId);
        Expect(tokenizer, EopLiteral, config.EopTokenId);
        if (config.EosTokenId is int eos)
        {
            Expect(tokenizer, EndOfTextLiteral, eos);
        }
        foreach (string literal in new[] { TurnStartLiteral, TurnEndLiteral })
        {
            if (tokenizer.SpecialId(literal) is not int id || id >= config.Lm.VocabSize)
            {
                throw new InvalidDataException($"The Haru tokenizer has no usable '{literal}' token.");
            }
        }
        return tokenizer;
    }

    private static void Expect(GgufTokenizer tokenizer, string literal, int expected)
    {
        int? id = tokenizer.SpecialId(literal);
        if (id != expected)
        {
            throw new InvalidDataException($"The Haru tokenizer maps '{literal}' to {id?.ToString() ?? "nothing"}; its config expects {expected}. "
                + "The tokenizer and the weights come from different builds.");
        }
    }
}
