namespace HartsyInference.Tokenizers;

/// <summary>Abstraction over a decoder-LLM tokenizer so the generation pipeline and chat templates work with any
/// model (not just Qwen). Implemented by <see cref="GgufTokenizer"/> (built from GGUF metadata) and by the
/// per-family tokenizers. Encoding is special-token aware so a rendered chat template's control-token literals
/// (e.g. <c>&lt;|im_start|&gt;</c>, <c>&lt;|begin_of_text|&gt;</c>, <c>[INST]</c>) map to their real ids.</summary>
public interface ILlmTokenizer
{
    /// <summary>Encodes text to ids. When <paramref name="addSpecial"/> is true, registered special/control
    /// tokens appearing literally in <paramref name="text"/> are emitted as their ids (the rest is BPE); when
    /// false, the whole string is ordinary BPE. Chat templates render to a string then encode with
    /// <paramref name="addSpecial"/> = true.</summary>
    int[] Encode(string text, bool addSpecial);

    /// <summary>Ordinary BPE encode with no special-token handling (plain content text).</summary>
    int[] EncodeOrdinary(string text);

    /// <summary>Decodes ids back to UTF-8 text (byte-level reversed), skipping special/control tokens.</summary>
    string Decode(IReadOnlyList<int> ids);

    /// <summary>Id of a special/control token by its literal string, or null if not a known special token.</summary>
    int? SpecialId(string token);

    /// <summary>Beginning-of-sequence id, or null if the model has none.</summary>
    int? BosId { get; }

    /// <summary>Primary end-of-sequence id, or null.</summary>
    int? EosId { get; }

    /// <summary>All ids that should stop generation (EOS plus end-of-turn / end-of-message variants).</summary>
    IReadOnlyList<int> StopIds { get; }

    /// <summary>BOS token literal (for chat templates that reference <c>bos_token</c>), or null.</summary>
    string? BosToken { get; }

    /// <summary>EOS token literal (for chat templates that reference <c>eos_token</c>), or null.</summary>
    string? EosToken { get; }
}
