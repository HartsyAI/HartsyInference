namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>YuE2's text tokenizer. Unlike every other model in the tree it ships no side tokenizer file — the whole
/// HuggingFace <c>tokenizers</c> document rides inside the checkpoint as the <c>text_encoders.yue2_tokenizer_json</c>
/// U8 tensor, so the only way to construct one is from bytes the checkpoint loader already holds.</summary>
/// <remarks>The vocabulary is a 184,704-entry superset of Qwen's: ordinary text below <c>EOD</c> (151,643), the
/// protocol's span markers just above it, and 32,768 codec ids from <c>CODEC_OFFSET</c> up. It is unrelated to
/// <see cref="YueTokenizer">YuE v1's</see>.</remarks>
public sealed class Yue2Tokenizer
{
    private readonly GgufTokenizer _bpe;

    /// <summary>Builds a tokenizer from the checkpoint's embedded document.</summary>
    public Yue2Tokenizer(ReadOnlySpan<byte> tokenizerJson)
    {
        if (tokenizerJson.IsEmpty)
            throw new ArgumentException("The YuE2 tokenizer document is empty.", nameof(tokenizerJson));
        using MemoryStream stream = new(tokenizerJson.ToArray(), writable: false);
        _bpe = HfTokenizerJson.LoadByteLevelBpe(stream);
    }

    /// <summary>Encodes prompt or score text. The protocol's span markers are added by
    /// <c>Yue2Protocol.TokenPrefix</c>, never by the tokenizer, so no special tokens are inserted here.</summary>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return _bpe.EncodeOrdinary(text);
    }

    /// <summary>Decodes planner output back to ABC notation.</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        return _bpe.Decode(ids);
    }
}
