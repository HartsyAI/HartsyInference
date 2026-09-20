namespace HartsyInference.ModelAssets.Tokenizers;

/// <summary>Turns a prompt into the fixed-length conditioning token sequence an LTX-2 pipeline feeds its text tower. LTX-2.3 uses Gemma 3's SentencePiece model (<see cref="GemmaTokenizer"/>) at 256 tokens; LTX-2.5 uses Gemma 4's rank-merge BPE (<see cref="Gemma4Tokenizer"/>) at 1024.</summary>
/// <remarks>Sequence length is part of the conditioning, not a padding detail: the text connector replaces learnable registers positionally, so encoding the same prompt at a different length produces different conditioning. That is why this returns a padded sequence rather than raw ids.</remarks>
public interface ILtx2PromptTokenizer
{
    /// <summary>Encodes <paramref name="text"/> to RAW ids (BOS + content, no padding). The pipeline assembles the conditioning sequence, because only it knows the connector's register multiple and it must mark which positions are real — pre-padding here would present pad tokens to the connector as content.</summary>
    int[] EncodeForConditioning(string text);

    /// <summary>Shortest conditioning sequence this family conditions at, or 0 to use only the register multiple. Gemma 4 is 1024; Gemma 3 keeps 0 so the shipping LTX-2.3 behaviour is unchanged.</summary>
    int MinimumConditioningLength { get; }

    /// <summary>One emphasis span's ids: CONTENT only, no sequence-start token and no padding. Needed because
    /// SwarmUI tokenizes each weighted leaf alone and splices it, and a per-span start token would put a stray
    /// sentence beginning in the middle of the caption.</summary>
    IReadOnlyList<int> EncodeSpan(string text);

    /// <summary>The single sequence-start id <see cref="EncodeForConditioning"/> puts before the content. Exposed
    /// so a caller assembling its own weighted sequence reproduces the same prefix instead of assuming one.</summary>
    int ConditioningStartId { get; }
}
