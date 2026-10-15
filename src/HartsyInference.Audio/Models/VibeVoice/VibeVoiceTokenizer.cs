using Microsoft.ML.Tokenizers;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Thin tokenizer wrapper for VibeVoice. Encodes with <see cref="Qwen2Tokenizer.EncodeExact"/> over the Qwen2.5
/// base vocabulary and the Qwen2.5 added-token table, which gives the same ids as upstream's Qwen2.5 tokenizer.</summary>
/// <remarks>
/// <para>This class exists for two reasons:
/// <list type="number">
///   <item>Expose <b>variable-length</b> Encode. Qwen3Tokenizer's main API pads to a
///         fixed window (suitable for diffusion text encoding), which is wrong for an AR
///         language model prompt where length matters at every position.</item>
///   <item>Pre-bind the <b>VibeVoice special-token IDs</b> — the multi-speaker pipeline
///         re-uses Qwen's <c>&lt;|vision_start/end/pad|&gt;</c> as
///         <c>speech_start/end/diffusion</c>. We don't extend the vocab; we just give those
///         existing IDs new logical names.</item>
/// </list></para></remarks>
public sealed class VibeVoiceTokenizer : IDisposable
{
    /// <summary>Speech-segment start. Re-uses Qwen's <c>&lt;|vision_start|&gt;</c> (ID 151 652).</summary>
    public const int SpeechStartTokenId = 151_652;

    /// <summary>Speech-segment end. Re-uses Qwen's <c>&lt;|vision_end|&gt;</c> (ID 151 653).</summary>
    public const int SpeechEndTokenId = 151_653;

    /// <summary>Per-frame diffusion sentinel; re-uses Qwen's <c>&lt;|vision_pad|&gt;</c> (ID 151 654). The LM emits one of these per 7.5 Hz acoustic frame.</summary>
    public const int SpeechDiffusionTokenId = 151_654;

    /// <summary><c>&lt;|endoftext|&gt;</c> — BOS and EOS for Qwen2.5.</summary>
    public const int EndOfTextTokenId = 151_643;

    /// <summary>Negative-prompt sentinel used by the streaming-0.5B CFG path (<c>&lt;|image_pad|&gt;</c>, ID 151 655); reserved for that variant only.</summary>
    public const int ImagePadTokenId = 151_655;

    /// <summary>The 22 added tokens of the Qwen2.5 tokenizer.json (ids 151643 to 151664). <c>Encode</c> recognizes exactly
    /// these literals; Qwen3-only ones such as <c>&lt;think&gt;</c> are absent, so they encode as ordinary text.</summary>
    private static readonly IReadOnlyDictionary<string, int> Qwen25AddedTokens = new Dictionary<string, int>(StringComparer.Ordinal)
    {
        ["<|endoftext|>"] = 151643, ["<|im_start|>"] = 151644, ["<|im_end|>"] = 151645,
        ["<|object_ref_start|>"] = 151646, ["<|object_ref_end|>"] = 151647, ["<|box_start|>"] = 151648,
        ["<|box_end|>"] = 151649, ["<|quad_start|>"] = 151650, ["<|quad_end|>"] = 151651,
        ["<|vision_start|>"] = 151652, ["<|vision_end|>"] = 151653, ["<|vision_pad|>"] = 151654,
        ["<|image_pad|>"] = 151655, ["<|video_pad|>"] = 151656, ["<tool_call>"] = 151657,
        ["</tool_call>"] = 151658, ["<|fim_prefix|>"] = 151659, ["<|fim_middle|>"] = 151660,
        ["<|fim_suffix|>"] = 151661, ["<|fim_pad|>"] = 151662, ["<|repo_name|>"] = 151663,
        ["<|file_sep|>"] = 151664,
    };

    private readonly Qwen2Tokenizer _qwen;
    private readonly Tokenizer _bpe;
    private int _disposed;

    public VibeVoiceTokenizer()
    {
        _qwen = Qwen2Tokenizer.CreateBaseVocabulary();
        using Stream vocab = EmbeddedTokenizerResources.OpenQwen3Vocab();
        using Stream merges = EmbeddedTokenizerResources.OpenQwen3Merges();
        _bpe = BpeTokenizer.Create(vocab, merges);
    }

    /// <summary>Encodes a single string into a variable-length token-id array; does NOT add BOS / EOS — the multi-speaker processor splices in <c>&lt;|vision_*|&gt;</c> sentinels and the final <c>&lt;|endoftext|&gt;</c> via the prompt template.</summary>
    public int[] Encode(string text)
    {
        ThrowIfDisposed();
        // Recognizes exactly the Qwen2.5 added tokens, as upstream's tokenizer does. Qwen3-only literals such as
        // <think> are not in the table, so they encode as ordinary text.
        return _qwen.EncodeExact(text, Qwen25AddedTokens);
    }

    /// <summary>Decodes a token-id sequence back to text; useful for logging the prompt the LM actually received.</summary>
    public string Decode(ReadOnlySpan<int> ids)
    {
        ThrowIfDisposed();
        int[] buf = new int[ids.Length];
        ids.CopyTo(buf);
        // Reverse the byte-level mapping applied in Encode so logged text reads normally (no Ġ/Ċ).
        return ByteLevelCodec.Decode(_bpe.Decode(buf) ?? string.Empty);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(VibeVoiceTokenizer));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _qwen.Dispose();
        // Microsoft.ML.Tokenizers BpeTokenizer doesn't implement IDisposable in our version;
        // the underlying streams were already closed in the ctor. Nothing else to free.
    }
}
