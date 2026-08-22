using Microsoft.ML.Tokenizers;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Thin tokenizer wrapper for VibeVoice, reusing <see cref="Qwen3Tokenizer"/>'s embedded vocab.json / merges.txt since Qwen2.5 and Qwen3 share the same 151 936-entry byte-level BPE vocab.</summary>
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

    private readonly Tokenizer _bpe;
    private int _disposed;

    public VibeVoiceTokenizer()
    {
        using Stream vocab = EmbeddedTokenizerResources.OpenQwen3Vocab();
        using Stream merges = EmbeddedTokenizerResources.OpenQwen3Merges();
        _bpe = BpeTokenizer.Create(vocab, merges);
    }

    /// <summary>Encodes a single string into a variable-length token-id array; does NOT add BOS / EOS — the multi-speaker processor splices in <c>&lt;|vision_*|&gt;</c> sentinels and the final <c>&lt;|endoftext|&gt;</c> via the prompt template.</summary>
    public int[] Encode(string text)
    {
        ThrowIfDisposed();
        // ML.Tokenizers' BpeTokenizer.Create(vocab, merges) does NOT apply the GPT-2 byte-level
        // pre-tokenizer, so it drops leading spaces (" Speaker" -> "Speaker") instead of matching the
        // Ġ-prefixed vocab keys ("ĠSpeaker"). Pre-encode into GPT-2 byte space so the prompt tokens
        // match Qwen2.5 exactly — otherwise the LM never sees the real script text and free-runs.
        IReadOnlyList<int> ids = _bpe.EncodeToIds(ByteLevelCodec.Encode(text));
        int[] result = new int[ids.Count];
        for (int i = 0; i < ids.Count; i++) result[i] = ids[i];
        return result;
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
        // Microsoft.ML.Tokenizers BpeTokenizer doesn't implement IDisposable in our version;
        // the underlying streams were already closed in the ctor. Nothing else to free.
    }
}
