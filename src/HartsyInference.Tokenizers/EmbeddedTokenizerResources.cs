using System.Reflection;

namespace HartsyInference.Tokenizers;

/// <summary>Streams over the canonical tokenizer vocabularies shipped inside this assembly. Each method returns a fresh <see cref="Stream"/> the caller owns and must dispose. Throws <see cref="InvalidOperationException"/> if the resource is missing (build-time misconfiguration).</summary>
public static class EmbeddedTokenizerResources
{
    /// <summary>OpenAI CLIP BPE vocabulary (49,408 tokens). Used by CLIP-L and CLIP-G —
    /// they share tokenization; only the encoder model size differs.</summary>
    public const string ClipVocabName   = "HartsyInference.Tokenizers.Resources.clip_vocab.json";

    /// <summary>OpenAI CLIP BPE merges. Pairs with <see cref="ClipVocabName"/>.</summary>
    public const string ClipMergesName  = "HartsyInference.Tokenizers.Resources.clip_merges.txt";

    /// <summary>T5 SentencePiece model (32,128 tokens). Same protobuf serves T5-base and
    /// T5-XXL; the encoder weights differ but the tokenizer is identical.</summary>
    public const string T5SpieceName    = "HartsyInference.Tokenizers.Resources.t5_spiece.model";

    /// <summary>umT5 multilingual SentencePiece model (256k vocab, from <c>google/umt5-xxl</c>).
    /// Used by Wan-Video's umT5-XXL conditioning — NOT interchangeable with <see cref="T5SpieceName"/>.</summary>
    public const string Umt5SpieceName  = "HartsyInference.Tokenizers.Resources.umt5_spiece.model";

    /// <summary>Qwen3-4B BPE vocabulary (151,936 tokens). Used by Flux.2 Klein and
    /// Z-Image text conditioning.</summary>
    public const string Qwen3VocabName  = "HartsyInference.Tokenizers.Resources.qwen3_vocab.json";

    /// <summary>Qwen3-4B BPE merges. Pairs with <see cref="Qwen3VocabName"/>.</summary>
    public const string Qwen3MergesName = "HartsyInference.Tokenizers.Resources.qwen3_merges.txt";

    /// <summary>Llama-3.1 byte-level BPE vocabulary (128,256 tokens). Used by HiDream-I1's fourth text
    /// encoder (Llama-3.1-8B-Instruct as a feature extractor). Extracted from the Llama-3.1 tokenizer.json
    /// into the vocab.json + merges.txt two-file form.</summary>
    public const string Llama3VocabName  = "HartsyInference.Tokenizers.Resources.llama3_vocab.json";

    /// <summary>Llama-3.1 byte-level BPE merges. Pairs with <see cref="Llama3VocabName"/>.</summary>
    public const string Llama3MergesName = "HartsyInference.Tokenizers.Resources.llama3_merges.txt";

    /// <summary>Canonical Llama-3 <c>tokenizer.json</c> (the single artifact the Llama-3.x repos ship; vocab
    /// 128,256). Consumed via <see cref="HfTokenizerJson"/> to build the byte-level BPE core directly — used by
    /// the Orpheus / CSM audio front-ends. Shared by every Llama-3.x model (3 / 3.1 / 3.2 use one tokenizer).</summary>
    public const string Llama3TokenizerJsonName = "HartsyInference.Tokenizers.Resources.llama3_tokenizer.json";

    /// <summary>Chatterbox text <c>tokenizer.json</c> (plain-char BPE, vocab 704, Whitespace pre-tokenizer).
    /// Consumed by <see cref="ChatterboxEnTokenizer"/>; embedded so Chatterbox TTS ships self-contained.</summary>
    public const string ChatterboxTokenizerJsonName = "HartsyInference.Tokenizers.Resources.chatterbox_tokenizer.json";

    private static readonly Assembly _asm = typeof(EmbeddedTokenizerResources).Assembly;

    /// <summary>Opens the named embedded resource. Caller owns the returned stream.</summary>
    public static Stream Open(string logicalName)
    {
        Stream? s = _asm.GetManifestResourceStream(logicalName);
        if (s is null)
        {
            throw new InvalidOperationException(
                $"Embedded tokenizer resource '{logicalName}' is missing from " +
                $"{_asm.GetName().Name}. Available resources: " +
                string.Join(", ", _asm.GetManifestResourceNames()));
        }
        return s;
    }

    public static Stream OpenClipVocab()   => Open(ClipVocabName);
    public static Stream OpenClipMerges()  => Open(ClipMergesName);
    public static Stream OpenT5Spiece()    => Open(T5SpieceName);
    public static Stream OpenUmt5Spiece()  => Open(Umt5SpieceName);
    public static Stream OpenQwen3Vocab()  => Open(Qwen3VocabName);
    public static Stream OpenQwen3Merges() => Open(Qwen3MergesName);
    public static Stream OpenLlama3Vocab() => Open(Llama3VocabName);
    public static Stream OpenLlama3Merges() => Open(Llama3MergesName);
    public static Stream OpenLlama3TokenizerJson() => Open(Llama3TokenizerJsonName);
    public static Stream OpenChatterboxTokenizerJson() => Open(ChatterboxTokenizerJsonName);

    /// <summary>True when the canonical Llama-3 <c>tokenizer.json</c> is embedded in this build.</summary>
    public static bool HasLlama3TokenizerJson =>
        _asm.GetManifestResourceStream(Llama3TokenizerJsonName) is not null;

    /// <summary>True when the Llama-3.1 tokenizer assets are embedded in this build. They ship separately
    /// from the source tree (the vocab.json is ~5 MB) and are included via a conditional csproj glob, so a
    /// checkout without the asset files still compiles — callers that need Llama tokenization (HiDream)
    /// should check this and surface a clear "drop the assets in Resources/" message rather than NRE.</summary>
    public static bool HasLlama3Assets =>
        _asm.GetManifestResourceStream(Llama3VocabName) is not null
        && _asm.GetManifestResourceStream(Llama3MergesName) is not null;
}
