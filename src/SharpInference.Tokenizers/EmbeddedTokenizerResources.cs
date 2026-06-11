using System.Reflection;

namespace SharpInference.Tokenizers;

/// <summary>Streams over the canonical tokenizer vocabularies shipped inside this assembly. Each method returns a fresh <see cref="Stream"/> the caller owns and must dispose. Throws <see cref="InvalidOperationException"/> if the resource is missing (build-time misconfiguration).</summary>
public static class EmbeddedTokenizerResources
{
    /// <summary>OpenAI CLIP BPE vocabulary (49,408 tokens). Used by CLIP-L and CLIP-G —
    /// they share tokenization; only the encoder model size differs.</summary>
    public const string ClipVocabName   = "SharpInference.Tokenizers.Resources.clip_vocab.json";

    /// <summary>OpenAI CLIP BPE merges. Pairs with <see cref="ClipVocabName"/>.</summary>
    public const string ClipMergesName  = "SharpInference.Tokenizers.Resources.clip_merges.txt";

    /// <summary>T5 SentencePiece model (32,128 tokens). Same protobuf serves T5-base and
    /// T5-XXL; the encoder weights differ but the tokenizer is identical.</summary>
    public const string T5SpieceName    = "SharpInference.Tokenizers.Resources.t5_spiece.model";

    /// <summary>umT5 multilingual SentencePiece model (256k vocab, from <c>google/umt5-xxl</c>).
    /// Used by Wan-Video's umT5-XXL conditioning — NOT interchangeable with <see cref="T5SpieceName"/>.</summary>
    public const string Umt5SpieceName  = "SharpInference.Tokenizers.Resources.umt5_spiece.model";

    /// <summary>Qwen3-4B BPE vocabulary (151,936 tokens). Used by Flux.2 Klein and
    /// Z-Image text conditioning.</summary>
    public const string Qwen3VocabName  = "SharpInference.Tokenizers.Resources.qwen3_vocab.json";

    /// <summary>Qwen3-4B BPE merges. Pairs with <see cref="Qwen3VocabName"/>.</summary>
    public const string Qwen3MergesName = "SharpInference.Tokenizers.Resources.qwen3_merges.txt";

    /// <summary>Llama-3.1 byte-level BPE vocabulary (128,256 tokens). Used by HiDream-I1's fourth text
    /// encoder (Llama-3.1-8B-Instruct as a feature extractor). Extracted from the Llama-3.1 tokenizer.json
    /// into the vocab.json + merges.txt two-file form.</summary>
    public const string Llama3VocabName  = "SharpInference.Tokenizers.Resources.llama3_vocab.json";

    /// <summary>Llama-3.1 byte-level BPE merges. Pairs with <see cref="Llama3VocabName"/>.</summary>
    public const string Llama3MergesName = "SharpInference.Tokenizers.Resources.llama3_merges.txt";

    private static readonly Assembly Asm = typeof(EmbeddedTokenizerResources).Assembly;

    /// <summary>Opens the named embedded resource. Caller owns the returned stream.</summary>
    public static Stream Open(string logicalName)
    {
        Stream? s = Asm.GetManifestResourceStream(logicalName);
        if (s is null)
        {
            throw new InvalidOperationException(
                $"Embedded tokenizer resource '{logicalName}' is missing from " +
                $"{Asm.GetName().Name}. Available resources: " +
                string.Join(", ", Asm.GetManifestResourceNames()));
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

    /// <summary>True when the Llama-3.1 tokenizer assets are embedded in this build. They ship separately
    /// from the source tree (the vocab.json is ~5 MB) and are included via a conditional csproj glob, so a
    /// checkout without the asset files still compiles — callers that need Llama tokenization (HiDream)
    /// should check this and surface a clear "drop the assets in Resources/" message rather than NRE.</summary>
    public static bool HasLlama3Assets =>
        Asm.GetManifestResourceStream(Llama3VocabName) is not null
        && Asm.GetManifestResourceStream(Llama3MergesName) is not null;
}
