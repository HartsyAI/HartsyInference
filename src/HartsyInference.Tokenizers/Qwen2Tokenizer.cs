using Microsoft.ML.Tokenizers;

namespace HartsyInference.Tokenizers;

/// <summary>Qwen2/Qwen2.5(-VL) byte-level BPE tokenizer with the ChatML template, used by Lance (chat-templated T2I/T2V prompts) and HunyuanImage 2.1 (Qwen2.5-VL MLLM encoder). Qwen2.5 and Qwen3 share the same base BPE vocab/merges (151,936 entries), so the default constructor reuses the embedded Qwen3 assets; pass explicit paths when a checkpoint ships its own <c>vocab.json</c>/<c>merges.txt</c> (Lance does).</summary>
public sealed class Qwen2Tokenizer : IDisposable
{
    /// <summary>Vocabulary size (matches Qwen2.5-VL's <c>config.json</c>).</summary>
    public const int VocabSize = 151936;

    /// <summary>Token id for <c>&lt;|endoftext|&gt;</c> — Qwen2's BOS and conventional pad token.</summary>
    public const int EndOfTextId = 151643;

    /// <summary>Token id for <c>&lt;|im_start|&gt;</c>.</summary>
    public const int ImStartId = 151644;

    /// <summary>Token id for <c>&lt;|im_end|&gt;</c> — Qwen2's EOS.</summary>
    public const int ImEndId = 151645;

    /// <summary>Token id for <c>&lt;|vision_start|&gt;</c> (Qwen2.5-VL / Lance sequence sentinel).</summary>
    public const int VisionStartId = 151652;

    /// <summary>Token id for <c>&lt;|vision_end|&gt;</c>.</summary>
    public const int VisionEndId = 151653;

    /// <summary>Token id for <c>&lt;|vision_pad|&gt;</c>.</summary>
    public const int VisionPadId = 151654;

    /// <summary>Token id for <c>&lt;|image_pad|&gt;</c>.</summary>
    public const int ImagePadId = 151655;

    /// <summary>Token id for <c>&lt;|video_pad|&gt;</c>.</summary>
    public const int VideoPadId = 151656;

    /// <summary>Default ChatML system prompt used by Qwen2.5-VL when none is supplied.</summary>
    public const string DefaultSystemPrompt = "You are a helpful assistant.";

    private readonly Tokenizer _tokenizer;
    private int _disposed;

    /// <summary>Creates a Qwen2 tokenizer from the embedded Qwen3 vocab/merges (identical base BPE for Qwen2.5; only the chat conventions differ, which this class owns).</summary>
    public Qwen2Tokenizer()
    {
        using Stream vocabStream = EmbeddedTokenizerResources.OpenQwen3Vocab();
        using Stream mergesStream = EmbeddedTokenizerResources.OpenQwen3Merges();
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Creates a Qwen2 tokenizer from <c>vocab.json</c> and <c>merges.txt</c> files (e.g. the ones shipped inside a Lance checkpoint folder).</summary>
    public Qwen2Tokenizer(string vocabPath, string mergesPath)
    {
        using Stream vocabStream = File.OpenRead(vocabPath);
        using Stream mergesStream = File.OpenRead(mergesPath);
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Encodes raw text to BPE token ids — no template, no special tokens, no padding.</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        return _tokenizer.EncodeToIds(text);
    }

    /// <summary>Encodes a prompt with the Qwen2.5 ChatML template, returning the raw (unpadded, variable-length) token ids. Format: <c>&lt;|im_start|&gt;system\n{system}&lt;|im_end|&gt;\n&lt;|im_start|&gt;user\n{prompt}&lt;|im_end|&gt;</c>, plus a trailing <c>\n&lt;|im_start|&gt;assistant\n</c> when <paramref name="addGenerationPrompt"/> is true (Lance uses true; HunyuanImage 2.1's encode template uses false).</summary>
    /// <param name="systemPrompt">System message; null uses <see cref="DefaultSystemPrompt"/>. Pass an empty string to omit the system turn entirely.</param>
    public int[] EncodeChat(string prompt, string? systemPrompt = null, bool addGenerationPrompt = true)
    {
        ThrowIfDisposed();
        string system = systemPrompt ?? DefaultSystemPrompt;

        List<int> ids = new(prompt.Length / 2 + 32);
        if (system.Length > 0)
        {
            ids.Add(ImStartId);
            AppendBpe(ids, "system\n" + system);
            ids.Add(ImEndId);
            AppendBpe(ids, "\n");
        }
        ids.Add(ImStartId);
        AppendBpe(ids, "user\n" + prompt);
        ids.Add(ImEndId);
        if (addGenerationPrompt)
        {
            AppendBpe(ids, "\n");
            ids.Add(ImStartId);
            AppendBpe(ids, "assistant\n");
        }
        return ids.ToArray();
    }

    /// <summary>Right-pads (or truncates) ids to <paramref name="length"/> using <see cref="EndOfTextId"/> — the HF convention for Qwen2 models with no dedicated pad token.</summary>
    public static int[] PadToLength(int[] ids, int length)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length));
        int[] result = new int[length];
        int copy = Math.Min(ids.Length, length);
        for (int i = 0; i < copy; i++)
            result[i] = ids[i];
        for (int i = copy; i < length; i++)
            result[i] = EndOfTextId;
        return result;
    }

    /// <summary>Builds a [seq] attention mask: 1 for real tokens, 0 for the trailing <see cref="EndOfTextId"/> padding run.</summary>
    public static int[] CreateAttentionMask(int[] tokenIds)
    {
        int[] mask = new int[tokenIds.Length];
        int lastReal = -1;
        for (int i = tokenIds.Length - 1; i >= 0; i--)
        {
            if (tokenIds[i] != EndOfTextId)
            {
                lastReal = i;
                break;
            }
        }
        for (int i = 0; i <= lastReal; i++)
            mask[i] = 1;
        return mask;
    }

    private void AppendBpe(List<int> dst, string text)
    {
        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(text);
        for (int i = 0; i < ids.Count; i++)
            dst.Add(ids[i]);
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Microsoft.ML.Tokenizers.Tokenizer is not IDisposable in v2.0; nothing to free.
        }
    }
}
