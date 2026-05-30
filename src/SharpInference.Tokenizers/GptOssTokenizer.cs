using Microsoft.ML.Tokenizers;

namespace SharpInference.Tokenizers;

/// <summary>OpenAI GPT-OSS byte-level BPE tokenizer used as the text-encoder front-end for Microsoft Lens. Wraps <see cref="BpeTokenizer"/> with <c>vocab.json</c> + <c>merges.txt</c> from <c>openai/gpt-oss-20b</c> (re-published in <c>microsoft/Lens/tokenizer/</c>). Vocab 201,088. Special tokens follow the GPT-OSS Harmony format: <c>&lt;|start|&gt;</c>, <c>&lt;|end|&gt;</c>, <c>&lt;|message|&gt;</c>, <c>&lt;|return|&gt;</c>, <c>&lt;|channel|&gt;</c>, <c>&lt;|constrain|&gt;</c>.
///
/// <para><b>Lens chat template</b> (see <c>lens/pipeline.py:_build_chat_inputs</c>): the system prompt and assistant-thinking wrapper always tokenize to exactly 97 tokens (<c>DEFAULT_TXT_OFFSET = 97</c>). The pipeline strips those 97 tokens off the front of the encoder output. <see cref="BuildChatInputs"/> renders the chat template; the pipeline still has to handle the offset because the user prompt's token count is variable.</para></summary>
public sealed class GptOssTokenizer : IDisposable
{
    /// <summary>Vocabulary size per <c>microsoft/Lens/text_encoder/config.json</c>.</summary>
    public const int VocabSize = 201088;

    /// <summary>System prompt the Lens pipeline always prepends. Verbatim from <c>lens/pipeline.py:_CHAT_SYSTEM</c>.</summary>
    public const string ChatSystemPrompt =
        "Describe the image by detailing the color, shape, size, texture, " +
        "quantity, text, spatial relationships of the objects and background.";

    /// <summary>Assistant thinking content Lens uses as the encoder-side prefix. Verbatim from <c>lens/pipeline.py:_CHAT_ASSISTANT_THINKING</c>.</summary>
    public const string ChatAssistantThinking = "Need to generate one image according to the description.";

    /// <summary>Token count of the chat-template wrapper (system + assistant thinking) — used by <see cref="SharpInference.Diffusion.Pipelines.LensPipeline"/> to strip the prefix off the encoder hidden states. Constant per upstream pipeline.</summary>
    public const int DefaultTxtOffset = 97;

    private readonly Tokenizer _tokenizer;
    private readonly int _maxLength;
    private readonly int _padTokenId;
    private int _disposed;

    /// <summary>Creates a GPT-OSS tokenizer from <c>vocab.json</c> + <c>merges.txt</c>. No embedded resource — GPT-OSS vocab is ~10 MB and is downloaded alongside the Lens checkpoint.</summary>
    /// <param name="vocabPath">Path to <c>vocab.json</c> from <c>microsoft/Lens/tokenizer/</c>.</param>
    /// <param name="mergesPath">Path to <c>merges.txt</c>.</param>
    /// <param name="maxLength">Truncation cap. Default 512 matches <c>LensPipeline.__call__.max_sequence_length</c>.</param>
    /// <param name="padTokenId">Pad token id. GPT-OSS' <c>tokenizer_config.json</c> sets <c>pad_token = eos_token</c>; for the published <c>openai/gpt-oss-20b</c> the eos token is <c>&lt;|return|&gt;</c> = 200002.</param>
    public GptOssTokenizer(string vocabPath, string mergesPath, int maxLength = 512, int padTokenId = 200002)
    {
        if (maxLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxLength));
        _maxLength = maxLength;
        _padTokenId = padTokenId;
        using Stream vocabStream = File.OpenRead(vocabPath);
        using Stream mergesStream = File.OpenRead(mergesPath);
        _tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
    }

    /// <summary>Encodes a user prompt using Lens' Harmony chat template. Format (rendered ahead of BPE):
    /// <code>
    /// &lt;|start|&gt;system&lt;|message|&gt;{ChatSystemPrompt}&lt;|end|&gt;
    /// &lt;|start|&gt;user&lt;|message|&gt;{prompt}&lt;|end|&gt;
    /// &lt;|start|&gt;assistant&lt;|channel|&gt;analysis&lt;|message|&gt;{ChatAssistantThinking}&lt;|end|&gt;
    /// &lt;|start|&gt;assistant&lt;|message|&gt;
    /// </code>
    /// Upstream uses HuggingFace's <c>apply_chat_template</c> then splits at <c>&lt;|return|&gt;</c> and keeps everything before. This matches that output structure. Right-pads to <see cref="DefaultTxtOffset"/>+prompt_tokens with <see cref="_padTokenId"/>.
    /// </summary>
    /// <param name="prompt">User prompt.</param>
    /// <returns>(token_ids, attention_mask) both length <see cref="_maxLength"/>. Mask is 1 for real tokens, 0 for pad.</returns>
    public (int[] tokenIds, int[] attentionMask) BuildChatInputs(string prompt)
    {
        ThrowIfDisposed();
        string rendered =
            $"<|start|>system<|message|>{ChatSystemPrompt}<|end|>" +
            $"<|start|>user<|message|>{prompt}<|end|>" +
            $"<|start|>assistant<|channel|>analysis<|message|>{ChatAssistantThinking}<|end|>" +
            $"<|start|>assistant<|message|>";

        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(rendered);

        int realLen = Math.Min(ids.Count, _maxLength);
        int[] tokenIds = new int[_maxLength];
        int[] mask = new int[_maxLength];
        for (int i = 0; i < realLen; i++)
        {
            tokenIds[i] = ids[i];
            mask[i] = 1;
        }
        for (int i = realLen; i < _maxLength; i++)
        {
            tokenIds[i] = _padTokenId;
            mask[i] = 0;
        }
        return (tokenIds, mask);
    }

    /// <summary>Encodes raw text without applying the chat template. Useful for tokenizer round-trip tests against the upstream Python loader.</summary>
    public IReadOnlyList<int> EncodeRaw(string text)
    {
        ThrowIfDisposed();
        return _tokenizer.EncodeToIds(text);
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
