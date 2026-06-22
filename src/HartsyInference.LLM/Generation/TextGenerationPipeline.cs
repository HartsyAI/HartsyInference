using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>End-to-end LLM text generation: chat template → tokenize → GPU-resident prefill → autoregressive
/// decode (per-token sampler chain) → stop on EOS/limit → detokenize. Composes a <see cref="GenericTransformer"/>,
/// a <see cref="Qwen2Tokenizer"/> (whose vocab covers Qwen2.5 and Qwen3), an <see cref="IChatTemplate"/>, and a
/// backend-selected <see cref="IBackend"/>. The pipeline does not own the model or tokenizer (passed in).</summary>
public sealed class TextGenerationPipeline
{
    private readonly GenericTransformer _model;
    private readonly Qwen2Tokenizer _tokenizer;
    private readonly IChatTemplate _template;
    private readonly IBackend _backend;

    /// <summary>Qwen end-of-turn token (<c>&lt;|im_end|&gt;</c>).</summary>
    public const int ImEndId = 151645;

    /// <summary>Qwen end-of-text token (<c>&lt;|endoftext|&gt;</c>).</summary>
    public const int EndOfTextId = 151643;

    /// <summary>Creates the pipeline. <paramref name="template"/> defaults to ChatML (Qwen2.5 / Qwen3).</summary>
    public TextGenerationPipeline(GenericTransformer model, Qwen2Tokenizer tokenizer, IBackend backend,
        IChatTemplate? template = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _backend = backend;
        _template = template ?? new ChatMlTemplate();
    }

    /// <summary>Generates text for <paramref name="request"/>. <paramref name="onToken"/> is invoked with each
    /// new token id as it is produced (for streaming).</summary>
    public GenerationResult Generate(GenerationRequest request, Action<int>? onToken = null)
    {
        int[] promptIds = BuildPromptIds(request);
        if (promptIds.Length == 0) throw new ArgumentException("Prompt produced zero tokens.", nameof(request));

        TransformerConfig cfg = _model.Config;
        SamplerChain sampler = SamplerChain.FromOptions(request.Sampling);
        List<int> generated = new(request.MaxTokens);

        using KvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim);

        bool stopped = false;
        // Prefill: run the whole prompt, sample the first token from the last position's logits.
        int next;
        using (Tensor hidden = _model.Forward(_backend, promptIds, 0, cache))
        using (Tensor logits = _model.ProjectLogits(_backend, hidden, promptIds.Length))
        {
            Span<float> lastRow = LastRow(logits, promptIds.Length, cfg.VocabSize);
            next = sampler.Next(lastRow, generated);
        }

        for (int step = 0; step < request.MaxTokens; step++)
        {
            if (IsStop(next, request.StopTokenIds)) { stopped = true; break; }
            generated.Add(next);
            onToken?.Invoke(next);

            using Tensor hidden = _model.Forward(_backend, [next], cache.CurrentLength, cache);
            using Tensor logits = _model.ProjectLogits(_backend, hidden, 1);
            Span<float> row = LastRow(logits, 1, cfg.VocabSize);
            next = sampler.Next(row, generated);
        }

        return new GenerationResult
        {
            TokenIds = generated,
            Text = _tokenizer.Decode(generated),
            PromptTokens = promptIds.Length,
            StoppedOnStopToken = stopped,
        };
    }

    private int[] BuildPromptIds(GenerationRequest request)
    {
        if (request.RawTokenIds is not null) return [.. request.RawTokenIds];
        if (request.Messages is not null) return _template.Encode(_tokenizer, request.Messages, addGenerationPrompt: true);
        if (request.Prompt is not null) return ChatMlTemplate.EncodeSingleTurn(_tokenizer, request.Prompt, request.SystemPrompt);
        throw new ArgumentException("Request must set RawTokenIds, Messages, or Prompt.", nameof(request));
    }

    private static unsafe Span<float> LastRow(Tensor logits, int t, int vocab)
    {
        // Last position row of a [1, T, vocab] logits tensor.
        float* p = (float*)logits.DataPointer;
        return new Span<float>(p + (long)(t - 1) * vocab, vocab);
    }

    private static bool IsStop(int tokenId, IReadOnlyList<int>? extra)
    {
        if (tokenId == ImEndId || tokenId == EndOfTextId) return true;
        if (extra is not null)
            for (int i = 0; i < extra.Count; i++) if (extra[i] == tokenId) return true;
        return false;
    }
}
