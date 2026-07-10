using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>End-to-end LLM text generation: chat template → tokenize → GPU-resident prefill → autoregressive
/// decode (per-token sampler chain) → stop on EOS/limit → detokenize. Composes a <see cref="GenericTransformer"/>,
/// an <see cref="ILlmTokenizer"/> (any model family), an <see cref="IChatTemplate"/>, and a backend-selected
/// <see cref="IBackend"/>. Stop tokens come from the tokenizer. The pipeline does not own the model/tokenizer.</summary>
public sealed class TextGenerationPipeline
{
    private readonly GenericTransformer _model;
    private readonly ILlmTokenizer _tokenizer;
    private readonly IChatTemplate _template;
    private readonly IBackend _backend;
    private readonly HashSet<int> _stopIds;

    /// <summary>Creates the pipeline. <paramref name="template"/> defaults to ChatML when not supplied.</summary>
    public TextGenerationPipeline(GenericTransformer model, ILlmTokenizer tokenizer, IBackend backend,
        IChatTemplate? template = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _backend = backend;
        _template = template ?? new ChatMlTemplate();
        _stopIds = [.. tokenizer.StopIds];
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
        HashSet<int> stops = _stopIds;
        if (request.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in request.StopTokenIds) stops.Add(s); }

        // Fixed-capacity KV (O(n) appends, bounded VRAM) sized for the prompt + the requested generation.
        int maxSeq = promptIds.Length + request.MaxTokens + 1;
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, maxSeq);

        bool stopped = false;
        int next;
        using (Tensor hidden = _model.Forward(_backend, promptIds, 0, cache))
        using (Tensor logits = _model.ProjectLogits(_backend, hidden, promptIds.Length))
        {
            Span<float> lastRow = LastRow(logits, promptIds.Length, cfg.VocabSize);
            next = sampler.Next(lastRow, generated);
        }

        for (int step = 0; step < request.MaxTokens; step++)
        {
            if (stops.Contains(next)) { stopped = true; break; }
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

    private int[] BuildPromptIds(GenerationRequest request) => PromptBuilder.BuildPromptIds(request, _tokenizer, _template);

    private static unsafe Span<float> LastRow(Tensor logits, int t, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        return new Span<float>(p + (long)(t - 1) * vocab, vocab);
    }
}
