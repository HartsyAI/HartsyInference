using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>Continuous-batching generation: runs many sequences through one shared model, batching the
/// compute-heavy projections/MLP across all active sequences in each decode step (one big GEMM instead of N
/// small ones) while attention stays per-sequence. Each sequence keeps its own KV cache and sampler; sequences
/// complete independently and drop out of the batch (it shrinks as they finish). Prompts are prefilled per
/// sequence (single-sequence path) on admission, then all decode in lockstep via
/// <see cref="GenericTransformer.ForwardBatchDecode"/>.
///
/// <para>Numerically equivalent to running each request through <see cref="TextGenerationPipeline"/> alone:
/// same weights, same per-sequence sampler, same add→forward→sample cycle. On a backend whose GEMM is
/// batch-invariant (the CPU backend computes each output row in a fixed reduction order regardless of batch
/// size) the output is bit-for-bit token-identical to single-sequence decode, and the
/// <c>ContinuousBatchTests</c> assert exactly that. On CUDA, cuBLAS may dispatch a different kernel for a
/// batched (M=B) GEMM than for the single-row (M=1) decode, so per-row logits can differ at the ~1e-3 level;
/// this is invisible except when a greedy argmax sits on a near-tie, where it can flip a token and the
/// sequences then diverge (the batched sequences still stay identical to <em>each other</em> — the path is
/// deterministic, not buggy). Chunked/batched prefill and dynamic mid-flight admission build on this.</para></summary>
public sealed class ContinuousBatchScheduler
{
    private readonly GenericTransformer _model;
    private readonly ILlmTokenizer _tokenizer;
    private readonly IChatTemplate _template;
    private readonly IBackend _backend;
    private readonly HashSet<int> _stopIds;

    public ContinuousBatchScheduler(GenericTransformer model, ILlmTokenizer tokenizer, IBackend backend,
        IChatTemplate? template = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _backend = backend;
        _template = template ?? new ChatMlTemplate();
        _stopIds = [.. tokenizer.StopIds];
    }

    private sealed class Seq
    {
        public required int Index;
        public required int[] PromptIds;
        public required FixedKvCache Cache;
        public required SamplerChain Sampler;
        public required HashSet<int> Stops;
        public required int MaxTokens;
        public required List<int> Generated;
        public int Next;
        public bool Done;
        public bool Stopped;
    }

    /// <summary>Generates completions for all <paramref name="requests"/> concurrently; results are returned in
    /// request order.</summary>
    public IReadOnlyList<GenerationResult> GenerateBatch(IReadOnlyList<GenerationRequest> requests)
    {
        if (requests.Count == 0) return [];
        TransformerConfig cfg = _model.Config;
        Seq[] seqs = new Seq[requests.Count];

        // Admission: build prompt, allocate KV, prefill, sample the first token (single-sequence).
        for (int i = 0; i < requests.Count; i++)
        {
            GenerationRequest req = requests[i];
            int[] promptIds = BuildPromptIds(req);
            if (promptIds.Length == 0) throw new ArgumentException($"Request {i} produced zero tokens.");
            HashSet<int> stops = _stopIds;
            if (req.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in req.StopTokenIds) stops.Add(s); }

            FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, promptIds.Length + req.MaxTokens + 1);
            Seq seq = new()
            {
                Index = i, PromptIds = promptIds, Cache = cache, Sampler = SamplerChain.FromOptions(req.Sampling),
                Stops = stops, MaxTokens = req.MaxTokens, Generated = new List<int>(req.MaxTokens),
            };
            using (Tensor hidden = _model.Forward(_backend, promptIds, 0, cache))
            using (Tensor logits = _model.ProjectLogits(_backend, hidden, promptIds.Length))
                seq.Next = seq.Sampler.Next(LastRow(logits, promptIds.Length, cfg.VocabSize), seq.Generated);
            seqs[i] = seq;
        }

        // Batched decode: each round feeds one token per still-running sequence.
        List<Seq> feeders = new(seqs.Length);
        while (true)
        {
            feeders.Clear();
            foreach (Seq seq in seqs)
            {
                if (seq.Done) continue;
                if (seq.Stops.Contains(seq.Next)) { seq.Done = true; seq.Stopped = true; continue; }
                if (seq.Generated.Count >= seq.MaxTokens) { seq.Done = true; continue; }
                seq.Generated.Add(seq.Next);
                feeders.Add(seq);
            }
            if (feeders.Count == 0) break;

            int bn = feeders.Count;
            int[] tokens = new int[bn];
            int[] positions = new int[bn];
            FixedKvCache[] caches = new FixedKvCache[bn];
            for (int b = 0; b < bn; b++)
            {
                tokens[b] = feeders[b].Next;
                positions[b] = feeders[b].Cache.CurrentLength;
                caches[b] = feeders[b].Cache;
            }

            using Tensor embeds = new(new TensorShape(1, bn, cfg.HiddenSize), DType.F32);
            _model.EmbedLookup(embeds, tokens);
            using Tensor hidden = _model.ForwardBatchDecode(_backend, embeds, positions, caches);
            using Tensor logits = _model.ProjectLogits(_backend, hidden, bn);
            for (int b = 0; b < bn; b++)
            {
                Seq seq = feeders[b];
                seq.Next = seq.Sampler.Next(LastRow(logits, b + 1, cfg.VocabSize), seq.Generated);
            }
        }

        GenerationResult[] results = new GenerationResult[seqs.Length];
        foreach (Seq seq in seqs)
        {
            results[seq.Index] = new GenerationResult
            {
                TokenIds = seq.Generated,
                Text = _tokenizer.Decode(seq.Generated),
                PromptTokens = seq.PromptIds.Length,
                StoppedOnStopToken = seq.Stopped,
            };
            seq.Cache.Dispose();
        }
        return results;
    }

    private int[] BuildPromptIds(GenerationRequest request)
    {
        if (request.RawTokenIds is not null) return [.. request.RawTokenIds];
        if (request.Messages is not null) return _template.Encode(_tokenizer, request.Messages, addGenerationPrompt: true);
        if (request.Prompt is not null)
        {
            List<ChatMessage> messages = new(2);
            if (!string.IsNullOrEmpty(request.SystemPrompt)) messages.Add(ChatMessage.System(request.SystemPrompt));
            messages.Add(ChatMessage.User(request.Prompt));
            return _template.Encode(_tokenizer, messages, addGenerationPrompt: true);
        }
        throw new ArgumentException("Request must set RawTokenIds, Messages, or Prompt.", nameof(request));
    }

    private static unsafe Span<float> LastRow(Tensor logits, int t, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        return new Span<float>(p + (long)(t - 1) * vocab, vocab);
    }
}
