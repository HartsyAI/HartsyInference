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
    /// new token id as it is produced (for streaming). <paramref name="ct"/> is checked once per generated
    /// token (both the eager loop and the graph-decode replay loop) — cancelling stops generation between
    /// tokens and throws <see cref="OperationCanceledException"/>; already-produced tokens are lost (callers
    /// that want partial output on cancellation should rely on <paramref name="onToken"/>, which still fires
    /// for every token generated before the cancellation is observed).</summary>
    public GenerationResult Generate(GenerationRequest request, Action<int>? onToken = null, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        int[] promptIds = BuildPromptIds(request);
        if (promptIds.Length == 0) throw new ArgumentException("Prompt produced zero tokens.", nameof(request));

        TransformerConfig cfg = _model.Config;
        SamplerChain sampler = SamplerChain.FromOptions(request.Sampling, _tokenizer, cfg.VocabSize);
        List<int> generated = new(request.MaxTokens);
        HashSet<int> stops = _stopIds;
        if (request.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in request.StopTokenIds) stops.Add(s); }

        // Fixed-capacity KV (O(n) appends, bounded VRAM) sized for the prompt + the requested generation.
        int maxSeq = promptIds.Length + request.MaxTokens + 1;
        // Gemma-4: local/SWA layers use a narrower head dim than global layers (HeadDimFor); every other
        // architecture's HeadDimFor is just the uniform HeadDim, so this is a no-op for them.
        int[] headDimPerLayer = new int[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) headDimPerLayer[i] = cfg.HeadDimFor(i);
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, headDimPerLayer, maxSeq);

        bool stopped = false;
        int next;
        using (Tensor hidden = _model.Forward(_backend, promptIds, 0, cache))
        using (Tensor logits = _model.ProjectLogits(_backend, hidden, promptIds.Length))
        {
            Span<float> lastRow = LastRow(logits, promptIds.Length, cfg.VocabSize);
            next = sampler.Next(lastRow, generated);
        }

        // CUDA-graph decode: collapses the ~600-700 kernel launches/token the plain loop below issues into one
        // cuGraphLaunch/step, removing the CPU launch-issuance bottleneck the perf grind identified as the
        // biggest remaining gap to llama.cpp (docs/Checklists/LLM_DECODE_PERF_GRIND.md Phase 6). Opt-in
        // (env-gated) and scoped to what's actually graph-safe: greedy only (the on-device argmax has no
        // sampler chain yet) and the plain dense GQA/RoPE decoder shape (SupportsGraphDecode) — MoE/MLA/
        // cross-attention/sliding-window models fall through to the verified default loop unchanged.
        bool graphDecodeRequested = request.GraphDecode ?? (Environment.GetEnvironmentVariable("HARTSY_GRAPH_DECODE") == "1");
        bool useGraphDecode = request.Sampling.Greedy
            && graphDecodeRequested
            && _model.SupportsGraphDecode(_backend);

        if (useGraphDecode)
        {
            stopped = GenerateGraphDecode(request, cache, promptIds.Length, next, generated, stops, onToken, ct);
        }
        else
        {
            for (int step = 0; step < request.MaxTokens; step++)
            {
                ct.ThrowIfCancellationRequested();
                if (stops.Contains(next)) { stopped = true; break; }
                generated.Add(next);
                onToken?.Invoke(next);

                using Tensor hidden = _model.Forward(_backend, [next], cache.CurrentLength, cache);
                using Tensor logits = _model.ProjectLogits(_backend, hidden, 1);
                Span<float> row = LastRow(logits, 1, cfg.VocabSize);
                next = sampler.Next(row, generated);
            }
        }

        return new GenerationResult
        {
            TokenIds = generated,
            Text = _tokenizer.Decode(generated),
            PromptTokens = promptIds.Length,
            StoppedOnStopToken = stopped,
        };
    }

    /// <summary>Greedy decode via one captured CUDA graph, replayed once per token. <paramref name="firstToken"/>
    /// is the token already sampled from the prefill's last position (mirrors the eager loop's starting state).
    /// Device state (position, current token id, the RoPE table, and — when a repetition penalty is requested —
    /// the token history) is refreshed OUTSIDE the graph before each replay — see IBackend's "Device-side decode
    /// position" docs for why that's what makes one capture valid for every step. Repetition penalty is the only
    /// sampler stage graph decode replicates (see <see cref="GenericTransformer.ForwardGraphDecodeStep"/> for why
    /// temperature/top-k/top-p/min-p are no-ops for a greedy pick); when the request's penalty is 1.0 the history
    /// buffers are still allocated (cheap, fixed-size) but the backend skips the append/penalty kernels entirely.
    /// Falls back silently to nothing extra on failure: if capture throws (an eligible-looking model hits
    /// something the graphed path doesn't actually support), the exception propagates — this path is opt-in
    /// (env-gated), so a user who turns it on and hits a gap gets a clear error, not silent mis-generation.</summary>
    private bool GenerateGraphDecode(GenerationRequest request, FixedKvCache cache, int promptLen, int firstToken,
        List<int> generated, HashSet<int> stops, Action<int>? onToken, CancellationToken ct)
    {
        TransformerConfig cfg = _model.Config;
        Tensor embedTable = _model.EnsureEmbedResidentForGraphDecode(_backend);
        (Tensor cosTable, Tensor sinTable) = _model.EnsureRopeTableForGraphDecode(_backend, cache.MaxSequenceLength);
        ulong devicePos = _backend.AllocDevicePos();
        ulong deviceTokenId = _backend.AllocDeviceTokenId();
        float repetitionPenalty = request.Sampling.RepetitionPenalty;
        ulong history = _backend.AllocDeviceHistory(cache.MaxSequenceLength);
        ulong historyCount = _backend.AllocDeviceCounter();
        object? graph = null;
        try
        {
            int pos = promptLen;   // absolute position of the token this step is about to generate
            _backend.WriteDeviceTokenId(deviceTokenId, firstToken);
            _backend.WriteDevicePos(devicePos, pos + 1, pos);
            _backend.WriteDeviceCounter(historyCount, 0);
            graph = _backend.CaptureGraph(() =>
                _model.ForwardGraphDecodeStep(_backend, embedTable, cache, cosTable, sinTable, devicePos, deviceTokenId,
                    history, historyCount, repetitionPenalty));

            int next = firstToken;
            for (int step = 0; step < request.MaxTokens; step++)
            {
                ct.ThrowIfCancellationRequested();
                if (stops.Contains(next)) return true;
                generated.Add(next);
                onToken?.Invoke(next);

                _backend.LaunchGraph(graph!);
                next = _backend.ReadDeviceTokenId(deviceTokenId);
                pos++;
                _backend.WriteDevicePos(devicePos, pos + 1, pos);   // prep for the NEXT replay
            }
            return false;
        }
        finally
        {
            if (graph is not null) _backend.DisposeGraph(graph);
            _backend.FreeDevicePos(devicePos);
            _backend.FreeDeviceTokenId(deviceTokenId);
            _backend.FreeDeviceHistory(history);
            _backend.FreeDeviceCounter(historyCount);
        }
    }

    private int[] BuildPromptIds(GenerationRequest request) => PromptBuilder.BuildPromptIds(request, _tokenizer, _template);

    private static unsafe Span<float> LastRow(Tensor logits, int t, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        return new Span<float>(p + (long)(t - 1) * vocab, vocab);
    }
}
