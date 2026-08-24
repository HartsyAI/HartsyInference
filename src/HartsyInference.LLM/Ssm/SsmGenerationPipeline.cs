using HartsyInference.Core.Backends;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Ssm;

/// <summary>End-to-end text generation for the recurrent (SSM) decoders, mirroring <see cref="TextGenerationPipeline"/>'s contract so <c>HartsyLocalLLMProvider</c> can drive either pipeline uniformly.</summary>
/// <remarks>No KV cache — instead the model itself carries a fixed-size recurrent state across calls (<see cref="ISsmModel.ResetState"/> at the start of a generation, then <see cref="ISsmModel.ForwardLastLogits"/> fed only the NEW tokens each call); true O(1)-per-decode-step, unlike a transformer's growing KV buffer.</remarks>
public sealed class SsmGenerationPipeline
{
    private readonly ISsmModel _model;
    private readonly ILlmTokenizer _tokenizer;
    private readonly IChatTemplate _template;
    private readonly IBackend _backend;
    private readonly HashSet<int> _stopIds;

    public SsmGenerationPipeline(ISsmModel model, ILlmTokenizer tokenizer, IBackend backend, IChatTemplate? template = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _backend = backend;
        _template = template ?? new ChatMlTemplate();
        _stopIds = [.. tokenizer.StopIds];
        // Headroom-guarded weight preload (same as TextGenerationPipeline): the SSM decode loop otherwise
        // re-uploads weights every step — 10.4 s of PCIe churn in a 32-token qwen3.5 run.
        try
        {
            long headroom = 2L << 30;
            if (backend.FreeMemoryBytes() is long free && free > headroom)
            {
                List<HartsyInference.Core.Tensors.Tensor> toPreload = [];
                long budget = free - headroom;
                foreach (HartsyInference.Core.Tensors.Tensor t in model.EnumerateWeights())
                {
                    long bytes = HartsyInference.Core.Tensors.Tensor.ComputeByteSize(t.Shape, t.DType);
                    if (budget - bytes < 0) continue;
                    budget -= bytes;
                    toPreload.Add(t);
                }
                backend.PreloadWeights(toPreload);
            }
        }
        catch (Exception ex) { HartsyInference.Core.Logging.Logs.Warning($"ssm weight preload failed (continuing lazy): {ex.Message}"); }
    }

    /// <summary><paramref name="ct"/> is checked once per generated token, mirroring <see cref="TextGenerationPipeline.Generate"/>.</summary>
    public GenerationResult Generate(GenerationRequest request, Action<int>? onToken = null, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        int[] promptIds = PromptBuilder.BuildPromptIds(request, _tokenizer, _template);
        if (promptIds.Length == 0) throw new ArgumentException("Prompt produced zero tokens.", nameof(request));

        SamplerChain sampler = SamplerChain.FromOptions(request.Sampling, _tokenizer, _model.VocabSize);
        List<int> generated = new(request.MaxTokens);
        HashSet<int> stops = _stopIds;
        if (request.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in request.StopTokenIds) stops.Add(s); }

        // Reset carried state, then prefill (whole prompt) and decode (exactly one new token per step) — the
        // model advances its own recurrent state, so each step after the prefill is O(1), not O(context length).
        _model.ResetState();
        int next = sampler.Next(_model.ForwardLastLogits(_backend, promptIds), generated);
        bool stopped = false;

        // Graph decode (Qwen3.5 class): greedy-only, penalty-free, capture-once + replay-per-token —
        // the transformer pipeline's design. The pre-capture warmup step is a REAL eager device step
        // (capture only records; execution happens at LaunchGraph), so it both primes every buffer
        // into the device caches (no memcpy nodes) and produces the second token.
        bool useGraph = request.Sampling.Greedy && !request.Sampling.HasJsonConstraint
            && request.Sampling.RepetitionPenalty == 1.0f && _model is ISsmGraphDecodable { GraphDecodeReady: true }
            && _backend.GraphDecodeSupported && Environment.GetEnvironmentVariable("HARTSY_SSM_GRAPH") != "0";
        if (useGraph && !stops.Contains(next) && request.MaxTokens > 0)
        {
            ISsmGraphDecodable g = (ISsmGraphDecodable)_model;
            int maxSeq = Math.Min(promptIds.Length + request.MaxTokens + 8, 8192);
            (HartsyInference.Core.Tensors.Tensor cos, HartsyInference.Core.Tensors.Tensor sin,
                HartsyInference.Core.Tensors.Tensor embed) = g.EnsureGraphDecodeResident(_backend, maxSeq);
            ulong devicePos = _backend.AllocDevicePos();
            ulong deviceTokenId = _backend.AllocDeviceTokenId();
            object? graph = null;
            int consumed = 0;
            try
            {
                int pos = g.CurrentPosition;
                _backend.WriteDeviceTokenId(deviceTokenId, next);
                _backend.WriteDevicePos(devicePos, pos + 1, pos);

                // Warmup = the first decode step, executed eagerly (also promotes states/caches).
                generated.Add(next); onToken?.Invoke(next); consumed++;
                g.ForwardGraphDecodeStep(_backend, embed, cos, sin, devicePos, deviceTokenId);
                next = _backend.ReadDeviceTokenId(deviceTokenId);
                pos++;
                _backend.WriteDevicePos(devicePos, pos + 1, pos);

                graph = _backend.CaptureGraph(() =>
                    g.ForwardGraphDecodeStep(_backend, embed, cos, sin, devicePos, deviceTokenId));

                for (int step = consumed; step < request.MaxTokens; step++)
                {
                    ct.ThrowIfCancellationRequested();
                    if (stops.Contains(next)) { stopped = true; break; }
                    generated.Add(next);
                    onToken?.Invoke(next);
                    consumed++;
                    _backend.LaunchGraph(graph!);
                    next = _backend.ReadDeviceTokenId(deviceTokenId);
                    pos++;
                    _backend.WriteDevicePos(devicePos, pos + 1, pos);
                }
                if (!stopped && stops.Contains(next)) stopped = true;
            }
            finally
            {
                if (graph is not null) _backend.DisposeGraph(graph);
                _backend.FreeDevicePos(devicePos);
                _backend.FreeDeviceTokenId(deviceTokenId);
                g.AdvancePosition(consumed);
            }
        }
        else
        {
            for (int step = 0; step < request.MaxTokens; step++)
            {
                ct.ThrowIfCancellationRequested();
                if (stops.Contains(next)) { stopped = true; break; }
                generated.Add(next);
                onToken?.Invoke(next);
                next = sampler.Next(_model.ForwardLastLogits(_backend, [next]), generated);
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
}
