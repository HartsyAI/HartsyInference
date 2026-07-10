using HartsyInference.Core.Backends;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.Ssm;

/// <summary>End-to-end text generation for the recurrent (SSM) decoders, mirroring
/// <see cref="TextGenerationPipeline"/>'s contract (chat template → tokenize → decode loop → stop on
/// EOS/limit → detokenize) so <c>HartsyLocalLLMProvider</c> can drive either pipeline uniformly.
///
/// <para>No KV cache — instead the model itself carries a fixed-size recurrent state across calls
/// (<see cref="ISsmModel.ResetState"/> at the start of a generation, then <see cref="ISsmModel.ForwardLastLogits"/>
/// fed only the NEW tokens each call). True O(1)-per-decode-step, unlike a transformer's growing KV buffer.</para></summary>
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
    }

    public GenerationResult Generate(GenerationRequest request, Action<int>? onToken = null)
    {
        int[] promptIds = PromptBuilder.BuildPromptIds(request, _tokenizer, _template);
        if (promptIds.Length == 0) throw new ArgumentException("Prompt produced zero tokens.", nameof(request));

        SamplerChain sampler = SamplerChain.FromOptions(request.Sampling);
        List<int> generated = new(request.MaxTokens);
        HashSet<int> stops = _stopIds;
        if (request.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in request.StopTokenIds) stops.Add(s); }

        // Reset carried state, then prefill (whole prompt) and decode (exactly one new token per step) — the
        // model advances its own recurrent state, so each step after the prefill is O(1), not O(context length).
        _model.ResetState();
        int next = sampler.Next(_model.ForwardLastLogits(_backend, promptIds), generated);
        bool stopped = false;
        for (int step = 0; step < request.MaxTokens; step++)
        {
            if (stops.Contains(next)) { stopped = true; break; }
            generated.Add(next);
            onToken?.Invoke(next);
            next = sampler.Next(_model.ForwardLastLogits(_backend, [next]), generated);
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
