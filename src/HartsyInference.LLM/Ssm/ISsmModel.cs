using HartsyInference.Core.Backends;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Ssm;

/// <summary>Common shape of the recurrent (non-transformer) decoder models — <see cref="MambaModel"/>,
/// <see cref="Mamba2Model"/>, <see cref="RwkvModel"/>, <see cref="Rwkv7Model"/> — so <see cref="SsmLanguageModel"/>
/// and <see cref="SsmGenerationPipeline"/> can drive any of them uniformly.</summary>
public interface ISsmModel : IDisposable
{
    int VocabSize { get; }

    /// <summary>The GGUF metadata (for building the tokenizer + chat template the same way a transformer model does).</summary>
    GgufMetadata Metadata { get; }

    /// <summary>Zeroes every layer's carried recurrent state. Call before the first <see cref="ForwardLastLogits"/>
    /// of a new generation — the model instance is reused across unrelated chat turns via the provider's device
    /// slot, so without this a fresh prompt would continue from the previous conversation's state.</summary>
    void ResetState();

    /// <summary>Runs the stack over <paramref name="ids"/> — the NEW tokens since the last call (the whole
    /// prompt for the first/prefill call, exactly one token per decode step) — advancing each layer's carried
    /// recurrent state, and returns the next-token logits (last position). True O(1)-per-decode-step: unlike a
    /// transformer's KV cache this needs no growing buffer, the state itself is the fixed-size summary.</summary>
    float[] ForwardLastLogits(IBackend backend, IReadOnlyList<int> ids);
}
