using HartsyInference.Core.Backends;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Ssm;

/// <summary>Common shape of the recurrent (non-transformer) decoder models — <see cref="MambaModel"/>,
/// <see cref="Mamba2Model"/>, <see cref="RwkvModel"/>, <see cref="Rwkv7Model"/> — so <see cref="SsmLanguageModel"/>
/// and <see cref="SsmGenerationPipeline"/> can drive any of them uniformly.</summary>
/// <summary>Optional capability: SSM models whose seq=1 decode step is fully device-side and
/// graph-capturable (currently Qwen3.5). The pipeline runs greedy decode as capture + replay.</summary>
public interface ISsmGraphDecodable
{
    (HartsyInference.Core.Tensors.Tensor cos, HartsyInference.Core.Tensors.Tensor sin, HartsyInference.Core.Tensors.Tensor embed)
        EnsureGraphDecodeResident(HartsyInference.Core.Backends.IBackend backend, int minCapacity);
    void ForwardGraphDecodeStep(HartsyInference.Core.Backends.IBackend backend,
        HartsyInference.Core.Tensors.Tensor embedTable, HartsyInference.Core.Tensors.Tensor cosTable,
        HartsyInference.Core.Tensors.Tensor sinTable, ulong devicePos, ulong deviceTokenId);
    void AdvancePosition(int tokens);
    int CurrentPosition { get; }

    /// <summary>Whether the fully-device graph-capture decode path is currently usable. Default true; a model
    /// returns false when a configuration can't be captured (e.g. Qwen3.5-MoE's host-side top-k expert routing,
    /// which can't run inside a replayed CUDA graph) so the pipeline falls back to eager decode.</summary>
    bool GraphDecodeReady => true;
}

public interface ISsmModel : IDisposable
{
    int VocabSize { get; }

    /// <summary>The GGUF metadata (for building the tokenizer + chat template the same way a transformer model does).</summary>
    GgufMetadata Metadata { get; }

    /// <summary>All weight tensors, for <see cref="HartsyInference.Core.Backends.IBackend.PreloadWeights"/> —
    /// without residency every SSM step re-uploads its weights over PCIe (measured: 10.4 s of H2D in a
    /// 32-token qwen3.5 run). Default: empty (models that haven't wired it keep lazy behavior).</summary>
    IEnumerable<HartsyInference.Core.Tensors.Tensor> EnumerateWeights() => [];

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
