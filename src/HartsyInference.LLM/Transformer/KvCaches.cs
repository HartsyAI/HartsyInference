using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Factory for the KV cache used by single-sequence autoregressive decode (prefill the prompt, then one token/frame per step) — the <b>one place</b> that decides which cache backing store the in-house transformer decoders use, so models never hand-roll a raw <c>StreamingKvCache</c> (O(n²) memcpy over a sequence) instead of calling <see cref="ForDecode"/> for a <see cref="FixedKvCache"/> (O(1) in-place appends, bounded VRAM). Swapping in a paged/quantized cache later is a one-line change here, not a per-model edit.</summary>
public static class KvCaches
{
    /// <summary>The <c>vram.kvF16</c> opt-in: F16 for every model type built through <see cref="ForDecode"/> and the direct <c>FixedKvCache</c> sites (audio decoders and others).
    /// Off by default; text LLMs follow <see cref="F16For"/>.</summary>
    public static bool F16Enabled => EngineKnobs.KvF16.Value;

    /// <summary>The KV cache element type choice for a text LLM running on <paramref name="backend"/>: <c>vram.textKvF16</c> when it is set (F16 only if the backend supports it), else the
    /// <c>vram.kvF16</c> opt-in, else F16 exactly when the backend can run attention over an F16 cache (<see cref="IBackend.SupportsF16KvCache"/>). F16 halves the cache and
    /// the bytes every decode step reads, which is 9-20% faster decode at 4K-16K context on a 12 GB card and what llama.cpp stores by default.</summary>
    public static bool F16For(IBackend? backend)
    {
        bool supported = backend?.SupportsF16KvCache == true;
        // vram.textKvF16 asks for F16 only where the backend can (true on a CPU backend would build a cache it cannot append to); the older vram.kvF16
        // stays a forced override whatever the backend reports.
        return EngineKnobs.TextKvF16.Value switch
        {
            false => false,
            true => supported,
            null => EngineKnobs.KvF16.Value || supported,
        };
    }

    /// <summary>The same choice for placement planning, which only plans CUDA devices (all of which support an F16 cache).</summary>
    public static bool F16ForCudaPlacement => EngineKnobs.TextKvF16.Value ?? true;

    /// <summary>Creates a decode KV cache for a batch-1 sequence of at most <paramref name="maxSeqLen"/> tokens; <paramref name="numLayers"/>/<paramref name="numKvHeads"/>/<paramref name="headDim"/> come from the model's transformer config. Dispose it when the utterance/song completes.</summary>
    public static IKvCache ForDecode(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
        => new FixedKvCache(numLayers, batch: 1, numKvHeads, headDim, System.Math.Max(1, maxSeqLen),
            F16Enabled ? DType.F16 : DType.F32);
}
