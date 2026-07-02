namespace HartsyInference.LLM.Transformer;

/// <summary>Factory for the KV cache used by single-sequence autoregressive decode (prefill the prompt, then one
/// token/frame per step). This is the <b>one place</b> that decides which cache backing store the in-house
/// transformer decoders use, so models never hand-roll <c>new StreamingKvCache(numLayers, 1, kvHeads, cap, headDim)</c>
/// (which copied the whole populated prefix on every append — O(n²) memcpy over a sequence). They call
/// <see cref="ForDecode"/> (or the <c>CreateDecodeCache</c> convenience on the model wrappers) and get a
/// <see cref="FixedKvCache"/>: O(1) in-place appends, bounded VRAM, and FlashAttention reads the valid length
/// separately. Swapping in a paged/quantized cache later is a one-line change here, not a per-model edit.</summary>
public static class KvCaches
{
    /// <summary>Creates a decode KV cache for a batch-1 sequence of at most <paramref name="maxSeqLen"/> tokens.
    /// <paramref name="numLayers"/>/<paramref name="numKvHeads"/>/<paramref name="headDim"/> come from the model's
    /// transformer config. Dispose it when the utterance/song completes (it is <see cref="IKvCache"/> :
    /// <see cref="System.IDisposable"/>, so <c>using</c> works without naming the concrete type).</summary>
    public static IKvCache ForDecode(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
        => new FixedKvCache(numLayers, batch: 1, numKvHeads, headDim, System.Math.Max(1, maxSeqLen));
}
