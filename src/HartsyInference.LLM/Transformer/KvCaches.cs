using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Factory for the KV cache used by single-sequence autoregressive decode (prefill the prompt, then one token/frame per step) — the <b>one place</b> that decides which cache backing store the in-house transformer decoders use, so models never hand-roll a raw <c>StreamingKvCache</c> (O(n²) memcpy over a sequence) instead of calling <see cref="ForDecode"/> for a <see cref="FixedKvCache"/> (O(1) in-place appends, bounded VRAM). Swapping in a paged/quantized cache later is a one-line change here, not a per-model edit.</summary>
public static class KvCaches
{
    /// <summary>F16-storage KV cache opt-in (halves VRAM; CUDA-only — see <see cref="FixedKvCache"/>'s kvDtype doc). Off by default per the plan's bar for flipping a concurrency/perf default ("after tests soak"); an operator opts in per the ROADMAP.md §1 note. Public so direct <c>FixedKvCache</c> construction sites that bypass <see cref="ForDecode"/> (e.g. <c>TextGenerationPipeline</c>, <c>DynamicBatchScheduler</c>) can still honor the same switch instead of drifting to their own env-var read.</summary>
    public static readonly bool F16Enabled = System.Environment.GetEnvironmentVariable("HARTSY_KV_F16") == "1";

    /// <summary>Creates a decode KV cache for a batch-1 sequence of at most <paramref name="maxSeqLen"/> tokens; <paramref name="numLayers"/>/<paramref name="numKvHeads"/>/<paramref name="headDim"/> come from the model's transformer config. Dispose it when the utterance/song completes.</summary>
    public static IKvCache ForDecode(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
        => new FixedKvCache(numLayers, batch: 1, numKvHeads, headDim, System.Math.Max(1, maxSeqLen),
            F16Enabled ? DType.F16 : DType.F32);
}
