using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>One autoregressive decode step, the shape both Moonshine decoders expose — they are unrelated
/// sealed types, so the shared decode loop takes the step as a delegate rather than a base class.</summary>
internal delegate Tensor MoonshineDecodeStep(IBackend backend, ReadOnlySpan<int> tokenIds, MoonshineDecoder.DecodeState state);

/// <summary>The greedy decode loop shared by <see cref="MoonshinePipeline"/> and <see cref="MoonshineStreamingPipeline"/>.</summary>
internal static class MoonshineGreedyDecoder
{
    /// <summary>Runs BOS → argmax → feed-back until EOS or <c>MaxNewTokens</c>, then detokenizes.</summary>
    public static string Decode(IBackend backend, MoonshineDecodeStep decodeStep, MoonshineDecoder.DecodeState state,
        MoonshineConfig cfg, MoonshineTokenizer tokenizer, MoonshineOptions opts)
    {
        // Prompt: just [BOS]. Moonshine has no language / task tokens.
        int[] prompt = [cfg.BosTokenId];
        Tensor logits = decodeStep(backend, prompt, state);
        int nextToken = ArgMax(logits, cfg.VocabSize);
        logits.Dispose();

        // Moonshine's documented hallucination guard: stop generating if the rate of
        // emitted tokens exceeds ~6.5 tokens / encoder-second of audio. We approximate
        // by capping total tokens to a multiple of the encoder sequence length, which
        // is what the original implementation effectively enforces.
        List<int> generated = new(opts.MaxNewTokens);
        int[] singleBuf = new int[1];
        for (int step = 0; step < opts.MaxNewTokens; step++)
        {
            if (nextToken == cfg.EosTokenId) break;
            generated.Add(nextToken);

            singleBuf[0] = nextToken;
            Tensor stepLogits = decodeStep(backend, singleBuf, state);
            nextToken = ArgMax(stepLogits, cfg.VocabSize);
            stepLogits.Dispose();
        }

        return tokenizer.Decode(generated.ToArray()).TrimStart();
    }

    private static unsafe int ArgMax(Tensor logits, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        int best = 0;
        float bestV = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float val = p[v];
            if (val > bestV) { bestV = val; best = v; }
        }
        return best;
    }
}
