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
    /// <summary>Upstream's hallucination bound: decoder tokens per second of audio.</summary>
    internal const double TokensPerSecond = 6.5;

    /// <summary>Floor on a derived budget, so a very short clip keeps room for its words.</summary>
    private const int MinDerivedTokens = 32;

    /// <summary>Decoder budget for a clip: tokens-per-second of its length, floored, clamped to the decoder's positions less BOS.</summary>
    internal static int DerivedTokenBudget(double seconds, int maxTextPositions)
    {
        double scaled = Math.Floor(seconds * TokensPerSecond);
        return (int)Math.Min(Math.Max(scaled, MinDerivedTokens), maxTextPositions - 1);
    }

    /// <summary>The caller's options with a 0 <c>MaxNewTokens</c> replaced by the budget derived from the clip's length; an explicit value is kept.</summary>
    internal static MoonshineOptions ResolveOptions(MoonshineOptions? options, int sampleCount, int sampleRate, int maxTextPositions)
    {
        MoonshineOptions opts = options ?? new MoonshineOptions();
        return opts.MaxNewTokens != 0
            ? opts
            : opts with { MaxNewTokens = DerivedTokenBudget((double)sampleCount / sampleRate, maxTextPositions) };
    }

    /// <summary>Longest clip one decode can cover before the decoder runs out of positions.</summary>
    internal static double WindowSecondsFor(int maxTextPositions) => (maxTextPositions - 1) / TokensPerSecond;

    /// <summary>Runs BOS → argmax → feed-back until EOS or <c>MaxNewTokens</c>, then detokenizes.</summary>
    public static string Decode(IBackend backend, MoonshineDecodeStep decodeStep, MoonshineDecoder.DecodeState state,
        MoonshineConfig cfg, MoonshineTokenizer tokenizer, MoonshineOptions opts)
    {
        if (opts.MaxNewTokens <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(opts), "MaxNewTokens must be resolved to a positive budget; pass the options through ResolveOptions first.");
        }

        // Prompt: just [BOS]. Moonshine has no language / task tokens.
        int[] prompt = [cfg.BosTokenId];
        Tensor logits = decodeStep(backend, prompt, state);
        int nextToken = ArgMax(logits, cfg.VocabSize);
        logits.Dispose();

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
