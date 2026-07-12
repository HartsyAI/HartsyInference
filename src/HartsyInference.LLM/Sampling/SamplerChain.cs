using System;
using System.Collections.Generic;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.Sampling;

/// <summary>Ordered pipeline of sampler steps plus a deterministic multinomial draw, built from a <see cref="SamplingOptions"/> and advanced once per generated token.</summary>
public sealed class SamplerChain
{
    // Fixed fallback so a zero seed still yields reproducible runs.
    private const uint DefaultSeed = 0x9E3779B9u;

    private readonly List<ISamplerStep> _steps;
    private readonly bool _greedy;
    private uint _rngState;

    private SamplerChain(List<ISamplerStep> steps, bool greedy, ulong seed)
    {
        _steps = steps;
        _greedy = greedy;
        uint state = seed == 0 ? DefaultSeed : (uint)(seed ^ (seed >> 32));
        // xorshift must never start at zero.
        _rngState = state == 0 ? DefaultSeed : state;
    }

    /// <summary>Builds the ordered chain (JSON grammar, repetition penalty, temperature, top-k, top-p, min-p)
    /// including only the steps that are active for the given options. The grammar step goes FIRST — it's a
    /// hard structural constraint (a token either can or can't extend valid JSON), not a probability-shaping
    /// setting, so it must exclude invalid tokens before anything else narrows the candidate set (though
    /// order doesn't actually change correctness here: -infinity logits stay excluded through every
    /// downstream elementwise transform regardless of where the mask is applied — first is just clearest).
    /// <paramref name="tokenizer"/>/<paramref name="vocabSize"/> are required when
    /// <see cref="SamplingOptions.JsonMode"/> or <see cref="SamplingOptions.JsonModeSentinel"/> is set (the
    /// grammar step must decode candidate token ids to text) and otherwise unused.</summary>
    public static SamplerChain FromOptions(SamplingOptions options, ILlmTokenizer? tokenizer = null, int vocabSize = 0)
    {
        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }
        List<ISamplerStep> steps = new List<ISamplerStep>();
        if (options.JsonMode)
        {
            if (tokenizer is null || vocabSize <= 0)
                throw new ArgumentException("JsonMode requires a tokenizer and vocabSize.", nameof(options));
            steps.Add(new JsonGrammarStep(tokenizer, vocabSize));
        }
        else if (!string.IsNullOrEmpty(options.JsonModeSentinel))
        {
            if (tokenizer is null || vocabSize <= 0)
                throw new ArgumentException("JsonModeSentinel requires a tokenizer and vocabSize.", nameof(options));
            steps.Add(new SentinelJsonGrammarStep(tokenizer, vocabSize, options.JsonModeSentinel));
        }
        if (options.RepetitionPenalty != 1.0f)
        {
            steps.Add(new RepetitionPenaltyStep(options.RepetitionPenalty));
        }
        if (options.Temperature != 1.0f && options.Temperature > 0.0f)
        {
            steps.Add(new TemperatureStep(options.Temperature));
        }
        if (options.TopK > 0)
        {
            steps.Add(new TopKStep(options.TopK));
        }
        if (options.TopP < 1.0f)
        {
            steps.Add(new TopPStep(options.TopP));
        }
        if (options.MinP > 0.0f)
        {
            steps.Add(new MinPStep(options.MinP));
        }
        return new SamplerChain(steps, options.Greedy, options.Seed);
    }

    /// <summary>Selects the next token id: argmax when greedy, otherwise the configured steps are applied and a multinomial draw is taken over the surviving probabilities.</summary>
    public int Next(Span<float> logits, IReadOnlyList<int> history)
    {
        if (logits.Length == 0)
        {
            throw new ArgumentException("Logits span must be non-empty.", nameof(logits));
        }
        // Greedy still runs the logit-adjustment steps (repetition penalty) — it only skips the RANDOM draw at
        // the end. Every step here is a monotonic-or-argmax-preserving transform (temperature scales positively,
        // top-k/top-p/min-p never drop the highest-scoring token), so running them before Argmax is safe and
        // matches llama.cpp/HF: greedy means "no randomness", not "no penalties". Skipping repetition penalty
        // here previously made every greedy decode loop into repeated phrases on small/weak models.
        for (int i = 0; i < _steps.Count; i++)
        {
            _steps[i].Apply(logits, history);
        }
        if (_greedy)
        {
            return Argmax(logits);
        }
        int count = logits.Length;
        float[] probs = new float[count];
        SamplerMath.Softmax(logits, probs);
        float sum = 0.0f;
        for (int i = 0; i < count; i++)
        {
            sum += probs[i];
        }
        if (sum <= 0.0f)
        {
            return Argmax(logits);
        }
        float r = NextFloat() * sum;
        float acc = 0.0f;
        int last = -1;
        for (int i = 0; i < count; i++)
        {
            if (probs[i] <= 0.0f)
            {
                continue;
            }
            last = i;
            acc += probs[i];
            if (r <= acc)
            {
                return i;
            }
        }
        // Floating-point drift can leave r just past the accumulated mass; fall back to the last survivor.
        return last >= 0 ? last : Argmax(logits);
    }

    private static int Argmax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bestValue = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestValue)
            {
                bestValue = logits[i];
                best = i;
            }
        }
        return best;
    }

    private float NextFloat()
    {
        uint state = _rngState;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        _rngState = state;
        // 24 high bits give a uniform float in [0, 1).
        return (state >> 8) * (1.0f / 16777216.0f);
    }
}
