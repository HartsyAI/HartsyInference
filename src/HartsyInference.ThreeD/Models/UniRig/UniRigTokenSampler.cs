using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Sampling;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// The token loop of skeleton generation, independent of any network: log-softmax, repetition penalty, grammar mask,
/// temperature, top-k and top-p, then one multinomial draw per token, until EOS. Single sequence, not upstream's 15-beam sample.
/// </summary>
public static class UniRigTokenSampler
{
    /// <summary>Runs the loop and returns <c>BOS, class, generated…, EOS</c>.</summary>
    /// <param name="firstLogits">Logits after the start tokens.</param>
    /// <param name="step">Feeds one token to the model and returns the next logits.</param>
    public static int[] Run(float[] firstLogits, Func<int, float[]> step, UniRigSamplingOptions options, int clsToken, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(firstLogits);
        ArgumentNullException.ThrowIfNull(step);
        ArgumentNullException.ThrowIfNull(options);
        UniRigGrammar grammar = new();
        grammar.Advance(UniRigTokenizer.BosToken);
        grammar.Advance(clsToken);
        List<ISamplerStep> chain =
        [
            new UniRigLogSoftmaxStep(),
            new RepetitionPenaltyStep(options.RepetitionPenalty),
            new UniRigGrammarStep(grammar),
            new TemperatureStep(options.Temperature),
            new TopKStep(options.TopK),
            new TopPStep(options.TopP),
        ];
        Random random = new(options.Seed);
        List<int> generated = [];
        float[] logits = firstLogits;
        float[] scratch = new float[firstLogits.Length];
        float[] probs = new float[firstLogits.Length];
        for (int n = 0; n < options.MaxNewTokens; n++)
        {
            cancel.ThrowIfCancellationRequested();
            logits.CopyTo(scratch, 0);
            foreach (ISamplerStep sampler in chain)
            {
                sampler.Apply(scratch, generated);
            }

            int token = Draw(scratch, probs, random);
            generated.Add(token);
            if (token == UniRigTokenizer.EosToken)
            {
                int[] tokens = new int[generated.Count + 2];
                tokens[0] = UniRigTokenizer.BosToken;
                tokens[1] = clsToken;
                generated.CopyTo(tokens, 2);
                return tokens;
            }

            logits = step(token);
        }

        throw new HartsyInferenceException($"UniRig generation hit {options.MaxNewTokens} tokens without an end-of-sequence token.");
    }

    private static int Draw(Span<float> logits, Span<float> probs, Random random)
    {
        float max = float.NegativeInfinity;
        foreach (float value in logits)
        {
            max = Math.Max(max, value);
        }

        if (float.IsNegativeInfinity(max))
        {
            throw new HartsyInferenceException("UniRig grammar left no allowed token.");
        }

        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = float.IsNegativeInfinity(logits[i]) ? 0f : MathF.Exp(logits[i] - max);
            sum += probs[i];
        }

        double target = random.NextDouble() * sum;
        double cumulative = 0;
        int last = -1;
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] <= 0f)
            {
                continue;
            }

            last = i;
            cumulative += probs[i];
            if (target < cumulative)
            {
                return i;
            }
        }

        return last;
    }
}
