using HartsyInference.LLM.Sampling;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Turns logits into log-probabilities first, as Hugging Face's beam-sample does before its processors, so the repetition penalty acts on log-probabilities.</summary>
internal sealed class UniRigLogSoftmaxStep : ISamplerStep
{
    public void Apply(Span<float> logits, IReadOnlyList<int> history)
    {
        float max = float.NegativeInfinity;
        foreach (float value in logits)
        {
            max = Math.Max(max, value);
        }

        double sum = 0;
        foreach (float value in logits)
        {
            sum += Math.Exp(value - max);
        }

        float logSum = max + (float)Math.Log(sum);
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] -= logSum;
        }
    }
}
