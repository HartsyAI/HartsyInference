namespace HartsyInference.LLM.Transformer.Heads;

/// <summary>A learned piecewise logit transform applied after the LM head (vLLM-ERNIE <c>multimax</c>):
/// <c>l + t0·relu(r0 − l) + t1·relu(l − r1) + t2·relu(r2 − l)² + t3·relu(l − r3)²</c>, with <c>r</c> the four thresholds and <c>t</c> the
/// four gains from the checkpoint. Computed in F32. Model-agnostic: any family with the same head supplies its own ranges and gains.</summary>
public static class MultimaxHead
{
    /// <summary>Transforms <paramref name="logits"/> in place.</summary>
    /// <param name="ranges">The four thresholds <c>r0..r3</c>.</param>
    /// <param name="gains">The four gains <c>t0..t3</c>.</param>
    public static void Apply(Span<float> logits, ReadOnlySpan<float> ranges, ReadOnlySpan<float> gains)
    {
        if (ranges.Length != 4 || gains.Length != 4) throw new ArgumentException("multimax takes four ranges and four gains.");
        float r0 = ranges[0], r1 = ranges[1], r2 = ranges[2], r3 = ranges[3];
        float t0 = gains[0], t1 = gains[1], t2 = gains[2], t3 = gains[3];
        for (int i = 0; i < logits.Length; i++)
        {
            float l = logits[i];
            float a = MathF.Max(r0 - l, 0f), b = MathF.Max(l - r1, 0f), c = MathF.Max(r2 - l, 0f), d = MathF.Max(l - r3, 0f);
            logits[i] = l + t0 * a + t1 * b + t2 * c * c + t3 * d * d;
        }
    }
}
