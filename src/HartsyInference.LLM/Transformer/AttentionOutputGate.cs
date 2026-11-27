namespace HartsyInference.LLM.Transformer;

/// <summary>Elementwise sigmoid gate on an attention output, <c>z *= sigmoid(logits)</c>; the logits come from the caller's own projection.</summary>
public static class AttentionOutputGate
{
    /// <summary>Gates <paramref name="z"/> in place by <c>sigmoid(logits)</c>; both must have the same width.</summary>
    public static void ApplySigmoid(Span<float> z, ReadOnlySpan<float> logits)
    {
        if (logits.Length != z.Length) throw new ArgumentException("gate logits must match the output width.", nameof(logits));
        for (int i = 0; i < z.Length; i++) z[i] *= 1f / (1f + MathF.Exp(-logits[i]));
    }
}
