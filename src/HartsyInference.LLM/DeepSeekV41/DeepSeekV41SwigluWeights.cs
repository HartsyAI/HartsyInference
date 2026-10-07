namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Dequantized F32 weights of one SwiGLU expert: <c>w1</c> and <c>w3</c> are <c>[Inter, Dim]</c>, <c>w2</c> is <c>[Dim, Inter]</c>, row-major.</summary>
/// <param name="Dim">Hidden width.</param>
/// <param name="Inter">Intermediate width.</param>
/// <param name="W1">Gate projection.</param>
/// <param name="W2">Down projection.</param>
/// <param name="W3">Up projection.</param>
public sealed record DeepSeekV41SwigluWeights(int Dim, int Inter, float[] W1, float[] W2, float[] W3)
{
    /// <summary>Checks the three matrices against <see cref="Dim"/> and <see cref="Inter"/>.</summary>
    public void Validate()
    {
        long n = (long)Dim * Inter;
        if (W1.Length != n || W2.Length != n || W3.Length != n)
            throw new ArgumentException($"SwiGLU weights must each hold {Dim} x {Inter} values.");
    }
}
