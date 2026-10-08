namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Weights of one SwiGLU expert, widened or in their stored form: <c>w1</c> and <c>w3</c> are <c>[Inter, Dim]</c>, <c>w2</c> is <c>[Dim, Inter]</c>, row-major.</summary>
/// <param name="Dim">Hidden width.</param>
/// <param name="Inter">Intermediate width.</param>
/// <param name="W1">Gate projection.</param>
/// <param name="W2">Down projection.</param>
/// <param name="W3">Up projection.</param>
public sealed record DeepSeekV41SwigluWeights(int Dim, int Inter, DeepSeekV41Weight W1, DeepSeekV41Weight W2, DeepSeekV41Weight W3)
{
    /// <summary>Checks the three matrices against <see cref="Dim"/> and <see cref="Inter"/>.</summary>
    public void Validate()
    {
        long n = (long)Dim * Inter;
        if (W1.Elements != n || W2.Elements != n || W3.Elements != n)
            throw new ArgumentException($"SwiGLU weights must each hold {Dim} x {Inter} values.");
    }
}
