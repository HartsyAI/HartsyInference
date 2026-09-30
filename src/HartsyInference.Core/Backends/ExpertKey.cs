namespace HartsyInference.Core.Backends;

/// <summary>One routed expert: the layer it sits in and its index within that layer.</summary>
public readonly record struct ExpertKey(int Layer, int Expert)
{
    /// <inheritdoc/>
    public override string ToString() => $"L{Layer}.E{Expert}";
}
