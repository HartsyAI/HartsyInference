namespace HartsyInference.Core.Backends;

/// <summary>
/// One routed expert: the layer it sits in, its index within that layer, and the bank that owns the layer. Two models
/// loaded into one cache each get their own bank, so the same layer number never collides. Bank 0 is the default.
/// </summary>
public readonly record struct ExpertKey(int Layer, int Expert, ushort Bank = 0)
{
    /// <summary>The cache identity of this expert's layer.</summary>
    public ExpertLayerKey LayerKey => new(Bank, Layer);

    /// <inheritdoc/>
    public override string ToString() => Bank == 0 ? $"L{Layer}.E{Expert}" : $"B{Bank}.L{Layer}.E{Expert}";
}
