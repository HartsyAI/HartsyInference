namespace HartsyInference.Core.Backends;

/// <summary>A layer within one expert bank: the unit the cache registers, protects and counts.</summary>
public readonly record struct ExpertLayerKey(ushort Bank, int Layer)
{
    /// <inheritdoc/>
    public override string ToString() => Bank == 0 ? $"L{Layer}" : $"B{Bank}.L{Layer}";
}
