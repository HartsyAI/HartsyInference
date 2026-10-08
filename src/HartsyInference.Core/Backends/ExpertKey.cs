namespace HartsyInference.Core.Backends;

/// <summary>
/// One routed expert: the layer it sits in, its index within that layer, and the bank that owns the layer. Two models loaded
/// into one cache each get their own bank, so the same layer number never collides. Bank 0 is the default.
/// </summary>
public readonly record struct ExpertKey(int Layer, int Expert, ushort Bank)
{
    /// <summary>The published two-argument form; bank 0.</summary>
    public ExpertKey(int Layer, int Expert) : this(Layer, Expert, 0)
    {
    }

    /// <summary>The cache identity of this expert's layer.</summary>
    public ExpertLayerKey LayerKey => new(Bank, Layer);

    /// <summary>The published two-output deconstruction, kept so existing callers still compile and link.</summary>
    public void Deconstruct(out int Layer, out int Expert)
    {
        Layer = this.Layer;
        Expert = this.Expert;
    }

    /// <inheritdoc/>
    public override string ToString() => Bank == 0 ? $"L{Layer}.E{Expert}" : $"B{Bank}.L{Layer}.E{Expert}";
}
