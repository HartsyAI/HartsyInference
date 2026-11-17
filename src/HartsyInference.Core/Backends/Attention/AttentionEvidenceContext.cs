namespace HartsyInference.Core.Backends.Attention;

/// <summary>Thread-local attribution around synchronous backend execution; it does not flow across awaits.</summary>
public readonly struct AttentionEvidenceContext : IDisposable
{
    private readonly string? _previousComponent;
    private readonly int _previousLayer;
    private readonly bool _active;

    internal AttentionEvidenceContext(string component, int layer)
    {
        _previousComponent = AttentionEvidence.Component;
        _previousLayer = AttentionEvidence.Layer;
        _active = true;
        AttentionEvidence.Component = component;
        AttentionEvidence.Layer = layer;
    }

    /// <summary>Restores the enclosing synchronous execution context.</summary>
    public void Dispose()
    {
        if (!_active) return;
        AttentionEvidence.Component = _previousComponent;
        AttentionEvidence.Layer = _previousLayer;
    }
}
