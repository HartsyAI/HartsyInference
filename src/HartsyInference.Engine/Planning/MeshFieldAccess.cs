namespace HartsyInference.Engine.Planning;

/// <summary>How one model treats a <see cref="Requests.MeshRequest"/> field.</summary>
public enum MeshFieldAccess
{
    /// <summary>The model reads the field.</summary>
    Used,

    /// <summary>The request is accepted and the field has no effect.</summary>
    Ignored,

    /// <summary>A request that sets the field to anything but its default fails before a checkpoint loads.</summary>
    Rejected,
}
