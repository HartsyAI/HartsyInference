namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>A serialization curve over voxel coordinates: Morton or Hilbert, optionally with x and y swapped.</summary>
public enum UtoniaSerialOrder
{
    /// <summary>Morton order over (x, y, z).</summary>
    ZOrder,

    /// <summary>Morton order over (y, x, z).</summary>
    ZOrderTrans,

    /// <summary>Hilbert curve over (x, y, z).</summary>
    Hilbert,

    /// <summary>Hilbert curve over (y, x, z).</summary>
    HilbertTrans
}
