namespace HartsyInference.ThreeD.Geometry;

/// <summary>glTF alpha modes.</summary>
public enum PbrAlphaMode
{
    /// <summary>Alpha is ignored (upstream TRELLIS.2 output).</summary>
    Opaque,

    /// <summary>Alpha is cut at 0.5.</summary>
    Mask,

    /// <summary>Alpha is blended.</summary>
    Blend,
}
