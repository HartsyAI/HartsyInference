namespace HartsyInference.ThreeD.Document;

/// <summary>How a material's alpha is used, as glTF names them.</summary>
public enum MeshAlphaMode
{
    /// <summary>Alpha is ignored.</summary>
    Opaque,

    /// <summary>Alpha below the cutoff is discarded.</summary>
    Mask,

    /// <summary>Alpha blends with what is behind.</summary>
    Blend,
}
