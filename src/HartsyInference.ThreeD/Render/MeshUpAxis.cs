namespace HartsyInference.ThreeD.Render;

/// <summary>Which axis of the stored geometry points up. Models return their own frames (glTF Y-up, or Z-up for some), so a renderer is told rather than guessing.</summary>
public enum MeshUpAxis
{
    /// <summary>Y is up, glTF's convention.</summary>
    Y,

    /// <summary>Z is up; the geometry is turned a quarter turn about X before it is viewed.</summary>
    Z,
}
