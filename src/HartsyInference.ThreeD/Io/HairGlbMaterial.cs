namespace HartsyInference.ThreeD.Io;

/// <summary>glTF metallic-roughness material: linear base colour RGBA, metallic and roughness factors, and double-sidedness.</summary>
public sealed record HairGlbMaterial(string Name, float[] BaseColorFactor, float MetallicFactor, float RoughnessFactor,
    bool DoubleSided);
