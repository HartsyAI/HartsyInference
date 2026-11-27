namespace HartsyInference.ThreeD.Geometry;

/// <summary>Where each PBR quantity sits among a volume's channels. Values lie in [0,1]; an <see cref="Alpha"/> of -1 means opaque.</summary>
/// <param name="BaseColor">First of three consecutive base-color channels (R, G, B).</param>
/// <param name="Metallic">Metallic channel.</param>
/// <param name="Roughness">Roughness channel.</param>
/// <param name="Alpha">Alpha channel, or -1 when the volume carries none.</param>
public readonly record struct PbrAttributeLayout(int BaseColor, int Metallic, int Roughness, int Alpha)
{
    /// <summary>The TRELLIS.2 layout: base color 0..2, metallic 3, roughness 4, alpha 5.</summary>
    public static PbrAttributeLayout Default => new(0, 3, 4, 5);

    /// <summary>Smallest channel count a volume needs to satisfy this layout.</summary>
    public int RequiredChannels => Math.Max(Math.Max(BaseColor + 3, Metallic + 1), Math.Max(Roughness + 1, Alpha + 1));

    /// <summary>Throws when an index is negative or the volume has too few channels.</summary>
    public void Validate(int channels)
    {
        if (BaseColor < 0 || Metallic < 0 || Roughness < 0 || Alpha < -1)
            throw new ArgumentException("Layout channel indices must be non-negative (alpha may be -1).");
        if (RequiredChannels > channels)
            throw new ArgumentException($"Layout needs {RequiredChannels} channels but the volume has {channels}.");
    }
}
