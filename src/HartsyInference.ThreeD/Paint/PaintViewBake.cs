namespace HartsyInference.ThreeD.Paint;

/// <summary>What one view contributes to the texture: for every covered texel of a <see cref="PaintTexelSurface"/> the sampled colour and the
/// viewing-angle weight (zero where the view does not see or trust the texel).</summary>
public sealed class PaintViewBake
{
    /// <summary>Creates the container.</summary>
    public PaintViewBake(int texelCount, int channels)
    {
        Channels = channels;
        Colors = new float[checked(texelCount * channels)];
        Weights = new float[texelCount];
    }

    /// <summary>Values per texel.</summary>
    public int Channels { get; }

    /// <summary>Colour per covered texel, <see cref="Channels"/> values each.</summary>
    public float[] Colors { get; }

    /// <summary>Cosine weight per covered texel; positive where the texel is painted.</summary>
    public float[] Weights { get; }
}
