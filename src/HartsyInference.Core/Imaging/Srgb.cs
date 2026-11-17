namespace HartsyInference.Core.Imaging;

/// <summary>The sRGB transfer function on host floats. Both directions are odd about zero, so a value slightly outside
/// 0..1 stays finite and keeps its sign.</summary>
public static class Srgb
{
    /// <summary>Encoded sRGB (0..1) to linear light.</summary>
    public static float ToLinear(float encoded)
    {
        float magnitude = MathF.Abs(encoded);
        float linear = magnitude <= 0.04045f ? magnitude / 12.92f : MathF.Pow((magnitude + 0.055f) / 1.055f, 2.4f);
        return encoded < 0f ? -linear : linear;
    }

    /// <summary>Linear light to encoded sRGB.</summary>
    public static float FromLinear(float linear)
    {
        float magnitude = MathF.Abs(linear);
        float encoded = magnitude <= 0.0031308f ? magnitude * 12.92f : 1.055f * MathF.Pow(magnitude, 1f / 2.4f) - 0.055f;
        return linear < 0f ? -encoded : encoded;
    }
}
