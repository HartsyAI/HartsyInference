namespace HartsyInference.Core.Imaging;

/// <summary>sRGB, linear sRGB and OKLab (Ottosson, 2020; the matrices of the published reference implementation), on
/// host floats. Planar spans hold three channels one after the other, <c>[3, pixels]</c>, the layout of a
/// <c>[1, 3, T, H, W]</c> frame tensor, so a tensor's span converts in place of copying through pixels.
/// <para>Channel normalization is a separate step and has no default values: a model that trains on normalized OKLab
/// owns the mean and standard deviation of its own data (<see cref="OkLabNormalization"/>).</para></summary>
public static class OkLab
{
    /// <summary>The sRGB transfer function's inverse (decode): encoded 0..1 to linear light. Odd about zero, so a value
    /// slightly outside the range stays finite and keeps its sign.</summary>
    public static float SrgbToLinear(float encoded)
    {
        float magnitude = MathF.Abs(encoded);
        float linear = magnitude <= 0.04045f ? magnitude / 12.92f : MathF.Pow((magnitude + 0.055f) / 1.055f, 2.4f);
        return encoded < 0f ? -linear : linear;
    }

    /// <summary>The sRGB transfer function (encode): linear light to encoded. Odd about zero.</summary>
    public static float LinearToSrgb(float linear)
    {
        float magnitude = MathF.Abs(linear);
        float encoded = magnitude <= 0.0031308f ? magnitude * 12.92f : 1.055f * MathF.Pow(magnitude, 1f / 2.4f) - 0.055f;
        return linear < 0f ? -encoded : encoded;
    }

    /// <summary>Linear sRGB to OKLab (L, a, b).</summary>
    public static (float L, float A, float B) FromLinearSrgb(float r, float g, float b)
    {
        float l = MathF.Cbrt(0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b);
        float m = MathF.Cbrt(0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b);
        float s = MathF.Cbrt(0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b);
        return (0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
                1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
                0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s);
    }

    /// <summary>OKLab to linear sRGB. The result is not clipped: a colour outside the sRGB gamut gives a component
    /// below 0 or above 1.</summary>
    public static (float R, float G, float B) ToLinearSrgb(float lightness, float a, float b)
    {
        float l = lightness + 0.3963377774f * a + 0.2158037573f * b;
        float m = lightness - 0.1055613458f * a - 0.0638541728f * b;
        float s = lightness - 0.0894841775f * a - 1.2914855480f * b;
        l = l * l * l;
        m = m * m * m;
        s = s * s * s;
        return (4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
                -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
                -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s);
    }

    /// <summary>Converts planar encoded sRGB (<c>[3, pixels]</c>, 0..1) to planar OKLab. The spans may not overlap.</summary>
    public static void FromSrgbPlanar(ReadOnlySpan<float> srgb, Span<float> okLab, int pixels)
    {
        CheckPlanar(srgb.Length, okLab.Length, pixels);
        for (int i = 0; i < pixels; i++)
        {
            (float l, float a, float b) = FromLinearSrgb(SrgbToLinear(srgb[i]), SrgbToLinear(srgb[pixels + i]), SrgbToLinear(srgb[2 * pixels + i]));
            okLab[i] = l;
            okLab[pixels + i] = a;
            okLab[2 * pixels + i] = b;
        }
    }

    /// <summary>Converts planar OKLab to planar encoded sRGB. With <paramref name="clip"/> each component is clipped to
    /// 0..1, which is the valid sRGB gamut; without it an out-of-gamut colour keeps its out-of-range components. The
    /// spans may not overlap.</summary>
    public static void ToSrgbPlanar(ReadOnlySpan<float> okLab, Span<float> srgb, int pixels, bool clip = true)
    {
        CheckPlanar(okLab.Length, srgb.Length, pixels);
        for (int i = 0; i < pixels; i++)
        {
            (float r, float g, float b) = ToLinearSrgb(okLab[i], okLab[pixels + i], okLab[2 * pixels + i]);
            float encodedR = LinearToSrgb(r), encodedG = LinearToSrgb(g), encodedB = LinearToSrgb(b);
            srgb[i] = clip ? Math.Clamp(encodedR, 0f, 1f) : encodedR;
            srgb[pixels + i] = clip ? Math.Clamp(encodedG, 0f, 1f) : encodedG;
            srgb[2 * pixels + i] = clip ? Math.Clamp(encodedB, 0f, 1f) : encodedB;
        }
    }

    /// <summary>Moves every colour of a planar OKLab frame into the sRGB gamut, in place, by clipping its linear sRGB
    /// components to 0..1 and converting back. Clipping the linear or the encoded component gives the same colour, since the
    /// transfer function fixes 0 and 1. This is a per-channel clip, not a chroma-preserving gamut map; a colour already in
    /// gamut comes back unchanged up to float rounding.</summary>
    public static void ClipToSrgbGamut(Span<float> okLab, int pixels)
    {
        CheckPlanar(okLab.Length, okLab.Length, pixels);
        for (int i = 0; i < pixels; i++)
        {
            (float r, float g, float b) = ToLinearSrgb(okLab[i], okLab[pixels + i], okLab[2 * pixels + i]);
            (float l, float a, float bb) = FromLinearSrgb(Math.Clamp(r, 0f, 1f), Math.Clamp(g, 0f, 1f), Math.Clamp(b, 0f, 1f));
            okLab[i] = l;
            okLab[pixels + i] = a;
            okLab[2 * pixels + i] = bb;
        }
    }

    private static void CheckPlanar(int sourceLength, int destinationLength, int pixels)
    {
        if (pixels < 0 || sourceLength != 3 * pixels || destinationLength != 3 * pixels)
            throw new ArgumentException($"Planar spans must hold 3 x {pixels} values; got {sourceLength} and {destinationLength}.");
    }
}

/// <summary>Per-channel mean and standard deviation that turn OKLab into the values a model trains on and back. There is
/// no default: the numbers belong to the model's training data.</summary>
public readonly record struct OkLabNormalization
{
    private readonly float[] _mean;
    private readonly float[] _std;

    /// <summary>Creates a normalization from the (L, a, b) means and standard deviations.</summary>
    public OkLabNormalization(ReadOnlySpan<float> mean, ReadOnlySpan<float> std)
    {
        if (mean.Length != 3 || std.Length != 3)
            throw new ArgumentException("Mean and standard deviation each need three values (L, a, b).");
        foreach (float value in std)
        {
            if (!(value > 0f)) throw new ArgumentException("Every standard deviation must be positive.", nameof(std));
        }
        _mean = mean.ToArray();
        _std = std.ToArray();
    }

    /// <summary>Planar OKLab to <c>(value - mean) / std</c> per channel, in place.</summary>
    public void Normalize(Span<float> okLab, int pixels)
    {
        Apply(okLab, pixels, static (value, mean, std) => (value - mean) / std);
    }

    /// <summary>Planar normalized OKLab back to OKLab, in place.</summary>
    public void Denormalize(Span<float> normalized, int pixels)
    {
        Apply(normalized, pixels, static (value, mean, std) => value * std + mean);
    }

    private void Apply(Span<float> planar, int pixels, Func<float, float, float, float> map)
    {
        if (_mean is null) throw new InvalidOperationException("The normalization has no values; build it with the constructor.");
        if (planar.Length != 3 * pixels)
            throw new ArgumentException($"A planar span must hold 3 x {pixels} values; got {planar.Length}.");
        for (int channel = 0; channel < 3; channel++)
        {
            Span<float> plane = planar.Slice(channel * pixels, pixels);
            float mean = _mean[channel], std = _std[channel];
            for (int i = 0; i < plane.Length; i++) plane[i] = map(plane[i], mean, std);
        }
    }
}
