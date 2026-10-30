using System.Globalization;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>A reference request's settings, read from its extras. Defaults are the released adapter's recommended ones.</summary>
public sealed record ReferenceAdapterSettings(string? Adapter, float Strength, long MaxPixels, ReferenceGuidanceSettings Guidance)
{
    /// <summary>The adapter's default reference area: one megapixel, as the node resizes it.</summary>
    public const long DefaultMaxPixels = 1024L * 1024L;

    /// <summary>The largest reference area accepted. Beyond it the VAE and attention cost grows past what the adapter was trained on.</summary>
    public const long MaxMaxPixels = 2L * 1024L * 1024L;

    /// <summary>The request's reference settings. Values outside their range are refused by name, never clamped.</summary>
    public static ReferenceAdapterSettings From(IReadOnlyDictionary<string, object>? extra)
    {
        string? adapter = RequestExtras.String(extra, RequestExtras.ReferenceAdapter);
        float strength = Read(extra, RequestExtras.ReferenceStrength, 1.0, 0.0, 4.0, "reference strength");
        float guidance = Read(extra, RequestExtras.ReferenceGuidance, 4.5, 0.0, 20.0, "reference guidance");
        float phi = Read(extra, RequestExtras.ReferenceRescalePhi, 0.5, 0.0, 1.0, "rescale phi");
        float start = Read(extra, RequestExtras.ReferenceStart, 0.0, 0.0, 1.0, "reference start");
        float end = Read(extra, RequestExtras.ReferenceEnd, 1.0, 0.0, 1.0, "reference end");
        double pixels = RequestExtras.Number(extra, RequestExtras.ReferenceMaxPixels, DefaultMaxPixels);
        if (pixels < 0.25 * 1024 * 1024 || pixels > MaxMaxPixels)
        {
            throw new ArgumentOutOfRangeException(RequestExtras.ReferenceMaxPixels, pixels,
                $"'{RequestExtras.ReferenceMaxPixels}' must be between 262144 and {MaxMaxPixels}.");
        }
        ReferenceCorrection correction = CorrectionFrom(extra);
        ReferenceOutside outside = OutsideFrom(extra);
        ReferenceGuidanceSettings guidanceSettings = new(guidance, correction, phi, start, end, outside);
        return new ReferenceAdapterSettings(adapter, strength, (long)pixels, guidanceSettings);
    }

    /// <summary>The adapter a request asks for, or null when it asks for no reference: a reference image turns the path on. An empty
    /// string names the released adapter. Naming an adapter without a reference is refused, since the adapter would load and do nothing.</summary>
    public static string? RequestedAdapter(ImageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        bool hasReferences = request.ReferenceImages is { Count: > 0 } || request.Img2Img is { Mode: Requests.Img2ImgMode.Reference };
        bool named = request.Extra.ContainsKey(RequestExtras.ReferenceAdapter);
        if (!hasReferences)
        {
            if (named)
                throw new ArgumentException($"'{RequestExtras.ReferenceAdapter}' needs a reference image; send one with the request.", RequestExtras.ReferenceAdapter);
            return null;
        }
        return RequestExtras.String(request.Extra, RequestExtras.ReferenceAdapter) ?? "";
    }

    private static ReferenceCorrection CorrectionFrom(IReadOnlyDictionary<string, object>? extra)
    {
        string name = RequestExtras.String(extra, RequestExtras.ReferenceCorrection, "apg")!.Trim().ToLowerInvariant();
        return name switch
        {
            "apg" => ReferenceCorrection.Apg,
            "rescale" => ReferenceCorrection.StdRescale,
            "none" => ReferenceCorrection.None,
            _ => throw new ArgumentOutOfRangeException(RequestExtras.ReferenceCorrection, name,
                $"'{RequestExtras.ReferenceCorrection}' must be 'apg', 'rescale' or 'none'."),
        };
    }

    private static ReferenceOutside OutsideFrom(IReadOnlyDictionary<string, object>? extra)
    {
        string name = RequestExtras.String(extra, RequestExtras.ReferenceOutside, "keep")!.Trim().ToLowerInvariant();
        return name switch
        {
            "keep" => ReferenceOutside.KeepReference,
            "drop" => ReferenceOutside.DropReference,
            _ => throw new ArgumentOutOfRangeException(RequestExtras.ReferenceOutside, name,
                $"'{RequestExtras.ReferenceOutside}' must be 'keep' or 'drop'."),
        };
    }

    private static float Read(IReadOnlyDictionary<string, object>? extra, string key, double fallback, double min, double max, string what)
    {
        double value = RequestExtras.Number(extra, key, fallback);
        if (!double.IsFinite(value) || value < min || value > max)
        {
            throw new ArgumentOutOfRangeException(key, value,
                $"The {what} must be between {min.ToString(CultureInfo.InvariantCulture)} and {max.ToString(CultureInfo.InvariantCulture)}.");
        }
        return (float)value;
    }
}
