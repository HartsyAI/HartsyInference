using HartsyInference.Engine.Requests;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Engine.Features;

/// <summary>Decodes an encoded image (PNG today) into the raw RGB24 <see cref="ImageData"/> the request contract
/// carries. Every transport needs this: the contract is deliberately raster-only so the Engine has no host image
/// dependency, but a CLI has a file path and an HTTP client has base64 — neither has loose RGB bytes.</summary>
public static class ImageDataCodec
{
    /// <summary>Decodes PNG bytes. Throws with the detected signature named when the format is not one we decode,
    /// because "invalid image" on a JPEG upload is a slow thing to diagnose from the client side.</summary>
    public static ImageData Decode(ReadOnlySpan<byte> encoded)
    {
        if (encoded.Length < 8)
        {
            throw new ArgumentException($"Image data is too short to identify ({encoded.Length} bytes).", nameof(encoded));
        }
        if (!IsPng(encoded))
        {
            throw new NotSupportedException(
                $"Unsupported image format (leading bytes {Describe(encoded)}). The engine decodes PNG; re-encode the image as PNG.");
        }
        (byte[] rgb, int width, int height) = PngDecoder.Decode(encoded);
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }

    /// <summary>Decodes a base64 payload, tolerating a <c>data:image/png;base64,</c> prefix since browser clients send
    /// data URIs verbatim.</summary>
    public static ImageData DecodeBase64(string base64)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(base64);
        string payload = base64.Trim();
        int comma = payload.IndexOf(',');
        if (payload.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && comma > 0)
        {
            payload = payload[(comma + 1)..];
        }
        byte[] bytes;
        try
        {
            bytes = Convert.FromBase64String(payload);
        }
        catch (FormatException ex)
        {
            throw new ArgumentException("Image is not valid base64.", nameof(base64), ex);
        }
        return Decode(bytes);
    }

    private static bool IsPng(ReadOnlySpan<byte> data) =>
        data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47;

    private static string Describe(ReadOnlySpan<byte> data)
    {
        if (data.Length >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF)
        {
            return "JPEG";
        }
        if (data.Length >= 2 && data[0] == 0x42 && data[1] == 0x4D)
        {
            return "BMP";
        }
        if (data.Length >= 4 && data[0] == 0x52 && data[1] == 0x49 && data[2] == 0x46 && data[3] == 0x46)
        {
            return "RIFF/WebP";
        }
        return $"0x{data[0]:X2}{data[1]:X2}";
    }
}
