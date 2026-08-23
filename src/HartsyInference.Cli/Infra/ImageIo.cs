using HartsyInference.Vision.Codec;

namespace HartsyInference.Cli.Infra;

/// <summary>Reads the two image formats the CLI itself produces and consumes.</summary>
public static class ImageIo
{
    /// <summary>Decodes an image file to RGB24 by extension — the CLI's own BMP artifacts, else PNG.</summary>
    /// <remarks>Dispatch is on the extension rather than a magic-byte sniff because <see cref="BmpEncoder"/> is the
    /// CLI's private artifact format and every path that reaches here was named by the CLI or typed by the user.</remarks>
    public static (byte[] Rgb, int Width, int Height) DecodeFile(string path)
    {
        if (path.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
            return BmpEncoder.Decode(File.ReadAllBytes(path));
        return PngDecoder.DecodeFromFile(path);
    }
}
