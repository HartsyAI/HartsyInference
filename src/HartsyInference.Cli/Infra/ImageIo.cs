using HartsyInference.Vision.Codec;


namespace HartsyInference.Cli.Infra;

/// <summary>Reads the two image formats the CLI itself produces and consumes.</summary>
public static class ImageIo
{
    /// <summary>Decodes an image file to RGB24: the CLI's own BMP artifacts by extension, else by magic bytes — baseline JPEG, else PNG.</summary>
    /// <remarks>BMP stays extension-keyed because <see cref="BmpEncoder"/> is the CLI's private artifact format. JPEG and PNG are sniffed from the
    /// file's first bytes, so a .jpg that is really a PNG (or the reverse) decodes as what it is.</remarks>
    public static (byte[] Rgb, int Width, int Height) DecodeFile(string path)
    {
        if (path.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase))
            return BmpEncoder.Decode(File.ReadAllBytes(path));
        byte[] bytes = File.ReadAllBytes(path);
        return JpegDecoder.IsJpeg(bytes) ? JpegDecoder.Decode(bytes) : PngDecoder.Decode(bytes);
    }
}
