using System.Buffers.Binary;
using System.IO.Compression;

namespace HartsyInference.Vision.Codec;

/// <summary>Minimal pure-C# PNG decoder supporting the two color types 99% of test fixtures use — RGB
/// (color type 2) and RGBA (color type 6, alpha dropped on output) — at 8 bits per channel, non-interlaced;
/// returns HWC-packed RGB u8, the same input format every <c>HartsyInference.Vision</c> pipeline expects.
/// Deliberately unsupported: 16-bit channels, grayscale/palette inputs, and Adam7 interlacing (obsolete);
/// unsupported formats throw <see cref="NotSupportedException"/> with a descriptive message so callers can
/// fall back to an external decoder or pre-convert images.</summary>
public static class PngDecoder
{
    private static readonly byte[] _pngSignature = [137, 80, 78, 71, 13, 10, 26, 10];

    /// <summary>Decodes a PNG file from disk.</summary>
    public static (byte[] rgb, int width, int height) DecodeFromFile(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        byte[] bytes = File.ReadAllBytes(path);
        return Decode(bytes);
    }

    /// <summary>Decodes a PNG from raw bytes. Returns HWC-packed RGB u8, the source width, and height; alpha and EXIF orientation are dropped.</summary>
    public static (byte[] rgb, int width, int height) Decode(ReadOnlySpan<byte> data)
    {
        DecodedPng png = Parse(data, applyOrientation: false);
        return (png.Rgb, png.Width, png.Height);
    }

    /// <summary>Decodes a PNG with its alpha (RGBA, or RGB with a colour-key <c>tRNS</c> chunk) and applies its <c>eXIf</c>
    /// orientation to the pixels the way PIL's <c>ImageOps.exif_transpose</c> does.</summary>
    public static DecodedPng DecodeWithAlpha(ReadOnlySpan<byte> data) => Parse(data, applyOrientation: true);

    private static DecodedPng Parse(ReadOnlySpan<byte> data, bool applyOrientation)
    {
        if (data.Length < 8 || !data.Slice(0, 8).SequenceEqual(_pngSignature))
            throw new ArgumentException("Not a PNG file — signature mismatch.", nameof(data));

        int offset = 8;
        int width = 0, height = 0, bitDepth = 0, colorType = -1, interlace = 0;
        List<byte[]> idatChunks = new();
        byte[]? transparency = null;
        byte[]? exif = null;

        while (offset < data.Length)
        {
            if (offset + 8 > data.Length)
                throw new InvalidDataException("PNG chunk header truncated.");
            int chunkLength = BinaryPrimitives.ReadInt32BigEndian(data.Slice(offset));
            string chunkType = System.Text.Encoding.ASCII.GetString(data.Slice(offset + 4, 4));
            offset += 8;
            if (offset + chunkLength + 4 > data.Length)
                throw new InvalidDataException($"PNG chunk '{chunkType}' truncated.");
            ReadOnlySpan<byte> body = data.Slice(offset, chunkLength);

            switch (chunkType)
            {
                case "IHDR":
                    if (body.Length < 13)
                        throw new InvalidDataException("IHDR chunk too short.");
                    width = BinaryPrimitives.ReadInt32BigEndian(body.Slice(0));
                    height = BinaryPrimitives.ReadInt32BigEndian(body.Slice(4));
                    bitDepth = body[8];
                    colorType = body[9];
                    int compression = body[10];
                    int filter = body[11];
                    interlace = body[12];
                    if (compression != 0)
                        throw new NotSupportedException($"PNG compression method {compression} not supported (only DEFLATE=0).");
                    if (filter != 0)
                        throw new NotSupportedException($"PNG filter method {filter} not supported (only 0).");
                    if (interlace != 0)
                        throw new NotSupportedException("Adam7 interlaced PNGs not supported.");
                    if (bitDepth != 8)
                        throw new NotSupportedException($"Only 8-bit PNGs are supported; got bit depth {bitDepth}.");
                    if (colorType != 2 && colorType != 6)
                        throw new NotSupportedException($"PNG color type {colorType} not supported (use RGB=2 or RGBA=6).");
                    break;
                case "IDAT":
                    idatChunks.Add(body.ToArray());
                    break;
                case "tRNS":
                    transparency = body.ToArray();
                    break;
                case "eXIf":
                    exif = body.ToArray();
                    break;
                case "IEND":
                    goto EndLoop;
                // All other chunks (tEXt, pHYs, gAMA, sRGB, PLTE without indexed color, etc.) ignored.
            }

            offset += chunkLength + 4; // chunk body + CRC32 we don't verify
        }
        EndLoop:

        if (width <= 0 || height <= 0 || colorType < 0)
            throw new InvalidDataException("PNG is missing or has invalid IHDR.");
        if (idatChunks.Count == 0)
            throw new InvalidDataException("PNG has no IDAT chunks.");

        // Concatenate IDAT data, then DEFLATE-decompress. The IDAT chunks together form a single
        // zlib stream (zlib = 2-byte header + DEFLATE + 4-byte Adler32). We skip the 2-byte zlib
        // header to feed the raw DEFLATE body into .NET's DeflateStream.
        int totalCompressed = 0;
        foreach (byte[] chunk in idatChunks) totalCompressed += chunk.Length;
        byte[] compressed = new byte[totalCompressed];
        int writeOffset = 0;
        foreach (byte[] chunk in idatChunks)
        {
            Buffer.BlockCopy(chunk, 0, compressed, writeOffset, chunk.Length);
            writeOffset += chunk.Length;
        }

        // Strip zlib 2-byte header (CMF + FLG).
        ReadOnlySpan<byte> deflated = compressed.AsSpan(2);
        // Output stride = 1 (filter byte) + bytes per scanline.
        int channelsIn = colorType == 2 ? 3 : 4;
        int bytesPerPixel = channelsIn; // 8-bit per channel.
        long expectedBytes = (1L + (long)width * bytesPerPixel) * height;
        if (expectedBytes > int.MaxValue || (long)width * height * 3 > int.MaxValue)
            throw new InvalidDataException($"PNG dimensions {width} x {height} are too large to decode.");
        int scanlineBytes = width * bytesPerPixel;
        int expected = (int)expectedBytes;

        // Inflate at most one byte past what the header promises, so a stream that expands far beyond its dimensions (a deflate bomb)
        // is refused without being held in memory.
        byte[] decompressed;
        using (MemoryStream ms = new(deflated.ToArray()))
        using (DeflateStream df = new(ms, CompressionMode.Decompress))
        // Presized to what the header promises so it does not grow by doubling, but only up to 64 MiB: a header can promise far more than the data holds.
        using (MemoryStream output = new(Math.Min(expected + 1, 64 << 20)))
        {
            byte[] block = new byte[81920];
            int read;
            while (output.Length <= expected && (read = df.Read(block, 0, (int)Math.Min(block.Length, expected + 1L - output.Length))) > 0)
                output.Write(block, 0, read);
            decompressed = output.ToArray();
        }
        if (decompressed.Length != expected)
            throw new InvalidDataException($"PNG decompressed data is {decompressed.Length} bytes; expected {expected}.");

        // Reverse per-scanline filtering and write to an HWC RGB output buffer simultaneously.
        // The filtering algorithms reference the previously-decoded scanline ("up") and the
        // previously-decoded pixel of the current scanline ("left") so we walk scanline by scanline.
        byte[] rawScanline = new byte[scanlineBytes]; // filter-undone current row
        byte[] prevScanline = new byte[scanlineBytes]; // filter-undone previous row
        byte[] rgb = new byte[width * height * 3];
        int[]? colorKey = colorType == 2 ? ReadColorKey(transparency) : null;
        byte[]? alpha = colorType == 6 || colorKey is not null ? new byte[width * height] : null;

        int srcOffset = 0;
        for (int y = 0; y < height; y++)
        {
            byte filterType = decompressed[srcOffset];
            srcOffset++;
            UnfilterScanline(filterType, decompressed.AsSpan(srcOffset, scanlineBytes), prevScanline, rawScanline, bytesPerPixel);
            srcOffset += scanlineBytes;

            int outBase = y * width * 3;
            if (colorType == 2)
            {
                Buffer.BlockCopy(rawScanline, 0, rgb, outBase, scanlineBytes);
                if (colorKey is not null)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int inOff = x * 3;
                        bool keyed = rawScanline[inOff] == colorKey[0] && rawScanline[inOff + 1] == colorKey[1]
                            && rawScanline[inOff + 2] == colorKey[2];
                        alpha![y * width + x] = keyed ? (byte)0 : (byte)255;
                    }
                }
            }
            else
            {
                for (int x = 0; x < width; x++)
                {
                    int inOff = x * 4;
                    int outOff = outBase + x * 3;
                    rgb[outOff + 0] = rawScanline[inOff + 0];
                    rgb[outOff + 1] = rawScanline[inOff + 1];
                    rgb[outOff + 2] = rawScanline[inOff + 2];
                    alpha![y * width + x] = rawScanline[inOff + 3];
                }
            }

            // Rotate buffers: this scanline becomes the previous one for the next iteration.
            (rawScanline, prevScanline) = (prevScanline, rawScanline);
        }

        int orientation = ExifOrientation.ReadTiff(exif);
        if (applyOrientation && orientation != ExifOrientation.Upright)
        {
            (rgb, alpha, width, height) = ExifOrientation.Apply(rgb, alpha, width, height, orientation);
        }

        return new DecodedPng(rgb, alpha, width, height, orientation, applyOrientation && orientation != ExifOrientation.Upright);
    }

    /// <summary>The three 8-bit colour-key samples of an RGB <c>tRNS</c> chunk, or null when there is none.</summary>
    private static int[]? ReadColorKey(byte[]? transparency)
    {
        if (transparency is not { Length: >= 6 })
            return null;
        return
        [
            BinaryPrimitives.ReadUInt16BigEndian(transparency.AsSpan(0)),
            BinaryPrimitives.ReadUInt16BigEndian(transparency.AsSpan(2)),
            BinaryPrimitives.ReadUInt16BigEndian(transparency.AsSpan(4)),
        ];
    }

    private static void UnfilterScanline(byte filterType, ReadOnlySpan<byte> filtered, byte[] prev, byte[] output, int bpp)
    {
        // PNG filter types: 0=None, 1=Sub, 2=Up, 3=Average, 4=Paeth.
        // For each: x is the current byte, a is the byte bpp positions to the left in the same
        // scanline (0 if past the start), b is the byte directly above (in prev scanline), c is
        // the byte above-and-left (0 if past the start).
        switch (filterType)
        {
            case 0: // None
                filtered.CopyTo(output);
                return;
            case 1: // Sub: x + a
                for (int i = 0; i < filtered.Length; i++)
                {
                    byte a = i >= bpp ? output[i - bpp] : (byte)0;
                    output[i] = (byte)(filtered[i] + a);
                }
                return;
            case 2: // Up: x + b
                for (int i = 0; i < filtered.Length; i++)
                    output[i] = (byte)(filtered[i] + prev[i]);
                return;
            case 3: // Average: x + (a + b) / 2
                for (int i = 0; i < filtered.Length; i++)
                {
                    byte a = i >= bpp ? output[i - bpp] : (byte)0;
                    byte b = prev[i];
                    output[i] = (byte)(filtered[i] + (a + b) / 2);
                }
                return;
            case 4: // Paeth: x + Paeth(a, b, c)
                for (int i = 0; i < filtered.Length; i++)
                {
                    byte a = i >= bpp ? output[i - bpp] : (byte)0;
                    byte b = prev[i];
                    byte c = i >= bpp ? prev[i - bpp] : (byte)0;
                    output[i] = (byte)(filtered[i] + PaethPredictor(a, b, c));
                }
                return;
            default:
                throw new NotSupportedException($"Unknown PNG filter type {filterType}.");
        }
    }

    /// <summary>The Paeth predictor — picks the neighbor (a, b, or c) closest to <c>a + b - c</c>.</summary>
    private static byte PaethPredictor(byte a, byte b, byte c)
    {
        int p = a + b - c;
        int pa = Math.Abs(p - a);
        int pb = Math.Abs(p - b);
        int pc = Math.Abs(p - c);
        if (pa <= pb && pa <= pc) return a;
        if (pb <= pc) return b;
        return c;
    }
}
