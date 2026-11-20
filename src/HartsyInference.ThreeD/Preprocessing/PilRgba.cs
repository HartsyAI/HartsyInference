using HartsyInference.Core.Imaging;

namespace HartsyInference.ThreeD.Preprocessing;

/// <summary>The 8-bit RGBA operations of Pillow that the model input preparations repeat: RGBA is filtered premultiplied (<c>RGBa</c>), and a
/// foreground is composited on black by a paste through its alpha. Written in Pillow's integer arithmetic so a prepared image matches the
/// reference's pixels.</summary>
internal static class PilRgba
{
    /// <summary>Multiplies each colour by its alpha, in place: <c>c·a / 255</c> rounded as Pillow's <c>MULDIV255</c>.</summary>
    public static void Premultiply(byte[] rgba)
    {
        for (int i = 0; i < rgba.Length; i += 4)
        {
            for (int c = 0; c < 3; c++)
            {
                int t = rgba[i + c] * rgba[i + 3] + 128;
                rgba[i + c] = (byte)((t + (t >> 8)) >> 8);
            }
        }
    }

    /// <summary>Divides each colour by its alpha, in place: <c>255·c / a</c> truncated and clipped; a transparent pixel becomes black.</summary>
    public static void Unpremultiply(byte[] rgba)
    {
        for (int i = 0; i < rgba.Length; i += 4)
        {
            int a = rgba[i + 3];
            for (int c = 0; c < 3; c++)
                rgba[i + c] = a == 0 ? (byte)0 : (byte)Math.Min(255 * rgba[i + c] / a, 255);
        }
    }

    /// <summary><c>Image.resize(size, LANCZOS)</c> of an RGBA image: premultiply, filter, divide back.</summary>
    public static byte[] ResizeLanczos(byte[] rgba, int width, int height, int outWidth, int outHeight)
    {
        byte[] premultiplied = (byte[])rgba.Clone();
        Premultiply(premultiplied);
        byte[] resized = new byte[outWidth * outHeight * 4];
        Resample.LanczosPil8(premultiplied, width, height, 4, resized, outWidth, outHeight);
        Unpremultiply(resized);
        return resized;
    }

    /// <summary><c>ImageFilter.MinFilter(3)</c> of a plane: the minimum over the 3 × 3 window, clipped at the borders.</summary>
    public static byte[] MinFilter3(byte[] plane, int width, int height)
    {
        byte[] result = new byte[plane.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte min = 255;
                for (int dy = Math.Max(y - 1, 0); dy <= Math.Min(y + 1, height - 1); dy++)
                {
                    for (int dx = Math.Max(x - 1, 0); dx <= Math.Min(x + 1, width - 1); dx++)
                        min = Math.Min(min, plane[dy * width + dx]);
                }

                result[y * width + x] = min;
            }
        }

        return result;
    }

    /// <summary><c>Image.new("RGB", size, black).paste(image, mask=alpha)</c>: <c>round(c·a / 255)</c> per colour, as Pillow's <c>BLEND</c> with a black background.</summary>
    public static byte[] CompositeOnBlack(byte[] rgba, int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int i = 0; i < width * height; i++)
        {
            int a = rgba[i * 4 + 3];
            for (int c = 0; c < 3; c++)
            {
                int t = rgba[i * 4 + c] * a + 128;
                rgb[i * 3 + c] = (byte)((t + (t >> 8)) >> 8);
            }
        }

        return rgb;
    }
}
