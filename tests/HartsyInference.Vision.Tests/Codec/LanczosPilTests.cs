using HartsyInference.Core.Imaging;

namespace HartsyInference.Vision.Tests.Codec;

/// <summary><see cref="Resample.LanczosPil8"/> against Pillow 13 <c>Image.resize(size, Image.LANCZOS)</c> on a 9×6 RGB pattern
/// with pixel <c>(x, y)</c> = <c>((37x + 11y) % 256, (13x² + 29y) % 256, (255 − 23x − 17y) % 256)</c>.</summary>
public sealed class LanczosPilTests
{
    private const int Width = 9, Height = 6;

    [Fact]
    public void Downscale_MatchesPillowBitForBit()
    {
        byte[] actual = new byte[5 * 4 * 3];

        Resample.LanczosPil8(Pattern(), Width, Height, 3, actual, 5, 4);

        Assert.Equal(
        [
            18, 7, 242, 82, 81, 201, 160, 174, 159, 180, 173, 117, 25, 111, 76, 34, 50, 217, 98, 119, 176, 175, 115, 134, 209, 112,
            92, 42, 140, 53, 51, 94, 190, 114, 172, 149, 197, 112, 107, 170, 105, 66, 53, 190, 15, 67, 152, 165, 128, 143, 124, 221,
            120, 85, 82, 132, 28, 58, 142, 90,
        ], actual);
    }

    [Fact]
    public void Upscale_MatchesPillowBitForBit()
    {
        byte[] actual = new byte[13 * 8 * 3];

        Resample.LanczosPil8(Pattern(), Width, Height, 3, actual, 13, 8);

        Assert.Equal(
        [
            0, 0, 255, 14, 3, 246, 46, 18, 227, 71, 48, 211, 96, 81, 196, 121, 157, 180, 147, 201, 164, 165, 91, 148, 221, 106, 132,
            210, 210, 117, 42, 156, 101, 0, 81, 82, 48, 59, 71, 4, 15, 247, 21, 19, 236, 53, 34, 217, 78, 64, 201, 103, 93, 186, 128,
            185, 170, 154, 251, 154, 172, 115, 138, 228, 133, 122, 217, 255, 107, 49, 181, 91, 4, 94, 72, 55, 77, 61, 13, 42, 232, 31,
            46, 221, 63, 60, 202, 88, 89, 186, 113, 137, 171, 138, 168, 155, 164, 157, 139, 182, 117, 123, 236, 122, 107, 223, 166,
            92, 58, 177, 76, 14, 135, 57, 65, 100, 46, 21, 61, 220, 38, 65, 209, 70, 79, 190, 95, 106, 174, 120, 183, 159, 145, 123,
            143, 171, 0, 127, 188, 100, 111, 249, 86, 95, 242, 3, 80, 68, 149, 64, 20, 177, 45, 72, 115, 35, 30, 84, 207, 47, 88, 196,
            79, 102, 177, 104, 129, 161, 129, 201, 146, 154, 145, 130, 180, 28, 114, 195, 125, 98, 255, 112, 82, 255, 35, 67, 79, 170,
            51, 28, 196, 34, 81, 138, 26, 37, 103, 195, 54, 108, 184, 86, 119, 165, 111, 146, 149, 136, 252, 134, 159, 205, 118, 187,
            59, 102, 229, 144, 86, 187, 132, 70, 88, 69, 56, 41, 234, 42, 58, 240, 1, 84, 153, 0, 47, 130, 180, 64, 132, 169, 96, 154,
            150, 121, 181, 134, 146, 172, 119, 168, 112, 103, 197, 82, 87, 255, 178, 71, 143, 168, 57, 0, 82, 38, 20, 127, 18, 84, 182,
            57, 91, 196, 103, 54, 146, 170, 71, 144, 159, 103, 182, 140, 128, 207, 124, 153, 59, 109, 175, 0, 93, 204, 98, 77, 254,
            203, 61, 164, 198, 51, 18, 86, 25, 35, 0, 0, 86, 97, 133, 98, 231, 255,
        ], actual);
    }

    [Fact]
    public void UnchangedSize_IsACopy()
    {
        byte[] source = Pattern();
        byte[] actual = new byte[source.Length];

        Resample.LanczosPil8(source, Width, Height, 3, actual, Width, Height);

        Assert.Equal(source, actual);
    }

    private static byte[] Pattern()
    {
        byte[] pixels = new byte[Width * Height * 3];
        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++)
            {
                int i = (y * Width + x) * 3;
                pixels[i] = (byte)((x * 37 + y * 11) % 256);
                pixels[i + 1] = (byte)((x * x * 13 + y * 29) % 256);
                pixels[i + 2] = (byte)(((255 - x * 23 - y * 17) % 256 + 256) % 256);
            }
        }
        return pixels;
    }
}
