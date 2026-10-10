using HartsyInference.Vision.Codec;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Unit-tier kernel correctness for <see cref="Resample"/> against
/// <c>torch.nn.functional.interpolate(mode="bicubic", antialias=True)</c> golden values (a = −0.75),
/// covering both the antialiased downscale path and the plain-bicubic upscale path.</summary>
public sealed class ResampleTests
{
    private static float[] Ramp(int count)
    {
        float[] data = new float[count];
        for (int i = 0; i < count; i++) data[i] = i;
        return data;
    }

    [Fact]
    public void BicubicPlane_Downscale_MatchesTorchAntialias()
    {
        // torch: F.interpolate(arange(48).reshape(1,1,6,8), (3,4), mode="bicubic", antialias=True)
        float[] expected =
        [
            4.895399f, 6.810924f, 8.892006f, 10.807532f,
            20.543934f, 22.459459f, 24.540541f, 26.456066f,
            36.192471f, 38.107994f, 40.189079f, 42.104607f,
        ];
        float[] dst = new float[12];
        Resample.BicubicPlane(Ramp(48), 8, 6, dst, 4, 3, a: -0.5f, antialias: true);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(dst[i] - expected[i]) < 1e-4f, $"[{i}] {dst[i]} != {expected[i]}");
    }

    [Fact]
    public void BicubicPlane_Upscale_MatchesTorch()
    {
        // torch: F.interpolate(arange(48).reshape(1,1,6,8), (9,12), mode="bicubic", antialias=True)
        // (antialias is a no-op on upscale — this pins the align_corners=False half-pixel phase).
        float[] expectedRow0 =
        [
            -0.592104f, -0.11455f, 0.640352f, 1.307019f, 1.973685f, 2.640352f,
            3.307019f, 3.973686f, 4.640352f, 5.307018f, 6.061922f, 6.539475f,
        ];
        float[] expectedLast =
        [
            40.460529f, 40.938084f, 41.692989f, 42.35965f, 43.026318f, 43.692986f,
            44.35965f, 45.026318f, 45.692986f, 46.35965f, 47.114555f, 47.592106f,
        ];
        float[] dst = new float[9 * 12];
        Resample.BicubicPlane(Ramp(48), 8, 6, dst, 12, 9, a: -0.5f, antialias: true);
        for (int i = 0; i < 12; i++)
        {
            Assert.True(Math.Abs(dst[i] - expectedRow0[i]) < 1e-4f, $"row0[{i}] {dst[i]} != {expectedRow0[i]}");
            Assert.True(Math.Abs(dst[8 * 12 + i] - expectedLast[i]) < 1e-4f, $"row8[{i}] {dst[8 * 12 + i]} != {expectedLast[i]}");
        }
    }

}
