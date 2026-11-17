using System.Text;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Geometry;

public sealed class GaussianSplatActivationsTests
{
    [Theory]
    [InlineData(0.004f)]
    [InlineData(0.5f)]
    [InlineData(3f)]
    public void SoftplusInverse_UndoesSoftplus(float x)
    {
        Assert.Equal(x, GaussianSplatActivations.Softplus(GaussianSplatActivations.SoftplusInverse(x)), 5);
    }

    [Fact]
    public void Softplus_ReturnsTheInputAboveTwenty_AsTorchDoes()
    {
        Assert.Equal(25f, GaussianSplatActivations.Softplus(25f));
        Assert.Equal(Math.Log(1.0 + Math.Exp(2.0)), GaussianSplatActivations.Softplus(2f), 5);
    }

    [Fact]
    public void LogScale_IsTheLogOfTheKernelFlooredScale()
    {
        const float bias = 0.004f, kernel = 0.0009f;
        float offset = GaussianSplatActivations.SoftplusInverse(bias);

        // A zero output gives the initial scale, floored by the kernel: sqrt(bias² + kernel²).
        Assert.Equal(Math.Log(Math.Sqrt(bias * (double)bias + kernel * (double)kernel)), GaussianSplatActivations.LogScale(0f, offset, kernel), 4);   // float32 SoftplusInverse of 0.004 is good to about 1e-5
        // A large raw value is the scale itself, as the floor stops mattering.
        Assert.Equal(Math.Log(Math.Sqrt(Math.Pow(Math.Log(1.0 + Math.Exp(1.5 + offset)), 2) + kernel * (double)kernel)),
            GaussianSplatActivations.LogScale(1.5f, offset, kernel), 4);
    }

    [Fact]
    public void TurnToYUp_IsAQuarterTurnAboutX()
    {
        // (w, x, y, z) = identity turns to (√½, √½, 0, 0), the turn itself.
        (float w, float x, float y, float z) = GaussianSplatActivations.TurnToYUp(1f, 0f, 0f, 0f);
        Assert.Equal(Math.Sqrt(0.5), w, 6);
        Assert.Equal(Math.Sqrt(0.5), x, 6);
        Assert.Equal(0f, y);
        Assert.Equal(0f, z);

        // The product keeps a unit quaternion unit, and rotating a y axis by it gives the z axis (and z gives -y), like positions (x, y, z) -> (x, -z, y).
        (float qw, float qx, float qy, float qz) = (0.5f, 0.5f, 0.5f, 0.5f);
        (float rw, float rx, float ry, float rz) = GaussianSplatActivations.TurnToYUp(qw, qx, qy, qz);
        Assert.Equal(1.0, rw * (double)rw + rx * (double)rx + ry * (double)ry + rz * (double)rz, 5);
    }
}

public sealed class PlyWriterSplatTests
{
    [Fact]
    public void WriteSplats_WritesTheInriaLayoutInOrder()
    {
        GaussianSplatCloud cloud = new()
        {
            Positions = [1f, 2f, 3f, 4f, 5f, 6f],
            Scales = [-1f, -2f, -3f, -4f, -5f, -6f],
            Rotations = [1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f],
            Opacities = [0.25f, -0.75f],
            ShCoefficients = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f],
            ShCoeffsPerSplat = 1,
        };

        byte[] ply = PlyWriter.WriteSplats(cloud);

        string marker = "end_header\n";
        string text = Encoding.ASCII.GetString(ply, 0, Math.Min(ply.Length, 800));
        int body = text.IndexOf(marker, StringComparison.Ordinal) + marker.Length;
        string header = text[..body];
        Assert.StartsWith("ply\nformat binary_little_endian 1.0\n", header);
        Assert.Contains("element vertex 2\n", header);
        string[] names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"];
        int[] at = names.Select(n => header.IndexOf($"property float {n}\n", StringComparison.Ordinal)).ToArray();
        Assert.All(at, i => Assert.True(i >= 0));
        Assert.Equal(at.OrderBy(i => i), at);
        Assert.Equal(body + 2 * names.Length * 4, ply.Length);

        float F(int splat, int property) => BitConverter.ToSingle(ply, body + (splat * names.Length + property) * 4);
        float[][] expected =
        [
            [1f, 2f, 3f, 0f, 0f, 0f, 0.1f, 0.2f, 0.3f, 0.25f, -1f, -2f, -3f, 1f, 0f, 0f, 0f],
            [4f, 5f, 6f, 0f, 0f, 0f, 0.4f, 0.5f, 0.6f, -0.75f, -4f, -5f, -6f, 0f, 1f, 0f, 0f],
        ];
        for (int s = 0; s < 2; s++)
            for (int p = 0; p < names.Length; p++)
                Assert.Equal(expected[s][p], F(s, p));
    }
}
