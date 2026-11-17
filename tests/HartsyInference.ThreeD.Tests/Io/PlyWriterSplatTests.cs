using System.Text;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Io;

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
