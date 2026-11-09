using System.Buffers.Binary;
using System.Text;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Tests.Io;

public sealed class PointCloudPlyReaderTests
{
    private static byte[] Ply(string header, byte[] body)
    {
        byte[] h = Encoding.ASCII.GetBytes(header);
        byte[] all = new byte[h.Length + body.Length];
        h.CopyTo(all, 0);
        body.CopyTo(all, h.Length);
        return all;
    }

    private static byte[] DoubleRow(params double[] values)
    {
        byte[] b = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteDoubleLittleEndian(b.AsSpan(i * 8), values[i]);
        }
        return b;
    }

    [Fact]
    public void Reads_xyz_and_normals_and_ignores_colours()
    {
        const string header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n" +
                              "property double x\nproperty double y\nproperty double z\n" +
                              "property double nx\nproperty double ny\nproperty double nz\n" +
                              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
        byte[] row0 = [.. DoubleRow(1, 2, 3, 0, 0, 1), 10, 20, 30];
        byte[] row1 = [.. DoubleRow(-1, 0.5, 4, 1, 0, 0), 40, 50, 60];
        PointCloudPlyReader.PointCloud pc = PointCloudPlyReader.Read(Ply(header, [.. row0, .. row1]));

        Assert.Equal(2, pc.Count);
        Assert.Equal([1, 2, 3, 0, 0, 1, -1, 0.5, 4, 1, 0, 0], pc.Data);
    }

    [Fact]
    public void Rejects_ascii()
    {
        const string header = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n" +
                              "property float nx\nproperty float ny\nproperty float nz\nend_header\n";
        Assert.Throws<InvalidDataException>(() => PointCloudPlyReader.Read(Ply(header, [])));
    }

    [Fact]
    public void Rejects_missing_normals()
    {
        const string header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty double x\nproperty double y\n" +
                              "property double z\nend_header\n";
        Assert.Throws<InvalidDataException>(() => PointCloudPlyReader.Read(Ply(header, DoubleRow(1, 2, 3))));
    }

    [Fact]
    public void Rejects_a_truncated_body()
    {
        const string header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty double x\nproperty double y\n" +
                              "property double z\nproperty double nx\nproperty double ny\nproperty double nz\nend_header\n";
        Assert.Throws<InvalidDataException>(() => PointCloudPlyReader.Read(Ply(header, DoubleRow(1, 2, 3, 0, 0, 1))));
    }

    [Fact]
    public void Reads_float_properties()
    {
        const string header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\n" +
                              "property float z\nproperty float nx\nproperty float ny\nproperty float nz\nend_header\n";
        byte[] body = new byte[24];
        float[] vals = [0.25f, -0.5f, 2f, 0f, 1f, 0f];
        for (int i = 0; i < 6; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(body.AsSpan(i * 4), vals[i]);
        }
        PointCloudPlyReader.PointCloud pc = PointCloudPlyReader.Read(Ply(header, body));

        Assert.Equal(new double[] { 0.25, -0.5, 2, 0, 1, 0 }, pc.Data);
    }
}
