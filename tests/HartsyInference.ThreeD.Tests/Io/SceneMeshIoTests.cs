using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Io;

public sealed class SceneMeshIoTests
{
    [Fact]
    public void ObjReader_FanTriangulatesQuadsAndResolvesRelativeIndices()
    {
        const string obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\nf -4 -3 -2\n";

        Mesh mesh = ObjReader.Parse(obj);

        Assert.Equal(4, mesh.VertexCount);
        Assert.Equal([0, 1, 2, 0, 2, 3, 0, 1, 2], mesh.Indices);
    }

    [Fact]
    public void ObjReader_RejectsFaceIndexOutsideTheDefinedVertices()
    {
        Assert.Throws<HartsyInferenceException>(() => ObjReader.Parse("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 5\n"));
    }

    [Fact]
    public void ObjReader_SplitsOnAnyWhitespaceAndSkipsKnownRecords()
    {
        Mesh mesh = ObjReader.Parse("# scene\r\nvn 0 0 1\r\nv\t0\t0\t0\r\nv 1 0 0\r\nv 0 1 0\r\nvt 0 0\r\nusemtl wood\r\nf\t1\t2\t3\r\n");

        Assert.Equal(3, mesh.VertexCount);
        Assert.Equal([0, 1, 2], mesh.Indices);
    }

    [Fact]
    public void GlbReader_AppliesNodeTransformAndMirrorsWindingForANegativeScale()
    {
        byte[] glb = TriangleGlb("{\"mesh\":0,\"translation\":[10,0,0],\"scale\":[-1,1,1]}", Positions);

        Mesh mesh = GlbReader.Parse(glb);

        Assert.Equal(new float[] { 10f, 0f, 0f, 9f, 0f, 0f, 10f, 1f, 0f }, mesh.Vertices);
        Assert.Equal([0, 2, 1], mesh.Indices);
    }

    [Theory]
    [InlineData("{\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"}")]
    [InlineData("{\"bufferView\":0,\"componentType\":5126,\"count\":-1,\"type\":\"VEC3\"}")]
    [InlineData("{\"bufferView\":7,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"}")]
    public void GlbReader_RefusesMalformedAccessorsWithADomainError(string position)
    {
        byte[] glb = TriangleGlb("{\"mesh\":0}", position);

        Assert.Throws<HartsyInferenceException>(() => GlbReader.Parse(glb));
    }

    private const string Positions = """{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}""";

    // One triangle in a 48-byte buffer: three float VEC3 positions, then three uint32 indices.
    private static byte[] TriangleGlb(string node, string position)
    {
        string json = $$"""
            {"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{{node}}],
             "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],
             "accessors":[{{position}},{"bufferView":1,"componentType":5125,"count":3,"type":"SCALAR"}],
             "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":12}],
             "buffers":[{"byteLength":48}]}
            """;
        MemoryStream bin = new();
        using (BinaryWriter writer = new(bin, Encoding.UTF8, leaveOpen: true))
        {
            foreach (float value in new float[] { 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f })
                writer.Write(value);
            foreach (uint index in new uint[] { 0, 1, 2 })
                writer.Write(index);
        }
        return Glb(json, bin.ToArray());
    }

    // Wraps a JSON chunk and a binary chunk in a GLB 2.0 container, padding the JSON with spaces to 4 bytes.
    private static byte[] Glb(string json, byte[] bin)
    {
        byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
        jsonBytes = [.. jsonBytes, .. new byte[(4 - jsonBytes.Length % 4) % 4].Select(_ => (byte)0x20)];
        MemoryStream stream = new();
        using (BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(0x46546C67u);
            writer.Write(2u);
            writer.Write((uint)(12 + 8 + jsonBytes.Length + 8 + bin.Length));
            writer.Write((uint)jsonBytes.Length);
            writer.Write(0x4E4F534Au);
            writer.Write(jsonBytes);
            writer.Write((uint)bin.Length);
            writer.Write(0x004E4942u);
            writer.Write(bin);
        }
        return stream.ToArray();
    }

    [Fact]
    public void GlbWriter_RoundTripsSeveralMeshesWithRebasedIndices()
    {
        Mesh first = new()
        {
            Vertices = [0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f],
            Indices = [0, 1, 2],
            Normals = [0f, 0f, 1f, 0f, 0f, 1f, 0f, 0f, 1f],
        };
        Mesh second = new()
        {
            Vertices = [2f, 2f, 2f, 3f, 2f, 2f, 2f, 3f, 2f, 3f, 3f, 2f],
            Indices = [0, 1, 2, 1, 3, 2],
        };

        byte[] glb = GlbWriter.Write([first, second]);
        Mesh read = GlbReader.Parse(glb);

        Assert.Equal(21, read.Vertices.Length);
        Assert.Equal([0, 1, 2, 3, 4, 5, 4, 6, 5], read.Indices);
        Assert.Equal(new float[] { 2f, 2f, 2f }, read.Vertices[9..12]);
    }

    [Fact]
    public void GlbWriter_RejectsEmptyMesh()
    {
        Mesh empty = new() { Vertices = [], Indices = [] };

        Assert.Throws<ArgumentException>(() => GlbWriter.Write([empty]));
    }

    [Fact]
    public void SceneMeshReader_RejectsUnsupportedExtensions()
    {
        Assert.Throws<NotSupportedException>(() => SceneMeshReader.FormatOf("C:/scenes/room.fbx"));
        Assert.Equal("glb", SceneMeshReader.FormatOf("C:/scenes/room.GLB"));
        Assert.Throws<NotSupportedException>(() => SceneMeshReader.Read(Encoding.UTF8.GetBytes("x"), "ply"));
    }
}
