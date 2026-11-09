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
