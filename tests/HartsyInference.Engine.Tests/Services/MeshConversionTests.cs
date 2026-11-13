using System.IO.Compression;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The conversion service over a glb the way a 3D model returns it: every target, the static-only report, the archive, and the refusals.</summary>
public sealed class MeshConversionTests
{
    private static byte[] Glb() => GlbWriter.Write(new Mesh
    {
        Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        Indices = [0, 1, 2, 2, 1, 3],
        Normals = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    });

    [Theory]
    [InlineData("obj", "model.obj")]
    [InlineData("ply", "model.ply")]
    [InlineData("stl", "model.stl")]
    [InlineData("glb", "model.glb")]
    public void AGlbConvertsToEveryTargetAndCountsWhatItWrote(string target, string file)
    {
        using InferenceEngine engine = new("cpu");

        MeshConversion result = engine.MeshConversion.Convert(Glb(), "glb", target);

        Assert.Equal(target, result.Target);
        Assert.Equal(file, result.Files[0].Name);
        Assert.Equal((4L, 2L), (result.Vertices, result.Triangles));
        Assert.False(result.StaticOnly);
        Assert.Empty(result.Notes);
    }

    [Fact]
    public void AnObjConvertsToGlbWithItsNormalsAndTheListsOfFormatsAreExposed()
    {
        using InferenceEngine engine = new("cpu");
        byte[] obj = System.Text.Encoding.UTF8.GetBytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n");

        MeshConversion result = engine.MeshConversion.Convert(obj, "OBJ", "GLB", "from obj");

        Assert.Equal("from_obj.glb", result.Files.Single().Name);
        Assert.NotNull(GlbReader.ParseDocument(result.Files[0].Data).Primitives.Single().Geometry.Normals);
        Assert.Equal(["glb", "obj"], engine.MeshConversion.Sources);
        Assert.Equal(["obj", "ply", "stl", "glb"], engine.MeshConversion.Targets);
    }

    [Fact]
    public void ArchiveHoldsEveryFileOfTheExport()
    {
        using InferenceEngine engine = new("cpu");

        MeshConversion result = engine.MeshConversion.Convert(Glb(), "glb", "obj");

        using ZipArchive zip = new(new MemoryStream(result.Archive()));
        Assert.Equal(result.Files.Select(f => f.Name), zip.Entries.Select(e => e.FullName));
    }

    [Fact]
    public void ASplatPlyAndAMeshPlyAreRefusedWithTheirOwnReason()
    {
        using InferenceEngine engine = new("cpu");
        byte[] splat = System.Text.Encoding.ASCII.GetBytes("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float f_dc_0\nproperty float opacity\nend_header\n0 0 0\n");
        byte[] meshPly = PlyWriter.WriteMeshBinary(new Mesh { Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0], Indices = [0, 1, 2] });

        Assert.Contains("Gaussian-splat", Assert.Throws<NotSupportedException>(() => engine.MeshConversion.Convert(splat, "ply", "glb")).Message);
        string mesh = Assert.Throws<NotSupportedException>(() => engine.MeshConversion.Convert(meshPly, "PLY", "glb")).Message;
        Assert.Contains("mesh PLY is not read", mesh);
        Assert.DoesNotContain("splat", mesh);
    }

    [Fact]
    public void AnUnknownTargetOrSourceNamesTheSupportedOnes()
    {
        using InferenceEngine engine = new("cpu");

        Assert.Contains("obj, ply, stl, glb", Assert.Throws<NotSupportedException>(() => engine.MeshConversion.Convert(Glb(), "glb", "fbx")).Message);
        Assert.Contains("glb, obj", Assert.Throws<NotSupportedException>(() => engine.MeshConversion.Convert(Glb(), "fbx", "glb")).Message);
    }

    [Fact]
    public void AMalformedFileIsADomainError()
    {
        using InferenceEngine engine = new("cpu");

        Assert.Throws<HartsyInferenceException>(() => engine.MeshConversion.Convert([1, 2, 3, 4], "glb", "obj"));
    }
}
