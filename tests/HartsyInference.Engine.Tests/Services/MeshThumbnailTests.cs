using System.Buffers.Binary;
using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The thumbnail service over a glb the way a 3D model returns it, a splat ply, and the refusals.</summary>
public sealed class MeshThumbnailTests
{
    private static byte[] Glb() => GlbWriter.Write(new Mesh
    {
        Vertices = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        Indices = [0, 1, 2, 2, 1, 3],
    });

    [Fact]
    public void AGlbIsDrawnToAPngOfTheRequestedSizeWithATransparentBackground()
    {
        using InferenceEngine engine = new("cpu");

        MeshThumbnail thumbnail = engine.MeshThumbnails.Render(Glb(), "glb", new MeshThumbnailRequest { Width = 96, Height = 48, View = "front" });

        DecodedPng png = PngDecoder.DecodeWithAlpha(thumbnail.Png);
        Assert.Equal((96, 48, "mesh"), (thumbnail.Width, thumbnail.Height, thumbnail.Kind));
        Assert.Equal((96, 48), (png.Width, png.Height));
        Assert.NotNull(png.Alpha);
        Assert.Equal(0, png.Alpha![0]);
        Assert.Equal(255, png.Alpha[24 * 96 + 48]);
    }

    [Fact]
    public void ABackgroundColourGivesAnOpaqueRgbPicture()
    {
        using InferenceEngine engine = new("cpu");

        MeshThumbnail thumbnail = engine.MeshThumbnails.Render(Glb(), "GLB", new MeshThumbnailRequest { Background = "#102030", Samples = 1 });

        DecodedPng png = PngDecoder.DecodeWithAlpha(thumbnail.Png);
        Assert.Null(png.Alpha);
        Assert.Equal([0x10, 0x20, 0x30], png.Rgb[..3]);
    }

    [Fact]
    public void AnObjIsDrawnAndAZUpFileCanBeTurnedUpright()
    {
        using InferenceEngine engine = new("cpu");
        byte[] obj = Encoding.UTF8.GetBytes("v 0 0 0\nv 1 0 0\nv 1 0 3\nv 0 0 3\nf 1 2 3\nf 1 3 4\n");

        MeshThumbnail yUp = engine.MeshThumbnails.Render(obj, "obj", new MeshThumbnailRequest { View = "front" });
        MeshThumbnail zUp = engine.MeshThumbnails.Render(obj, "obj", new MeshThumbnailRequest { View = "front", Up = "z" });

        Assert.NotEqual(yUp.Png, zUp.Png);
    }

    [Fact]
    public void ASplatPlyIsDrawnAsPointsAndSaysHowManyFaintSplatsWereLeftOut()
    {
        using InferenceEngine engine = new("cpu");
        string header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float opacity\nend_header\n";
        byte[] head = Encoding.ASCII.GetBytes(header);
        byte[] ply = new byte[head.Length + 3 * 7 * 4];
        head.CopyTo(ply, 0);
        float[] rows = [0, 0, 0, 1, 1, 1, 5, 1, 0, 0, 1, 1, 1, 5, 0, 1, 0, 1, 1, 1, -9];
        for (int i = 0; i < rows.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(ply.AsSpan(head.Length + i * 4), rows[i]);

        MeshThumbnail thumbnail = engine.MeshThumbnails.Render(ply, "ply");

        Assert.Equal("points", thumbnail.Kind);
        Assert.Contains(thumbnail.Notes, n => n.StartsWith("1 splat(s) fainter"));
    }

    [Fact]
    public void BadRequestsAreArgumentErrorsAndABadFileIsADomainError()
    {
        using InferenceEngine engine = new("cpu");

        Assert.Contains("front, back", Assert.Throws<ArgumentException>(() => engine.MeshThumbnails.Render(Glb(), "glb", new MeshThumbnailRequest { View = "sideways" })).Message);
        Assert.Throws<ArgumentException>(() => engine.MeshThumbnails.Render(Glb(), "glb", new MeshThumbnailRequest { Up = "x" }));
        Assert.Throws<ArgumentException>(() => engine.MeshThumbnails.Render(Glb(), "glb", new MeshThumbnailRequest { Background = "red" }));
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.MeshThumbnails.Render(Glb(), "glb", new MeshThumbnailRequest { Width = 4 }));
        Assert.Throws<NotSupportedException>(() => engine.MeshThumbnails.Render(Glb(), "fbx"));
        Assert.Throws<HartsyInferenceException>(() => engine.MeshThumbnails.Render([1, 2, 3, 4], "glb"));
        Assert.Equal(["glb", "obj", "ply"], engine.MeshThumbnails.Formats);
    }
}
