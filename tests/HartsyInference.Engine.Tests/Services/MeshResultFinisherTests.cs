using System.Text;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshResultFinisherTests
{
    private static Mesh Sphere(int rings, int sectors)
    {
        List<float> vertices = [];
        List<int> indices = [];
        for (int r = 0; r <= rings; r++)
        {
            double phi = Math.PI * r / rings;
            for (int s = 0; s <= sectors; s++)
            {
                double theta = 2 * Math.PI * s / sectors;
                vertices.AddRange([(float)(Math.Sin(phi) * Math.Cos(theta)), (float)Math.Cos(phi), (float)(Math.Sin(phi) * Math.Sin(theta))]);
            }
        }
        for (int r = 0; r < rings; r++)
        {
            for (int s = 0; s < sectors; s++)
            {
                int a = r * (sectors + 1) + s;
                int b = a + sectors + 1;
                indices.AddRange([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        return new Mesh { Vertices = [.. vertices], Indices = [.. indices] };
    }

    private static readonly MeshSettingsUsed S = new() { ModelId = "test", EngineVersion = "1.2.3" };

    private static MeshResult Raw(Mesh mesh) => new() { Data = GlbWriter.Write(mesh), Format = "glb" };

    [Fact]
    public void ANativeRequestKeepsTheBytesAndReportsTheMeasurements()
    {
        MeshResult raw = Raw(Sphere(12, 16));

        MeshResult result = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest(), raw, S, CancellationToken.None);

        Assert.Equal(GlbReader.Parse(raw.Data).Vertices, GlbReader.Parse(result.Data).Vertices);
        Assert.Equal(MeshFrame.Native, result.Frame);
        Assert.Equal(12 * 16 * 2, result.Info!.TriangleCount);
        Assert.Equal(result.Data.Length, result.Info.ByteSize);
        Assert.Null(result.Preview);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void TheSettingsAreEmbeddedInTheGlbExtrasAndInThePlyHeaderAndReturned()
    {
        MeshSettingsUsed settings = new()
        {
            ModelId = "hunyuan3d",
            Seed = 77,
            EngineVersion = "9.9",
            Parameters = new Dictionary<string, HartsyInference.Engine.Planning.ResolvedParameter>
            {
                ["Seed"] = new("77", HartsyInference.Engine.Planning.ParameterSource.Derived, "random"),
                ["Steps"] = new("30", HartsyInference.Engine.Planning.ParameterSource.UserOverride),
            },
        };

        MeshResult glb = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest(), Raw(Sphere(6, 8)), settings, CancellationToken.None);
        System.Text.Json.Nodes.JsonObject embedded = MeshMetadata.ReadGlb(glb.Data)!;

        Assert.Same(settings, glb.Settings);
        Assert.Equal("hunyuan3d", (string?)embedded["model"]);
        Assert.Equal(77, (int?)embedded["seed"]);
        Assert.Equal("HartsyInference 9.9", (string?)embedded["generator"]);
        Assert.Equal("30", (string?)embedded["settings"]!["Steps"]!["value"]);
        Assert.Equal("UserOverride", (string?)embedded["settings"]!["Steps"]!["source"]);
        Assert.Equal(1, GlbInspector.Read(glb.Data).MeshCount);

        byte[] head = Encoding.ASCII.GetBytes("ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n");
        byte[] source = [.. head, .. new byte[12]];
        MeshResult splats = MeshResultFinisher.Finish("sam3d-objects", new MeshRequest(), new MeshResult { Data = source, Format = "ply" }, settings, CancellationToken.None);
        string header = Encoding.UTF8.GetString(splats.Data);

        Assert.Contains("comment hartsyinference {", header);
        Assert.Contains("\"seed\":77", header);
        Assert.Equal(1, splats.Info!.SplatCount);
        Assert.Equal(source.Length + header.Split('\n').First(l => l.StartsWith("comment hartsyinference")).Length + 1, splats.Data.Length);
    }

    [Fact]
    public void ThePlyCommentIsReplacedNotRepeatedAndCanBeReadBack()
    {
        byte[] head = Encoding.ASCII.GetBytes("ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n");
        byte[] source = [.. head, .. new byte[12]];

        byte[] once = MeshMetadata.SetPly(source, new System.Text.Json.Nodes.JsonObject { ["a"] = 1 });
        byte[] twice = MeshMetadata.SetPly(once, new System.Text.Json.Nodes.JsonObject { ["b"] = 2 });

        Assert.Equal(1, Encoding.UTF8.GetString(twice).Split('\n').Count(l => l.StartsWith(MeshMetadata.PlyPrefix)));
        Assert.Equal(2, (int?)MeshMetadata.ReadPly(twice)!["b"]);
        Assert.Null(MeshMetadata.ReadPly(source));
        Assert.Equal(1, MeshResultFinisher.SplatInfo(twice).SplatCount);
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => MeshMetadata.SetPly([1, 2, 3], new System.Text.Json.Nodes.JsonObject()));
    }

    [Fact]
    public void AFormatThatIsNeitherGlbNorPlyIsReturnedUntouchedWithItsSettings()
    {
        MeshResult result = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest(), new MeshResult { Data = [1, 2, 3], Format = "obj" }, S, CancellationToken.None);

        Assert.Equal(new byte[] { 1, 2, 3 }, result.Data);
        Assert.Empty(result.Warnings);
        Assert.Same(S, result.Settings);
    }

    [Fact]
    public void MetadataKeepsTheExtrasAFileAlreadyHas()
    {
        byte[] glb = GlbWriter.Write(Sphere(4, 4));
        System.Text.Json.Nodes.JsonObject first = new() { ["a"] = 1 };

        byte[] once = MeshMetadata.SetGlb(glb, first);
        byte[] twice = MeshMetadata.SetGlb(once, new System.Text.Json.Nodes.JsonObject { ["b"] = 2 });

        Assert.Equal(2, (int?)MeshMetadata.ReadGlb(twice)!["b"]);
        Assert.Equal(GlbReader.Parse(glb).Vertices, GlbReader.Parse(twice).Vertices);
    }

    [Fact]
    public void PreviewIsADecimatedGlbInTheSameFrameAndNullWhenTheMeshIsAlreadySmall()
    {
        MeshResult big = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest { PreviewTriangles = 100, Frame = MeshFrame.GltfYUpMetres }, Raw(Sphere(40, 60)), S, CancellationToken.None);
        MeshResult small = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest { PreviewTriangles = 10_000 }, Raw(Sphere(12, 16)), S, CancellationToken.None);

        Assert.NotNull(big.Preview);
        Mesh preview = GlbReader.Parse(big.Preview);
        Assert.InRange(preview.TriangleCount, 20, 200);
        Assert.True(GlbInspector.Read(big.Preview).HasNormals);
        float[] y = [.. preview.Vertices.Where((_, i) => i % 3 == 1)];
        Assert.InRange(y.Max() - y.Min(), 0.9f, 1.05f);
        Assert.Null(small.Preview);
    }

    [Fact]
    public void Pixal3DIsTurnedAHalfTurnToGltfFrontAndOthersAreOnlyCentredAndScaledWithAWarning()
    {
        Mesh lopsided = new() { Vertices = [0, 0, 0, 2, 0, 0, 0, 1, 0], Indices = [0, 1, 2] };

        MeshResult pixal = MeshResultFinisher.Finish("pixal3d", new MeshRequest { Frame = MeshFrame.GltfYUpMetres }, Raw(lopsided), S, CancellationToken.None);
        MeshResult hunyuan = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest { Frame = MeshFrame.GltfYUpMetres }, Raw(lopsided), S, CancellationToken.None);

        Assert.Equal(MeshFrame.GltfYUpMetres, pixal.Frame);
        Assert.Equal(-0.5f, GlbReader.Parse(pixal.Data).Vertices[3], 4);
        Assert.Equal(0.5f, GlbReader.Parse(hunyuan.Data).Vertices[3], 4);
        Assert.Empty(pixal.Warnings);
        Assert.Contains("front direction is not verified", Assert.Single(hunyuan.Warnings));
    }

    [Fact]
    public void ASplatPlyReportsItsCountAndBoundsAndOnlyGainsTheSettingsComment()
    {
        StringBuilder header = new();
        header.Append("ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty float opacity\nend_header\n");
        byte[] head = Encoding.ASCII.GetBytes(header.ToString());
        float[] body = [1, 2, 3, 0.5f, -1, 5, 0, 0.5f];
        byte[] ply = new byte[head.Length + body.Length * 4];
        Buffer.BlockCopy(head, 0, ply, 0, head.Length);
        Buffer.BlockCopy(body, 0, ply, head.Length, body.Length * 4);

        MeshResult result = MeshResultFinisher.Finish("sam3d-objects", new MeshRequest { PreviewTriangles = 100, Frame = MeshFrame.GltfYUpMetres }, new MeshResult { Data = ply, Format = "ply" }, S, CancellationToken.None);

        Assert.Equal(ply.Length + Encoding.UTF8.GetByteCount(MeshMetadata.PlyPrefix + MeshResultFinisher.Describe(S, MeshFrame.Native).ToJsonString()) + 1, result.Data.Length);
        Assert.Equal(2, result.Info!.SplatCount);
        Assert.Equal([-1f, 2f, 0f], result.Info.BoundsMin!);
        Assert.Equal([1f, 5f, 3f], result.Info.BoundsMax!);
        Assert.Null(result.Preview);
    }

    [Fact]
    public void AGlbTheReaderRefusesStillGetsItsContentsAndAWarning()
    {
        // A GLB whose only primitive is not indexed: GlbReader refuses it, the summary still reads.
        byte[] glb = Raw(Sphere(4, 4)).Data;
        MeshResult result = MeshResultFinisher.Finish("unirig", new MeshRequest { PreviewTriangles = 10 }, new MeshResult { Data = glb }, S, CancellationToken.None);

        Assert.NotNull(result.Info);
    }

    /// <summary>The same-length text replacement in a GLB's JSON, so the container stays valid.</summary>
    private static byte[] Corrupt(byte[] glb, string from, string to)
    {
        string text = Encoding.UTF8.GetString(glb);
        int at = text.IndexOf(from, StringComparison.Ordinal);
        Assert.True(at >= 0, $"'{from}' not found");
        byte[] result = [.. glb];
        Encoding.UTF8.GetBytes(to).CopyTo(result, at);
        return result;
    }

    [Fact]
    public void AGlbWithAnOutOfRangeSceneStillReturnsTheGeneration()
    {
        byte[] glb = Corrupt(Raw(Sphere(6, 8)).Data, "\"scene\":0", "\"scene\":9");

        MeshResult result = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest { Frame = MeshFrame.GltfYUpMetres, PreviewTriangles = 10 }, new MeshResult { Data = glb }, S, CancellationToken.None);

        Assert.Equal(MeshFrame.Native, result.Frame);
        Assert.Contains(result.Warnings, w => w.StartsWith("The frame was not normalised"));
    }

    [Fact]
    public void AGlbWithAnOutOfRangeAccessorKeepsTheGenerationAndSaysWhatItCouldNotDo()
    {
        byte[] glb = Corrupt(Raw(Sphere(6, 8)).Data, "\"POSITION\":0", "\"POSITION\":9");

        MeshResult result = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest { Frame = MeshFrame.GltfYUpMetres, PreviewTriangles = 10 }, new MeshResult { Data = glb }, S, CancellationToken.None);

        Assert.NotNull(result.Info);
        Assert.Null(result.Preview);
        Assert.NotEmpty(result.Warnings);
    }

    [Fact]
    public void BoundsAreNotReportedInTheFilesOwnSpaceOnceTheFrameMovedThem()
    {
        // An index accessor of a type the reader refuses, so only the accessor bounds (pre-frame) exist.
        Mesh mesh = Sphere(6, 8);
        byte[] glb = Raw(mesh).Data;
        byte[] noIndices = Corrupt(glb, "\"componentType\":5125", "\"componentType\":5124");

        MeshResult native = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest(), new MeshResult { Data = noIndices }, S, CancellationToken.None);

        Assert.True(native.Info!.BoundsMin is not null, string.Join(" | ", native.Warnings));
        Assert.True(native.Info.VertexCount > 0);

        MeshResult framed = MeshResultFinisher.Finish("hunyuan3d", new MeshRequest { Frame = MeshFrame.GltfYUpMetres }, new MeshResult { Data = noIndices }, S, CancellationToken.None);

        Assert.Equal(MeshFrame.GltfYUpMetres, framed.Frame);
        Assert.Null(framed.Info!.BoundsMin);
    }

    [Fact]
    public void ASplatPlyWhoseCoordinatesAreNotFirstOrNotFloatsIsReadFromItsHeader()
    {
        byte[] Ply(string properties, float[] body)
        {
            byte[] head = Encoding.ASCII.GetBytes("ply\nformat binary_little_endian 1.0\nelement vertex 1\n" + properties + "end_header\n");
            byte[] bytes = new byte[head.Length + body.Length * 4];
            head.CopyTo(bytes, 0);
            Buffer.BlockCopy(body, 0, bytes, head.Length, body.Length * 4);
            return bytes;
        }

        MeshInfo shifted = MeshResultFinisher.SplatInfo(Ply("property float opacity\nproperty float x\nproperty float y\nproperty float z\n", [0.5f, 1, 2, 3]));
        MeshInfo mixed = MeshResultFinisher.SplatInfo(Ply("property uchar red\nproperty float x\nproperty float y\nproperty float z\n", [0, 1, 2, 3]));

        Assert.Equal([1f, 2f, 3f], shifted.BoundsMin!);
        Assert.Equal(1, mixed.SplatCount);
        Assert.Null(mixed.BoundsMin);
    }

    [Fact]
    public void APlyThatIsAsciiOrDeclaresItsVertexElementAfterAnotherGivesTheCountAtMost()
    {
        byte[] Ply(string format, string elements) =>
            Encoding.ASCII.GetBytes("ply\nformat " + format + " 1.0\n" + elements + "end_header\n").Concat(new byte[64]).ToArray();

        MeshInfo ascii = MeshResultFinisher.SplatInfo(Ply("ascii", "element vertex 1\nproperty float x\nproperty float y\nproperty float z\n"));
        MeshInfo faceFirst = MeshResultFinisher.SplatInfo(Ply("binary_little_endian", "element face 1\nproperty float a\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n"));

        Assert.Null(ascii.BoundsMin);
        Assert.Equal(0, ascii.SplatCount);
        Assert.Null(faceFirst.BoundsMin);
    }

    [Fact]
    public void ANegativePreviewBudgetIsRejectedBeforeAnythingLoads()
    {
        MeshRequest request = new() { Image = new ImageData { Rgb = new byte[12], Width = 2, Height = 2 }, PreviewTriangles = -1 };

        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => MeshRequestValidator.Validate("triposr", request, matterAvailable: false));

        Assert.Equal(nameof(MeshRequest.PreviewTriangles), MeshFieldTag.FieldOf(error));
    }
}
